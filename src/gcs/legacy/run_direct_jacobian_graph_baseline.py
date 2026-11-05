#!/usr/bin/env python3
"""Compute the frozen direct-Jacobian graph baseline on registered 1L runs."""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import hashlib
import io
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Imports use the installed package.

from gcs.analysis.direct_jacobian_graph import (  # noqa: E402
    compute_batched_jacobian_sketch,
    make_fixed_random_directions,
    summarize_direct_jacobian_graph,
)
from gcs.analysis.paths import create_path_forward_function  # noqa: E402
from gcs.legacy.diagnostics_common import RUNS_1L, load_gcs, resolve_path  # noqa: E402
from gcs.legacy.reanalyze_gcs import load_checkpoint  # noqa: E402


PATH_CONFIGS = {
    "attention_path": ("hook_resid_pre", "hook_attn_out"),
    "mlp_path": ("hook_resid_mid", "hook_mlp_out"),
    "overall_path": ("hook_resid_pre", "hook_resid_post"),
}


def load_pickle_cpu(path: Path) -> Any:
    """Load geometry snapshots even when they were serialized on another GPU."""

    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda data: torch.load(
                    io.BytesIO(data),
                    map_location="cpu",
                    weights_only=False,
                )
            return super().find_class(module, name)

    with open(path, "rb") as handle:
        return CPUUnpickler(handle).load()


def save_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle)
    tmp_path.replace(path)


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def run_slug(act: str, task: str, seed: int) -> str:
    return f"{act.lower()}_{task}_s{int(seed)}"


def evenly_subsample(values: list[int], max_items: int | None) -> list[int]:
    if max_items is None or len(values) <= max_items:
        return values
    if max_items <= 1:
        return [values[-1]]
    indices = np.linspace(0, len(values) - 1, num=max_items, dtype=int)
    return [values[int(index)] for index in indices]


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def output_is_complete(
    path: Path,
    *,
    target_epochs: list[int],
    paths: list[str],
    expected_params: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_pickle_cpu(path)
    except Exception:
        return False
    if not bool(payload.get("complete", False)):
        return False
    if [int(epoch) for epoch in payload.get("target_epochs", [])] != target_epochs:
        return False
    if [int(epoch) for epoch in payload.get("epochs", [])] != target_epochs:
        return False
    if list(payload.get("paths", [])) != paths:
        return False
    params = payload.get("analysis_params", {})
    if not all(params.get(key) == value for key, value in expected_params.items()):
        return False
    expected_length = len(target_epochs)
    path_scores = payload.get("path_scores", {})
    path_summaries = payload.get("path_summaries", {})
    return all(
        len(path_scores.get(path_name, [])) == expected_length
        and len(path_summaries.get(path_name, [])) == expected_length
        for path_name in paths
    )


def initialize_or_resume(
    output_path: Path,
    *,
    metadata: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        payload = load_pickle_cpu(output_path)
        expected_epochs = list(metadata["target_epochs"])
        processed = [int(epoch) for epoch in payload.get("epochs", [])]
        if processed != expected_epochs[: len(processed)]:
            raise RuntimeError(
                f"Existing partial output has a non-prefix epoch schedule: {output_path}"
            )
        for key in (
            "act",
            "task",
            "seed",
            "source_exp_dir",
            "geometry_path",
            "prompt_sha256",
            "target_epochs",
            "full_saved_epoch_count",
            "paths",
            "analysis_params",
        ):
            if payload.get(key) != metadata.get(key):
                raise RuntimeError(
                    f"Existing output metadata mismatch for {key}: {output_path}"
                )
        return payload

    return {
        **metadata,
        "schema_version": 1,
        "complete": False,
        "epochs": [],
        "path_scores": {path: [] for path in metadata["paths"]},
        "path_summaries": {path: [] for path in metadata["paths"]},
        "checkpoint_elapsed_seconds": [],
    }


def compute_checkpoint(
    checkpoint_path: Path,
    *,
    prompts: torch.Tensor,
    paths: list[str],
    direction_count: int,
    direction_seed: int,
    graph_k: int,
    target_layer: int,
    device: str,
) -> tuple[dict[str, float], dict[str, dict[str, Any]], str]:
    model, _ = load_checkpoint(checkpoint_path, device=device)
    dtype = next(model.parameters()).dtype
    prompts_device = prompts.to(device=device, dtype=torch.int64)
    with torch.no_grad():
        embeddings = model.embed(prompts_device).detach()

    directions = make_fixed_random_directions(
        count=direction_count,
        dimension=int(embeddings.shape[-1]),
        seed=direction_seed,
        dtype=dtype,
        device=device,
    )
    directions_hash = tensor_sha256(
        make_fixed_random_directions(
            count=direction_count,
            dimension=int(embeddings.shape[-1]),
            seed=direction_seed,
            dtype=torch.float64,
            device="cpu",
        )
    )

    scores: dict[str, float] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for path_name in paths:
        start_hook, end_hook = PATH_CONFIGS[path_name]
        path_forward = create_path_forward_function(
            model,
            start_hook,
            end_hook,
            target_layer=target_layer,
        )
        sketches = compute_batched_jacobian_sketch(
            path_forward,
            embeddings,
            directions,
            probe_position=-1,
        )
        summary = summarize_direct_jacobian_graph(sketches, k=graph_k)
        score = float(summary["direct_jacobian_score"])
        scores[path_name] = score
        summaries[path_name] = {
            key: value
            for key, value in summary.items()
            if key not in {"coherence_matrix", "similarity_matrix", "knn_indices"}
        }

    del model, embeddings, directions
    return scores, summaries, directions_hash


def run_one(
    act: str,
    task: str,
    seed: int,
    base: str,
    *,
    output_dir: Path,
    paths: list[str],
    direction_count: int,
    direction_seed: int,
    graph_k: int,
    expected_n: int,
    target_layer: int,
    max_checkpoints: int | None,
    save_every: int,
    device: str,
    overwrite: bool,
) -> dict[str, Any]:
    exp_dir = resolve_path(base)
    geometry_path = exp_dir / "final_geometry.pkl"
    if not geometry_path.exists():
        raise FileNotFoundError(f"Missing saved input geometry: {geometry_path}")
    geometry = load_pickle_cpu(geometry_path)
    prompts = geometry["X_prompts"].detach().to(device="cpu", dtype=torch.int64)
    if int(prompts.shape[0]) != expected_n:
        raise ValueError(
            f"Expected N={expected_n}, got {prompts.shape[0]} in {geometry_path}"
        )

    saved_epochs, _ = load_gcs(base)
    checkpoint_map = {
        checkpoint_epoch(path): path
        for path in (exp_dir / "checkpoints").glob("*.pth")
    }
    full_target_epochs = [int(epoch) for epoch in saved_epochs]
    missing = [epoch for epoch in full_target_epochs if epoch not in checkpoint_map]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} saved epochs lack checkpoints in {exp_dir}: {missing[:5]}"
        )
    target_epochs = evenly_subsample(full_target_epochs, max_checkpoints)

    analysis_params = {
        "N": expected_n,
        "graph_k": graph_k,
        "direction_count": direction_count,
        "direction_seed": direction_seed,
        "probe_position": "final_token",
        "embedding_evaluation": "current_checkpoint",
        "input_prompt_source": "saved_final_geometry_prompts_only",
        "similarity": "absolute_cosine",
        "sparsification": "directed_topk_union",
        "self_loop": 1.0,
        "spectral_aggregation": "submitted_entropy_N_minus_2_pow_H",
        "target_layer": target_layer,
        "max_checkpoints": max_checkpoints,
    }
    expected_params = dict(analysis_params)
    out_path = output_dir / "trajectories" / f"{run_slug(act, task, seed)}.pkl"
    if output_is_complete(
        out_path,
        target_epochs=target_epochs,
        paths=paths,
        expected_params=expected_params,
    ) and not overwrite:
        return {
            "act": act,
            "task": task,
            "seed": int(seed),
            "exp_dir": str(exp_dir),
            "output": str(out_path),
            "status": "skipped_complete",
            "n_epochs": len(target_epochs),
        }

    metadata = {
        "act": act,
        "task": task,
        "seed": int(seed),
        "source_exp_dir": str(exp_dir),
        "geometry_path": str(geometry_path),
        "prompt_sha256": tensor_sha256(prompts),
        "target_epochs": target_epochs,
        "full_saved_epoch_count": len(full_target_epochs),
        "paths": paths,
        "analysis_params": analysis_params,
    }
    payload = initialize_or_resume(out_path, metadata=metadata, overwrite=overwrite)
    start_index = len(payload["epochs"])

    progress = tqdm(
        target_epochs[start_index:],
        desc=f"{run_slug(act, task, seed)} direct-J",
        leave=False,
    )
    for offset, epoch in enumerate(progress, start=start_index):
        started = time.perf_counter()
        scores, summaries, directions_hash = compute_checkpoint(
            checkpoint_map[epoch],
            prompts=prompts,
            paths=paths,
            direction_count=direction_count,
            direction_seed=direction_seed,
            graph_k=graph_k,
            target_layer=target_layer,
            device=device,
        )
        payload["epochs"].append(int(epoch))
        payload["directions_sha256"] = directions_hash
        for path_name in paths:
            payload["path_scores"][path_name].append(scores[path_name])
            payload["path_summaries"][path_name].append(summaries[path_name])
        elapsed = time.perf_counter() - started
        payload["checkpoint_elapsed_seconds"].append(float(elapsed))
        progress.set_postfix(seconds=f"{elapsed:.2f}")

        if (offset + 1) % max(save_every, 1) == 0:
            save_pickle_atomic(out_path, payload)

    payload["complete"] = len(payload["epochs"]) == len(target_epochs)
    payload["total_elapsed_seconds"] = float(
        sum(payload["checkpoint_elapsed_seconds"])
    )
    save_pickle_atomic(out_path, payload)
    return {
        "act": act,
        "task": task,
        "seed": int(seed),
        "exp_dir": str(exp_dir),
        "output": str(out_path),
        "status": "completed",
        "n_epochs": len(target_epochs),
        "elapsed_seconds": payload["total_elapsed_seconds"],
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the frozen direct-Jacobian graph baseline."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/rebuttal_direct_jacobian_graph_20260723"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="123,777")
    parser.add_argument("--paths", default="attention_path,mlp_path,overall_path")
    parser.add_argument("--direction-count", type=int, default=8)
    parser.add_argument("--direction-seed", type=int, default=20260723)
    parser.add_argument("--graph-k", type=int, default=15)
    parser.add_argument("--expected-N", type=int, default=200)
    parser.add_argument("--target-layer", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)

    seeds = {int(value) for value in args.seeds.split(",") if value.strip()}
    paths = [value.strip() for value in args.paths.split(",") if value.strip()]
    unknown_paths = [path for path in paths if path not in PATH_CONFIGS]
    if unknown_paths:
        raise ValueError(f"Unknown paths: {unknown_paths}")

    runs = [run for run in RUNS_1L if int(run[2]) in seeds]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    if not runs:
        raise SystemExit("No registered runs matched --seeds.")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "frozen_protocol": str(
            ROOT / "rebuttal_2026" / "DIRECT_JACOBIAN_GRAPH_PROTOCOL_20260723.md"
        ),
        "device": args.device,
        "seeds": sorted(seeds),
        "paths": paths,
        "direction_count": args.direction_count,
        "direction_seed": args.direction_seed,
        "graph_k": args.graph_k,
        "expected_N": args.expected_N,
        "target_layer": args.target_layer,
        "max_runs": args.max_runs,
        "max_checkpoints": args.max_checkpoints,
        "registered_run_count": len(runs),
    }
    with open(output_dir / "run_config.json", "w") as handle:
        json.dump(run_config, handle, indent=2)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(
        f"Direct-Jacobian baseline: runs={len(runs)} device={args.device} "
        f"d={args.direction_count} k={args.graph_k} paths={','.join(paths)}"
    )
    for index, (act, task, seed, base) in enumerate(runs, start=1):
        label = f"[{index}/{len(runs)}] {act}/{task}/s{seed}"
        print(f"{label} START", flush=True)
        try:
            row = run_one(
                act,
                task,
                seed,
                base,
                output_dir=output_dir,
                paths=paths,
                direction_count=args.direction_count,
                direction_seed=args.direction_seed,
                graph_k=args.graph_k,
                expected_n=args.expected_N,
                target_layer=args.target_layer,
                max_checkpoints=args.max_checkpoints,
                save_every=args.save_every,
                device=args.device,
                overwrite=args.overwrite,
            )
            rows.append(row)
            print(
                f"{label} {row['status'].upper()} epochs={row['n_epochs']} "
                f"elapsed={row.get('elapsed_seconds', 0.0):.1f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "act": act,
                "task": task,
                "seed": int(seed),
                "exp_dir": str(resolve_path(base)),
                "status": "failed",
                "error": repr(exc),
            }
            failures.append(failure)
            rows.append(failure)
            print(f"{label} FAILED: {exc!r}", flush=True)

        write_manifest(output_dir / "run_manifest.csv", rows)
        with open(output_dir / "failures.json", "w") as handle:
            json.dump(failures, handle, indent=2)

    if failures:
        raise SystemExit(f"{len(failures)} run(s) failed; see {output_dir / 'failures.json'}")
    print(f"All {len(rows)} registered runs complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
