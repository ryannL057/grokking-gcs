#!/usr/bin/env python3
"""
Shuffled-reference negative control for GCS.

The main protocol builds a final reference geometry and keeps the pairing
between each reference point, its prompt, and its embedding. This control keeps
the final representation geometry, k-NN graph, and tangent bases fixed, but
randomly permutes the prompt/embedding assigned to each reference point. It
therefore preserves the marginal final geometry while destroying the
input-to-solution-frame pairing.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import io
import pickle
import re
import sys
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Imports use the installed package.

from gcs.legacy.reanalyze_gcs import load_checkpoint
from gcs.legacy.recompute_gcs_existing_geometry import (
    compute_three_paths,
    evenly_subsample_epochs,
    infer_num_layers,
    load_existing_result,
    target_layers_for_recompute,
)


PATH_KEYS = ("overall_path", "attention_path", "mlp_path")


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_pickle(path: Path) -> Any:
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
            return super().find_class(module, name)

    with open(path, "rb") as handle:
        return CPUUnpickler(handle).load()


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def load_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_selected_epochs(exp_dir: Path, max_checkpoints: int | None) -> list[int]:
    _, _, epochs, _ = load_existing_result(exp_dir)
    checkpoint_map = {checkpoint_epoch(path): path for path in (exp_dir / "checkpoints").glob("*.pth")}
    selected = [int(epoch) for epoch in epochs if int(epoch) in checkpoint_map]
    return evenly_subsample_epochs(selected, max_checkpoints)


def load_reference_geometry(exp_dir: Path, target_layer: int) -> dict[str, Any]:
    candidates = [exp_dir / f"final_geometry_layer{target_layer}.pkl"]
    if target_layer == 0:
        candidates.append(exp_dir / "final_geometry.pkl")
    candidates.append(exp_dir / "final_geometry.pkl")
    for path in candidates:
        if path.exists():
            return load_pickle(path)
    raise FileNotFoundError(f"No final geometry found for {exp_dir} layer {target_layer}")


def shuffled_pairing_geometry(geometry: dict[str, Any], seed: int) -> dict[str, Any]:
    """Keep geometry fixed, but permute prompts and embeddings across points."""
    n = int(geometry["N"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    shuffled = copy(geometry)
    shuffled["X_prompts"] = geometry["X_prompts"][perm].clone()
    shuffled["X_embeddings"] = [geometry["X_embeddings"][int(idx)].clone() for idx in perm]
    shuffled["pairing_permutation"] = perm.tolist()
    shuffled["control_type"] = "shuffled_prompt_embedding_pairing"
    return shuffled


def output_path(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / f"{row['activation']}_{row['task']}_s{row['seed']}_shuffled_reference.pkl"


def init_payload(row: dict[str, str], exp_dir: Path, selected_epochs: list[int], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "title": f"Shuffled-reference GCS control: {row['activation']}/{row['task']}/s{row['seed']}",
        "experiment_dir": str(exp_dir.relative_to(ROOT) if exp_dir.is_relative_to(ROOT) else exp_dir),
        "activation": row["activation"],
        "task": row["task"],
        "seed": int(row["seed"]),
        "geometry_type": "final_reference_with_shuffled_prompt_embedding_pairing",
        "selected_epochs": selected_epochs,
        "epochs": [],
        "path_gcs_scores": {key: [] for key in PATH_KEYS},
        "analysis_params": {
            "max_checkpoints": args.max_checkpoints,
            "token_probe": args.token_probe,
            "shuffle_seed_offset": args.shuffle_seed_offset,
        },
    }


def load_partial(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = load_pickle(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_complete(payload: dict[str, Any], selected_epochs: list[int]) -> bool:
    return list(map(int, payload.get("epochs", []))) == list(map(int, selected_epochs))


def run_one(row: dict[str, str], args: argparse.Namespace) -> Path:
    exp_dir = repo_path(row["exp_dir"])
    selected_epochs = expected_selected_epochs(exp_dir, args.max_checkpoints)
    out_path = output_path(args.output_dir, row)
    payload = load_partial(out_path)
    if payload is not None and is_complete(payload, selected_epochs):
        print(f"SKIP complete {out_path}")
        return out_path
    if payload is None:
        payload = init_payload(row, exp_dir, selected_epochs, args)

    done_epochs = set(map(int, payload.get("epochs", [])))
    checkpoint_map = {checkpoint_epoch(path): path for path in (exp_dir / "checkpoints").glob("*.pth")}
    num_layers = infer_num_layers(exp_dir, device=args.device)
    target_layers = target_layers_for_recompute(num_layers)
    target_layer = int(row.get("target_layer", target_layers[-1]) or target_layers[-1])

    geometry = load_reference_geometry(exp_dir, target_layer)
    shuffle_seed = int(row["seed"]) + args.shuffle_seed_offset
    geometry = shuffled_pairing_geometry(geometry, shuffle_seed)

    missing = [epoch for epoch in selected_epochs if int(epoch) not in done_epochs]
    label = f"{row['activation']}/{row['task']}/s{row['seed']} shuffled"
    for epoch in tqdm(missing, desc=label):
        model, _ = load_checkpoint(checkpoint_map[int(epoch)], device=args.device)
        scores = compute_three_paths(model, geometry, target_layer, args.device, args.token_probe)
        payload["epochs"].append(int(epoch))
        for key in PATH_KEYS:
            payload["path_gcs_scores"].setdefault(key, []).append(float(scores[key]))
        save_pickle(out_path, payload)
        del model
        torch.cuda.empty_cache()

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("scripts/manifests/current_geometry_subset_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/shuffled_reference_control_20260428"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-checkpoints", type=int, default=101)
    parser.add_argument("--token-probe", choices=("final", "operands", "all"), default="final")
    parser.add_argument("--shuffle-seed-offset", type=int, default=50_000)
    parser.add_argument("--limit-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = repo_path(args.manifest)
    args.output_dir = repo_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    rows = load_manifest(args.manifest)
    if args.limit_runs is not None:
        rows = rows[: args.limit_runs]
    print(f"Shuffled-reference control: runs={len(rows)} max_ckpt={args.max_checkpoints} device={args.device}")
    if args.dry_run:
        for row in rows:
            print(row["activation"], row["task"], row["seed"], repo_path(row["exp_dir"]))
        return
    outputs = [run_one(row, args) for row in rows]
    manifest_path = args.output_dir / "outputs.txt"
    manifest_path.write_text("\n".join(str(path.relative_to(ROOT)) for path in outputs) + "\n")
    print(f"Saved output manifest: {manifest_path}")


if __name__ == "__main__":
    main()
