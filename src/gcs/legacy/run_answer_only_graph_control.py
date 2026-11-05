#!/usr/bin/env python3
"""Balanced answer-only reference graph control for attention-GCS triage.

This control asks whether the GCS trajectory is explained by answer-class
membership alone.  For each addition run, it builds a small answer-balanced
reference sample from the training set, then recomputes the attention-path GCS
trajectory under three graphs on the same prompts:

  - final_knn: k-NN in the final block-output representation;
  - final_prediction_graph: neighbors share the final model's predicted answer;
  - answer_graph: neighbors are the other prompts with the same answer;
  - random_graph: random non-self neighbors.

No training is performed.  The control is deliberately small so it can be run
late in paper polishing without changing the main experimental contract.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import copy
import io
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis.geometry import build_knn_graph, compute_tangent_space  # noqa: E402
from gcs.analysis.gmn_tsa import compute_path_gmn_tsa  # noqa: E402
from gcs.core.data import create_data  # noqa: E402
from gcs.legacy.diagnostics_common import (  # noqa: E402
    RUNS_1L,
    detect_grok_epoch,
    load_metrics,
    resolve_path,
)
from gcs.legacy.evaluate_workflow_selection_heldout import (  # noqa: E402
    build_targets,
    load_equivariance,
    score_window,
    select_first_prominent_post_valley,
    window_bounds,
)
from gcs.legacy.recompute_gcs_existing_geometry import (  # noqa: E402
    cast_geometry,
    checkpoint_epoch,
    infer_config,
)
from gcs.legacy.reanalyze_gcs import load_checkpoint  # noqa: E402


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def task_answer(prompts: torch.Tensor, p: int, task: str) -> np.ndarray:
    a = prompts[:, 0].detach().cpu().numpy().astype(int)
    b = prompts[:, 1].detach().cpu().numpy().astype(int)
    if task == "add":
        return (a + b) % p
    if task == "sub":
        return (a - b) % p
    if task == "mul":
        return (a * b) % p
    if task == "div":
        out = np.zeros_like(a)
        for idx, (aa, bb) in enumerate(zip(a, b)):
            out[idx] = 0 if bb == 0 else (int(aa) * pow(int(bb), p - 2, p)) % p
        return out
    raise ValueError(f"Unknown task: {task}")


def final_checkpoint(exp_dir: Path) -> Path:
    ckpts = sorted((exp_dir / "checkpoints").glob("*.pth"), key=checkpoint_epoch)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {exp_dir / 'checkpoints'}")
    return ckpts[-1]


def selected_checkpoints(exp_dir: Path, max_checkpoints: int | None) -> list[tuple[int, Path]]:
    ckpts = sorted((exp_dir / "checkpoints").glob("*.pth"), key=checkpoint_epoch)
    pairs = [(checkpoint_epoch(path), path) for path in ckpts]
    pairs = [(epoch, path) for epoch, path in pairs if epoch >= 0]
    if max_checkpoints is None or len(pairs) <= max_checkpoints:
        return pairs
    idx = np.linspace(0, len(pairs) - 1, num=max_checkpoints, dtype=int)
    return [pairs[int(i)] for i in idx]


def balanced_prompts(train_data: Any, p: int, task: str, per_answer: int, seed: int) -> torch.Tensor:
    groups: dict[int, list[torch.Tensor]] = defaultdict(list)
    for idx in range(len(train_data)):
        prompt, _ = train_data[idx]
        ans = int(task_answer(prompt.unsqueeze(0), p, task)[0])
        groups[ans].append(prompt.detach().cpu())

    rng = np.random.default_rng(seed)
    prompts: list[torch.Tensor] = []
    missing: list[int] = []
    for ans in range(p):
        candidates = groups.get(ans, [])
        if len(candidates) < per_answer:
            missing.append(ans)
            continue
        chosen = rng.choice(len(candidates), size=per_answer, replace=False)
        prompts.extend(candidates[int(i)] for i in chosen)
    if missing:
        raise ValueError(f"Not enough training prompts for answers: {missing[:8]}...")
    return torch.stack(prompts, dim=0)


def extract_final_geometry(
    model: Any,
    prompts_cpu: torch.Tensor,
    *,
    device: str,
    layer_idx: int,
    k: int,
    d: int,
) -> dict[str, Any]:
    prompts = prompts_cpu.to(device=device).to(torch.float64)
    with torch.no_grad():
        cache: dict[str, torch.Tensor] = {}
        model.cache_all(cache, incl_bwd=False)
        _ = model(prompts.to(torch.int64))
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        if hook_name not in cache:
            raise KeyError(f"Missing hook {hook_name}")
        X_sample = cache[hook_name][:, -1, :].detach()
        model.remove_all_hooks()

    knn_indices, knn_distances = build_knn_graph(X_sample, k=k)
    tangents = [compute_tangent_space(i, X_sample, knn_indices, d) for i in range(X_sample.shape[0])]

    X_embeddings = []
    with torch.no_grad():
        for prompt in prompts.to(torch.int64):
            X_embeddings.append(model.embed(prompt.unsqueeze(0)).squeeze(0).detach())

    return {
        "X_sample": X_sample,
        "X_prompts": prompts,
        "knn_indices": knn_indices,
        "knn_distances": knn_distances,
        "N": int(X_sample.shape[0]),
        "k": int(k),
        "d": int(d),
        "d_model": int(X_sample.shape[1]),
        "X_embeddings": X_embeddings,
        "all_input_tangent_bases": tangents,
        "layer_idx": int(layer_idx),
        "hook_point": "hook_resid_post",
        "model_type": "transformer",
        "embedding_type": None,
        "control_type": "final_knn",
    }


def answer_graph_indices(answers: np.ndarray, per_answer: int) -> np.ndarray:
    # balanced_prompts orders answers in blocks, but use answers explicitly to
    # keep the helper robust if the prompt order changes.
    n = len(answers)
    k = per_answer - 1
    graph = np.zeros((n, k), dtype=np.int64)
    for i, ans in enumerate(answers):
        peers = np.where(answers == ans)[0]
        peers = peers[peers != i]
        if len(peers) != k:
            raise ValueError(f"Expected {k} same-answer peers, got {len(peers)}")
        graph[i] = peers
    return graph


def class_graph_indices(classes: np.ndarray, k: int) -> np.ndarray:
    """Deterministic same-class graph for predicted endpoint classes."""
    n = len(classes)
    graph = np.zeros((n, k), dtype=np.int64)
    for i, cls in enumerate(classes):
        peers = np.where(classes == cls)[0]
        peers = peers[peers != i]
        if len(peers) < k:
            raise ValueError(f"Class {cls} has only {len(peers)} peers; need {k}")
        graph[i] = peers[:k]
    return graph


def random_graph_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    graph = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        candidates = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        graph[i] = rng.choice(candidates, size=k, replace=False)
    return graph


def final_predictions(model: Any, prompts_cpu: torch.Tensor, p: int, *, device: str) -> np.ndarray:
    with torch.no_grad():
        logits = model(prompts_cpu.to(device=device, dtype=torch.long))[:, -1, :p]
        return logits.argmax(dim=-1).detach().cpu().numpy().astype(int)


def geometry_with_graph(base: dict[str, Any], graph: np.ndarray, control_type: str) -> dict[str, Any]:
    geo = copy.copy(base)
    X_sample = base["X_sample"]
    d = int(base["d"])
    tangents = [compute_tangent_space(i, X_sample, graph, d) for i in range(X_sample.shape[0])]
    geo["knn_indices"] = graph
    geo["knn_distances"] = np.zeros_like(graph, dtype=np.float64)
    geo["all_input_tangent_bases"] = tangents
    geo["control_type"] = control_type
    return geo


def trajectory_correlation(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    rho, _ = stats.spearmanr(a[mask], b[mask])
    return float(rho)


def compute_attention_trajectory(
    exp_dir: Path,
    geometry: dict[str, Any],
    ckpts: list[tuple[int, Path]],
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    epochs: list[int] = []
    values: list[float] = []
    for epoch, ckpt_path in tqdm(ckpts, desc=f"{exp_dir.name}:{geometry['control_type']}", leave=False):
        model, _ = load_checkpoint(ckpt_path, device=device)
        dtype = next(model.parameters()).dtype
        geo = cast_geometry(geometry, device=device, dtype=dtype)
        value = compute_path_gmn_tsa(
            model,
            geo,
            "hook_resid_pre",
            "hook_attn_out",
            target_layer=int(geo.get("layer_idx", 0)),
            device=device,
            token_probe="final",
            alignment_method="cosine",
        )
        epochs.append(int(epoch))
        values.append(float(value))
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return np.asarray(epochs, dtype=int), np.asarray(values, dtype=float)


def evaluate_variant(
    *,
    act: str,
    task: str,
    seed: int,
    exp_dir: Path,
    variant: str,
    epochs: np.ndarray,
    series: np.ndarray,
    final_series: np.ndarray,
    metrics: dict[str, Any],
    equivariance: dict[Any, Any],
    window_len: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = build_targets(act, task, seed, exp_dir, metrics, epochs, equivariance)
    center = select_first_prominent_post_valley(series, window_len)
    start, end = window_bounds(center, len(series), window_len)
    rho = trajectory_correlation(series, final_series)
    for target in targets:
        if target["target"] not in {"test_accuracy_rise", "attention_equivariance_drop"}:
            continue
        scored = score_window(target["values"], window_len, start, target["mode"])
        rows.append({
            "activation": act,
            "task": task,
            "seed": int(seed),
            "variant": variant,
            "target": target["target"],
            "n_epochs": int(len(epochs)),
            "window_len": int(window_len),
            "selected_start_epoch": int(epochs[start]),
            "selected_end_epoch": int(epochs[end]),
            "selected_center_epoch": int(epochs[center]),
            "percentile": float(scored["percentile"]),
            "top20_hit": bool(float(scored["percentile"]) >= 0.8),
            "selected_score": float(scored["selected_score"]),
            "best_score": float(scored["best_score"]),
            "rho_vs_final_knn": float(rho),
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), str(row["target"]))].append(row)
    out: list[dict[str, Any]] = []
    for (variant, target), items in sorted(groups.items()):
        percentiles = np.asarray([float(r["percentile"]) for r in items], dtype=float)
        rhos = np.asarray([float(r["rho_vs_final_knn"]) for r in items], dtype=float)
        out.append({
            "variant": variant,
            "target": target,
            "n_runs": len({(r["activation"], r["task"], int(r["seed"])) for r in items}),
            "median_percentile": float(np.median(percentiles)),
            "top20_hit_rate": float(np.mean(percentiles >= 0.8)),
            "median_rho_vs_final_knn": float(np.nanmedian(rhos)),
            "mean_rho_vs_final_knn": float(np.nanmean(rhos)),
        })
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/answer_only_graph_control_20260505"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activations", default="ReLU,GeLU,SiLU")
    parser.add_argument("--seeds", default="42,123,777")
    parser.add_argument("--limit-runs", type=int, default=5)
    parser.add_argument("--max-checkpoints", type=int, default=31)
    parser.add_argument("--per-answer", type=int, default=3)
    parser.add_argument("--window-len", type=int, default=5)
    parser.add_argument("--equivariance", type=Path, default=Path("experiments/equivariance_analysis/equivariance_all_activations.pkl"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    torch.set_default_dtype(torch.float64)

    acts = {item.strip() for item in args.activations.split(",") if item.strip()}
    seeds = {int(item.strip()) for item in args.seeds.split(",") if item.strip()}
    candidate_runs = [
        (act, task, int(seed), base)
        for act, task, seed, base in RUNS_1L
        if task == "add" and act in acts and int(seed) in seeds
    ]
    candidate_runs.sort(key=lambda row: (row[0], row[2]))
    selected_runs: list[tuple[str, str, int, str]] = []
    for act, task, seed, base in candidate_runs:
        exp_dir = resolve_path(base)
        if not (exp_dir / "checkpoints").exists():
            continue
        try:
            metrics = load_metrics(base)
            if detect_grok_epoch(metrics) is None:
                continue
        except Exception:
            continue
        selected_runs.append((act, task, seed, base))
        if args.limit_runs and len(selected_runs) >= args.limit_runs:
            break

    equivariance = load_equivariance(resolve_path(args.equivariance))
    all_rows: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, int, str], dict[str, Any]] = {}

    print(
        f"Answer-only graph control: runs={len(selected_runs)} "
        f"per_answer={args.per_answer} k=d={args.per_answer - 1} "
        f"max_ckpt={args.max_checkpoints} device={args.device}"
    )
    for act, task, seed, base in selected_runs:
        exp_dir = resolve_path(base)
        metrics = load_metrics(base)
        cfg = infer_config(exp_dir)
        p = int(cfg.get("p", 113))
        fraction = float(cfg.get("fraction", 0.3))
        train_seed = int(cfg.get("seed", seed))
        train_data, _ = create_data(p=p, fraction=fraction, seed=train_seed, device=args.device, task=task)
        prompts_cpu = balanced_prompts(
            train_data,
            p=p,
            task=task,
            per_answer=args.per_answer,
            seed=100_000 + int(seed),
        )

        final_model, _ = load_checkpoint(final_checkpoint(exp_dir), device=args.device)
        k = args.per_answer - 1
        base_geo = extract_final_geometry(
            final_model,
            prompts_cpu,
            device=args.device,
            layer_idx=0,
            k=k,
            d=k,
        )
        answers = task_answer(base_geo["X_prompts"], p, task)
        preds = final_predictions(final_model, base_geo["X_prompts"], p, device=args.device)
        pred_geo = geometry_with_graph(
            base_geo,
            class_graph_indices(preds, k),
            "final_prediction_graph",
        )
        answer_geo = geometry_with_graph(
            base_geo,
            answer_graph_indices(answers, args.per_answer),
            "answer_graph",
        )
        random_geo = geometry_with_graph(
            base_geo,
            random_graph_indices(int(base_geo["N"]), k, seed=200_000 + int(seed)),
            "random_graph",
        )
        del final_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        ckpts = selected_checkpoints(exp_dir, args.max_checkpoints)
        expected_epochs = np.asarray([epoch for epoch, _ in ckpts], dtype=int)
        run_cache = args.output_dir / f"{act}_add_s{seed}_trajectories.pkl"
        if run_cache.exists() and not args.force:
            payload = load_pickle(run_cache)
        else:
            payload = {}
        for variant, geo in (
            ("final_knn", base_geo),
            ("final_prediction_graph", pred_geo),
            ("answer_graph", answer_geo),
            ("random_graph", random_geo),
        ):
            needs_compute = variant not in payload or args.force
            if not needs_compute:
                cached_epochs = np.asarray(payload[variant].get("epochs", []), dtype=int)
                needs_compute = not np.array_equal(cached_epochs, expected_epochs)
            if needs_compute:
                epochs, series = compute_attention_trajectory(exp_dir, geo, ckpts, device=args.device)
                payload[variant] = {"epochs": epochs, "attention_gcs": series}
        save_pickle(run_cache, payload)

        final_epochs = np.asarray(payload["final_knn"]["epochs"], dtype=int)
        final_series = np.asarray(payload["final_knn"]["attention_gcs"], dtype=float)
        for variant, item in payload.items():
            epochs = np.asarray(item["epochs"], dtype=int)
            series = np.asarray(item["attention_gcs"], dtype=float)
            if not np.array_equal(epochs, final_epochs):
                raise ValueError(f"Epoch mismatch for {variant} in {exp_dir}")
            rows = evaluate_variant(
                act=act,
                task=task,
                seed=seed,
                exp_dir=exp_dir,
                variant=variant,
                epochs=epochs,
                series=series,
                final_series=final_series,
                metrics=metrics,
                equivariance=equivariance,
                window_len=args.window_len,
            )
            all_rows.extend(rows)
            trajectories[(act, seed, variant)] = item

    write_csv(args.output_dir / "per_run.csv", all_rows)
    summary_rows = summarize(all_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    save_pickle(args.output_dir / "trajectories.pkl", trajectories)

    print(f"Saved {args.output_dir / 'per_run.csv'}")
    print(f"Saved {args.output_dir / 'summary.csv'}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
