#!/usr/bin/env python3
"""Method-level sanity checks for the GCS statistic.

This script produces three reviewer-facing checks:

1. A synthetic shared-rule vs lookup-memorization toy coherence field.
2. A reduced-cohort attention-path sensitivity analysis for diagonal/self-loop
   spectral choices.
3. Fixed rank-permutation and random-tangent-basis negative controls for
   SVD-order matching and final-geometry basis choice.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis.gmn_tsa import compute_path_gmn_tsa, summarize_coherence_matrix
from gcs.legacy.diagnostics_common import RUNS_1L, resolve_path
from gcs.legacy.recompute_gcs_existing_geometry import (
    cast_geometry,
    checkpoint_epoch,
    ensure_geometries,
    evenly_subsample_epochs,
    infer_num_layers,
    load_checkpoint,
    load_existing_result,
    target_layers_for_recompute,
)


ATTENTION_PATH = ("hook_resid_pre", "hook_attn_out")

DEFAULT_SELECTED = {
    "experiments/relu_1layer_add_20260213_004649",
    "experiments/relu_1layer_mul_p113_s42_20260301_051814",
    "experiments/relu_1layer_sub_p113_s42_20260301_074151",
    "experiments/relu_1layer_div_p113_s42_20260301_085906",
    "experiments/gelu_1layer_add_p113_s42_20260301_041540",
    "experiments/gelu_1layer_mul_p113_s42_20260301_063635",
    "experiments/silu_1layer_add_s42_20260213_000145",
    "experiments/silu_1layer_mul_p113_s42_20260301_055000",
}


def circular_knn(n: int, k: int) -> np.ndarray:
    offsets: list[int] = []
    step = 1
    while len(offsets) < k:
        offsets.append(step)
        if len(offsets) < k:
            offsets.append(-step)
        step += 1
    return np.asarray([[(i + off) % n for off in offsets[:k]] for i in range(n)], dtype=int)


def iter_union_edges(knn_indices: np.ndarray) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, neigh in enumerate(knn_indices):
        for j in neigh:
            j = int(j)
            if i == j:
                continue
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def smooth_frame(theta: float, d: int, width: int) -> np.ndarray:
    frame = np.zeros((d, width), dtype=np.float64)
    for ell in range(d):
        angle = theta * (0.35 + 0.07 * ell)
        frame[ell, 2 * ell] = np.cos(angle)
        frame[ell, 2 * ell + 1] = np.sin(angle)
    return frame


def random_frame(rng: np.random.Generator, d: int, width: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(width, d)))
    return q.T


def coherence_matrix_from_bases(bases: np.ndarray, knn_indices: np.ndarray) -> np.ndarray:
    n = bases.shape[0]
    g = np.eye(n, dtype=np.float64)
    for i, j in iter_union_edges(knn_indices):
        score = np.mean(np.abs(np.sum(bases[i] * bases[j], axis=1)))
        g[i, j] = score
        g[j, i] = score
    return g


def summarize_variants(g: np.ndarray, knn_indices: np.ndarray) -> dict[str, dict[str, float]]:
    n = g.shape[0]
    no_diag = np.array(g, copy=True)
    np.fill_diagonal(no_diag, 0.0)

    degree = np.sum(np.abs(g), axis=1)
    inv_sqrt = np.where(degree > 1e-12, 1.0 / np.sqrt(degree), 0.0)
    sym_norm = (inv_sqrt[:, None] * g) * inv_sqrt[None, :]

    return {
        "main": summarize_coherence_matrix(g, knn_indices=knn_indices, N=n),
        "no_diag": summarize_coherence_matrix(no_diag, knn_indices=knn_indices, N=n),
        "sym_norm": summarize_coherence_matrix(sym_norm, knn_indices=knn_indices, N=n),
    }


def run_toy(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    n, k, d, width = args.toy_n, args.toy_k, args.d, max(2 * args.d, args.toy_width)
    knn = circular_knn(n, k)
    thetas = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    smooth = np.stack([smooth_frame(float(theta), d, width) for theta in thetas], axis=0)

    q, _ = np.linalg.qr(rng.normal(size=(width, width)))
    shared = np.einsum("ndw,wo->ndo", smooth, q)
    shared = normalize_rows(shared.reshape(-1, width)).reshape(n, d, width)
    shared += args.toy_noise * rng.normal(size=shared.shape)
    shared = normalize_rows(shared.reshape(-1, width)).reshape(n, d, width)

    lookup = np.stack([random_frame(rng, d, width) for _ in range(n)], axis=0)
    shuffled = shared[rng.permutation(n)]

    rows: list[dict[str, Any]] = []
    for condition, bases in (
        ("shared_smooth_rule", shared),
        ("shuffled_rule_pairing", shuffled),
        ("lookup_memorization", lookup),
    ):
        g = coherence_matrix_from_bases(bases, knn)
        variants = summarize_variants(g, knn)
        for variant, summary in variants.items():
            rows.append(
                {
                    "condition": condition,
                    "variant": variant,
                    "gcs_score": float(summary["gcs_score"]),
                    "gmn_tsa_score": float(summary["gmn_tsa_score"]),
                    "mean_edge_coherence": float(summary["mean_edge_coherence"]),
                    "participation_rank": float(summary["participation_rank"]),
                }
            )
    return {"rows": rows, "n": n, "k": k, "d": d, "width": width, "noise": args.toy_noise}


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x0, y0) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def normalize_01(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def summarize_condition_against_main(values: dict[str, list[float]]) -> list[dict[str, Any]]:
    main = np.asarray(values["main"], dtype=np.float64)
    out: list[dict[str, Any]] = []
    for condition, raw in values.items():
        if condition == "main":
            continue
        arr = np.asarray(raw, dtype=np.float64)
        out.append(
            {
                "condition": condition,
                "spearman_vs_main": spearman(main, arr),
                "normalized_mae_vs_main": float(np.mean(np.abs(normalize_01(main) - normalize_01(arr)))),
                "argmax_idx_delta": int(np.argmax(arr) - np.argmax(main)),
                "final_main": float(main[-1]),
                "final_condition": float(arr[-1]),
            }
        )
    return out


def compute_attention_details(
    model: Any,
    geometry: dict[str, Any],
    target_layer: int,
    device: str,
    alignment_method: str,
) -> dict[str, Any]:
    dtype = next(model.parameters()).dtype
    geo = cast_geometry(geometry, device=device, dtype=dtype)
    start_hook, end_hook = ATTENTION_PATH
    return compute_path_gmn_tsa(
        model,
        geo,
        start_hook,
        end_hook,
        target_layer=target_layer,
        device=device,
        alignment_method=alignment_method,
        token_probe="final",
        return_details=True,
    )


def random_tangent_geometry(geometry: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Keep prompts and kNN graph fixed but replace each tangent basis by random orthonormal rows."""
    out = copy.deepcopy(geometry)
    bases = geometry["all_input_tangent_bases"]
    random_bases = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    for basis in bases:
        basis_cpu = basis.detach().cpu()
        d, width = basis_cpu.shape
        q, _ = torch.linalg.qr(torch.randn(width, d, generator=gen, dtype=basis_cpu.dtype), mode="reduced")
        random_bases.append(q.T.contiguous())
    out["all_input_tangent_bases"] = random_bases
    return out


def selected_runs(exp_dirs: list[str]) -> list[tuple[Any, Path]]:
    if exp_dirs:
        return [((None, None, None, path), resolve_path(path)) for path in exp_dirs]
    run_meta = [run for run in RUNS_1L if run[3] in DEFAULT_SELECTED]
    return [(meta, resolve_path(meta[3])) for meta in run_meta]


def run_reduced_controls(args: argparse.Namespace) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    runs = selected_runs(args.exp_dir)

    for meta, exp_dir in tqdm(runs, desc="runs"):
        activation, task, seed, base = meta
        _, _, epochs, _ = load_existing_result(exp_dir)
        checkpoint_map = {checkpoint_epoch(p): p for p in (exp_dir / "checkpoints").glob("*.pth")}
        selected_epochs = [epoch for epoch in epochs if epoch in checkpoint_map]
        selected_epochs = evenly_subsample_epochs(selected_epochs, args.max_checkpoints)
        if not selected_epochs:
            raise RuntimeError(f"No matching checkpoints for {exp_dir}")

        num_layers = infer_num_layers(exp_dir, device=args.device)
        target_layer = target_layers_for_recompute(num_layers)[-1]
        geometries = ensure_geometries(
            exp_dir,
            [target_layer],
            num_layers,
            args.device,
            N=args.N,
            k=args.k,
            d=args.d,
            force_rebuild=not args.use_saved_geometry,
        )
        geometry = geometries[target_layer]
        random_geometry = random_tangent_geometry(
            geometry,
            seed=int(args.seed) + 1009 * (len(run_rows) + len(point_rows) + 1),
        )
        knn = np.asarray(geometry["knn_indices"], dtype=int)

        values: dict[str, list[float]] = {
            "main": [],
            "no_diag": [],
            "sym_norm": [],
            "rank_permuted": [],
            "random_tangent": [],
        }

        for epoch in tqdm(selected_epochs, desc=Path(base).name, leave=False):
            model, _ = load_checkpoint(checkpoint_map[int(epoch)], device=args.device)
            main_details = compute_attention_details(
                model, geometry, target_layer, args.device, alignment_method="cosine"
            )
            variants = summarize_variants(np.asarray(main_details["coherence_matrix"]), knn)
            rank_details = compute_attention_details(
                model, geometry, target_layer, args.device, alignment_method="rank_permuted"
            )
            random_details = compute_attention_details(
                model, random_geometry, target_layer, args.device, alignment_method="cosine"
            )

            values["main"].append(float(variants["main"]["gcs_score"]))
            values["no_diag"].append(float(variants["no_diag"]["gcs_score"]))
            values["sym_norm"].append(float(variants["sym_norm"]["gcs_score"]))
            values["rank_permuted"].append(float(rank_details["gcs_score"]))
            values["random_tangent"].append(float(random_details["gcs_score"]))
            point_rows.append(
                {
                    "activation": activation,
                    "task": task,
                    "seed": seed,
                    "exp_dir": str(base),
                    "epoch": int(epoch),
                    "main": values["main"][-1],
                    "no_diag": values["no_diag"][-1],
                    "sym_norm": values["sym_norm"][-1],
                    "rank_permuted": values["rank_permuted"][-1],
                    "random_tangent": values["random_tangent"][-1],
                    "mean_edge": float(variants["main"]["mean_edge_coherence"]),
                    "gmn_main": float(variants["main"]["gmn_tsa_score"]),
                }
            )

        for row in summarize_condition_against_main(values):
            row.update(
                {
                    "activation": activation,
                    "task": task,
                    "seed": seed,
                    "exp_dir": str(base),
                    "n_checkpoints": len(selected_epochs),
                }
            )
            run_rows.append(row)

    return {"run_rows": run_rows, "point_rows": point_rows, "n_runs": len(runs)}


def aggregate_run_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in sorted({row["condition"] for row in rows}):
        sub = [row for row in rows if row["condition"] == condition]
        rhos = np.asarray([row["spearman_vs_main"] for row in sub], dtype=np.float64)
        maes = np.asarray([row["normalized_mae_vs_main"] for row in sub], dtype=np.float64)
        deltas = np.asarray([abs(row["argmax_idx_delta"]) for row in sub], dtype=np.float64)
        out[condition] = {
            "n": len(sub),
            "median_spearman_vs_main": float(np.nanmedian(rhos)),
            "positive_spearman_count": int(np.sum(rhos > 0)),
            "median_normalized_mae_vs_main": float(np.median(maes)),
            "argmax_within_one_count": int(np.sum(deltas <= 1)),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="experiments/gcs_method_sanity_20260502")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exp-dir", action="append", default=[])
    parser.add_argument("--max-checkpoints", type=int, default=9)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--use-saved-geometry", action="store_true")
    parser.add_argument("--skip-reduced-controls", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--toy-n", type=int, default=120)
    parser.add_argument("--toy-k", type=int, default=10)
    parser.add_argument("--toy-width", type=int, default=32)
    parser.add_argument("--toy-noise", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    toy = run_toy(args)
    write_csv(out_dir / "toy_sanity_rows.csv", toy["rows"])

    controls: dict[str, Any] = {"run_rows": [], "point_rows": [], "n_runs": 0}
    if not args.skip_reduced_controls:
        controls = run_reduced_controls(args)
        write_csv(out_dir / "attention_method_control_runs.csv", controls["run_rows"])
        write_csv(out_dir / "attention_method_control_points.csv", controls["point_rows"])

    summary = {
        "toy": toy,
        "reduced_controls": {
            "n_runs": controls["n_runs"],
            "n_rows": len(controls["run_rows"]),
            "aggregation": aggregate_run_rows(controls["run_rows"]) if controls["run_rows"] else {},
            "N": args.N,
            "k": args.k,
            "d": args.d,
            "max_checkpoints": args.max_checkpoints,
            "use_saved_geometry": bool(args.use_saved_geometry),
        },
        "files": {
            "toy_csv": str(out_dir / "toy_sanity_rows.csv"),
            "run_csv": str(out_dir / "attention_method_control_runs.csv"),
            "point_csv": str(out_dir / "attention_method_control_points.csv"),
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
