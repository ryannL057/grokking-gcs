#!/usr/bin/env python3
"""
Held-out benchmark for first-stage checkpoint-window selection.

This script evaluates whether label-free selectors choose short windows that
are enriched for independent transition targets.  Unlike the older cohort
summary, it reports a fixed seed split:

  - development: seed 42
  - held-out: seeds 123 and 777

All selectors use the same centered window budget.  The main GCS selector uses
only a path-specific GCS trajectory and never uses accuracy, t_grok, endpoint
equivariance, Fourier progress, or task algebra labels during selection.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

pass  # Package imports do not search the experiment-data directory.

from gcs.legacy.diagnostics_common import (  # noqa: E402
    RUNS_1L,
    detect_grok_epoch,
    load_gcs,
    load_metrics,
    resolve_path,
    smooth_series,
)


COMMUTATIVE_TASKS = {"add", "mul"}
NONCOMMUTATIVE_TASKS = {"sub", "div"}


SELECTOR_METADATA: dict[str, dict[str, str]] = {
    "gcs_attention": {"family": "GCS", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "gcs_mlp": {"family": "GCS", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "gcs_overall": {"family": "GCS", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "mean_edge_attention": {"family": "mean_edge", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "mean_edge_mlp": {"family": "mean_edge", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "mean_edge_overall": {"family": "mean_edge", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "spectral_pr_attention": {"family": "spectral_pr", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "spectral_pr_mlp": {"family": "spectral_pr", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "spectral_pr_overall": {"family": "spectral_pr", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "weight_final_distance_drop": {"family": "weight_final_distance", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "final_repr_l2_drop": {"family": "final_repr_distance", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "final_repr_cka_rise": {"family": "final_repr_cka", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "path_l2_attention_drop": {"family": "path_output_l2", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "path_l2_mlp_drop": {"family": "path_output_l2", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "path_l2_overall_drop": {"family": "path_output_l2", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "path_cka_attention_rise": {"family": "path_output_cka", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "path_cka_mlp_rise": {"family": "path_output_cka", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "path_cka_overall_rise": {"family": "path_output_cka", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "path_pr_attention_drop": {"family": "path_output_pr", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "attention_output_variance_abs": {"family": "attention_scalar", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "param_norm_slope": {"family": "norm", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "train_loss_drop": {"family": "train_loss", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "task_symmetry_readout": {"family": "task_symmetry", "uses_endpoint_label": "True", "uses_test_accuracy": "False"},
    "swap_distance_abs": {"family": "task_symmetry_abs", "uses_endpoint_label": "True", "uses_test_accuracy": "False"},
    "random": {"family": "random", "uses_endpoint_label": "False", "uses_test_accuracy": "False"},
    "accuracy_oracle": {"family": "oracle", "uses_endpoint_label": "False", "uses_test_accuracy": "True"},
}


def finite_array(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)
    if arr.size and np.isnan(arr).any():
        med = np.nanmedian(arr)
        arr = np.where(np.isnan(arr), med, arr)
    return arr


def split_for_seed(seed: int, extra_heldout_seeds: set[int] | None = None) -> str:
    if int(seed) == 42:
        return "dev_seed42"
    if int(seed) in {123, 777} or int(seed) in (extra_heldout_seeds or set()):
        return "heldout_seed123_777"
    return "other"


def load_extra_runs(path: Path | None) -> list[tuple[str, str, int, str]]:
    if path is None:
        return []
    rows: list[tuple[str, str, int, str]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["act"], row["task"], int(row["seed"]), row["path"]))
    return rows


def align_to_epochs(
    source_epochs: Iterable[Any],
    values: Iterable[Any],
    target_epochs: np.ndarray,
) -> Optional[np.ndarray]:
    src_e = np.asarray(list(source_epochs), dtype=float)
    src_v = finite_array(values)
    if src_e.size == 0 or src_v.size == 0 or src_e.size != src_v.size:
        return None
    mask = np.isfinite(src_e) & np.isfinite(src_v)
    if mask.sum() < 2:
        return None
    src_e = src_e[mask]
    src_v = src_v[mask]
    order = np.argsort(src_e)
    src_e = src_e[order]
    src_v = src_v[order]
    return np.interp(target_epochs.astype(float), src_e, src_v)


def window_bounds(center_idx: int, n: int, window_len: int) -> Tuple[int, int]:
    half = window_len // 2
    start = max(0, int(center_idx) - half)
    end = start + window_len - 1
    if end >= n:
        end = n - 1
        start = max(0, end - window_len + 1)
    return start, end


def all_window_scores(arr: np.ndarray, window_len: int, mode: str) -> np.ndarray:
    scores = []
    for start in range(0, arr.size - window_len + 1):
        end = start + window_len - 1
        if mode == "rise":
            score = arr[end] - arr[start]
        elif mode == "drop":
            score = arr[start] - arr[end]
        elif mode == "abs":
            score = abs(arr[end] - arr[start])
        else:
            raise ValueError(f"Unknown target mode: {mode}")
        scores.append(float(score))
    return np.asarray(scores, dtype=float)


def score_window(
    arr: np.ndarray,
    window_len: int,
    start_idx: int,
    mode: str,
) -> Dict[str, Any]:
    scores = all_window_scores(arr, window_len, mode)
    if scores.size == 0:
        raise ValueError("Window length exceeds trajectory length")
    selected = float(scores[start_idx])
    best_idx = int(np.argmax(scores))
    percentile = float(np.mean(scores <= selected))
    return {
        "selected_score": selected,
        "percentile": percentile,
        "best_window_start_idx": best_idx,
        "best_score": float(scores[best_idx]),
    }


def select_first_prominent_post_valley(series: np.ndarray, window_len: int) -> int:
    """Select a post-initialization peak without accuracy or t_grok."""
    sm = smooth_series(series, window=3)
    n = sm.size
    if n <= window_len:
        return n // 2

    valid_lo = min(max(1, window_len // 2), n - 1)
    valid_hi = max(valid_lo + 1, n - window_len // 2)
    valley_search_hi = max(valid_lo + 1, int(0.45 * n))
    valley_idx = valid_lo + int(np.argmin(sm[valid_lo:valley_search_hi]))
    peak_start = min(max(valley_idx + 1, valid_lo), valid_hi - 1)
    post = sm[peak_start:valid_hi]
    if post.size == 0:
        return int(np.argmax(sm[valid_lo:valid_hi]) + valid_lo)

    post_range = float(np.nanmax(post) - sm[valley_idx])
    min_prominence = max(0.05 * max(abs(float(sm[valley_idx])), 1.0), 0.15 * post_range)
    for idx in range(peak_start + 1, valid_hi - 1):
        is_local_peak = sm[idx - 1] <= sm[idx] and sm[idx] >= sm[idx + 1]
        prominent = (sm[idx] - sm[valley_idx]) >= min_prominence
        if is_local_peak and prominent:
            return int(idx)

    threshold = sm[valley_idx] + 0.75 * post_range
    crossing = np.where(post >= threshold)[0]
    if crossing.size:
        return int(peak_start + crossing[0])
    return int(np.argmax(post) + peak_start)


def select_largest_slope(
    source_epochs: Iterable[Any],
    values: Iterable[Any],
    target_epochs: np.ndarray,
    *,
    mode: str,
) -> Optional[int]:
    steps = np.asarray(list(source_epochs), dtype=float)
    vals = finite_array(values)
    if steps.size < 3 or vals.size != steps.size:
        return None
    if mode == "rise":
        slopes = np.diff(vals) / np.maximum(np.diff(steps), 1.0)
    elif mode == "drop":
        slopes = -np.diff(vals) / np.maximum(np.diff(steps), 1.0)
    elif mode == "abs":
        slopes = np.abs(np.diff(vals) / np.maximum(np.diff(steps), 1.0))
    else:
        raise ValueError(f"Unknown slope mode: {mode}")
    if slopes.size == 0 or not np.isfinite(slopes).any():
        return None
    slope_idx = int(np.nanargmax(slopes))
    mid_epoch = float((steps[slope_idx] + steps[slope_idx + 1]) / 2.0)
    return int(np.argmin(np.abs(target_epochs - mid_epoch)))


def load_equivariance(path: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


def load_fourier_progress(exp_dir: Path, target_epochs: np.ndarray) -> Optional[np.ndarray]:
    csv_path = exp_dir / "nanda_progress_comparison" / "comparison_timeseries.csv"
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    if not rows:
        return None
    epochs = [float(r["epoch"]) for r in rows]
    if "restricted_progress" in rows[0]:
        values = [float(r["restricted_progress"]) for r in rows]
    elif "trig_loss_raw" in rows[0]:
        values = [float(r["trig_loss_raw"]) for r in rows]
    else:
        return None
    return align_to_epochs(epochs, values, target_epochs)


def load_aggregation_selectors(
    exp_dir: Path,
    target_epochs: np.ndarray,
    aggregation_suffix: str,
) -> dict[str, np.ndarray]:
    """Load alternative aggregation trajectories and orient them as coherence scores."""
    if not aggregation_suffix:
        return {}
    pkl_path = exp_dir / f"gcs_aggregation_baselines{aggregation_suffix}.pkl"
    if not pkl_path.exists():
        return {}
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    epochs = payload.get("epochs", [])
    metric_scores = payload.get("metric_scores", {})
    out: dict[str, np.ndarray] = {}
    path_map = {
        "attention": "attention_path",
        "mlp": "mlp_path",
        "overall": "overall_path",
    }

    mean_edge = metric_scores.get("mean_edge_coherence", {})
    for short_name, path_key in path_map.items():
        arr = align_to_epochs(epochs, mean_edge.get(path_key, []), target_epochs)
        if arr is not None:
            out[f"mean_edge_{short_name}"] = arr

    # Participation-rank-style summaries count effective modes; lower mode count
    # corresponds to greater transformation coherence.  Negate before applying
    # the same post-valley peak rule so all selectors are oriented consistently.
    spectral_pr = metric_scores.get("participation_rank", {})
    for short_name, path_key in path_map.items():
        arr = align_to_epochs(epochs, spectral_pr.get(path_key, []), target_epochs)
        if arr is not None:
            out[f"spectral_pr_{short_name}"] = -arr

    return out


def load_auxiliary_selectors(
    exp_dir: Path,
    target_epochs: np.ndarray,
    auxiliary_suffix: str,
) -> dict[str, np.ndarray]:
    """Load scalar baseline trajectories produced by compute_workflow_selector_aux_baselines.py."""
    if not auxiliary_suffix:
        return {}
    pkl_path = exp_dir / f"workflow_selector_aux_baselines{auxiliary_suffix}.pkl"
    if not pkl_path.exists():
        return {}
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    epochs = payload.get("epochs", [])
    series = payload.get("series", {})
    out: dict[str, np.ndarray] = {}
    for key in (
        "weight_final_distance",
        "final_repr_l2_distance",
        "final_repr_cka",
        "path_l2_attention",
        "path_l2_mlp",
        "path_l2_overall",
        "path_cka_attention",
        "path_cka_mlp",
        "path_cka_overall",
        "path_pr_attention",
        "attention_output_variance",
    ):
        if key in series:
            arr = align_to_epochs(epochs, series[key], target_epochs)
            if arr is not None:
                out[key] = arr
    return out


def build_targets(
    act: str,
    task: str,
    seed: int,
    exp_dir: Path,
    metrics: Dict[str, Any],
    epochs: np.ndarray,
    equivariance: Dict[Tuple[str, str, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    acc = align_to_epochs(metrics.get("log_steps", []), metrics.get("test_accuracies", []), epochs)
    if acc is not None:
        targets.append({"target": "test_accuracy_rise", "mode": "rise", "values": acc})

    eq = equivariance.get((act, task, seed))
    if eq is not None:
        norm = np.asarray(eq["norm_scale"], dtype=float)
        dist = np.asarray(eq["equi_dist"], dtype=float)
        rel = dist / np.maximum(norm, 1e-8)
        rel_aligned = align_to_epochs(eq["epochs"], rel, epochs)
        if rel_aligned is not None:
            if task in COMMUTATIVE_TASKS:
                targets.append({
                    "target": "attention_equivariance_drop",
                    "mode": "drop",
                    "values": rel_aligned,
                })
            elif task in NONCOMMUTATIVE_TASKS:
                targets.append({
                    "target": "attention_order_sensitivity_rise",
                    "mode": "rise",
                    "values": rel_aligned,
                })

    fourier = load_fourier_progress(exp_dir, epochs)
    if fourier is not None:
        targets.append({"target": "fourier_restricted_progress_rise", "mode": "rise", "values": fourier})

    return targets


def selector_centers(
    scores: dict[str, Any],
    metrics: Dict[str, Any],
    epochs: np.ndarray,
    window_len: int,
    extra_selector_scores: Optional[dict[str, np.ndarray]] = None,
    auxiliary_selector_scores: Optional[dict[str, np.ndarray]] = None,
) -> dict[str, Optional[int]]:
    centers: dict[str, Optional[int]] = {}
    for selector, path_key in (
        ("gcs_attention", "attention_path"),
        ("gcs_mlp", "mlp_path"),
        ("gcs_overall", "overall_path"),
    ):
        if path_key in scores:
            centers[selector] = select_first_prominent_post_valley(
                finite_array(scores[path_key]),
                window_len,
            )

    for selector, series in (extra_selector_scores or {}).items():
        centers[selector] = select_first_prominent_post_valley(
            finite_array(series),
            window_len,
        )

    centers["param_norm_slope"] = select_largest_slope(
        metrics.get("log_steps", []),
        metrics.get("norms", []),
        epochs,
        mode="rise",
    )
    centers["train_loss_drop"] = select_largest_slope(
        metrics.get("log_steps", []),
        metrics.get("train_losses", []),
        epochs,
        mode="drop",
    )
    centers["accuracy_oracle"] = select_largest_slope(
        metrics.get("log_steps", []),
        metrics.get("test_accuracies", []),
        epochs,
        mode="rise",
    )

    aux = auxiliary_selector_scores or {}
    if "weight_final_distance" in aux:
        centers["weight_final_distance_drop"] = select_largest_slope(
            epochs, aux["weight_final_distance"], epochs, mode="drop"
        )
    if "final_repr_l2_distance" in aux:
        centers["final_repr_l2_drop"] = select_largest_slope(
            epochs, aux["final_repr_l2_distance"], epochs, mode="drop"
        )
    if "final_repr_cka" in aux:
        centers["final_repr_cka_rise"] = select_largest_slope(
            epochs, aux["final_repr_cka"], epochs, mode="rise"
        )
    for site in ("attention", "mlp", "overall"):
        l2_key = f"path_l2_{site}"
        cka_key = f"path_cka_{site}"
        if l2_key in aux:
            centers[f"{l2_key}_drop"] = select_largest_slope(
                epochs, aux[l2_key], epochs, mode="drop"
            )
        if cka_key in aux:
            centers[f"{cka_key}_rise"] = select_largest_slope(
                epochs, aux[cka_key], epochs, mode="rise"
            )
    if "path_pr_attention" in aux:
        centers["path_pr_attention_drop"] = select_largest_slope(
            epochs, aux["path_pr_attention"], epochs, mode="drop"
        )
    if "attention_output_variance" in aux:
        centers["attention_output_variance_abs"] = select_largest_slope(
            epochs, aux["attention_output_variance"], epochs, mode="abs"
        )
    return centers


def target_conditioned_centers(target: Dict[str, Any], epochs: np.ndarray) -> dict[str, Optional[int]]:
    """Selectors that are only defined for a task-specific validation readout."""
    if target["target"] not in {
        "attention_equivariance_drop",
        "attention_order_sensitivity_rise",
    }:
        return {}
    return {
        "task_symmetry_readout": select_largest_slope(
            epochs,
            target["values"],
            epochs,
            mode=target["mode"],
        ),
        "swap_distance_abs": select_largest_slope(
            epochs,
            target["values"],
            epochs,
            mode="abs",
        ),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]], window_len: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["window_len"]) != int(window_len):
            continue
        key = (row["split"], row["selector"], row["target"])
        groups[key].append(row)

    out = []
    for (split, selector, target), items in sorted(groups.items()):
        vals = np.asarray([float(r["percentile"]) for r in items], dtype=float)
        run_keys = {(r["act"], r["task"], int(r["seed"])) for r in items}
        meta = SELECTOR_METADATA.get(selector, {})
        out.append({
            "split": split,
            "selector": selector,
            "family": meta.get("family", ""),
            "uses_endpoint_label": meta.get("uses_endpoint_label", ""),
            "uses_test_accuracy": meta.get("uses_test_accuracy", ""),
            "target": target,
            "window_len": int(window_len),
            "n_runs": len(run_keys),
            "n_rows": len(items),
            "mean_percentile": float(np.mean(vals)),
            "median_percentile": float(np.median(vals)),
            "top10_hit_rate": float(np.mean(vals >= 0.90)),
            "top20_hit_rate": float(np.mean(vals >= 0.80)),
        })
    return out


def summarize_margins(rows: List[Dict[str, Any]], window_len: int) -> list[dict[str, Any]]:
    """Run-level GCS-attention margins over simple label-free baselines."""
    by_key: dict[tuple[str, str, int, str, int], dict[str, float]] = defaultdict(dict)
    split_by_key: dict[tuple[str, str, int, str, int], str] = {}
    for row in rows:
        if int(row["window_len"]) != int(window_len):
            continue
        key = (row["act"], row["task"], int(row["seed"]), row["target"], int(row["window_len"]))
        by_key[key][row["selector"]] = float(row["percentile"])
        split_by_key[key] = row["split"]

    baselines = ["param_norm_slope", "train_loss_drop", "gcs_mlp", "gcs_overall"]
    out = []
    for split in ("dev_seed42", "heldout_seed123_777", "all"):
        for target in sorted({key[3] for key in by_key}):
            margins = []
            wins = []
            for key, vals in by_key.items():
                if key[3] != target:
                    continue
                if split != "all" and split_by_key.get(key) != split:
                    continue
                if "gcs_attention" not in vals:
                    continue
                avail = [vals[b] for b in baselines if b in vals]
                if not avail:
                    continue
                best = max(avail)
                margin = vals["gcs_attention"] - best
                margins.append(margin)
                wins.append(margin > 0)
            if margins:
                arr = np.asarray(margins, dtype=float)
                out.append({
                    "split": split,
                    "target": target,
                    "window_len": int(window_len),
                    "n_runs": int(arr.size),
                    "median_margin_vs_best_simple_baseline": float(np.median(arr)),
                    "mean_margin_vs_best_simple_baseline": float(np.mean(arr)),
                    "win_rate_vs_best_simple_baseline": float(np.mean(wins)),
                })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out benchmark for GCS-guided window selection.")
    parser.add_argument("--output-dir", type=str, default="experiments/workflow_selection_heldout")
    parser.add_argument("--equivariance", type=str, default="experiments/equivariance_analysis/equivariance_all_activations.pkl")
    parser.add_argument("--window-lens", type=str, default="7")
    parser.add_argument("--tasks", type=str, default="add,sub,mul,div")
    parser.add_argument("--activations", type=str, default="ReLU,GeLU,SiLU")
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument(
        "--aggregation-suffix",
        type=str,
        default="",
        help="Optional suffix for gcs_aggregation_baselines<SUFFIX>.pkl files.",
    )
    parser.add_argument(
        "--auxiliary-suffix",
        type=str,
        default="",
        help="Optional suffix for workflow_selector_aux_baselines<SUFFIX>.pkl files.",
    )
    parser.add_argument(
        "--extra-runs-csv",
        type=str,
        default="",
        help="Optional CSV with columns act,task,seed,path to append to RUNS_1L.",
    )
    parser.add_argument(
        "--extra-heldout-seeds",
        type=str,
        default="",
        help="Comma-separated extra seeds that should be assigned to the held-out split.",
    )
    args = parser.parse_args()

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    window_lens = [int(x.strip()) for x in args.window_lens.split(",") if x.strip()]
    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    activations = {x.strip() for x in args.activations.split(",") if x.strip()}
    equivariance = load_equivariance(resolve_path(args.equivariance))
    extra_runs_path = resolve_path(args.extra_runs_csv) if args.extra_runs_csv else None
    registered_runs = list(RUNS_1L) + load_extra_runs(extra_runs_path)
    extra_heldout_seeds = {int(x.strip()) for x in args.extra_heldout_seeds.split(",") if x.strip()}

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for act, task, seed, base in registered_runs:
        if task not in tasks or act not in activations:
            continue
        exp_dir = resolve_path(base)
        try:
            metrics = load_metrics(base)
            t_grok = detect_grok_epoch(metrics)
            if t_grok is None:
                skipped.append({"act": act, "task": task, "seed": seed, "reason": "no_t_grok"})
                continue
            epochs, scores = load_gcs(base)
            epochs = np.asarray(epochs, dtype=int)
        except Exception as exc:
            skipped.append({"act": act, "task": task, "seed": seed, "reason": str(exc)})
            continue

        targets = build_targets(act, task, seed, exp_dir, metrics, epochs, equivariance)
        if not targets:
            skipped.append({"act": act, "task": task, "seed": seed, "reason": "no_targets"})
            continue

        split = split_for_seed(seed, extra_heldout_seeds)
        extra_selectors = load_aggregation_selectors(exp_dir, epochs, args.aggregation_suffix)
        auxiliary_selectors = load_auxiliary_selectors(exp_dir, epochs, args.auxiliary_suffix)
        for window_len in window_lens:
            if window_len % 2 == 0:
                raise ValueError("Window lengths must be odd")
            if len(epochs) < window_len:
                continue
            centers = selector_centers(
                scores,
                metrics,
                epochs,
                window_len,
                extra_selectors,
                auxiliary_selectors,
            )
            n_starts = len(epochs) - window_len + 1
            random_starts = rng.integers(0, n_starts, size=args.random_trials)

            for target in targets:
                target_centers = dict(centers)
                target_centers.update(target_conditioned_centers(target, epochs))
                for selector, center_idx in target_centers.items():
                    if center_idx is None:
                        continue
                    start_idx, end_idx = window_bounds(center_idx, len(epochs), window_len)
                    try:
                        stats = score_window(target["values"], window_len, start_idx, target["mode"])
                    except Exception as exc:
                        skipped.append({
                            "act": act,
                            "task": task,
                            "seed": seed,
                            "reason": f"{target['target']}:{exc}",
                        })
                        continue
                    best_start = int(stats["best_window_start_idx"])
                    best_end = best_start + window_len - 1
                    meta = SELECTOR_METADATA.get(selector, {})
                    rows.append({
                        "split": split,
                        "act": act,
                        "task": task,
                        "seed": int(seed),
                        "exp_dir": str(exp_dir),
                        "t_grok": int(t_grok),
                        "selector": selector,
                        "selector_family": meta.get("family", ""),
                        "target": target["target"],
                        "target_mode": target["mode"],
                        "window_len": int(window_len),
                        "center_epoch": int(epochs[center_idx]),
                        "window_start_epoch": int(epochs[start_idx]),
                        "window_end_epoch": int(epochs[end_idx]),
                        "selected_score": stats["selected_score"],
                        "percentile": stats["percentile"],
                        "best_window_start_epoch": int(epochs[best_start]),
                        "best_window_end_epoch": int(epochs[min(best_end, len(epochs) - 1)]),
                        "best_score": stats["best_score"],
                        "random_trial": "",
                    })

                all_scores = all_window_scores(target["values"], window_len, target["mode"])
                for trial_idx, start_idx in enumerate(random_starts):
                    start_idx = int(start_idx)
                    selected = float(all_scores[start_idx])
                    percentile = float(np.mean(all_scores <= selected))
                    rows.append({
                        "split": split,
                        "act": act,
                        "task": task,
                        "seed": int(seed),
                        "exp_dir": str(exp_dir),
                        "t_grok": int(t_grok),
                        "selector": "random",
                        "selector_family": "random",
                        "target": target["target"],
                        "target_mode": target["mode"],
                        "window_len": int(window_len),
                        "center_epoch": "",
                        "window_start_epoch": int(epochs[start_idx]),
                        "window_end_epoch": int(epochs[start_idx + window_len - 1]),
                        "selected_score": selected,
                        "percentile": percentile,
                        "best_window_start_epoch": "",
                        "best_window_end_epoch": "",
                        "best_score": "",
                        "random_trial": int(trial_idx),
                    })

    write_csv(out_dir / "window_scores_heldout.csv", rows)
    write_csv(out_dir / "skipped.csv", skipped)

    all_summary: list[dict[str, Any]] = []
    all_margins: list[dict[str, Any]] = []
    for window_len in window_lens:
        all_summary.extend(summarize(rows, window_len))
        all_margins.extend(summarize_margins([r for r in rows if r["selector"] != "random"], window_len))
    write_csv(out_dir / "summary_heldout.csv", all_summary)
    write_csv(out_dir / "margins_vs_simple_baselines.csv", all_margins)
    with open(out_dir / "summary_heldout.json", "w") as f:
        json.dump({
            "summary": all_summary,
            "margins_vs_simple_baselines": all_margins,
            "n_rows": len(rows),
            "skipped": skipped,
            "random_trials": args.random_trials,
            "seed_split": {"dev": [42], "heldout": [123, 777] + sorted(extra_heldout_seeds)},
            "extra_runs_csv": str(extra_runs_path) if extra_runs_path else "",
        }, f, indent=2)

    print(f"Saved {len(rows)} rows to {out_dir / 'window_scores_heldout.csv'}")
    print(f"Saved summary to {out_dir / 'summary_heldout.csv'}")
    print(f"Saved margins to {out_dir / 'margins_vs_simple_baselines.csv'}")
    if skipped:
        print(f"Skipped {len(skipped)} run/target cases; see {out_dir / 'skipped.csv'}")


if __name__ == "__main__":
    main()
