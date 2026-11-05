#!/usr/bin/env python3
"""Residual complementarity of GCS against final-state proximity baselines.

This analysis answers a reviewer-facing question: if direct alignment to the
final state is already a strong retrospective locator, does attention-GCS still
carry a distinct signal?

For each run and validation target, we score every legal fixed-length window.
Within that run, we residualize both the target-window score and the
attention-GCS window-rise score against one or more final-state proximity
features, then summarize the residual association and incremental R^2 at the
run level.  Windows overlap, so the inferential unit reported in the summary is
the run, not the individual window.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = _gcs_data_root()
pass  # Imports use the installed package.

from gcs.legacy.diagnostics_common import (  # noqa: E402
    RUNS_1L,
    detect_grok_epoch,
    load_gcs,
    load_metrics,
    resolve_path,
)
from gcs.legacy.evaluate_workflow_selection_heldout import (  # noqa: E402
    all_window_scores,
    build_targets,
    finite_array,
    load_auxiliary_selectors,
    load_equivariance,
    load_extra_runs,
    split_for_seed,
)


BASELINE_FEATURES: dict[str, list[tuple[str, str]]] = {
    "final_repr": [
        ("final_repr_l2_distance", "drop"),
        ("final_repr_cka", "rise"),
    ],
    "attention_state": [
        ("path_l2_attention", "drop"),
        ("path_cka_attention", "rise"),
    ],
    "combined_final_state": [
        ("final_repr_l2_distance", "drop"),
        ("final_repr_cka", "rise"),
        ("path_l2_attention", "drop"),
        ("path_cka_attention", "rise"),
    ],
}


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks, zero indexed."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    sorted_vals = arr[order]
    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = rank
        i = j
    return ranks


def standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    out = x.copy()
    for col in range(out.shape[1]):
        vals = out[:, col]
        sd = float(np.std(vals))
        if sd < 1e-12:
            out[:, col] = 0.0
        else:
            out[:, col] = (vals - float(np.mean(vals))) / sd
    return out


def ols_fit_predict(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    x = standardize_columns(features)
    x = np.column_stack([np.ones(x.shape[0]), x])
    y = np.asarray(target, dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    r2 = 0.0 if ss_tot < 1e-12 else max(float(1.0 - ss_res / ss_tot), 0.0)
    return pred, r2


def residualize(target: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, float]:
    pred, r2 = ols_fit_predict(features, target)
    return np.asarray(target, dtype=float) - pred, r2


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def window_feature(series: np.ndarray, window_len: int, mode: str) -> np.ndarray:
    return all_window_scores(finite_array(series), window_len, mode)


def gcs_window_features(series: np.ndarray, window_len: int) -> dict[str, np.ndarray]:
    arr = finite_array(series)
    n_windows = arr.size - window_len + 1
    out: dict[str, list[float]] = {
        "gcs_rise": [],
        "gcs_abs_change": [],
        "gcs_center": [],
        "gcs_max": [],
        "gcs_range": [],
        "gcs_peak_prominence": [],
    }
    half = window_len // 2
    for start in range(n_windows):
        end = start + window_len - 1
        window = arr[start:end + 1]
        center = start + half
        prior = arr[: start + 1]
        out["gcs_rise"].append(float(arr[end] - arr[start]))
        out["gcs_abs_change"].append(float(abs(arr[end] - arr[start])))
        out["gcs_center"].append(float(arr[center]))
        out["gcs_max"].append(float(np.max(window)))
        out["gcs_range"].append(float(np.max(window) - np.min(window)))
        out["gcs_peak_prominence"].append(float(np.max(window) - np.min(prior)))
    return {key: np.asarray(vals, dtype=float) for key, vals in out.items()}


def exact_sign_p_ge(k: int, n: int) -> float:
    """One-sided exact sign-test p-value under p=0.5 for >= k positives."""
    if n <= 0:
        return float("nan")
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["target"], row["baseline"], row["gcs_feature"])].append(row)

    out: list[dict[str, Any]] = []
    for (split, target, baseline, gcs_feature), items in sorted(groups.items()):
        corrs = np.asarray([float(r["residual_spearman"]) for r in items], dtype=float)
        deltas = np.asarray([float(r["delta_rank_r2"]) for r in items], dtype=float)
        corrs = corrs[np.isfinite(corrs)]
        deltas = deltas[np.isfinite(deltas)]
        pos_corr = int(np.sum(corrs > 0))
        pos_delta = int(np.sum(deltas > 1e-10))
        out.append({
            "split": split,
            "target": target,
            "baseline": baseline,
            "gcs_feature": gcs_feature,
            "n_runs": len(items),
            "n_finite_corr": int(corrs.size),
            "median_residual_spearman": float(np.median(corrs)) if corrs.size else float("nan"),
            "mean_residual_spearman": float(np.mean(corrs)) if corrs.size else float("nan"),
            "positive_residual_corr": pos_corr,
            "sign_p_residual_corr_ge": exact_sign_p_ge(pos_corr, int(corrs.size)),
            "median_delta_rank_r2": float(np.median(deltas)) if deltas.size else float("nan"),
            "mean_delta_rank_r2": float(np.mean(deltas)) if deltas.size else float("nan"),
            "positive_delta_rank_r2": pos_delta,
            "sign_p_delta_rank_r2_ge": exact_sign_p_ge(pos_delta, int(deltas.size)),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/gcs_final_state_residuals_20260502")
    parser.add_argument("--equivariance", default="experiments/equivariance_analysis/equivariance_all_activations.pkl")
    parser.add_argument("--auxiliary-suffix", default="_wsel_aux_pathalign20260502")
    parser.add_argument("--window-len", type=int, default=11)
    parser.add_argument("--tasks", default="add,sub,mul,div")
    parser.add_argument("--activations", default="ReLU,GeLU,SiLU")
    parser.add_argument("--seeds", default="123,777")
    parser.add_argument(
        "--extra-runs-csv",
        default="",
        help="Optional CSV with columns act,task,seed,path to append to RUNS_1L.",
    )
    parser.add_argument(
        "--extra-heldout-seeds",
        default="",
        help="Comma-separated extra seeds assigned to the held-out split.",
    )
    args = parser.parse_args()

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    activations = {item.strip() for item in args.activations.split(",") if item.strip()}
    seeds = {int(item.strip()) for item in args.seeds.split(",") if item.strip()}
    extra_runs_path = resolve_path(args.extra_runs_csv) if args.extra_runs_csv else None
    registered_runs = list(RUNS_1L) + load_extra_runs(extra_runs_path)
    extra_heldout_seeds = {int(item.strip()) for item in args.extra_heldout_seeds.split(",") if item.strip()}
    equivariance = load_equivariance(resolve_path(args.equivariance))

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for act, task, seed, base in registered_runs:
        if act not in activations or task not in tasks or int(seed) not in seeds:
            continue
        exp_dir = resolve_path(base)
        try:
            metrics = load_metrics(base)
            if detect_grok_epoch(metrics) is None:
                skipped.append({"act": act, "task": task, "seed": seed, "reason": "no_t_grok"})
                continue
            epochs, scores = load_gcs(base)
            epochs = np.asarray(epochs, dtype=int)
            if epochs.size < args.window_len:
                skipped.append({"act": act, "task": task, "seed": seed, "reason": "too_few_epochs"})
                continue
            aux = load_auxiliary_selectors(exp_dir, epochs, args.auxiliary_suffix)
            targets = build_targets(act, task, seed, exp_dir, metrics, epochs, equivariance)
            if not targets:
                skipped.append({"act": act, "task": task, "seed": seed, "reason": "no_targets"})
                continue
            gcs_features = gcs_window_features(np.asarray(scores["attention_path"], dtype=float), args.window_len)
        except Exception as exc:
            skipped.append({"act": act, "task": task, "seed": seed, "reason": repr(exc)})
            continue

        for target in targets:
            if target["target"] not in {"test_accuracy_rise", "attention_equivariance_drop"}:
                continue
            target_windows = window_feature(np.asarray(target["values"], dtype=float), args.window_len, target["mode"])
            if target_windows.size != next(iter(gcs_features.values())).size:
                skipped.append({
                    "act": act,
                    "task": task,
                    "seed": seed,
                    "target": target["target"],
                    "reason": "window_size_mismatch",
                })
                continue

            # Rank-transform within run so the residual correlation is a
            # partial Spearman-style association over legal windows.
            y_rank = rankdata(target_windows)
            for baseline, feature_specs in BASELINE_FEATURES.items():
                feature_cols = []
                missing = []
                for key, mode in feature_specs:
                    if key not in aux:
                        missing.append(key)
                        continue
                    feature = window_feature(np.asarray(aux[key], dtype=float), args.window_len, mode)
                    if feature.size != target_windows.size:
                        missing.append(key)
                        continue
                    feature_cols.append(rankdata(feature))
                if missing or not feature_cols:
                    skipped.append({
                        "act": act,
                        "task": task,
                        "seed": seed,
                        "target": target["target"],
                        "baseline": baseline,
                        "reason": "missing_aux:" + ",".join(missing),
                    })
                    continue
                x = np.column_stack(feature_cols)
                y_resid, baseline_r2 = residualize(y_rank, x)
                for gcs_feature_name, gcs_windows in gcs_features.items():
                    gcs_rank = rankdata(gcs_windows)
                    gcs_resid, _ = residualize(gcs_rank, x)
                    resid_corr = safe_corr(y_resid, gcs_resid)
                    _, with_gcs_r2 = ols_fit_predict(np.column_stack([x, gcs_rank]), y_rank)
                    rows.append({
                        "split": split_for_seed(seed, extra_heldout_seeds),
                        "act": act,
                        "task": task,
                        "seed": int(seed),
                        "target": target["target"],
                        "baseline": baseline,
                        "gcs_feature": gcs_feature_name,
                        "window_len": int(args.window_len),
                        "n_windows": int(target_windows.size),
                        "baseline_rank_r2": float(baseline_r2),
                        "with_gcs_rank_r2": float(with_gcs_r2),
                        "delta_rank_r2": float(with_gcs_r2 - baseline_r2),
                        "residual_spearman": float(resid_corr),
                        "target_best_score": float(np.max(target_windows)),
                        "gcs_window_score_median": float(np.median(gcs_windows)),
                    })

    summary = summarize(rows)
    write_csv(out_dir / "per_run_residuals.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "skipped.csv", skipped)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "n_rows": len(rows),
                "n_skipped": len(skipped),
                "window_len": int(args.window_len),
                "baseline_features": BASELINE_FEATURES,
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"Saved {out_dir / 'summary.csv'}")
    for row in summary:
        print(
            f"{row['split']:22s} {row['target']:28s} {row['baseline']:20s} "
            f"{row['gcs_feature']:20s} n={row['n_runs']:2d} "
            f"med_resid_rho={row['median_residual_spearman']:.3f} "
            f"pos={row['positive_residual_corr']}/{row['n_finite_corr']} "
            f"med_dR2={row['median_delta_rank_r2']:.3f}"
        )


if __name__ == "__main__":
    main()
