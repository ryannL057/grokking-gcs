#!/usr/bin/env python3
"""Held-out selector benchmark for GrokAlign/centroid alignment.

This script reuses cached centroid-alignment trajectories from
``experiments/grokalign_systematic/grokalign_all_runs.pkl`` and evaluates them
under the same window-selection protocol used for the paper's Table 4.  It is
intentionally post-hoc and scalar-only: centroid alignment is not path-resolved,
so the comparison asks whether it can nominate useful checkpoint windows, not
whether it can route to an attention/MLP path.
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
from typing import Any

import numpy as np

pass  # Package imports do not search the experiment-data directory.

from gcs.legacy.diagnostics_common import (  # noqa: E402
    RUNS_1L,
    detect_grok_epoch,
    load_gcs,
    load_metrics,
    resolve_path,
)
from gcs.legacy.evaluate_workflow_selection_heldout import (  # noqa: E402
    align_to_epochs,
    build_targets,
    load_equivariance,
    score_window,
    select_first_prominent_post_valley,
    select_largest_slope,
    split_for_seed,
    window_bounds,
)


SELECTOR_DISPLAY = {
    "gcs_attention": "Attention-GCS",
    "path_cka_attention_rise": "Attn-output CKA-to-final",
    "path_l2_attention_drop": "Attn-output L2-to-final",
    "final_repr_cka_rise": "Final-repr CKA",
    "centroid_alignment_peak": "Centroid alignment, peak rule",
    "centroid_alignment_abs": "Centroid alignment, largest abs. change",
    "centroid_alignment_rise": "Centroid alignment, largest rise",
    "centroid_alignment_drop": "Centroid alignment, largest drop",
    "random": "Random window",
}

MAIN_TABLE_SELECTORS = [
    "gcs_attention",
    "path_cka_attention_rise",
    "path_l2_attention_drop",
    "final_repr_cka_rise",
    "centroid_alignment_peak",
    "centroid_alignment_abs",
    "centroid_alignment_rise",
    "centroid_alignment_drop",
    "random",
]

MAIN_TARGETS = {
    "acc": "test_accuracy_rise",
    "equiv": "attention_equivariance_drop",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def load_grokalign_cache(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in payload.values():
        out[(item["act"], item["task"], int(item["seed"]))] = item
    return out


def centroid_centers(epochs: np.ndarray, centroid: np.ndarray, window_len: int) -> dict[str, int | None]:
    return {
        "centroid_alignment_peak": select_first_prominent_post_valley(centroid, window_len),
        "centroid_alignment_abs": select_largest_slope(epochs, centroid, epochs, mode="abs"),
        "centroid_alignment_rise": select_largest_slope(epochs, centroid, epochs, mode="rise"),
        "centroid_alignment_drop": select_largest_slope(epochs, centroid, epochs, mode="drop"),
    }


def deterministic_summary(rows: list[dict[str, Any]], window_len: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["window_len"]) != int(window_len):
            continue
        groups[(row["split"], row["selector"], row["target"])].append(row)

    out = []
    for (split, selector, target), items in sorted(groups.items()):
        vals = np.asarray([float(r["percentile"]) for r in items], dtype=float)
        run_keys = {(r["act"], r["task"], int(r["seed"])) for r in items}
        out.append({
            "split": split,
            "selector": selector,
            "selector_display": SELECTOR_DISPLAY.get(selector, selector),
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


def run_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (row["act"], row["task"], int(row["seed"]))


def row_percentiles_for(
    rows: list[dict[str, Any]],
    *,
    split: str,
    selector: str,
    target: str,
    window_len: int,
) -> np.ndarray:
    vals = [
        float(row["percentile"])
        for row in rows
        if row["split"] == split
        and row["selector"] == selector
        and row["target"] == target
        and int(row["window_len"]) == int(window_len)
    ]
    return np.asarray(vals, dtype=float)


def summarize_with_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "median": float("nan"),
            "median_ci_low": float("nan"),
            "median_ci_high": float("nan"),
            "top20": float("nan"),
            "top20_ci_low": float("nan"),
            "top20_ci_high": float("nan"),
        }
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[idx]
    med = np.median(samples, axis=1)
    top = np.mean(samples >= 0.80, axis=1)
    return {
        "median": float(np.median(values)),
        "median_ci_low": float(np.percentile(med, 2.5)),
        "median_ci_high": float(np.percentile(med, 97.5)),
        "top20": float(np.mean(values >= 0.80)),
        "top20_ci_low": float(np.percentile(top, 2.5)),
        "top20_ci_high": float(np.percentile(top, 97.5)),
    }


def compact_table(
    rows: list[dict[str, Any]],
    *,
    split: str,
    window_len: int,
    rng: np.random.Generator,
    n_boot: int,
) -> list[dict[str, Any]]:
    out = []
    for selector in MAIN_TABLE_SELECTORS:
        row: dict[str, Any] = {
            "selector": selector,
            "selector_display": SELECTOR_DISPLAY.get(selector, selector),
        }
        for prefix, target in MAIN_TARGETS.items():
            vals = row_percentiles_for(
                rows,
                split=split,
                selector=selector,
                target=target,
                window_len=window_len,
            )
            stats = summarize_with_ci(vals, rng, n_boot)
            row[f"{prefix}_n"] = int(vals.size)
            row[f"{prefix}_median"] = "" if np.isnan(stats["median"]) else f"{stats['median']:.3f}"
            row[f"{prefix}_median_ci"] = (
                "" if np.isnan(stats["median"]) else
                f"[{stats['median_ci_low']:.3f}, {stats['median_ci_high']:.3f}]"
            )
            row[f"{prefix}_top20"] = "" if np.isnan(stats["top20"]) else f"{stats['top20']:.3f}"
            row[f"{prefix}_top20_ci"] = (
                "" if np.isnan(stats["top20"]) else
                f"[{stats['top20_ci_low']:.3f}, {stats['top20_ci_high']:.3f}]"
            )
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GrokAlign/centroid alignment as held-out window selectors.")
    parser.add_argument("--grokalign-cache", default="experiments/grokalign_systematic/grokalign_all_runs.pkl")
    parser.add_argument("--existing-window-scores", default="experiments/workflow_selection_heldout_pathalign_20260502/window_scores_heldout.csv")
    parser.add_argument("--equivariance", default="experiments/equivariance_analysis/equivariance_all_activations.pkl")
    parser.add_argument("--output-dir", default="experiments/grokalign_selector_benchmark_20260503")
    parser.add_argument("--window-len", type=int, default=11)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    window_len = int(args.window_len)
    if window_len % 2 == 0:
        raise ValueError("window-len must be odd")

    cache = load_grokalign_cache(resolve_path(args.grokalign_cache))
    equivariance = load_equivariance(resolve_path(args.equivariance))

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for act, task, seed, base in RUNS_1L:
        key = (act, task, int(seed))
        if key not in cache:
            skipped.append({"act": act, "task": task, "seed": seed, "reason": "missing_grokalign_cache"})
            continue
        try:
            metrics = load_metrics(base)
            t_grok = detect_grok_epoch(metrics)
            if t_grok is None:
                skipped.append({"act": act, "task": task, "seed": seed, "reason": "no_t_grok"})
                continue
            epochs, _scores = load_gcs(base)
            epochs = np.asarray(epochs, dtype=int)
        except Exception as exc:
            skipped.append({"act": act, "task": task, "seed": seed, "reason": str(exc)})
            continue

        ga = cache[key]["grokalign"]
        centroid = align_to_epochs(ga["epochs"], ga["grokalign_scores"], epochs)
        if centroid is None:
            skipped.append({"act": act, "task": task, "seed": seed, "reason": "centroid_alignment_failed"})
            continue

        targets = build_targets(act, task, int(seed), resolve_path(base), metrics, epochs, equivariance)
        if not targets:
            skipped.append({"act": act, "task": task, "seed": seed, "reason": "no_targets"})
            continue

        centers = centroid_centers(epochs, centroid, window_len)
        split = split_for_seed(int(seed))
        for target in targets:
            for selector, center_idx in centers.items():
                if center_idx is None:
                    continue
                start_idx, end_idx = window_bounds(center_idx, len(epochs), window_len)
                stats = score_window(target["values"], window_len, start_idx, target["mode"])
                best_start = int(stats["best_window_start_idx"])
                rows.append({
                    "split": split,
                    "act": act,
                    "task": task,
                    "seed": int(seed),
                    "exp_dir": str(resolve_path(base)),
                    "t_grok": int(t_grok),
                    "selector": selector,
                    "selector_family": "centroid_alignment",
                    "target": target["target"],
                    "target_mode": target["mode"],
                    "window_len": int(window_len),
                    "center_epoch": int(epochs[center_idx]),
                    "window_start_epoch": int(epochs[start_idx]),
                    "window_end_epoch": int(epochs[end_idx]),
                    "selected_score": stats["selected_score"],
                    "percentile": stats["percentile"],
                    "best_window_start_epoch": int(epochs[best_start]),
                    "best_window_end_epoch": int(epochs[min(best_start + window_len - 1, len(epochs) - 1)]),
                    "best_score": stats["best_score"],
                })

    write_csv(out_dir / "window_scores_grokalign.csv", rows)
    write_csv(out_dir / "skipped.csv", skipped)

    summary = deterministic_summary(rows, window_len)
    write_csv(out_dir / "summary_grokalign.csv", summary)

    existing_rows: list[dict[str, Any]] = []
    existing_path = resolve_path(args.existing_window_scores)
    if existing_path.exists():
        existing_rows = [dict(row) for row in read_csv(existing_path)]
    joined_rows = existing_rows + rows
    write_csv(out_dir / "window_scores_joined.csv", joined_rows)

    compact = compact_table(
        joined_rows,
        split="heldout_seed123_777",
        window_len=window_len,
        rng=rng,
        n_boot=int(args.n_boot),
    )
    write_csv(out_dir / "main_table_grokalign_selector_w11.csv", compact)

    payload = {
        "window_len": window_len,
        "grokalign_cache": str(resolve_path(args.grokalign_cache)),
        "existing_window_scores": str(existing_path),
        "n_grokalign_rows": len(rows),
        "n_joined_rows": len(joined_rows),
        "skipped": skipped,
        "summary": summary,
        "main_table": compact,
        "interpretation": (
            "Centroid alignment is evaluated as a scalar stage/window selector. "
            "It is not path-resolved, so it cannot by itself route to the "
            "attention/MLP interface; the comparison is window-selection utility."
        ),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved GrokAlign rows: {out_dir / 'window_scores_grokalign.csv'}")
    print(f"Saved joined table:   {out_dir / 'main_table_grokalign_selector_w11.csv'}")
    if skipped:
        print(f"Skipped {len(skipped)} cases; see {out_dir / 'skipped.csv'}")


if __name__ == "__main__":
    main()
