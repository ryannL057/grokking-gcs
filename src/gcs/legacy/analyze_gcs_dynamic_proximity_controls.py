#!/usr/bin/env python3
"""Matched-dynamic CKA/L2 controls for the all-window GCS residual audit.

The July 6 residual analysis compared dynamic GCS window summaries with signed
endpoint motion in CKA/L2.  This frozen follow-up gives every proximity
trajectory the same absolute-change and within-window-range summaries.  Both
the target and GCS feature are rank transformed and separately residualized
within run.  Runs, rather than overlapping windows, are the reporting unit.
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

from gcs.legacy.analyze_gcs_final_state_residuals import (  # noqa: E402
    gcs_window_features,
    rankdata,
    residualize,
    safe_corr,
)
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


GCS_FEATURES = ("gcs_range", "gcs_abs_change", "gcs_rise")

ATTENTION_SIGNED = (
    "attention_l2_signed_drop",
    "attention_cka_signed_rise",
)
ATTENTION_SIGNED_RANGE = ATTENTION_SIGNED + (
    "attention_l2_range",
    "attention_cka_range",
)
ATTENTION_FULL_DYNAMIC = ATTENTION_SIGNED + (
    "attention_l2_abs_change",
    "attention_l2_range",
    "attention_cka_abs_change",
    "attention_cka_range",
)
FINAL_SIGNED = (
    "final_l2_signed_drop",
    "final_cka_signed_rise",
)
FINAL_SIGNED_RANGE = FINAL_SIGNED + (
    "final_l2_range",
    "final_cka_range",
)
FINAL_FULL_DYNAMIC = FINAL_SIGNED + (
    "final_l2_abs_change",
    "final_l2_range",
    "final_cka_abs_change",
    "final_cka_range",
)
COMBINED_FULL_DYNAMIC = ATTENTION_FULL_DYNAMIC + FINAL_FULL_DYNAMIC

CONTROL_SETS: dict[str, tuple[str, ...]] = {
    "attention_signed": ATTENTION_SIGNED,
    "attention_signed_range": ATTENTION_SIGNED_RANGE,
    "attention_full_dynamic": ATTENTION_FULL_DYNAMIC,
    "final_signed": FINAL_SIGNED,
    "final_signed_range": FINAL_SIGNED_RANGE,
    "final_full_dynamic": FINAL_FULL_DYNAMIC,
    "combined_full_dynamic": COMBINED_FULL_DYNAMIC,
}

TARGET_CONTROL_SETS = {
    "attention_equivariance_drop": (
        "attention_signed",
        "attention_signed_range",
        "attention_full_dynamic",
        "combined_full_dynamic",
    ),
    "test_accuracy_rise": (
        "final_signed",
        "final_signed_range",
        "final_full_dynamic",
        "combined_full_dynamic",
    ),
}

PRIMARY_CONFIG = (
    "attention_equivariance_drop",
    "attention_full_dynamic",
    "gcs_range",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def exact_binomial_upper(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n))


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    finite_indices = [i for i, value in enumerate(p_values) if math.isfinite(value)]
    order = sorted(finite_indices, key=lambda index: p_values[index])
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def bootstrap_median_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    sample_indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    medians = np.median(values[sample_indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def trajectory_window_features(
    series: np.ndarray,
    *,
    window_len: int,
    prefix: str,
    signed_mode: str,
) -> dict[str, np.ndarray]:
    arr = finite_array(series)
    signed_label = "signed_drop" if signed_mode == "drop" else "signed_rise"
    signed = all_window_scores(arr, window_len, signed_mode)
    absolute = all_window_scores(arr, window_len, "abs")
    ranges = np.asarray(
        [
            float(np.max(arr[start : start + window_len]) - np.min(arr[start : start + window_len]))
            for start in range(arr.size - window_len + 1)
        ],
        dtype=float,
    )
    return {
        f"{prefix}_{signed_label}": signed,
        f"{prefix}_abs_change": absolute,
        f"{prefix}_range": ranges,
    }


def proximity_window_features(aux: dict[str, np.ndarray], window_len: int) -> dict[str, np.ndarray]:
    specs = (
        ("path_l2_attention", "attention_l2", "drop"),
        ("path_cka_attention", "attention_cka", "rise"),
        ("final_repr_l2_distance", "final_l2", "drop"),
        ("final_repr_cka", "final_cka", "rise"),
    )
    output: dict[str, np.ndarray] = {}
    missing = [key for key, _, _ in specs if key not in aux]
    if missing:
        raise KeyError("missing auxiliary trajectories: " + ",".join(missing))
    for key, prefix, signed_mode in specs:
        output.update(
            trajectory_window_features(
                np.asarray(aux[key], dtype=float),
                window_len=window_len,
                prefix=prefix,
                signed_mode=signed_mode,
            )
        )
    return output


def summarize(
    rows: list[dict[str, Any]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target"], row["control_set"], row["gcs_feature"])].append(row)

    output: list[dict[str, Any]] = []
    for (target, control_set, feature), items in sorted(grouped.items()):
        correlations = np.asarray([float(item["residual_spearman"]) for item in items])
        finite = correlations[np.isfinite(correlations)]
        nonzero = finite[np.abs(finite) > 1e-12]
        positives = int(np.sum(nonzero > 0))
        negatives = int(np.sum(nonzero < 0))
        ties = int(finite.size - nonzero.size)
        ci_low, ci_high = bootstrap_median_ci(
            finite,
            rng=rng,
            n_bootstrap=n_bootstrap,
        )
        config = (target, control_set, feature)
        output.append(
            {
                "target": target,
                "control_set": control_set,
                "gcs_feature": feature,
                "primary_preregistered": config == PRIMARY_CONFIG,
                "n_controls": len(CONTROL_SETS[control_set]),
                "n_runs": len(items),
                "n_finite": int(finite.size),
                "median_residual_spearman": float(np.median(finite)) if finite.size else float("nan"),
                "mean_residual_spearman": float(np.mean(finite)) if finite.size else float("nan"),
                "bootstrap_median_ci_low": ci_low,
                "bootstrap_median_ci_high": ci_high,
                "positive_runs": positives,
                "negative_runs": negatives,
                "tied_runs": ties,
                "sign_p_one_sided": exact_binomial_upper(positives, int(nonzero.size)),
            }
        )

    by_target: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        by_target[row["target"]].append(index)
    for indices in by_target.values():
        adjusted = holm_adjust([float(output[index]["sign_p_one_sided"]) for index in indices])
        for index, value in zip(indices, adjusted):
            output[index]["holm_p_all12_within_target"] = value

    return output


def render_readme(summary_rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    primary = next(row for row in summary_rows if row["primary_preregistered"])
    secondary = next(
        row
        for row in summary_rows
        if row["target"] == "attention_equivariance_drop"
        and row["control_set"] == "attention_full_dynamic"
        and row["gcs_feature"] == "gcs_abs_change"
    )
    if primary["median_residual_spearman"] > 0 and primary["bootstrap_median_ci_low"] > 0:
        decision = "Confirmatory"
    elif primary["median_residual_spearman"] > 0:
        decision = "Directional"
    else:
        decision = "Unsupported"

    lines = [
        "# Matched Dynamic-Proximity Control Audit",
        "",
        "Postprocess-only analysis under the protocol frozen on 2026-07-13. No new training or GCS computation.",
        "",
        "## Frozen primary result",
        "",
        f"- Frozen six-control decision: **{decision}**",
        "- Target: attention-equivariance drop",
        "- GCS feature: within-window range",
        "- Controls: signed motion, absolute endpoint motion, and within-window range for attention-output L2 and CKA",
        (
            f"- Median partial Spearman: {primary['median_residual_spearman']:.6f} "
            f"(20,000-run-bootstrap 95% CI "
            f"[{primary['bootstrap_median_ci_low']:.6f}, {primary['bootstrap_median_ci_high']:.6f}])"
        ),
        (
            f"- Positive runs: {primary['positive_runs']}/{primary['n_finite']}; "
            f"one-sided exact sign p={primary['sign_p_one_sided']:.6g}; "
            f"Holm p across all 12 target-specific rows={primary['holm_p_all12_within_target']:.6g}"
        ),
        (
            f"- Predeclared secondary `gcs_abs_change`: median "
            f"{secondary['median_residual_spearman']:.6f}, 95% CI "
            f"[{secondary['bootstrap_median_ci_low']:.6f}, "
            f"{secondary['bootstrap_median_ci_high']:.6f}], "
            f"{secondary['positive_runs']}/{secondary['n_finite']} positive, "
            f"all-12 Holm p={secondary['holm_p_all12_within_target']:.6g}"
        ),
        "",
        "## Interpretation boundary",
        "",
        "The frozen primary remains `gcs_range`; the numerically larger secondary is supporting evidence rather than a replacement headline. "
        "This analysis addresses feature-definition matching, not causality, scalar-selector dominance, or stage-independent information. "
        "A later training-stage sensitivity attenuates the primary to 0.117 [-0.032, 0.272] with linear stage and 0.088 [-0.054, 0.215] with linear plus quadratic stage. "
        "Reviewer-facing wording is therefore conditional residual association beyond the six named CKA/L2 summaries. "
        "The accuracy rows remain secondary because the leave-one-run-out top-20 accuracy increment is negative. "
        "Overlapping windows are never treated as independent inferential units.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/analyze_gcs_dynamic_proximity_controls.py \\",
        f"  --output-dir {args.output_dir} \\",
        f"  --equivariance {args.equivariance} \\",
        f"  --extra-runs-csv {args.extra_runs_csv} \\",
        f"  --auxiliary-suffix {args.auxiliary_suffix} \\",
        f"  --window-len {args.window_len} \\",
        f"  --seeds {args.seeds} \\",
        f"  --extra-heldout-seeds {args.extra_heldout_seeds}",
        "```",
        "",
        "Files: `window_rows.csv`, `per_run_correlations.csv`, `summary.csv`, `summary.json`, and `skipped.csv`.",
        "See `experiments/rebuttal_training_stage_sensitivity_20260713/README.md` for the claim-calibrating stage sensitivity.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="experiments/rebuttal_dynamic_proximity_controls_20260713",
    )
    parser.add_argument(
        "--equivariance",
        default=(
            "experiments/equivariance_analysis_workflow_seed_extension_20260611/"
            "equivariance_all_activations_plus_seed_extension.pkl"
        ),
    )
    parser.add_argument(
        "--extra-runs-csv",
        default="experiments/workflow_seed_extension_20260610/runs_manifest.csv",
    )
    parser.add_argument("--auxiliary-suffix", default="_wsel_aux_seedext20260614")
    parser.add_argument("--window-len", type=int, default=11)
    parser.add_argument("--tasks", default="add,sub,mul,div")
    parser.add_argument("--activations", default="ReLU,GeLU,SiLU")
    parser.add_argument("--seeds", default="123,777,31415,271828,161803,57721")
    parser.add_argument("--extra-heldout-seeds", default="31415,271828,161803,57721")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--rng-seed", type=int, default=20260713)
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    activations = {item.strip() for item in args.activations.split(",") if item.strip()}
    seeds = {int(item.strip()) for item in args.seeds.split(",") if item.strip()}
    heldout_extra = {
        int(item.strip()) for item in args.extra_heldout_seeds.split(",") if item.strip()
    }
    registered_runs = list(RUNS_1L) + load_extra_runs(resolve_path(args.extra_runs_csv))
    equivariance = load_equivariance(resolve_path(args.equivariance))

    window_rows: list[dict[str, Any]] = []
    per_run_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for act, task, seed, base in registered_runs:
        if act not in activations or task not in tasks or int(seed) not in seeds:
            continue
        experiment_dir = resolve_path(base)
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
            aux = load_auxiliary_selectors(experiment_dir, epochs, args.auxiliary_suffix)
            proximity = proximity_window_features(aux, args.window_len)
            gcs = gcs_window_features(
                np.asarray(scores["attention_path"], dtype=float),
                args.window_len,
            )
            targets = build_targets(act, task, seed, experiment_dir, metrics, epochs, equivariance)
        except Exception as exc:
            skipped.append({"act": act, "task": task, "seed": seed, "reason": repr(exc)})
            continue

        for target in targets:
            target_name = target["target"]
            if target_name not in TARGET_CONTROL_SETS:
                continue
            target_windows = all_window_scores(
                finite_array(target["values"]),
                args.window_len,
                target["mode"],
            )
            n_windows = int(target_windows.size)
            all_arrays = list(proximity.values()) + [gcs[name] for name in GCS_FEATURES]
            if any(array.size != n_windows for array in all_arrays):
                skipped.append(
                    {
                        "act": act,
                        "task": task,
                        "seed": seed,
                        "target": target_name,
                        "reason": "window_size_mismatch",
                    }
                )
                continue

            split = split_for_seed(seed, heldout_extra)
            run_identifier = f"{act}/{task}/s{int(seed)}"
            target_rank = rankdata(target_windows)
            for window_index in range(n_windows):
                row: dict[str, Any] = {
                    "split": split,
                    "run_id": run_identifier,
                    "act": act,
                    "task": task,
                    "seed": int(seed),
                    "target": target_name,
                    "window_len": int(args.window_len),
                    "window_index": window_index,
                    "window_start_epoch": int(epochs[window_index]),
                    "window_end_epoch": int(epochs[window_index + args.window_len - 1]),
                    "target_score": float(target_windows[window_index]),
                    "target_rank": float(target_rank[window_index]),
                }
                for name in GCS_FEATURES:
                    row[name] = float(gcs[name][window_index])
                for name, values in proximity.items():
                    row[name] = float(values[window_index])
                window_rows.append(row)

            for control_name in TARGET_CONTROL_SETS[target_name]:
                control_names = CONTROL_SETS[control_name]
                control_matrix = np.column_stack(
                    [rankdata(proximity[name]) for name in control_names]
                )
                target_residual, _ = residualize(target_rank, control_matrix)
                for feature_name in GCS_FEATURES:
                    feature_rank = rankdata(gcs[feature_name])
                    feature_residual, _ = residualize(feature_rank, control_matrix)
                    correlation = safe_corr(target_residual, feature_residual)
                    per_run_rows.append(
                        {
                            "split": split,
                            "run_id": run_identifier,
                            "act": act,
                            "task": task,
                            "seed": int(seed),
                            "target": target_name,
                            "control_set": control_name,
                            "control_features": ";".join(control_names),
                            "gcs_feature": feature_name,
                            "window_len": int(args.window_len),
                            "n_windows": n_windows,
                            "residual_spearman": float(correlation),
                        }
                    )

    rng = np.random.default_rng(args.rng_seed)
    summary_rows = summarize(per_run_rows, rng=rng, n_bootstrap=args.bootstrap)
    primary = next((row for row in summary_rows if row["primary_preregistered"]), None)
    if primary is None:
        raise RuntimeError("Frozen primary configuration was not produced")

    write_csv(output_dir / "window_rows.csv", window_rows)
    write_csv(output_dir / "per_run_correlations.csv", per_run_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "skipped.csv", skipped)
    payload = {
        "protocol_frozen": "2026-07-13",
        "window_len": args.window_len,
        "bootstrap_replicates": args.bootstrap,
        "rng_seed": args.rng_seed,
        "n_window_rows": len(window_rows),
        "n_per_run_rows": len(per_run_rows),
        "n_skipped": len(skipped),
        "control_sets": {key: list(value) for key, value in CONTROL_SETS.items()},
        "target_control_sets": {key: list(value) for key, value in TARGET_CONTROL_SETS.items()},
        "primary_config": list(PRIMARY_CONFIG),
        "summary": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "README.md").write_text(render_readme(summary_rows, args))

    print(f"Saved matched-dynamic audit to {output_dir}")
    print(
        "PRIMARY "
        f"rho={primary['median_residual_spearman']:.6f} "
        f"CI=[{primary['bootstrap_median_ci_low']:.6f},"
        f"{primary['bootstrap_median_ci_high']:.6f}] "
        f"positive={primary['positive_runs']}/{primary['n_finite']} "
        f"p={primary['sign_p_one_sided']:.6g} "
        f"Holm={primary['holm_p_all12_within_target']:.6g}"
    )


if __name__ == "__main__":
    main()
