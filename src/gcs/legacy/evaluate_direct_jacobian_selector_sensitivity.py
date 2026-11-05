#!/usr/bin/env python3
"""Cached selector-rule sensitivity and paired audit for Direct-J attention."""

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

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis.direct_jacobian_sensitivity import (  # noqa: E402
    paired_percentile_summary,
    paired_top20_table,
    select_change_center,
)
from gcs.legacy.diagnostics_common import (  # noqa: E402
    RUNS_1L,
    detect_grok_epoch,
    load_gcs,
    load_metrics,
    resolve_path,
)
from gcs.legacy.evaluate_workflow_selection_heldout import (  # noqa: E402
    build_targets,
    finite_array,
    load_equivariance,
    score_window,
    select_first_prominent_post_valley,
    window_bounds,
)


RULES = {
    "post_valley_peak": None,
    "largest_absolute_change": "abs",
    "largest_rise": "rise",
    "largest_drop": "drop",
}
TARGETS = {
    "test_accuracy_rise": "Accuracy rise",
    "attention_equivariance_drop": "Attention equivariance drop",
}
EXPECTED_DENOMINATORS = {
    "test_accuracy_rise": 23,
    "attention_equivariance_drop": 11,
}


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_slug(act: str, task: str, seed: int) -> str:
    return f"{act.lower()}_{task}_s{int(seed)}"


def run_key(act: str, task: str, seed: int) -> tuple[str, str, int]:
    return str(act), str(task), int(seed)


def summarize_sensitivity(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["rule"]), str(row["target"]))].append(
            float(row["percentile"])
        )

    output: list[dict[str, Any]] = []
    for rule in RULES:
        for target in TARGETS:
            values = np.asarray(grouped.get((rule, target), []), dtype=float)
            if values.size == 0:
                continue
            output.append(
                {
                    "rule": rule,
                    "target": target,
                    "window_len": int(rows[0]["window_len"]),
                    "n_runs": int(values.size),
                    "median_percentile": float(np.median(values)),
                    "mean_percentile": float(np.mean(values)),
                    "top20_hit_count": int(np.sum(values >= 0.8)),
                    "top20_hit_rate": float(np.mean(values >= 0.8)),
                }
            )
    return output


def compare_with_main_evaluation(
    recomputed_rows: list[dict[str, Any]],
    canonical_path: Path,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    canonical_rows = read_csv(canonical_path)
    canonical = {
        (
            row["act"],
            row["task"],
            int(row["seed"]),
            row["selector"],
            row["target"],
        ): row
        for row in canonical_rows
        if row["selector"] in {"gcs_attention", "direct_jacobian_attention"}
    }
    recomputed = {
        (
            row["act"],
            row["task"],
            int(row["seed"]),
            row["selector"],
            row["target"],
        ): row
        for row in recomputed_rows
    }
    shared = sorted(set(canonical) & set(recomputed))
    missing = sorted(set(canonical) - set(recomputed))
    unexpected = sorted(set(recomputed) - set(canonical))
    maximum_difference = 0.0
    mismatches: list[str] = []
    for key in shared:
        old = canonical[key]
        new = recomputed[key]
        for field in ("center_epoch", "window_start_epoch", "window_end_epoch"):
            if int(old[field]) != int(new[field]):
                mismatches.append(f"{key}/{field}: {old[field]} != {new[field]}")
        difference = abs(float(old["percentile"]) - float(new["percentile"]))
        maximum_difference = max(maximum_difference, difference)
        if difference > tolerance:
            mismatches.append(
                f"{key}/percentile: {old['percentile']} != {new['percentile']}"
            )
    return {
        "canonical_rows": len(canonical),
        "recomputed_rows": len(recomputed),
        "shared_rows": len(shared),
        "missing_rows": len(missing),
        "unexpected_rows": len(unexpected),
        "maximum_percentile_difference": maximum_difference,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def format_hit(row: dict[str, Any]) -> str:
    return (
        f"{int(row['top20_hit_count'])}/{int(row['n_runs'])} "
        f"({float(row['top20_hit_rate']):.3f})"
    )


def build_readme(
    sensitivity_summary: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    *,
    main_reference: dict[str, dict[str, float | int]],
    audit_status: str,
    main_reproduction: dict[str, Any],
) -> str:
    indexed = {
        (str(row["rule"]), str(row["target"])): row
        for row in sensitivity_summary
    }
    paired = {str(row["target"]): row for row in paired_summary}
    lines = [
        "# Direct-Jacobian Selector Sensitivity and Paired Audit",
        "",
        f"Status: **{audit_status}**",
        "",
        "This audit uses only the cached direct-Jacobian trajectories. It does",
        "not recompute JVPs or tune a rule on held-out targets. All four frozen",
        "rules are reported.",
        "",
        "## Selector-rule sensitivity",
        "",
        "| Selector | Accuracy median | Accuracy top-20 | "
        "Equivariance median | Equivariance top-20 |",
        "|---|---:|---:|---:|---:|",
    ]
    gcs_accuracy = main_reference["test_accuracy_rise"]
    gcs_equivariance = main_reference["attention_equivariance_drop"]
    lines.append(
        "| Attention-GCS post-valley peak | "
        f"{float(gcs_accuracy['median_percentile']):.3f} | "
        f"{int(gcs_accuracy['top20_hit_count'])}/"
        f"{int(gcs_accuracy['n_runs'])} "
        f"({float(gcs_accuracy['top20_hit_rate']):.3f}) | "
        f"{float(gcs_equivariance['median_percentile']):.3f} | "
        f"{int(gcs_equivariance['top20_hit_count'])}/"
        f"{int(gcs_equivariance['n_runs'])} "
        f"({float(gcs_equivariance['top20_hit_rate']):.3f}) |"
    )
    labels = {
        "post_valley_peak": "Direct-J attention: post-valley peak",
        "largest_absolute_change": "Direct-J attention: largest absolute change",
        "largest_rise": "Direct-J attention: largest rise",
        "largest_drop": "Direct-J attention: largest drop",
    }
    for rule in RULES:
        accuracy = indexed[(rule, "test_accuracy_rise")]
        equivariance = indexed[(rule, "attention_equivariance_drop")]
        lines.append(
            f"| {labels[rule]} | "
            f"{float(accuracy['median_percentile']):.3f} | "
            f"{format_hit(accuracy)} | "
            f"{float(equivariance['median_percentile']):.3f} | "
            f"{format_hit(equivariance)} |"
        )

    lines.extend(
        [
            "",
            "## Same-run paired primary comparison",
            "",
            "Differences are Attention-GCS minus Direct-J attention under the",
            "original frozen post-valley-peak rule.",
            "",
            "| Target | Median percentile difference [95% paired bootstrap CI] | "
            "Positive / tied / negative runs | "
            "Both / GCS-only / Direct-J-only / neither top-20 | "
            "Exact McNemar p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for target in TARGETS:
        row = paired[target]
        lines.append(
            f"| {TARGETS[target]} | "
            f"{float(row['median_difference']):.3f} "
            f"[{float(row['median_bootstrap_ci_low']):.3f}, "
            f"{float(row['median_bootstrap_ci_high']):.3f}] | "
            f"{int(row['positive_runs'])} / {int(row['tied_runs'])} / "
            f"{int(row['negative_runs'])} | "
            f"{int(row['both_hit'])} / {int(row['gcs_only_hit'])} / "
            f"{int(row['direct_only_hit'])} / {int(row['neither_hit'])} | "
            f"{float(row['mcnemar_exact_p_two_sided']):.4g} |"
        )

    fully_recovering_rules = [
        rule
        for rule in RULES
        if rule != "post_valley_peak"
        and all(
            float(indexed[(rule, target)]["median_percentile"])
            >= float(main_reference[target]["median_percentile"])
            and float(indexed[(rule, target)]["top20_hit_rate"])
            >= float(main_reference[target]["top20_hit_rate"])
            for target in TARGETS
        )
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if fully_recovering_rules:
        readable = " and ".join(
            rule.replace("_", "-") for rule in fully_recovering_rules
        )
        lines.extend(
            [
                f"The {readable} rules reach or exceed Attention-GCS",
                "on both the median percentile and top-20 rate for both targets.",
                "This activates the recovery branch frozen in the sensitivity protocol:",
                "the strict post-valley drop-in gap is real, as confirmed by the",
                "paired audit, but it is selector-shape dependent and cannot support",
                "a general claim that the solution frame is necessary or superior",
                "for attention-window localization.",
                "",
                "The supported split result is narrower: GCS exposes the submitted",
                "solution-conditioned non-monotone coherence peak, while abrupt",
                "changes—especially drops—in the direct attention-Jacobian graph",
                "provide a strong alternative transition locator. The change-rule",
                "result strengthens rather than weakens the need to describe the",
                "main experiment as a frozen drop-in construction ablation, not as",
                "an optimized comparison of method families.",
            ]
        )
    else:
        lines.extend(
            [
                "No single fixed change rule reaches the Attention-GCS median and",
                "top-20 rate on both targets. The detailed table should be reported",
                "without choosing a held-out winner.",
            ]
        )
    lines.extend(
        [
            "",
            "This remains a cached sensitivity, not an optimization of the best",
            "possible Direct-J method. It does not isolate final neighborhoods from",
            "local directions, establish uniform dominance, or change the strong",
            "CKA/L2 boundary.",
            "",
            "## Integrity checks",
            "",
            f"- Main evaluator rows reproduced: "
            f"{main_reproduction['shared_rows']}/"
            f"{main_reproduction['canonical_rows']}.",
            f"- Maximum main percentile discrepancy: "
            f"{main_reproduction['maximum_percentile_difference']:.3g}.",
            "- Expected denominators: 23 accuracy and 11 equivariance.",
            "- Bootstrap resamples: 20,000 paired runs.",
            "- Independent reconstruction: see `INDEPENDENT_AUDIT.md`.",
            "",
            "Files: `selector_sensitivity_per_run.csv`,",
            "`selector_sensitivity_summary.csv`, `paired_per_run.csv`,",
            "`paired_summary.csv`, `audit.json`, and `independent_audit.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Direct-J attention selector rules and paired results."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/rebuttal_direct_jacobian_graph_20260723"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "rebuttal_2026/direct_jacobian_selector_paired_audit_20260723"
        ),
    )
    parser.add_argument(
        "--equivariance",
        type=Path,
        default=Path(
            "experiments/equivariance_analysis/equivariance_all_activations.pkl"
        ),
    )
    parser.add_argument("--seeds", default="123,777")
    parser.add_argument("--window-len", type=int, default=11)
    parser.add_argument("--bootstrap-trials", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_len <= 0 or args.window_len % 2 == 0:
        raise ValueError("--window-len must be a positive odd integer")

    experiment_dir = resolve_path(args.experiment_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equivariance = load_equivariance(resolve_path(args.equivariance))
    seeds = {int(value) for value in args.seeds.split(",") if value.strip()}
    runs = [run for run in RUNS_1L if int(run[2]) in seeds]

    sensitivity_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for act, task, seed, base in runs:
        trajectory_path = (
            experiment_dir / "trajectories" / f"{run_slug(act, task, seed)}.pkl"
        )
        payload = load_pickle(trajectory_path)
        if not bool(payload.get("complete", False)):
            raise RuntimeError(f"Incomplete trajectory: {trajectory_path}")
        epochs = np.asarray(payload["epochs"], dtype=int)
        if not np.array_equal(epochs, np.asarray(payload["target_epochs"], dtype=int)):
            raise RuntimeError(f"Incomplete epoch schedule: {trajectory_path}")
        direct = finite_array(payload["path_scores"]["attention_path"])
        existing_epochs, existing_scores = load_gcs(base)
        if not np.array_equal(epochs, np.asarray(existing_epochs, dtype=int)):
            raise RuntimeError(f"GCS epoch mismatch: {act}/{task}/s{seed}")
        gcs = finite_array(existing_scores["attention_path"])

        metrics = load_metrics(base)
        if detect_grok_epoch(metrics) is None:
            raise RuntimeError(f"Held-out grok gate failed: {act}/{task}/s{seed}")
        targets = build_targets(
            act,
            task,
            int(seed),
            resolve_path(base),
            metrics,
            epochs,
            equivariance,
        )
        targets = [target for target in targets if target["target"] in TARGETS]

        centers = {
            "post_valley_peak": select_first_prominent_post_valley(
                direct,
                args.window_len,
            )
        }
        for rule, mode in RULES.items():
            if mode is None:
                continue
            centers[rule] = select_change_center(
                epochs,
                direct,
                window_len=args.window_len,
                mode=mode,
                smooth_window=3,
            )
        gcs_center = select_first_prominent_post_valley(gcs, args.window_len)

        for target in targets:
            target_name = str(target["target"])
            target_values = np.asarray(target["values"], dtype=float)
            target_mode = str(target["mode"])
            run_identifier = run_key(act, task, int(seed))

            gcs_start, gcs_end = window_bounds(
                gcs_center,
                len(epochs),
                args.window_len,
            )
            gcs_stats = score_window(
                target_values,
                args.window_len,
                gcs_start,
                target_mode,
            )
            main_rows.append(
                {
                    "act": act,
                    "task": task,
                    "seed": int(seed),
                    "selector": "gcs_attention",
                    "target": target_name,
                    "center_epoch": int(epochs[gcs_center]),
                    "window_start_epoch": int(epochs[gcs_start]),
                    "window_end_epoch": int(epochs[gcs_end]),
                    "percentile": float(gcs_stats["percentile"]),
                }
            )

            direct_main_percentile: float | None = None
            for rule in RULES:
                center = centers[rule]
                start, end = window_bounds(center, len(epochs), args.window_len)
                stats = score_window(
                    target_values,
                    args.window_len,
                    start,
                    target_mode,
                )
                row = {
                    "act": act,
                    "task": task,
                    "seed": int(seed),
                    "exp_dir": str(resolve_path(base)),
                    "rule": rule,
                    "target": target_name,
                    "target_mode": target_mode,
                    "window_len": int(args.window_len),
                    "center_epoch": int(epochs[center]),
                    "window_start_epoch": int(epochs[start]),
                    "window_end_epoch": int(epochs[end]),
                    "selected_score": float(stats["selected_score"]),
                    "percentile": float(stats["percentile"]),
                    "top20_hit": bool(float(stats["percentile"]) >= 0.8),
                    "best_window_start_epoch": int(
                        epochs[int(stats["best_window_start_idx"])]
                    ),
                    "best_score": float(stats["best_score"]),
                }
                sensitivity_rows.append(row)
                if rule == "post_valley_peak":
                    direct_main_percentile = float(stats["percentile"])
                    main_rows.append(
                        {
                            "act": act,
                            "task": task,
                            "seed": int(seed),
                            "selector": "direct_jacobian_attention",
                            "target": target_name,
                            "center_epoch": int(epochs[center]),
                            "window_start_epoch": int(epochs[start]),
                            "window_end_epoch": int(epochs[end]),
                            "percentile": float(stats["percentile"]),
                        }
                    )
            if direct_main_percentile is None:
                raise AssertionError("Missing post-valley Direct-J percentile")
            paired_rows.append(
                {
                    "act": run_identifier[0],
                    "task": run_identifier[1],
                    "seed": run_identifier[2],
                    "target": target_name,
                    "gcs_percentile": float(gcs_stats["percentile"]),
                    "direct_jacobian_percentile": direct_main_percentile,
                    "gcs_minus_direct_difference": (
                        float(gcs_stats["percentile"]) - direct_main_percentile
                    ),
                    "gcs_top20_hit": bool(float(gcs_stats["percentile"]) >= 0.8),
                    "direct_jacobian_top20_hit": bool(
                        direct_main_percentile >= 0.8
                    ),
                }
            )

    sensitivity_summary = summarize_sensitivity(sensitivity_rows)
    for row in sensitivity_summary:
        expected = EXPECTED_DENOMINATORS[str(row["target"])]
        if int(row["n_runs"]) != expected:
            raise RuntimeError(f"Unexpected denominator: {row}")

    paired_summary: list[dict[str, Any]] = []
    main_reference: dict[str, dict[str, float | int]] = {}
    for target in TARGETS:
        items = [row for row in paired_rows if row["target"] == target]
        gcs_values = np.asarray([row["gcs_percentile"] for row in items], dtype=float)
        direct_values = np.asarray(
            [row["direct_jacobian_percentile"] for row in items],
            dtype=float,
        )
        if gcs_values.size != EXPECTED_DENOMINATORS[target]:
            raise RuntimeError(f"Unexpected paired denominator for {target}")
        percentile_summary = paired_percentile_summary(
            gcs_values,
            direct_values,
            bootstrap_trials=args.bootstrap_trials,
            bootstrap_seed=args.bootstrap_seed,
        )
        hit_table = paired_top20_table(gcs_values, direct_values)
        paired_summary.append(
            {
                "target": target,
                **percentile_summary,
                **{
                    key: value
                    for key, value in hit_table.items()
                    if key not in {"threshold", "n_runs"}
                },
            }
        )
        main_reference[target] = {
            "n_runs": int(gcs_values.size),
            "median_percentile": float(np.median(gcs_values)),
            "top20_hit_count": int(np.sum(gcs_values >= 0.8)),
            "top20_hit_rate": float(np.mean(gcs_values >= 0.8)),
        }

    main_reproduction = compare_with_main_evaluation(
        main_rows,
        experiment_dir / "per_run_windows.csv",
    )
    errors: list[str] = []
    if main_reproduction["missing_rows"]:
        errors.append("Missing canonical main rows")
    if main_reproduction["unexpected_rows"]:
        errors.append("Unexpected recomputed main rows")
    if main_reproduction["mismatch_count"]:
        errors.append("Main evaluation reproduction mismatch")

    write_csv(output_dir / "selector_sensitivity_per_run.csv", sensitivity_rows)
    write_csv(output_dir / "selector_sensitivity_summary.csv", sensitivity_summary)
    write_csv(output_dir / "paired_per_run.csv", paired_rows)
    write_csv(output_dir / "paired_summary.csv", paired_summary)
    audit = {
        "status": "PASS" if not errors else "FAIL",
        "protocol": str(
            ROOT
            / "rebuttal_2026"
            / "DIRECT_JACOBIAN_SELECTOR_PAIRED_AUDIT_PROTOCOL_20260723.md"
        ),
        "experiment_dir": str(experiment_dir),
        "heldout_runs": len(runs),
        "expected_denominators": EXPECTED_DENOMINATORS,
        "rules": list(RULES),
        "window_len": args.window_len,
        "smoothing_window": 3,
        "bootstrap_trials": args.bootstrap_trials,
        "bootstrap_seed": args.bootstrap_seed,
        "main_reproduction": main_reproduction,
        "errors": errors,
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    readme = build_readme(
        sensitivity_summary,
        paired_summary,
        main_reference=main_reference,
        audit_status=str(audit["status"]),
        main_reproduction=main_reproduction,
    )
    (output_dir / "README.md").write_text(readme)
    print(readme)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
