#!/usr/bin/env python3
"""Evaluate frozen selectors against cached attention-specific replacement cost."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcs.reproduce.mechanistic_statistics import (  # noqa: E402
    evaluate_selected_donor,
    mechanistic_percentile,
    paired_selector_summary,
    select_oracle_center,
)


PROTOCOL_VERSION = "mechanistic_yield_selector_audit_v1_20260723"
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_723
EXPECTED_RUNS = 11

DONOR_SELECTORS = {
    "attention_gcs": "gcs_window_center",
    "attention_cka": "attn_cka_window_center",
    "attention_l2": "attn_l2_window_center",
    "final_repr_cka": "final_cka_window_center",
}
DIRECT_J_RULES = (
    "post_valley_peak",
    "largest_absolute_change",
    "largest_rise",
    "largest_drop",
)
SELECTOR_ORDER = (
    "attention_gcs",
    "attention_cka",
    "attention_l2",
    "final_repr_cka",
    "accuracy_oracle",
    "equivariance_oracle",
    "direct_j_post_valley_peak",
    "direct_j_largest_absolute_change",
    "direct_j_largest_rise",
    "direct_j_largest_drop",
)
DISPLAY_NAMES = {
    "attention_gcs": "Attention-GCS",
    "attention_cka": "Attention-output CKA",
    "attention_l2": "Attention-output L2",
    "final_repr_cka": "Final-representation CKA",
    "accuracy_oracle": "Accuracy-rise oracle",
    "equivariance_oracle": "Equivariance-improvement oracle",
    "direct_j_post_valley_peak": "Direct-J post-valley peak",
    "direct_j_largest_absolute_change": "Direct-J largest absolute change",
    "direct_j_largest_rise": "Direct-J largest rise",
    "direct_j_largest_drop": "Direct-J largest drop",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def run_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (
        row["act"].strip().lower(),
        row["task"].strip().lower(),
        int(row["seed"]),
    )


def run_id(key: tuple[str, str, int]) -> str:
    act, task, seed = key
    return f"{act}_{task}_s{seed}"


def integer_epoch(value: str | int | float) -> int:
    numeric = float(value)
    rounded = int(round(numeric))
    if not np.isfinite(numeric) or not np.isclose(numeric, rounded):
        raise ValueError(f"non-integral epoch: {value!r}")
    return rounded


def index_unique_epochs(
    rows: list[dict[str, str]],
    *,
    value_field: str,
) -> dict[tuple[str, str, int], int]:
    indexed: dict[tuple[str, str, int], int] = {}
    for row in rows:
        key = run_key(row)
        value = integer_epoch(row[value_field])
        if key in indexed:
            raise ValueError(f"duplicate selector row for {run_id(key)}")
        indexed[key] = value
    return indexed


def bootstrap_median_interval(values: np.ndarray) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        values,
        size=(BOOTSTRAP_RESAMPLES, values.size),
        replace=True,
    )
    medians = np.median(samples, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def build_selector_epochs(
    donor_rows: list[dict[str, str]],
    direct_rows: list[dict[str, str]],
    dynamic_rows: list[dict[str, str]],
) -> tuple[
    dict[str, dict[tuple[str, str, int], int]],
    dict[str, int],
]:
    selectors: dict[str, dict[tuple[str, str, int], int]] = {}
    source_counts: dict[str, int] = {}

    for selector, donor_label in DONOR_SELECTORS.items():
        rows = [
            row
            for row in donor_rows
            if row["donor_label"] == donor_label
            and row["window_split"] == "heldout_seed123_777"
        ]
        selectors[selector] = index_unique_epochs(rows, value_field="donor_epoch")
        source_counts[selector] = len(rows)

    direct_by_rule_target: dict[
        tuple[str, str], dict[tuple[str, str, int], int]
    ] = {}
    for rule in DIRECT_J_RULES:
        for target in ("test_accuracy_rise", "attention_equivariance_drop"):
            rows = [
                row
                for row in direct_rows
                if row["rule"] == rule and row["target"] == target
            ]
            direct_by_rule_target[(rule, target)] = index_unique_epochs(
                rows,
                value_field="center_epoch",
            )
        accuracy = direct_by_rule_target[(rule, "test_accuracy_rise")]
        equivariance = direct_by_rule_target[(rule, "attention_equivariance_drop")]
        for key, center in equivariance.items():
            if accuracy.get(key) != center:
                raise ValueError(
                    f"Direct-J center differs by target for {rule}/{run_id(key)}"
                )
        selector = f"direct_j_{rule}"
        selectors[selector] = equivariance
        source_counts[selector] = len(equivariance)

    dynamic_by_target: dict[
        str, dict[tuple[str, str, int], list[dict[str, str]]]
    ] = {
        "test_accuracy_rise": defaultdict(list),
        "attention_equivariance_drop": defaultdict(list),
    }
    for row in dynamic_rows:
        target = row["target"]
        if target in dynamic_by_target:
            dynamic_by_target[target][run_key(row)].append(row)
    for target, selector in (
        ("test_accuracy_rise", "accuracy_oracle"),
        ("attention_equivariance_drop", "equivariance_oracle"),
    ):
        selectors[selector] = {
            key: select_oracle_center(rows)
            for key, rows in dynamic_by_target[target].items()
        }
        source_counts[selector] = len(selectors[selector])

    return selectors, source_counts


def build_selected_rows(
    all_window_rows: list[dict[str, str]],
    selector_epochs: dict[str, dict[tuple[str, str, int], int]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[str, str, int]],
]:
    all_by_run: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in all_window_rows:
        all_by_run[run_key(row)].append(row)

    shared_keys = set(all_by_run)
    for selector in SELECTOR_ORDER:
        shared_keys &= set(selector_epochs[selector])
    keys = sorted(shared_keys)
    if len(keys) != EXPECTED_RUNS:
        observed = ", ".join(run_id(key) for key in keys)
        raise ValueError(
            f"expected {EXPECTED_RUNS} shared runs, got {len(keys)}: {observed}"
        )
    expected_keys = {
        ("relu", "add", 123),
        ("relu", "add", 777),
        ("relu", "mul", 123),
        ("relu", "mul", 777),
        ("gelu", "add", 123),
        ("gelu", "add", 777),
        ("gelu", "mul", 123),
        ("gelu", "mul", 777),
        ("silu", "add", 777),
        ("silu", "mul", 123),
        ("silu", "mul", 777),
    }
    if set(keys) != expected_keys:
        raise ValueError("shared cohort differs from the frozen 11-run cohort")

    selected_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for key in keys:
        run_rows = sorted(
            all_by_run[key],
            key=lambda row: integer_epoch(row["donor_epoch"]),
        )
        legal_outcomes = np.asarray(
            [float(row["attn_minus_mlp_delta"]) for row in run_rows],
            dtype=float,
        )
        legal_percentiles = np.asarray(
            [
                mechanistic_percentile(legal_outcomes, outcome)
                for outcome in legal_outcomes
            ]
        )
        first = run_rows[0]
        random_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": run_id(key),
                "act": first["act"],
                "task": first["task"],
                "seed": int(first["seed"]),
                "n_legal_donors": len(run_rows),
                "expected_mechanistic_target": float(np.mean(legal_outcomes)),
                "expected_mechanistic_percentile": float(
                    np.mean(legal_percentiles)
                ),
                "expected_top20_rate": float(np.mean(legal_percentiles >= 0.8)),
                "expected_positive_target_rate": float(
                    np.mean(legal_outcomes > 0.0)
                ),
            }
        )
        for selector in SELECTOR_ORDER:
            selected_epoch = selector_epochs[selector][key]
            evaluated = evaluate_selected_donor(
                run_rows,
                selected_epoch=selected_epoch,
            )
            selected_rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": run_id(key),
                    "act": first["act"],
                    "task": first["task"],
                    "seed": int(first["seed"]),
                    "selector": selector,
                    "selector_display": DISPLAY_NAMES[selector],
                    **evaluated,
                }
            )
    return selected_rows, random_rows, keys


def summarize_selectors(
    selected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[str(row["selector"])].append(row)
    summaries: list[dict[str, Any]] = []
    for selector in SELECTOR_ORDER:
        rows = grouped[selector]
        if len(rows) != EXPECTED_RUNS:
            raise ValueError(f"{selector} has {len(rows)} rows")
        percentiles = np.asarray(
            [float(row["mechanistic_percentile"]) for row in rows]
        )
        targets = np.asarray([float(row["mechanistic_target"]) for row in rows])
        timing = np.asarray([float(row["donor_minus_t_grok"]) for row in rows])
        accuracy = np.asarray([float(row["donor_test_accuracy"]) for row in rows])
        low, high = bootstrap_median_interval(percentiles)
        summaries.append(
            {
                "selector": selector,
                "selector_display": DISPLAY_NAMES[selector],
                "n_runs": len(rows),
                "median_mechanistic_percentile": float(np.median(percentiles)),
                "median_percentile_bootstrap_ci_low": low,
                "median_percentile_bootstrap_ci_high": high,
                "top20_hit_count": int(np.sum(percentiles >= 0.8)),
                "top20_hit_rate": float(np.mean(percentiles >= 0.8)),
                "positive_target_count": int(np.sum(targets > 0.0)),
                "positive_target_rate": float(np.mean(targets > 0.0)),
                "median_mechanistic_target": float(np.median(targets)),
                "median_donor_minus_t_grok": float(np.median(timing)),
                "median_donor_test_accuracy": float(np.median(accuracy)),
            }
        )
    return summaries


def summarize_random(
    random_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fields = (
        "expected_mechanistic_target",
        "expected_mechanistic_percentile",
        "expected_top20_rate",
        "expected_positive_target_rate",
    )
    summary: dict[str, Any] = {
        "selector": "uniform_random_legal_expectation",
        "selector_display": "Uniform-random legal donor expectation",
        "n_runs": len(random_rows),
    }
    for field in fields:
        values = np.asarray([float(row[field]) for row in random_rows])
        summary[f"median_{field}"] = float(np.median(values))
        summary[f"mean_{field}"] = float(np.mean(values))
    return summary


def summarize_pairs(
    selected_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[str(row["selector"])].append(row)
    gcs = grouped["attention_gcs"]
    pairwise: list[dict[str, Any]] = []
    for selector in SELECTOR_ORDER:
        if selector == "attention_gcs":
            continue
        comparator = grouped[selector]
        paired = paired_selector_summary(
            gcs,
            comparator,
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            bootstrap_seed=BOOTSTRAP_SEED,
        )
        gcs_by_run = {str(row["run_id"]): row for row in gcs}
        comparator_by_run = {str(row["run_id"]): row for row in comparator}
        common = sorted(gcs_by_run)
        target_differences = np.asarray(
            [
                float(gcs_by_run[key]["mechanistic_target"])
                - float(comparator_by_run[key]["mechanistic_target"])
                for key in common
            ]
        )
        timing_differences = np.asarray(
            [
                float(gcs_by_run[key]["donor_minus_t_grok"])
                - float(comparator_by_run[key]["donor_minus_t_grok"])
                for key in common
            ]
        )
        pairwise.append(
            {
                "comparator": selector,
                "comparator_display": DISPLAY_NAMES[selector],
                **paired,
                "median_mechanistic_target_difference": float(
                    np.median(target_differences)
                ),
                "median_timing_difference_epochs": float(
                    np.median(timing_differences)
                ),
            }
        )
    return pairwise


def format_ratio(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.3f})"


def build_readme(
    summaries: list[dict[str, Any]],
    random_summary: dict[str, Any],
    pairwise: list[dict[str, Any]],
    keys: list[tuple[str, str, int]],
) -> str:
    lines = [
        "# Mechanistic-Yield Selector Audit",
        "",
        "Status: **PASS**",
        "",
        "This post-result audit joins frozen selector centers to the cached",
        "all-window final-base surgery grid. The primary outcome is attention-",
        "specific replacement cost: attention-reset accuracy cost minus MLP-reset",
        "accuracy cost. Higher within-run percentiles indicate larger attention-",
        "specific cost. The run is the summary unit.",
        "",
        "## Held-out selector comparison",
        "",
        "| Selector | Median percentile [95% run-bootstrap CI] | Top-20 | M>0 | Median M | Median donor - grok | Median donor acc. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        total = int(row["n_runs"])
        lines.append(
            "| "
            f"{row['selector_display']} | "
            f"{float(row['median_mechanistic_percentile']):.3f} "
            f"[{float(row['median_percentile_bootstrap_ci_low']):.3f}, "
            f"{float(row['median_percentile_bootstrap_ci_high']):.3f}] | "
            f"{format_ratio(int(row['top20_hit_count']), total)} | "
            f"{format_ratio(int(row['positive_target_count']), total)} | "
            f"{float(row['median_mechanistic_target']):.3f} | "
            f"{float(row['median_donor_minus_t_grok']):.0f} | "
            f"{float(row['median_donor_test_accuracy']):.3f} |"
        )
    lines.extend(
        [
            "",
            "Uniformly sampling a legal donor has a median across-run expected",
            f"percentile of {float(random_summary['median_expected_mechanistic_percentile']):.3f}, "
            f"top-20 probability {float(random_summary['median_expected_top20_rate']):.3f}, "
            "and positive-target probability "
            f"{float(random_summary['median_expected_positive_target_rate']):.3f}.",
            "",
            "## Same-run GCS comparisons",
            "",
            "Differences are Attention-GCS minus the named comparator.",
            "",
            "| Comparator | Median percentile difference [95% CI] | Positive / tied / negative | Both / GCS-only / comparator-only / neither top-20 | McNemar p | Median M difference | Median timing difference |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise:
        lines.append(
            "| "
            f"{row['comparator_display']} | "
            f"{float(row['median_difference']):.3f} "
            f"[{float(row['median_bootstrap_ci_low']):.3f}, "
            f"{float(row['median_bootstrap_ci_high']):.3f}] | "
            f"{int(row['positive_runs'])} / {int(row['tied_runs'])} / "
            f"{int(row['negative_runs'])} | "
            f"{int(row['both_top20'])} / {int(row['gcs_only_top20'])} / "
            f"{int(row['comparator_only_top20'])} / "
            f"{int(row['neither_top20'])} | "
            f"{float(row['mcnemar_exact_two_sided_p']):.4g} | "
            f"{float(row['median_mechanistic_target_difference']):.3f} | "
            f"{float(row['median_timing_difference_epochs']):.0f} |"
        )

    non_gcs = [row for row in summaries if row["selector"] != "attention_gcs"]
    gcs = next(row for row in summaries if row["selector"] == "attention_gcs")
    comparator_matches_or_exceeds = [
        row
        for row in non_gcs
        if float(row["median_mechanistic_percentile"])
        >= float(gcs["median_mechanistic_percentile"])
        or int(row["top20_hit_count"]) >= int(gcs["top20_hit_count"])
    ]
    lines.extend(
        [
            "",
            "## Frozen interpretation boundary",
            "",
        ]
    )
    if comparator_matches_or_exceeds:
        names = ", ".join(
            str(row["selector_display"]) for row in comparator_matches_or_exceeds
        )
        lines.extend(
            [
                "At least one comparator matches or exceeds Attention-GCS on the",
                f"median percentile or top-20 count ({names}). The audit therefore",
                "does not support unique or superior mechanistic-yield recovery by",
                "GCS. Any useful difference must be described as timing, analysis-",
                "object, or hypothesis complementarity and checked against the paired",
                "rows above.",
            ]
        )
    else:
        lines.extend(
            [
                "Attention-GCS has the largest aggregate median percentile and top-20",
                "count in this frozen table. Because this is a small post-result audit",
                "with multiple comparisons, it supports matched follow-up yield but",
                "not unique discovery or a universal superiority claim.",
            ]
        )
    lines.extend(
        [
            "",
            "The Direct-J transition sensitivity remains binding regardless of this",
            "table: the full solution-frame construction is not necessary for generic",
            "transition localization.",
            "",
            "## Integrity",
            "",
            f"- Shared held-out cohort: {len(keys)}/{EXPECTED_RUNS} runs.",
            f"- Selected donor rows: {len(summaries) * len(keys)}.",
            f"- Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}.",
            "- Every selected center matched exactly one cached legal donor.",
            "- Cached M was checked against delta-attention minus delta-MLP.",
            "- See `INDEPENDENT_AUDIT.md` for a separate reconstruction.",
            "",
            "Files: `selected_donors_per_run.csv`, `selector_summary.csv`,",
            "`gcs_pairwise_summary.csv`, `uniform_random_per_run.csv`,",
            "`uniform_random_summary.json`, and `audit.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-window",
        type=Path,
        default=ROOT
        / "experiments/rebuttal_all_window_intervention_yield_20260712"
        / "analysis/merged_windows.csv",
    )
    parser.add_argument(
        "--donor-yield",
        type=Path,
        default=ROOT
        / "experiments/rebuttal_donor_yield_gcs_cka_l2_20260706"
        / "donor_yield_surgery_per_run.csv",
    )
    parser.add_argument(
        "--direct-j",
        type=Path,
        default=ROOT
        / "rebuttal_2026/direct_jacobian_selector_paired_audit_20260723"
        / "selector_sensitivity_per_run.csv",
    )
    parser.add_argument(
        "--dynamic-windows",
        type=Path,
        default=ROOT
        / "experiments/rebuttal_dynamic_proximity_controls_20260713"
        / "window_rows.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "rebuttal_2026/mechanistic_yield_selector_audit_20260723",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_window_rows = read_csv(args.all_window)
    donor_rows = read_csv(args.donor_yield)
    direct_rows = read_csv(args.direct_j)
    dynamic_rows = read_csv(args.dynamic_windows)

    selector_epochs, source_counts = build_selector_epochs(
        donor_rows,
        direct_rows,
        dynamic_rows,
    )
    selected, random_rows, keys = build_selected_rows(
        all_window_rows,
        selector_epochs,
    )
    summaries = summarize_selectors(selected)
    random_summary = summarize_random(random_rows)
    pairwise = summarize_pairs(selected)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "selected_donors_per_run.csv", selected)
    write_csv(output_dir / "selector_summary.csv", summaries)
    write_csv(output_dir / "gcs_pairwise_summary.csv", pairwise)
    write_csv(output_dir / "uniform_random_per_run.csv", random_rows)
    (output_dir / "uniform_random_summary.json").write_text(
        json.dumps(random_summary, indent=2, sort_keys=True) + "\n"
    )
    audit = {
        "status": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "expected_runs": EXPECTED_RUNS,
        "registered_runs": len(keys),
        "run_ids": [run_id(key) for key in keys],
        "selected_rows": len(selected),
        "selector_summaries": len(summaries),
        "pairwise_summaries": len(pairwise),
        "random_rows": len(random_rows),
        "source_counts": source_counts,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inputs": {
            "all_window": str(args.all_window),
            "donor_yield": str(args.donor_yield),
            "direct_j": str(args.direct_j),
            "dynamic_windows": str(args.dynamic_windows),
        },
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    readme = build_readme(summaries, random_summary, pairwise, keys)
    (output_dir / "README.md").write_text(readme)
    print(readme)


if __name__ == "__main__":
    main()

