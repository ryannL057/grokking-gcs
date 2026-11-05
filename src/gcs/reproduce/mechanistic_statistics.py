"""Helpers for the frozen mechanistic-yield selector audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gcs.reproduce.direct_statistics import (
    exact_mcnemar_two_sided,
    paired_percentile_summary,
    paired_top20_table,
)


def mechanistic_percentile(values: Sequence[float], selected: float) -> float:
    """Return the submitted-style empirical percentile of a selected outcome."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a non-empty finite one-dimensional array")
    selected_value = float(selected)
    if not np.isfinite(selected_value):
        raise ValueError("selected must be finite")
    return float(np.mean(array <= selected_value))


def _integer_epoch(value: str | int | float) -> int:
    numeric = float(value)
    rounded = int(round(numeric))
    if not np.isfinite(numeric) or not np.isclose(numeric, rounded):
        raise ValueError(f"epoch must be finite and integral, got {value!r}")
    return rounded


def select_oracle_center(rows: Sequence[Mapping[str, Any]]) -> int:
    """Select the earliest center among windows with the largest target score."""
    if not rows:
        raise ValueError("oracle selector requires at least one legal window")
    starts = sorted({_integer_epoch(row["window_start_epoch"]) for row in rows})
    positive_steps = [
        second - first
        for first, second in zip(starts, starts[1:], strict=False)
        if second > first
    ]
    if not positive_steps:
        raise ValueError("cannot infer checkpoint spacing from one oracle window")
    checkpoint_step = min(positive_steps)

    declared_lengths = {
        int(row["window_len"])
        for row in rows
        if str(row.get("window_len", "")).strip()
    }
    if declared_lengths:
        if len(declared_lengths) != 1:
            raise ValueError("oracle rows disagree on window length")
        window_len = declared_lengths.pop()
    else:
        regular_spans = [
            _integer_epoch(row["window_end_epoch"])
            - _integer_epoch(row["window_start_epoch"])
            for row in rows
        ]
        divisible = [span for span in regular_spans if span % checkpoint_step == 0]
        if not divisible:
            raise ValueError("cannot infer oracle window length")
        window_len = max(divisible) // checkpoint_step + 1
    if window_len <= 0 or window_len % 2 == 0:
        raise ValueError("oracle window length must be positive and odd")

    candidates: list[tuple[float, int]] = []
    for row in rows:
        start = _integer_epoch(row["window_start_epoch"])
        end = _integer_epoch(row["window_end_epoch"])
        center = start + (window_len // 2) * checkpoint_step
        if end < center:
            raise ValueError(f"window ends before its center: {start}, {end}")
        score = float(row["target_score"])
        if not np.isfinite(score):
            raise ValueError("oracle target score must be finite")
        candidates.append((score, center))
    best_score = max(score for score, _ in candidates)
    return min(center for score, center in candidates if score == best_score)


def evaluate_selected_donor(
    all_window_rows: Sequence[Mapping[str, Any]],
    *,
    selected_epoch: int,
) -> dict[str, Any]:
    """Join one selected epoch to a run's cached all-window intervention grid."""
    if not all_window_rows:
        raise ValueError("all-window rows cannot be empty")
    epochs = [_integer_epoch(row["donor_epoch"]) for row in all_window_rows]
    if len(set(epochs)) != len(epochs):
        raise ValueError("all-window rows contain duplicate donor epochs")
    matches = [
        row
        for row, epoch in zip(all_window_rows, epochs, strict=True)
        if epoch == int(selected_epoch)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"selected donor epoch {selected_epoch} has {len(matches)} matches"
        )

    outcomes = np.asarray(
        [float(row["attn_minus_mlp_delta"]) for row in all_window_rows],
        dtype=float,
    )
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("mechanistic outcomes must all be finite")
    selected_row = matches[0]
    selected_outcome = float(selected_row["attn_minus_mlp_delta"])
    recomputed = float(selected_row["delta_attn"]) - float(selected_row["delta_mlp"])
    if not np.isclose(selected_outcome, recomputed, atol=1e-12, rtol=1e-12):
        raise ValueError("cached attention-specific replacement cost is inconsistent")

    percentile = mechanistic_percentile(outcomes, selected_outcome)
    donor_epoch = _integer_epoch(selected_row["donor_epoch"])
    t_grok = _integer_epoch(selected_row["t_grok"])
    return {
        "selected_epoch": donor_epoch,
        "mechanistic_target": selected_outcome,
        "mechanistic_percentile": percentile,
        "mechanistic_top20_hit": bool(percentile >= 0.8),
        "mechanistic_target_positive": bool(selected_outcome > 0.0),
        "donor_minus_t_grok": donor_epoch - t_grok,
        "donor_over_t_grok": float(donor_epoch / t_grok),
        "donor_test_accuracy": float(selected_row["donor_test_acc"]),
        "n_legal_donors": len(all_window_rows),
    }


def paired_selector_summary(
    gcs_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Summarize same-run GCS-minus-comparator mechanistic percentiles."""
    gcs_by_run = {str(row["run_id"]): row for row in gcs_rows}
    comparator_by_run = {str(row["run_id"]): row for row in comparator_rows}
    if len(gcs_by_run) != len(gcs_rows) or len(comparator_by_run) != len(
        comparator_rows
    ):
        raise ValueError("paired rows must have unique run ids")
    if set(gcs_by_run) != set(comparator_by_run):
        raise ValueError("paired selectors must cover identical run ids")
    run_ids = sorted(gcs_by_run)
    gcs = np.asarray(
        [float(gcs_by_run[run_id]["mechanistic_percentile"]) for run_id in run_ids]
    )
    comparator = np.asarray(
        [
            float(comparator_by_run[run_id]["mechanistic_percentile"])
            for run_id in run_ids
        ]
    )
    percentiles = paired_percentile_summary(
        gcs,
        comparator,
        bootstrap_trials=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    hit_table = paired_top20_table(gcs, comparator)
    return {
        "n_runs": len(run_ids),
        **percentiles,
        "both_top20": int(hit_table["both_hit"]),
        "gcs_only_top20": int(hit_table["gcs_only_hit"]),
        "comparator_only_top20": int(hit_table["direct_only_hit"]),
        "neither_top20": int(hit_table["neither_hit"]),
        "discordant_runs": int(hit_table["discordant_runs"]),
        "mcnemar_exact_two_sided_p": exact_mcnemar_two_sided(
            int(hit_table["gcs_only_hit"]),
            int(hit_table["direct_only_hit"]),
        ),
    }
