"""Pure helpers for the cached direct-Jacobian selector and paired audits."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected a one-dimensional series, got {arr.shape}")
    if arr.size and not np.isfinite(arr).all():
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("Series contains no finite values")
        arr = np.where(np.isfinite(arr), arr, np.median(finite))
    return arr


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        raise ValueError(f"Expected a positive odd smoothing window, got {window}")
    arr = _finite_array(values)
    if arr.size < window or window == 1:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: arr.size]


def select_change_center(
    epochs: np.ndarray,
    series: np.ndarray,
    *,
    window_len: int,
    mode: str,
    smooth_window: int = 3,
) -> int:
    """Select the earliest largest incoming slope with a full centered window."""
    epoch_values = np.asarray(epochs, dtype=float)
    score_values = _finite_array(series)
    if epoch_values.ndim != 1 or epoch_values.size != score_values.size:
        raise ValueError(
            "Epochs and scores must be one-dimensional arrays of equal length"
        )
    if window_len <= 0 or window_len % 2 == 0:
        raise ValueError(f"Expected a positive odd window length, got {window_len}")
    if score_values.size < window_len:
        raise ValueError(
            f"Window length {window_len} exceeds trajectory length {score_values.size}"
        )
    if not np.isfinite(epoch_values).all():
        raise ValueError("Epochs must be finite")

    smoothed = _smooth_series(score_values, smooth_window)
    half = window_len // 2
    centers = np.arange(max(half, 1), score_values.size - half, dtype=int)
    if centers.size == 0:
        raise ValueError("No center admits a full window")
    elapsed = epoch_values[centers] - epoch_values[centers - 1]
    if np.any(elapsed <= 0):
        raise ValueError("Epochs must be strictly increasing at candidate centers")
    slopes = (smoothed[centers] - smoothed[centers - 1]) / elapsed

    if mode == "rise":
        criterion = slopes
    elif mode == "drop":
        criterion = -slopes
    elif mode == "abs":
        criterion = np.abs(slopes)
    else:
        raise ValueError(f"Unknown change mode: {mode}")
    return int(centers[int(np.argmax(criterion))])


def paired_percentile_summary(
    gcs_percentiles: np.ndarray,
    direct_percentiles: np.ndarray,
    *,
    bootstrap_trials: int,
    bootstrap_seed: int,
    tie_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Summarize same-run GCS-minus-Direct-J percentile differences."""
    gcs = _finite_array(gcs_percentiles)
    direct = _finite_array(direct_percentiles)
    if gcs.size == 0 or gcs.size != direct.size:
        raise ValueError("Paired percentile arrays must have equal non-zero length")
    if bootstrap_trials <= 0:
        raise ValueError("bootstrap_trials must be positive")

    differences = gcs - direct
    rng = np.random.default_rng(int(bootstrap_seed))
    samples = rng.choice(
        differences,
        size=(int(bootstrap_trials), differences.size),
        replace=True,
    )
    bootstrapped_medians = np.median(samples, axis=1)
    low, high = np.quantile(bootstrapped_medians, [0.025, 0.975])
    positive = differences > tie_tolerance
    negative = differences < -tie_tolerance
    tied = ~(positive | negative)
    return {
        "n_runs": int(differences.size),
        "median_difference": float(np.median(differences)),
        "median_bootstrap_ci_low": float(low),
        "median_bootstrap_ci_high": float(high),
        "mean_difference": float(np.mean(differences)),
        "positive_runs": int(np.sum(positive)),
        "tied_runs": int(np.sum(tied)),
        "negative_runs": int(np.sum(negative)),
    }


def exact_mcnemar_two_sided(first_only: int, second_only: int) -> float:
    """Conventional two-sided exact conditional McNemar p-value."""
    first = int(first_only)
    second = int(second_only)
    if first < 0 or second < 0:
        raise ValueError("Discordant counts must be non-negative")
    discordant = first + second
    if discordant == 0:
        return 1.0
    lower = min(first, second)
    tail_numerator = sum(math.comb(discordant, value) for value in range(lower + 1))
    probability = 2.0 * tail_numerator / float(2**discordant)
    return float(min(1.0, probability))


def paired_top20_table(
    gcs_percentiles: np.ndarray,
    direct_percentiles: np.ndarray,
    *,
    threshold: float = 0.8,
) -> dict[str, Any]:
    """Build the paired top-20 table and exact McNemar test."""
    gcs = _finite_array(gcs_percentiles)
    direct = _finite_array(direct_percentiles)
    if gcs.size == 0 or gcs.size != direct.size:
        raise ValueError("Paired percentile arrays must have equal non-zero length")

    gcs_hit = gcs >= float(threshold)
    direct_hit = direct >= float(threshold)
    both = int(np.sum(gcs_hit & direct_hit))
    gcs_only = int(np.sum(gcs_hit & ~direct_hit))
    direct_only = int(np.sum(~gcs_hit & direct_hit))
    neither = int(np.sum(~gcs_hit & ~direct_hit))
    return {
        "threshold": float(threshold),
        "n_runs": int(gcs.size),
        "both_hit": both,
        "gcs_only_hit": gcs_only,
        "direct_only_hit": direct_only,
        "neither_hit": neither,
        "discordant_runs": gcs_only + direct_only,
        "mcnemar_exact_p_two_sided": exact_mcnemar_two_sided(
            gcs_only,
            direct_only,
        ),
    }
