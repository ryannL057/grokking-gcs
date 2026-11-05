#!/usr/bin/env python3
"""Evaluate the frozen near-solved endpoint queue and paired prefix diagnostic."""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import io
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import beta, spearmanr

ROOT = _gcs_data_root()
pass  # Imports use the installed package.

from gcs.analysis.near_solved_provenance import (  # noqa: E402
    reference_geometry_semantic_sha256,
    sha256_file,
    validate_file_record,
)
from gcs.legacy.diagnostics_common import resolve_path  # noqa: E402
from gcs.legacy.select_near_solved_endpoints_20260724 import (  # noqa: E402
    _checkpoint_map,
    selection_semantic_sha256,
)


DEFAULT_EXPERIMENT_DIR = Path(
    "experiments/rebuttal_near_solved_endpoint_queue_20260724"
)
PATHS = ("overall_path", "attention_path", "mlp_path")
INDICATORS = ("late_attention", "early_mlp", "combined_signature")


def load_pickle_cpu(path: Path) -> Any:
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda data: torch.load(
                    io.BytesIO(data),
                    map_location="cpu",
                    weights_only=False,
                )
            return super().find_class(module, name)

    with path.open("rb") as handle:
        return CPUUnpickler(handle).load()


def edge_smooth_three(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Three-point moving average with replicated endpoint values."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Expected a non-empty one-dimensional trajectory")
    if not np.all(np.isfinite(array)):
        raise ValueError("Trajectory contains non-finite values")
    padded = np.pad(array, (1, 1), mode="edge")
    return np.convolve(
        padded,
        np.ones(3, dtype=np.float64) / 3.0,
        mode="valid",
    )


def normalize_zero_one(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum <= 1e-12:
        return np.zeros_like(array)
    return (array - minimum) / (maximum - minimum)


def endpoint_signature(
    *,
    epochs: Sequence[int] | np.ndarray,
    path_scores: Mapping[str, Sequence[float] | np.ndarray],
    reference_epoch: int,
) -> dict[str, Any]:
    """Apply the preregistered endpoint-relative locator-like signature."""
    epoch_array = np.asarray(epochs, dtype=np.int64)
    reference_epoch = int(reference_epoch)
    if epoch_array.ndim != 1 or epoch_array.size < 3:
        raise ValueError("Endpoint signature requires at least three epochs")
    if reference_epoch <= 0:
        raise ValueError("Reference epoch must be positive")
    if np.any(np.diff(epoch_array) <= 0):
        raise ValueError("Epochs must be strictly increasing")
    if int(epoch_array[-1]) != reference_epoch:
        raise ValueError("Trajectory prefix must end at the reference epoch")

    normalized: dict[str, np.ndarray] = {}
    for path in PATHS:
        if path not in path_scores:
            raise KeyError(f"Missing {path}")
        values = np.asarray(path_scores[path], dtype=np.float64)
        if values.shape != epoch_array.shape:
            raise ValueError(
                f"{path} shape {values.shape} != epoch shape {epoch_array.shape}"
            )
        normalized[path] = normalize_zero_one(edge_smooth_three(values))

    x = epoch_array.astype(np.float64) / float(reference_epoch)
    attention = normalized["attention_path"]
    attention_candidates = np.flatnonzero((x >= 0.30) & (x <= 0.95))
    if attention_candidates.size == 0:
        raise ValueError("No epoch lies in the frozen attention window [0.30, 0.95]")
    attention_peak_index = int(
        attention_candidates[
            int(np.argmax(attention[attention_candidates]))
        ]
    )
    attention_peak_x = float(x[attention_peak_index])
    late_attention = bool(attention_peak_x > 0.45)

    mlp = normalized["mlp_path"]
    mlp_trough_index = int(np.argmin(mlp))
    recovery_candidates = np.flatnonzero(
        (np.arange(mlp.size) > mlp_trough_index) & (mlp >= 0.75)
    )
    mlp_recovery_index = (
        int(recovery_candidates[0])
        if recovery_candidates.size
        else None
    )
    mlp_recovery_x = (
        float(x[mlp_recovery_index])
        if mlp_recovery_index is not None
        else None
    )
    early_mlp = bool(
        mlp_recovery_x is not None and mlp_recovery_x <= 0.70
    )

    return {
        "n_gcs_points": int(epoch_array.size),
        "attention_peak_epoch": int(epoch_array[attention_peak_index]),
        "attention_peak_x": attention_peak_x,
        "late_attention": late_attention,
        "mlp_trough_epoch": int(epoch_array[mlp_trough_index]),
        "mlp_trough_x": float(x[mlp_trough_index]),
        "mlp_recovery_epoch": (
            int(epoch_array[mlp_recovery_index])
            if mlp_recovery_index is not None
            else None
        ),
        "mlp_recovery_x": mlp_recovery_x,
        "early_mlp": early_mlp,
        "combined_signature": bool(late_attention and early_mlp),
    }


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Exact equal-tailed two-sided binomial interval."""
    successes = int(successes)
    trials = int(trials)
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError(
            f"Expected 0 <= successes <= positive trials, got "
            f"{successes}/{trials}"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lower, upper


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.size < 2:
        raise ValueError("Spearman inputs must have equal shape and length >= 2")
    if np.ptp(left_array) <= 1e-12 or np.ptp(right_array) <= 1e-12:
        return None
    statistic = float(spearmanr(left_array, right_array).statistic)
    return statistic if np.isfinite(statistic) else None


def _canonical_solved_prefix(
    selection: Mapping[str, Any],
    epochs: np.ndarray,
) -> dict[str, np.ndarray]:
    payload = load_pickle_cpu(Path(selection["canonical_solved_gcs_path"]))
    if "epochs" in payload:
        solved_epochs = np.asarray(payload["epochs"], dtype=np.int64)
        solved_scores = payload["path_gcs_scores"]
    else:
        solved_epochs = np.asarray(
            payload["gcs_steps_final_geo"],
            dtype=np.int64,
        )
        solved_scores = payload["path_gcs_scores_final_geo"]
    if len(set(solved_epochs.tolist())) != int(solved_epochs.size):
        raise ValueError(
            f"{selection['run_id']}: duplicate solved-final GCS epochs"
        )
    index_by_epoch = {
        int(epoch): index for index, epoch in enumerate(solved_epochs)
    }
    missing = [int(epoch) for epoch in epochs if int(epoch) not in index_by_epoch]
    if missing:
        raise ValueError(
            f"{selection['run_id']}: solved-final trajectory lacks common "
            f"prefix epochs {missing}"
        )
    indices = [index_by_epoch[int(epoch)] for epoch in epochs]
    return {
        path: np.asarray(solved_scores[path], dtype=np.float64)[indices]
        for path in PATHS
    }


def _validate_trajectory_provenance(
    selection: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    expected_bindings = {
        "selection_manifest_path": str(
            Path(selection["selection_manifest_path"]).resolve()
        ),
        "selection_manifest_sha256": str(
            selection["selection_manifest_sha256"]
        ),
        "selection_semantic_sha256": str(
            selection["selection_semantic_sha256"]
        ),
        "protocol_path": str(Path(selection["protocol_path"]).resolve()),
        "protocol_sha256": str(selection["protocol_sha256"]),
        "reference_checkpoint_path": str(
            Path(selection["reference_checkpoint_path"]).resolve()
        ),
        "reference_checkpoint_sha256": str(
            selection["reference_checkpoint_sha256"]
        ),
        "reference_checkpoint_size_bytes": int(
            selection["reference_checkpoint_size_bytes"]
        ),
    }
    for key, expected in expected_bindings.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"{selection['run_id']}: trajectory {key} provenance mismatch"
            )
    if sha256_file(expected_bindings["selection_manifest_path"]) != str(
        expected_bindings["selection_manifest_sha256"]
    ):
        raise ValueError(
            f"{selection['run_id']}: stale selection manifest"
        )
    if sha256_file(expected_bindings["protocol_path"]) != str(
        expected_bindings["protocol_sha256"]
    ):
        raise ValueError(f"{selection['run_id']}: stale frozen protocol")
    validate_file_record(
        {
            "epoch": int(selection["reference_epoch"]),
            "path": expected_bindings["reference_checkpoint_path"],
            "sha256": expected_bindings["reference_checkpoint_sha256"],
            "size_bytes": expected_bindings[
                "reference_checkpoint_size_bytes"
            ],
        },
        expected_path=selection["reference_checkpoint_path"],
        expected_epoch=int(selection["reference_epoch"]),
    )

    target_records = list(payload.get("target_checkpoints", []))
    processed_records = list(payload.get("processed_checkpoints", []))
    target_epochs = [int(epoch) for epoch in selection["target_epochs"]]
    if len(target_records) != len(target_epochs):
        raise ValueError(
            f"{selection['run_id']}: target checkpoint ledger length mismatch"
        )
    checkpoint_map = _checkpoint_map(Path(selection["source_exp_dir"]))
    for epoch, record in zip(target_epochs, target_records):
        if epoch not in checkpoint_map:
            raise FileNotFoundError(
                f"{selection['run_id']}: checkpoint {epoch} disappeared"
            )
        validate_file_record(
            record,
            expected_path=checkpoint_map[epoch],
            expected_epoch=epoch,
        )
    if processed_records != target_records:
        raise ValueError(
            f"{selection['run_id']}: complete processed ledger is mixed/stale"
        )
    for epoch, record in zip(target_epochs, processed_records):
        validate_file_record(
            record,
            expected_path=checkpoint_map[epoch],
            expected_epoch=epoch,
        )

    geometry_path = Path(payload["reference_geometry_path"]).resolve()
    geometry = load_pickle_cpu(geometry_path)
    geometry_sha = reference_geometry_semantic_sha256(geometry)
    if geometry_sha != str(
        payload.get("reference_geometry_semantic_sha256")
    ):
        raise ValueError(
            f"{selection['run_id']}: stale reference geometry"
        )
    geometry_metadata = geometry.get("near_solved_reference_metadata", {})
    for key, expected in {
        **expected_bindings,
        "run_id": str(selection["run_id"]),
        "reference_epoch": int(selection["reference_epoch"]),
        "canonical_prompt_sha256": str(
            selection["canonical_prompt_sha256"]
        ),
        "reference_geometry_semantic_sha256": geometry_sha,
    }.items():
        if geometry_metadata.get(key) != expected:
            raise ValueError(
                f"{selection['run_id']}: geometry {key} provenance mismatch"
            )


def _validate_trajectory(
    selection: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not bool(payload.get("complete", False)):
        raise ValueError(f"{selection['run_id']}: trajectory is incomplete")
    if str(payload.get("run_id")) != str(selection["run_id"]):
        raise ValueError(f"{selection['run_id']}: trajectory run_id mismatch")
    if int(payload.get("reference_epoch", -1)) != int(
        selection["reference_epoch"]
    ):
        raise ValueError(
            f"{selection['run_id']}: trajectory reference mismatch"
        )
    if payload.get("canonical_prompt_sha256") != selection.get(
        "canonical_prompt_sha256"
    ):
        raise ValueError(
            f"{selection['run_id']}: trajectory prompt hash mismatch"
        )
    for flag in ("exact_band", "overshoot"):
        if bool(payload.get(flag)) != bool(selection.get(flag)):
            raise ValueError(
                f"{selection['run_id']}: trajectory {flag} mismatch"
            )
    _validate_trajectory_provenance(selection, payload)
    epochs = np.asarray(payload["epochs"], dtype=np.int64)
    target = np.asarray(selection["target_epochs"], dtype=np.int64)
    if not np.array_equal(epochs, target):
        raise ValueError(
            f"{selection['run_id']}: trajectory does not match frozen prefix"
        )
    params = payload.get("analysis_params", {})
    expected_params = {
        "N": 200,
        "k": 15,
        "d": 8,
        "dtype": "float64",
        "target_layer": 0,
        "canonical_epoch_schedule": (
            "all_saved_canonical_epochs_at_or_before_reference"
        ),
        "reference_hook": "hook_resid_post",
        "token_probe": "final",
        "alignment_method": "cosine",
        "similarity": "rowwise_absolute_cosine",
        "aggregation": "spectral_entropy",
        "compute": "batch_equivalent_jvp",
    }
    for key, expected in expected_params.items():
        if params.get(key) != expected:
            raise ValueError(
                f"{selection['run_id']}: {key}={params.get(key)!r}, "
                f"expected {expected!r}"
            )
    scores = {
        path: np.asarray(payload["path_scores"][path], dtype=np.float64)
        for path in PATHS
    }
    if any(values.shape != epochs.shape for values in scores.values()):
        raise ValueError(f"{selection['run_id']}: invalid path-score length")
    if not all(np.all(np.isfinite(values)) for values in scores.values()):
        raise ValueError(f"{selection['run_id']}: non-finite path score")
    return epochs, scores


def evaluate_run(
    *,
    selection: Mapping[str, Any],
    trajectory_path: Path,
) -> dict[str, Any]:
    payload = load_pickle_cpu(trajectory_path)
    epochs, near_scores = _validate_trajectory(selection, payload)
    solved_scores = _canonical_solved_prefix(selection, epochs)
    reference_epoch = int(selection["reference_epoch"])
    near_signature = endpoint_signature(
        epochs=epochs,
        path_scores=near_scores,
        reference_epoch=reference_epoch,
    )
    solved_signature = endpoint_signature(
        epochs=epochs,
        path_scores=solved_scores,
        reference_epoch=reference_epoch,
    )

    row: dict[str, Any] = {
        "run_id": selection["run_id"],
        "activation": selection["activation"],
        "task": selection["task"],
        "seed": int(selection["seed"]),
        "reference_epoch": reference_epoch,
        "reference_train_accuracy": float(
            selection["reference_train_accuracy"]
        ),
        "reference_test_accuracy": float(
            selection["reference_test_accuracy"]
        ),
        "exact_band": bool(selection["exact_band"]),
        "overshoot": bool(selection["overshoot"]),
        "trajectory_path": str(trajectory_path.resolve()),
    }
    for key, value in near_signature.items():
        row[key] = value
    for key, value in solved_signature.items():
        row[f"solved_prefix_{key}"] = value
    for path in PATHS:
        row[f"{path}_spearman_near_vs_solved_prefix"] = _safe_spearman(
            near_scores[path],
            solved_scores[path],
        )
    for indicator in INDICATORS:
        row[f"{indicator}_agreement"] = bool(
            near_signature[indicator] == solved_signature[indicator]
        )
    return row


def summarize_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Cannot summarize empty group {label}")
    result: dict[str, Any] = {
        "group": label,
        "n": len(rows),
        "reference_test_accuracy": {
            "min": float(
                np.min([float(row["reference_test_accuracy"]) for row in rows])
            ),
            "median": float(
                np.median(
                    [float(row["reference_test_accuracy"]) for row in rows]
                )
            ),
            "max": float(
                np.max([float(row["reference_test_accuracy"]) for row in rows])
            ),
        },
    }
    for indicator in INDICATORS:
        count = int(sum(bool(row[indicator]) for row in rows))
        lower, upper = clopper_pearson_interval(count, len(rows))
        result[indicator] = {
            "count": count,
            "n": len(rows),
            "rate": float(count / len(rows)),
            "clopper_pearson_95_low": lower,
            "clopper_pearson_95_high": upper,
        }
    result["paired_diagnostic"] = {
        indicator: {
            "agreement_count": int(
                sum(bool(row[f"{indicator}_agreement"]) for row in rows)
            ),
            "agreement_rate": float(
                np.mean(
                    [bool(row[f"{indicator}_agreement"]) for row in rows]
                )
            ),
        }
        for indicator in INDICATORS
    }
    result["paired_diagnostic"]["pathwise_spearman_median"] = {
        path: (
            float(np.median(values))
            if (
                values := [
                    float(row[f"{path}_spearman_near_vs_solved_prefix"])
                    for row in rows
                    if row[f"{path}_spearman_near_vs_solved_prefix"] is not None
                ]
            )
            else None
        )
        for path in PATHS
    }
    return result


def subgroup_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_group: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for factor in ("activation", "task"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[factor])].append(row)
        for level, members in sorted(grouped.items()):
            summary = summarize_group(
                members,
                label=f"{parent_group}:{factor}={level}",
            )
            for indicator in INDICATORS:
                item = summary[indicator]
                output.append(
                    {
                        "parent_group": parent_group,
                        "analysis_label": "descriptive_not_confirmatory",
                        "factor": factor,
                        "level": level,
                        "indicator": indicator,
                        **item,
                    }
                )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_selection(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    manifest_sha = sha256_file(path)
    payload = json.loads(path.read_text())
    rows = [dict(row) for row in payload["rows"]]
    if len(rows) != 35:
        raise ValueError(f"Selection manifest has {len(rows)} rows, expected 35")
    if sum(bool(row["exact_band"]) for row in rows) != 24:
        raise ValueError("Selection manifest exact-band denominator is not 24")
    if sum(bool(row["overshoot"]) for row in rows) != 11:
        raise ValueError("Selection manifest overshoot denominator is not 11")
    protocol_path = Path(payload["protocol_path"]).resolve()
    protocol_sha = sha256_file(protocol_path)
    if protocol_sha != str(payload.get("protocol_sha256")):
        raise ValueError("Selection references a stale protocol")
    semantic_sha = selection_semantic_sha256(
        rows,
        protocol_sha256=protocol_sha,
    )
    if semantic_sha != str(payload.get("selection_semantic_sha256")):
        raise ValueError("Selection semantic hash mismatch")
    for row in rows:
        if row.get("protocol_path") != str(protocol_path):
            raise ValueError(f"{row['run_id']}: protocol path mismatch")
        if row.get("protocol_sha256") != protocol_sha:
            raise ValueError(f"{row['run_id']}: protocol hash mismatch")
        if row.get("selection_semantic_sha256") != semantic_sha:
            raise ValueError(
                f"{row['run_id']}: selection semantic hash mismatch"
            )
        validate_file_record(
            {
                "epoch": int(row["reference_epoch"]),
                "path": row["reference_checkpoint_path"],
                "sha256": row["reference_checkpoint_sha256"],
                "size_bytes": int(
                    row["reference_checkpoint_size_bytes"]
                ),
            },
            expected_path=row["reference_checkpoint_path"],
            expected_epoch=int(row["reference_epoch"]),
        )
        row["selection_manifest_path"] = str(path)
        row["selection_manifest_sha256"] = manifest_sha
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cohort",
        choices=("primary", "full"),
        default="full",
        help="Primary requires 24 exact-band runs; full requires all 35.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = resolve_path(args.experiment_dir)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else experiment_dir / "evaluation"
    )
    selections = _load_selection(experiment_dir / "selection_manifest.json")
    if args.cohort == "primary":
        selected = [row for row in selections if bool(row["exact_band"])]
    else:
        selected = selections

    rows = [
        evaluate_run(
            selection=selection,
            trajectory_path=(
                experiment_dir
                / "trajectories"
                / f"{selection['run_id']}.pkl"
            ),
        )
        for selection in selected
    ]
    exact_rows = [row for row in rows if bool(row["exact_band"])]
    overshoot_rows = [row for row in rows if bool(row["overshoot"])]
    summaries = {
        "primary_exact_band_24": summarize_group(
            exact_rows,
            label="primary_exact_band_24",
        )
    }
    subgroup = subgroup_rows(
        exact_rows,
        parent_group="primary_exact_band_24",
    )
    if args.cohort == "full":
        summaries["secondary_full_queue_35"] = summarize_group(
            rows,
            label="secondary_full_queue_35",
        )
        summaries["secondary_overshoot_11"] = summarize_group(
            overshoot_rows,
            label="secondary_overshoot_11",
        )
        subgroup.extend(
            subgroup_rows(
                rows,
                parent_group="secondary_full_queue_35",
            )
        )

    result = {
        "schema_version": 2,
        "cohort_mode": args.cohort,
        "provenance": {
            "selection_manifest_path": selections[0][
                "selection_manifest_path"
            ],
            "selection_manifest_sha256": selections[0][
                "selection_manifest_sha256"
            ],
            "selection_semantic_sha256": selections[0][
                "selection_semantic_sha256"
            ],
            "protocol_path": selections[0]["protocol_path"],
            "protocol_sha256": selections[0]["protocol_sha256"],
            "target_checkpoint_validation": (
                "all target paths and SHA-256 values rehashed"
            ),
            "reference_geometry_validation": (
                "semantic SHA-256 recomputed"
            ),
        },
        "frozen_signature": {
            "smoothing": "three_point_moving_average_edge_replicated",
            "path_normalization": "independent_minmax_over_prefix",
            "time": "epoch/reference_epoch",
            "attention_window": [0.30, 0.95],
            "late_attention": "attention_peak_x > 0.45",
            "early_mlp": "first_post_trough_normalized_mlp_ge_0.75 at x <= 0.70",
            "combined_signature": "late_attention AND early_mlp",
            "paired_spearman": "raw_GCS_on_identical_canonical_epoch_prefix",
        },
        "summaries": summaries,
        "rows": rows,
    }
    _write_csv(output_dir / "per_run_results.csv", rows)
    _write_csv(output_dir / "descriptive_subgroups.csv", subgroup)
    _write_json(output_dir / "summary.json", result)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
