#!/usr/bin/env python3
"""Freeze the checkpoint-available near-solved endpoint queue.

This script is deliberately CPU-only.  It selects references from logged
accuracies at epochs that have a real saved checkpoint, anchors each run to the
ordered prompts in its submitted final geometry, and records the exact
canonical GCS prefix that the runner must recompute.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import hashlib
import io
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis.near_solved_provenance import (  # noqa: E402
    canonical_json_sha256,
    make_file_record,
    sha256_file,
)
from gcs.legacy.diagnostics_common import RUNS_1L, resolve_path  # noqa: E402


PROTOCOL_DATE = "2026-07-24"
LOWER_ACCURACY = 0.90
UPPER_ACCURACY = 0.95
EXPECTED_RUNS = 35
EXPECTED_EXACT_BAND = 24
EXPECTED_OVERSHOOT = 11
DEFAULT_OUTPUT_DIR = Path(
    "experiments/rebuttal_near_solved_endpoint_queue_20260724"
)
PROTOCOL_PATH = (
    ROOT / "rebuttal_2026" / "NEAR_SOLVED_ENDPOINT_PROTOCOL_20260724.md"
).resolve()
CANONICAL_GCS_FILENAMES = (
    "gcs_results_union.pkl",
    "gcs_results.pkl",
    "training_metrics_final_geo_union.pkl",
    "training_metrics_final_geo.pkl",
)


def load_pickle_cpu(path: Path) -> Any:
    """Load pickle payloads whose tensors may have been saved on a GPU."""

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


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint epoch from {path}")
    return int(match.group(1))


def run_id(activation: str, task: str, seed: int) -> str:
    return f"{activation.lower()}_{task}_s{int(seed)}"


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def select_reference(
    *,
    log_steps: Sequence[int],
    train_accuracies: Sequence[float],
    test_accuracies: Sequence[float],
    checkpoint_epochs: Iterable[int],
    lower: float = LOWER_ACCURACY,
    upper: float = UPPER_ACCURACY,
) -> dict[str, Any]:
    """Select the first saved checkpoint with logged test accuracy >= lower."""
    if not (
        len(log_steps) == len(train_accuracies) == len(test_accuracies)
    ):
        raise ValueError("Metric step/train/test arrays must have equal length")
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(f"Invalid accuracy band [{lower}, {upper}]")

    metrics_by_epoch: dict[int, tuple[float, float]] = {}
    for raw_epoch, raw_train, raw_test in zip(
        log_steps,
        train_accuracies,
        test_accuracies,
    ):
        epoch = int(raw_epoch)
        if epoch in metrics_by_epoch:
            raise ValueError(f"Duplicate logged metric epoch {epoch}")
        train_accuracy = float(raw_train)
        test_accuracy = float(raw_test)
        if not np.isfinite(train_accuracy) or not np.isfinite(test_accuracy):
            raise ValueError(f"Non-finite logged accuracy at epoch {epoch}")
        metrics_by_epoch[epoch] = (train_accuracy, test_accuracy)

    saved = sorted(
        int(epoch)
        for epoch in set(checkpoint_epochs)
        if int(epoch) in metrics_by_epoch
    )
    if not saved:
        raise ValueError("No saved checkpoints have matching logged accuracies")

    selected_index = next(
        (
            index
            for index, epoch in enumerate(saved)
            if metrics_by_epoch[epoch][1] >= lower
        ),
        None,
    )
    if selected_index is None:
        raise ValueError(
            f"No saved checkpoint reaches test accuracy {lower:.2f}"
        )

    reference_epoch = saved[selected_index]
    train_accuracy, test_accuracy = metrics_by_epoch[reference_epoch]
    previous_epoch = saved[selected_index - 1] if selected_index > 0 else None
    previous_test = (
        metrics_by_epoch[previous_epoch][1]
        if previous_epoch is not None
        else None
    )
    exact_band = lower <= test_accuracy <= upper
    overshoot = test_accuracy > upper
    return {
        "reference_epoch": int(reference_epoch),
        "reference_train_accuracy": float(train_accuracy),
        "reference_test_accuracy": float(test_accuracy),
        "exact_band": bool(exact_band),
        "overshoot": bool(overshoot),
        "previous_saved_epoch": (
            int(previous_epoch) if previous_epoch is not None else None
        ),
        "previous_saved_test_accuracy": (
            float(previous_test) if previous_test is not None else None
        ),
        "previous_saved_below_lower": bool(
            previous_test is not None and previous_test < lower
        ),
        "selection_rule": "first_saved_checkpoint_with_logged_test_accuracy_ge_0.90",
    }


def canonical_prefix_epochs(
    *,
    canonical_epochs: Sequence[int],
    checkpoint_epochs: Iterable[int],
    reference_epoch: int,
) -> list[int]:
    """Return every saved canonical GCS epoch through the reference endpoint."""
    checkpoint_set = {int(epoch) for epoch in checkpoint_epochs}
    reference_epoch = int(reference_epoch)
    if reference_epoch not in checkpoint_set:
        raise ValueError(
            f"Reference epoch {reference_epoch} has no saved checkpoint"
        )
    canonical = [int(epoch) for epoch in canonical_epochs]
    if len(set(canonical)) != len(canonical):
        raise ValueError("Canonical GCS schedule contains duplicate epochs")
    if any(right <= left for left, right in zip(canonical, canonical[1:])):
        raise ValueError("Canonical GCS schedule must be strictly increasing")
    if reference_epoch not in canonical:
        raise ValueError(
            f"Reference epoch {reference_epoch} is absent from canonical GCS "
            "schedule; paired solved-prefix comparison is undefined"
        )

    ordered = [
        epoch
        for epoch in canonical
        if epoch <= reference_epoch and epoch in checkpoint_set
    ]
    if not ordered or ordered[-1] != reference_epoch:
        raise AssertionError("Canonical prefix must end at the reference")
    return ordered


def canonical_gcs_payload(exp_dir: Path) -> tuple[Path, dict[str, Any]]:
    for filename in CANONICAL_GCS_FILENAMES:
        path = exp_dir / filename
        if not path.exists():
            continue
        payload = load_pickle_cpu(path)
        if "epochs" in payload and "path_gcs_scores" in payload:
            return path, payload
        if "gcs_steps_final_geo" in payload and "path_gcs_scores_final_geo" in payload:
            return path, {
                "epochs": payload["gcs_steps_final_geo"],
                "path_gcs_scores": payload["path_gcs_scores_final_geo"],
            }
    raise FileNotFoundError(f"No canonical solved-final GCS payload in {exp_dir}")


def _checkpoint_map(exp_dir: Path) -> dict[int, Path]:
    paths = list((exp_dir / "checkpoints").glob("*.pth"))
    if not paths:
        paths = list(exp_dir.glob("*.pth"))
    paths = sorted(paths, key=checkpoint_epoch)
    result: dict[int, Path] = {}
    for path in paths:
        epoch = checkpoint_epoch(path)
        if epoch in result:
            raise ValueError(f"Duplicate checkpoint epoch {epoch} in {exp_dir}")
        result[epoch] = path.resolve()
    if not result:
        raise FileNotFoundError(f"No checkpoints found in {exp_dir}")
    return result


def build_selection_rows(
    runs: Sequence[tuple[str, str, int, str]] = RUNS_1L,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort_index, (activation, task, seed, base) in enumerate(runs):
        exp_dir = resolve_path(base).resolve()
        metrics_path = exp_dir / "training_metrics.pkl"
        geometry_path = exp_dir / "final_geometry.pkl"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        if not geometry_path.exists():
            raise FileNotFoundError(geometry_path)

        metrics = load_pickle_cpu(metrics_path)
        checkpoint_map = _checkpoint_map(exp_dir)
        selected = select_reference(
            log_steps=metrics["log_steps"],
            train_accuracies=metrics["train_accuracies"],
            test_accuracies=metrics["test_accuracies"],
            checkpoint_epochs=checkpoint_map,
        )
        reference_epoch = int(selected["reference_epoch"])
        reference_record = make_file_record(
            checkpoint_map[reference_epoch],
            epoch=reference_epoch,
        )

        geometry = load_pickle_cpu(geometry_path)
        prompts = geometry["X_prompts"].detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        if tuple(prompts.shape) != (200, 3):
            raise ValueError(
                f"{geometry_path}: expected prompt shape (200, 3), "
                f"got {tuple(prompts.shape)}"
            )

        canonical_path, canonical_payload = canonical_gcs_payload(exp_dir)
        target_epochs = canonical_prefix_epochs(
            canonical_epochs=canonical_payload["epochs"],
            checkpoint_epochs=checkpoint_map,
            reference_epoch=reference_epoch,
        )
        missing_path_keys = {
            "overall_path",
            "attention_path",
            "mlp_path",
        } - set(canonical_payload["path_gcs_scores"])
        if missing_path_keys:
            raise KeyError(
                f"{canonical_path}: missing canonical paths "
                f"{sorted(missing_path_keys)}"
            )

        rows.append(
            {
                "cohort_index": int(cohort_index),
                "run_id": run_id(activation, task, seed),
                "activation": activation,
                "task": task,
                "seed": int(seed),
                "registered_base": str(base),
                "source_exp_dir": str(exp_dir),
                "metrics_path": str(metrics_path.resolve()),
                "submitted_final_geometry_path": str(geometry_path.resolve()),
                "canonical_solved_gcs_path": str(canonical_path.resolve()),
                "canonical_prompt_count": int(prompts.shape[0]),
                "canonical_prompt_sha256": tensor_sha256(prompts),
                "reference_checkpoint_path": str(
                    checkpoint_map[reference_epoch]
                ),
                "reference_checkpoint_sha256": reference_record["sha256"],
                "reference_checkpoint_size_bytes": reference_record[
                    "size_bytes"
                ],
                "reference_epoch": reference_epoch,
                "reference_train_accuracy": selected[
                    "reference_train_accuracy"
                ],
                "reference_test_accuracy": selected[
                    "reference_test_accuracy"
                ],
                "previous_saved_epoch": selected["previous_saved_epoch"],
                "previous_saved_test_accuracy": selected[
                    "previous_saved_test_accuracy"
                ],
                "previous_saved_below_lower": selected[
                    "previous_saved_below_lower"
                ],
                "exact_band": selected["exact_band"],
                "overshoot": selected["overshoot"],
                "primary_exact_band_included": selected["exact_band"],
                "secondary_full_queue_included": True,
                "selection_rule": selected["selection_rule"],
                "target_epoch_count": len(target_epochs),
                "target_first_epoch": target_epochs[0],
                "target_last_epoch": target_epochs[-1],
                "canonical_epoch_schedule": (
                    "all_saved_canonical_epochs_at_or_before_reference"
                ),
                "reference_epoch_in_canonical_payload": True,
                "protocol_path": str(PROTOCOL_PATH),
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "target_epochs": target_epochs,
            }
        )
    return rows


SELECTION_SEMANTIC_KEYS = (
    "cohort_index",
    "run_id",
    "activation",
    "task",
    "seed",
    "source_exp_dir",
    "metrics_path",
    "submitted_final_geometry_path",
    "canonical_solved_gcs_path",
    "canonical_prompt_count",
    "canonical_prompt_sha256",
    "reference_checkpoint_path",
    "reference_checkpoint_sha256",
    "reference_checkpoint_size_bytes",
    "reference_epoch",
    "reference_train_accuracy",
    "reference_test_accuracy",
    "previous_saved_epoch",
    "previous_saved_test_accuracy",
    "previous_saved_below_lower",
    "exact_band",
    "overshoot",
    "selection_rule",
    "canonical_epoch_schedule",
    "target_epochs",
)


def selection_semantic_sha256(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol_sha256: str,
) -> str:
    ledger = {
        "schema": "near_solved_selection_v1",
        "protocol_sha256": str(protocol_sha256),
        "rows": [
            {key: row[key] for key in SELECTION_SEMANTIC_KEYS}
            for row in rows
        ],
    }
    return canonical_json_sha256(ledger)


def validate_frozen_cohort(rows: Sequence[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} runs, observed {len(rows)}")
    exact = [row for row in rows if bool(row["exact_band"])]
    overshoot = [row for row in rows if bool(row["overshoot"])]
    if len(exact) != EXPECTED_EXACT_BAND:
        raise ValueError(
            f"Expected {EXPECTED_EXACT_BAND} exact-band runs, "
            f"observed {len(exact)}"
        )
    if len(overshoot) != EXPECTED_OVERSHOOT:
        raise ValueError(
            f"Expected {EXPECTED_OVERSHOOT} overshoots, "
            f"observed {len(overshoot)}"
        )
    for row in rows:
        accuracy = float(row["reference_test_accuracy"])
        if bool(row["exact_band"]) == bool(row["overshoot"]):
            raise ValueError(
                f"{row['run_id']}: exact-band/overshoot flags are not exclusive"
            )
        if bool(row["exact_band"]) and not (
            LOWER_ACCURACY <= accuracy <= UPPER_ACCURACY
        ):
            raise ValueError(f"{row['run_id']}: invalid exact-band accuracy")
        if bool(row["overshoot"]):
            if accuracy <= UPPER_ACCURACY:
                raise ValueError(f"{row['run_id']}: invalid overshoot accuracy")
            if not bool(row["previous_saved_below_lower"]):
                raise ValueError(
                    f"{row['run_id']}: overshoot does not jump from below "
                    f"{LOWER_ACCURACY:.2f}"
                )


def summarize_selection(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exact = [row for row in rows if bool(row["exact_band"])]
    overshoot = [row for row in rows if bool(row["overshoot"])]
    accuracies = np.asarray(
        [float(row["reference_test_accuracy"]) for row in rows],
        dtype=float,
    )
    exact_accuracies = np.asarray(
        [float(row["reference_test_accuracy"]) for row in exact],
        dtype=float,
    )
    selection_hashes = {
        str(row["selection_semantic_sha256"]) for row in rows
    }
    protocol_hashes = {str(row["protocol_sha256"]) for row in rows}
    if len(selection_hashes) != 1 or len(protocol_hashes) != 1:
        raise ValueError("Rows do not share one selection/protocol hash")
    return {
        "protocol_date": PROTOCOL_DATE,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": next(iter(protocol_hashes)),
        "selection_semantic_sha256": next(iter(selection_hashes)),
        "selection_viewed_new_gcs": False,
        "selection_rule": (
            "first saved checkpoint with logged test accuracy >= 0.90"
        ),
        "primary_estimand": "exact saved checkpoint in inclusive [0.90, 0.95]",
        "secondary_estimand": "first saved checkpoint at or above 0.90",
        "registered_run_count": len(rows),
        "primary_exact_band_count": len(exact),
        "secondary_overshoot_count": len(overshoot),
        "secondary_reference_accuracy": {
            "min": float(np.min(accuracies)),
            "median": float(np.median(accuracies)),
            "max": float(np.max(accuracies)),
        },
        "primary_reference_accuracy": {
            "min": float(np.min(exact_accuracies)),
            "median": float(np.median(exact_accuracies)),
            "max": float(np.max(exact_accuracies)),
        },
        "analysis_params": {
            "N": 200,
            "k": 15,
            "d": 8,
            "dtype": "float64",
            "target_layer": 0,
            "canonical_epoch_schedule": (
                "all_saved_canonical_epochs_at_or_before_reference"
            ),
            "reference_hook": "hook_resid_post",
            "paths": [
                "overall_path",
                "attention_path",
                "mlp_path",
            ],
            "token_probe": "final",
            "alignment": "rowwise_absolute_cosine",
            "aggregation": "spectral_entropy",
        },
    }


def _json_ready_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in row.items()
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty selection manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    flattened = []
    for row in rows:
        item = _json_ready_row(dict(row))
        item["target_epochs"] = json.dumps(item["target_epochs"])
        flattened.append(item)
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print the frozen selection without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_selection_rows()
    validate_frozen_cohort(rows)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    semantic_sha = selection_semantic_sha256(
        rows,
        protocol_sha256=protocol_sha,
    )
    for row in rows:
        row["selection_semantic_sha256"] = semantic_sha
    summary = summarize_selection(rows)
    payload = {
        "schema_version": 2,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "selection_semantic_sha256": semantic_sha,
        "summary": summary,
        "rows": [_json_ready_row(row) for row in rows],
    }
    if args.check_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    output_dir = resolve_path(args.output_dir)
    write_csv_atomic(output_dir / "selection_manifest.csv", rows)
    manifest_path = output_dir / "selection_manifest.json"
    write_json_atomic(manifest_path, payload)
    manifest_hash = sha256_file(manifest_path)
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    sidecar.write_text(f"{manifest_hash}  {manifest_path.name}\n")
    summary["selection_manifest_path"] = str(manifest_path.resolve())
    summary["selection_manifest_sha256"] = manifest_hash
    write_json_atomic(output_dir / "selection_summary.json", summary)
    print(
        f"Wrote {len(rows)} rows: primary exact-band={EXPECTED_EXACT_BAND}, "
        f"overshoot={EXPECTED_OVERSHOOT} to {output_dir}"
    )


if __name__ == "__main__":
    main()
