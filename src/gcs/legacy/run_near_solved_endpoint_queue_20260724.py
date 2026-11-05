#!/usr/bin/env python3
"""Run the frozen near-solved endpoint GCS queue.

The safe default is ``--mode plan``.  Batched computation is fail-closed:
``--mode run`` is refused until ``--mode canary`` has compared all three paths
against the legacy implementation on a real registered checkpoint and written
a passing certificate with maximum discrepancy at most 1e-10.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis.batched_gcs import compute_path_gcs_batched  # noqa: E402
from gcs.analysis.gmn_tsa import compute_path_gmn_tsa  # noqa: E402
from gcs.analysis.near_solved_provenance import (  # noqa: E402
    canonical_json_sha256,
    make_file_record,
    reference_geometry_semantic_sha256,
    sha256_file,
    validate_file_record,
)
from gcs.analysis.reference_geometry import (  # noqa: E402
    build_reference_geometry_from_prompts,
)
from gcs.legacy.diagnostics_common import resolve_path  # noqa: E402
from gcs.legacy.reanalyze_gcs import load_checkpoint  # noqa: E402
from gcs.legacy.select_near_solved_endpoints_20260724 import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    _checkpoint_map,
    load_pickle_cpu,
    selection_semantic_sha256,
    tensor_sha256,
)


CANARY_TOLERANCE = 1e-10
PATH_CONFIGS = {
    "overall_path": ("hook_resid_pre", "hook_resid_post"),
    "attention_path": ("hook_resid_pre", "hook_attn_out"),
    "mlp_path": ("hook_resid_mid", "hook_mlp_out"),
}
PATHS = tuple(PATH_CONFIGS)
ANALYSIS_PARAMS = {
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

def save_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(path)


def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _geometry_to_device(
    geometry: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    moved = copy.copy(dict(geometry))
    if torch.is_tensor(moved.get("X_sample")):
        moved["X_sample"] = moved["X_sample"].to(
            device=device,
            dtype=torch.float64,
        )
    if torch.is_tensor(moved.get("X_prompts")):
        moved["X_prompts"] = moved["X_prompts"].to(
            device=device,
            dtype=torch.int64,
        )
    for key in ("X_embeddings", "all_input_tangent_bases"):
        moved[key] = [
            value.to(device=device, dtype=torch.float64)
            for value in moved[key]
        ]
    return moved


def _geometry_to_cpu(geometry: Mapping[str, Any]) -> dict[str, Any]:
    return _geometry_to_device(geometry, device="cpu")


def _load_selections(experiment_dir: Path) -> list[dict[str, Any]]:
    path = (experiment_dir / "selection_manifest.json").resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; first run "
            "scripts/select_near_solved_endpoints_20260724.py"
        )
    manifest_sha = sha256_file(path)
    payload = json.loads(path.read_text())
    rows = [dict(row) for row in payload["rows"]]
    if len(rows) != 35:
        raise ValueError(f"Expected 35 frozen selections, observed {len(rows)}")
    if sum(bool(row["exact_band"]) for row in rows) != 24:
        raise ValueError("Frozen selection no longer contains exact-band n=24")
    if sum(bool(row["overshoot"]) for row in rows) != 11:
        raise ValueError("Frozen selection no longer contains 11 overshoots")
    protocol_path = Path(payload["protocol_path"]).resolve()
    protocol_sha = sha256_file(protocol_path)
    if protocol_sha != str(payload.get("protocol_sha256")):
        raise ValueError("Frozen protocol content no longer matches selection")
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
        reference_record = {
            "epoch": int(row["reference_epoch"]),
            "path": str(row["reference_checkpoint_path"]),
            "sha256": str(row["reference_checkpoint_sha256"]),
            "size_bytes": int(row["reference_checkpoint_size_bytes"]),
        }
        validate_file_record(
            reference_record,
            expected_path=row["reference_checkpoint_path"],
            expected_epoch=int(row["reference_epoch"]),
        )
        row["selection_manifest_path"] = str(path)
        row["selection_manifest_sha256"] = manifest_sha
    return rows


def _select_queue(
    rows: Sequence[dict[str, Any]],
    *,
    cohort: str,
    requested_run_ids: set[str] | None,
    max_runs: int | None,
) -> list[dict[str, Any]]:
    selected = (
        [row for row in rows if bool(row["exact_band"])]
        if cohort == "primary"
        else list(rows)
    )
    if requested_run_ids is not None:
        available = {str(row["run_id"]) for row in selected}
        unknown = requested_run_ids - available
        if unknown:
            raise ValueError(
                f"Requested run ids outside {cohort} queue: {sorted(unknown)}"
            )
        selected = [
            row for row in selected if str(row["run_id"]) in requested_run_ids
        ]
    if max_runs is not None:
        selected = selected[: int(max_runs)]
    if not selected:
        raise ValueError("Selected queue is empty")
    return selected


def _validate_prompt_source(selection: Mapping[str, Any]) -> torch.Tensor:
    geometry = load_pickle_cpu(
        Path(selection["submitted_final_geometry_path"])
    )
    prompts = geometry["X_prompts"].detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    if tuple(prompts.shape) != (200, 3):
        raise ValueError(
            f"{selection['run_id']}: expected canonical prompts (200, 3), "
            f"got {tuple(prompts.shape)}"
        )
    observed_hash = tensor_sha256(prompts)
    if observed_hash != str(selection["canonical_prompt_sha256"]):
        raise ValueError(
            f"{selection['run_id']}: submitted prompt hash changed"
        )
    return prompts


def _validate_saved_reference_geometry(
    geometry: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    metadata = geometry.get("near_solved_reference_metadata", {})
    expected_metadata = {
        "run_id": str(selection["run_id"]),
        "reference_epoch": int(selection["reference_epoch"]),
        "canonical_prompt_sha256": str(
            selection["canonical_prompt_sha256"]
        ),
        "prompt_source": "submitted_final_geometry_X_prompts_only",
        "reference_checkpoint_path": str(
            Path(selection["reference_checkpoint_path"]).resolve()
        ),
        "reference_checkpoint_sha256": str(
            selection["reference_checkpoint_sha256"]
        ),
        "reference_checkpoint_size_bytes": int(
            selection["reference_checkpoint_size_bytes"]
        ),
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
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{selection['run_id']}: saved geometry {key} mismatch"
            )
    for key in ("N", "k", "d"):
        if int(geometry[key]) != int(ANALYSIS_PARAMS[key]):
            raise ValueError(
                f"{selection['run_id']}: saved geometry {key} mismatch"
            )
    prompts = geometry["X_prompts"].detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    if tensor_sha256(prompts) != expected_metadata["canonical_prompt_sha256"]:
        raise ValueError(
            f"{selection['run_id']}: reference geometry prompt hash mismatch"
        )
    validate_file_record(
        {
            "epoch": int(selection["reference_epoch"]),
            "path": str(selection["reference_checkpoint_path"]),
            "sha256": str(selection["reference_checkpoint_sha256"]),
            "size_bytes": int(
                selection["reference_checkpoint_size_bytes"]
            ),
        },
        expected_path=selection["reference_checkpoint_path"],
        expected_epoch=int(selection["reference_epoch"]),
    )
    if sha256_file(selection["selection_manifest_path"]) != str(
        selection["selection_manifest_sha256"]
    ):
        raise ValueError(
            f"{selection['run_id']}: selection manifest became stale"
        )
    if sha256_file(selection["protocol_path"]) != str(
        selection["protocol_sha256"]
    ):
        raise ValueError(f"{selection['run_id']}: protocol became stale")
    observed_geometry_sha = reference_geometry_semantic_sha256(geometry)
    if metadata.get(
        "reference_geometry_semantic_sha256"
    ) != observed_geometry_sha:
        raise ValueError(
            f"{selection['run_id']}: reference geometry semantic hash mismatch"
        )


def build_or_load_reference_geometry(
    selection: Mapping[str, Any],
    *,
    geometry_path: Path,
    device: str,
) -> dict[str, Any]:
    """Build selected-checkpoint geometry from canonical prompts only."""
    if geometry_path.exists():
        geometry_cpu = load_pickle_cpu(geometry_path)
        _validate_saved_reference_geometry(geometry_cpu, selection)
        return _geometry_to_device(geometry_cpu, device=device)

    prompts = _validate_prompt_source(selection)
    checkpoint_path = Path(selection["reference_checkpoint_path"])
    validate_file_record(
        {
            "epoch": int(selection["reference_epoch"]),
            "path": str(checkpoint_path.resolve()),
            "sha256": str(selection["reference_checkpoint_sha256"]),
            "size_bytes": int(
                selection["reference_checkpoint_size_bytes"]
            ),
        },
        expected_path=checkpoint_path,
        expected_epoch=int(selection["reference_epoch"]),
    )
    model, _ = load_checkpoint(checkpoint_path, device=device)
    model = model.to(device=device, dtype=torch.float64)
    try:
        geometry = build_reference_geometry_from_prompts(
            model=model,
            prompts=prompts,
            N=ANALYSIS_PARAMS["N"],
            k=ANALYSIS_PARAMS["k"],
            d=ANALYSIS_PARAMS["d"],
            layer_idx=ANALYSIS_PARAMS["target_layer"],
            device=device,
            hook_point=ANALYSIS_PARAMS["reference_hook"],
        )
        geometry_semantic_sha = reference_geometry_semantic_sha256(geometry)
        geometry["near_solved_reference_metadata"] = {
            "schema_version": 2,
            "run_id": str(selection["run_id"]),
            "reference_epoch": int(selection["reference_epoch"]),
            "reference_checkpoint_path": str(checkpoint_path.resolve()),
            "reference_checkpoint_sha256": str(
                selection["reference_checkpoint_sha256"]
            ),
            "reference_checkpoint_size_bytes": int(
                selection["reference_checkpoint_size_bytes"]
            ),
            "canonical_prompt_sha256": str(
                selection["canonical_prompt_sha256"]
            ),
            "selection_manifest_path": str(
                Path(selection["selection_manifest_path"]).resolve()
            ),
            "selection_manifest_sha256": str(
                selection["selection_manifest_sha256"]
            ),
            "selection_semantic_sha256": str(
                selection["selection_semantic_sha256"]
            ),
            "protocol_path": str(
                Path(selection["protocol_path"]).resolve()
            ),
            "protocol_sha256": str(selection["protocol_sha256"]),
            "reference_geometry_semantic_sha256": geometry_semantic_sha,
            "prompt_source_path": str(
                Path(
                    selection["submitted_final_geometry_path"]
                ).resolve()
            ),
            "prompt_source": "submitted_final_geometry_X_prompts_only",
            "excluded_final_geometry_fields": [
                "X_sample",
                "knn_indices",
                "knn_distances",
                "all_input_tangent_bases",
                "X_embeddings",
            ],
            "reference_hook": ANALYSIS_PARAMS["reference_hook"],
        }
        geometry_cpu = _geometry_to_cpu(geometry)
        _validate_saved_reference_geometry(geometry_cpu, selection)
        save_pickle_atomic(geometry_path, geometry_cpu)
        return _geometry_to_device(geometry_cpu, device=device)
    finally:
        del model
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def build_target_checkpoint_records(
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checkpoint_map = _checkpoint_map(Path(selection["source_exp_dir"]))
    records: list[dict[str, Any]] = []
    for raw_epoch in selection["target_epochs"]:
        epoch = int(raw_epoch)
        if epoch not in checkpoint_map:
            raise FileNotFoundError(
                f"{selection['run_id']}: missing target checkpoint {epoch}"
            )
        records.append(make_file_record(checkpoint_map[epoch], epoch=epoch))
    return records


def validate_trajectory_checkpoint_provenance(
    payload: Mapping[str, Any],
) -> None:
    """Rehash every frozen target and verify the processed prefix ledger."""
    targets = list(payload.get("target_checkpoints", []))
    processed = list(payload.get("processed_checkpoints", []))
    epochs = [int(epoch) for epoch in payload.get("epochs", [])]
    if not targets:
        raise ValueError("Trajectory lacks target checkpoint provenance")
    target_epochs = [int(record.get("epoch", -1)) for record in targets]
    if len(set(target_epochs)) != len(target_epochs) or any(
        right <= left
        for left, right in zip(target_epochs, target_epochs[1:])
    ):
        raise ValueError("Target checkpoint epochs are not strictly increasing")
    for record in targets:
        validate_file_record(
            record,
            expected_path=record["path"],
            expected_epoch=int(record["epoch"]),
        )
    if len(processed) != len(epochs):
        raise ValueError("Processed checkpoint ledger length mismatch")
    if processed != targets[: len(processed)]:
        raise ValueError(
            "Processed checkpoint ledger is not the frozen target prefix"
        )
    if epochs != target_epochs[: len(epochs)]:
        raise ValueError("Processed epochs are not the frozen target prefix")
    for record in processed:
        validate_file_record(
            record,
            expected_path=record["path"],
            expected_epoch=int(record["epoch"]),
        )


def _compute_path_details(
    *,
    model: Any,
    geometry: Mapping[str, Any],
    path: str,
    device: str,
    batched: bool,
) -> dict[str, Any]:
    start_hook, end_hook = PATH_CONFIGS[path]
    common = {
        "model": model,
        "geometric_data": geometry,
        "start_hook": start_hook,
        "end_hook": end_hook,
        "target_layer": ANALYSIS_PARAMS["target_layer"],
        "device": device,
        "alignment_method": ANALYSIS_PARAMS["alignment_method"],
        "token_probe": ANALYSIS_PARAMS["token_probe"],
        "return_details": True,
    }
    if batched:
        return compute_path_gcs_batched(**common)
    return compute_path_gmn_tsa(**common)


def run_batched_canary(
    selection: Mapping[str, Any],
    *,
    experiment_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Compare batched and legacy implementations on a real checkpoint."""
    geometry_path = (
        experiment_dir
        / "reference_geometries"
        / f"{selection['run_id']}.pkl"
    )
    geometry = build_or_load_reference_geometry(
        selection,
        geometry_path=geometry_path,
        device=device,
    )
    checkpoint_path = Path(selection["reference_checkpoint_path"])
    model, _ = load_checkpoint(checkpoint_path, device=device)
    model = model.to(device=device, dtype=torch.float64)
    comparisons: dict[str, dict[str, float]] = {}
    maximum_difference = 0.0
    started = time.perf_counter()
    try:
        for path in PATHS:
            legacy = _compute_path_details(
                model=model,
                geometry=geometry,
                path=path,
                device=device,
                batched=False,
            )
            batched = _compute_path_details(
                model=model,
                geometry=geometry,
                path=path,
                device=device,
                batched=True,
            )
            score_difference = abs(
                float(legacy["gcs_score"]) - float(batched["gcs_score"])
            )
            matrix_difference = float(
                np.max(
                    np.abs(
                        np.asarray(legacy["coherence_matrix"], dtype=float)
                        - np.asarray(batched["coherence_matrix"], dtype=float)
                    )
                )
            )
            path_maximum = max(score_difference, matrix_difference)
            maximum_difference = max(maximum_difference, path_maximum)
            comparisons[path] = {
                "legacy_gcs": float(legacy["gcs_score"]),
                "batched_gcs": float(batched["gcs_score"]),
                "gcs_absolute_difference": score_difference,
                "coherence_matrix_max_absolute_difference": matrix_difference,
                "maximum_absolute_difference": path_maximum,
            }
    finally:
        del model
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    geometry_semantic_sha = reference_geometry_semantic_sha256(geometry)
    payload = {
        "schema_version": 2,
        "real_checkpoint": True,
        "run_id": str(selection["run_id"]),
        "reference_epoch": int(selection["reference_epoch"]),
        "reference_checkpoint_path": str(checkpoint_path.resolve()),
        "reference_checkpoint_sha256": str(
            selection["reference_checkpoint_sha256"]
        ),
        "reference_checkpoint_size_bytes": int(
            selection["reference_checkpoint_size_bytes"]
        ),
        "reference_geometry_path": str(geometry_path.resolve()),
        "reference_geometry_semantic_sha256": geometry_semantic_sha,
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
        "canonical_prompt_sha256": str(
            selection["canonical_prompt_sha256"]
        ),
        "analysis_params": dict(ANALYSIS_PARAMS),
        "legacy_implementation": "analysis.gmn_tsa.compute_path_gmn_tsa",
        "batched_implementation": "analysis.batched_gcs.compute_path_gcs_batched",
        "tolerance": CANARY_TOLERANCE,
        "maximum_absolute_difference": maximum_difference,
        "passed": bool(maximum_difference <= CANARY_TOLERANCE),
        "comparisons": comparisons,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    save_json_atomic(experiment_dir / "batched_canary.json", payload)
    if not payload["passed"]:
        raise RuntimeError(
            "Batched GCS canary failed: maximum absolute difference "
            f"{maximum_difference:.3e} > {CANARY_TOLERANCE:.1e}"
        )
    return payload


def validate_canary_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("real_checkpoint") is not True:
        raise ValueError("Canary is not marked as a real-checkpoint comparison")
    if payload.get("passed") is not True:
        raise ValueError("Batched GCS canary did not pass")
    if float(payload.get("tolerance", np.inf)) != CANARY_TOLERANCE:
        raise ValueError("Canary tolerance does not match the frozen 1e-10")
    if float(payload.get("maximum_absolute_difference", np.inf)) > CANARY_TOLERANCE:
        raise ValueError("Canary maximum discrepancy exceeds 1e-10")
    if set(payload.get("comparisons", {})) != set(PATHS):
        raise ValueError("Canary does not cover the frozen three paths")
    for key, expected in ANALYSIS_PARAMS.items():
        if payload.get("analysis_params", {}).get(key) != expected:
            raise ValueError(f"Canary analysis parameter {key} mismatch")
    validate_file_record(
        {
            "epoch": int(payload["reference_epoch"]),
            "path": str(payload["reference_checkpoint_path"]),
            "sha256": str(payload["reference_checkpoint_sha256"]),
            "size_bytes": int(payload["reference_checkpoint_size_bytes"]),
        },
        expected_path=payload["reference_checkpoint_path"],
        expected_epoch=int(payload["reference_epoch"]),
    )
    if sha256_file(payload["selection_manifest_path"]) != str(
        payload["selection_manifest_sha256"]
    ):
        raise ValueError("Canary selection manifest content changed")
    if sha256_file(payload["protocol_path"]) != str(
        payload["protocol_sha256"]
    ):
        raise ValueError("Canary protocol content changed")
    geometry_path = Path(payload["reference_geometry_path"])
    geometry = load_pickle_cpu(geometry_path)
    observed_geometry_sha = reference_geometry_semantic_sha256(geometry)
    if observed_geometry_sha != str(
        payload["reference_geometry_semantic_sha256"]
    ):
        raise ValueError("Canary reference geometry content changed")


def require_passing_canary(
    experiment_dir: Path,
    selections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = experiment_dir / "batched_canary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run this script with --mode canary first"
        )
    payload = json.loads(path.read_text())
    validate_canary_payload(payload)
    matches = [
        selection
        for selection in selections
        if str(selection["run_id"]) == str(payload["run_id"])
    ]
    if len(matches) != 1:
        raise ValueError("Canary run is absent from the frozen selection")
    selection = matches[0]
    if not bool(selection["exact_band"]):
        raise ValueError("Canary run is not in the primary exact-band cohort")
    expected = {
        "reference_epoch": int(selection["reference_epoch"]),
        "canonical_prompt_sha256": str(
            selection["canonical_prompt_sha256"]
        ),
        "reference_checkpoint_path": str(
            Path(selection["reference_checkpoint_path"]).resolve()
        ),
        "reference_checkpoint_sha256": str(
            selection["reference_checkpoint_sha256"]
        ),
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
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Canary {key} no longer matches selection")
    return payload


def _trajectory_metadata(
    selection: Mapping[str, Any],
    *,
    geometry_path: Path,
    geometry_semantic_sha256: str,
    target_checkpoints: Sequence[Mapping[str, Any]],
    canary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "complete": False,
        "run_id": str(selection["run_id"]),
        "activation": str(selection["activation"]),
        "task": str(selection["task"]),
        "seed": int(selection["seed"]),
        "source_exp_dir": str(selection["source_exp_dir"]),
        "reference_epoch": int(selection["reference_epoch"]),
        "reference_train_accuracy": float(
            selection["reference_train_accuracy"]
        ),
        "reference_test_accuracy": float(
            selection["reference_test_accuracy"]
        ),
        "exact_band": bool(selection["exact_band"]),
        "overshoot": bool(selection["overshoot"]),
        "reference_checkpoint_path": str(
            Path(selection["reference_checkpoint_path"]).resolve()
        ),
        "reference_checkpoint_sha256": str(
            selection["reference_checkpoint_sha256"]
        ),
        "reference_checkpoint_size_bytes": int(
            selection["reference_checkpoint_size_bytes"]
        ),
        "reference_geometry_path": str(geometry_path.resolve()),
        "reference_geometry_semantic_sha256": str(
            geometry_semantic_sha256
        ),
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
        "canonical_prompt_sha256": str(
            selection["canonical_prompt_sha256"]
        ),
        "target_epochs": [
            int(epoch) for epoch in selection["target_epochs"]
        ],
        "target_checkpoints": [
            dict(record) for record in target_checkpoints
        ],
        "paths": list(PATHS),
        "analysis_params": dict(ANALYSIS_PARAMS),
        "batched_canary": {
            "run_id": canary["run_id"],
            "reference_checkpoint_sha256": canary[
                "reference_checkpoint_sha256"
            ],
            "maximum_absolute_difference": canary[
                "maximum_absolute_difference"
            ],
            "tolerance": canary["tolerance"],
            "selection_manifest_sha256": canary[
                "selection_manifest_sha256"
            ],
            "selection_semantic_sha256": canary[
                "selection_semantic_sha256"
            ],
            "protocol_sha256": canary["protocol_sha256"],
            "reference_geometry_semantic_sha256": canary[
                "reference_geometry_semantic_sha256"
            ],
        },
        "epochs": [],
        "processed_checkpoints": [],
        "path_scores": {path: [] for path in PATHS},
        "checkpoint_elapsed_seconds": [],
    }


def _load_or_initialize_trajectory(
    path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(dict(metadata))
    payload = load_pickle_cpu(path)
    immutable = (
        "run_id",
        "activation",
        "task",
        "seed",
        "source_exp_dir",
        "reference_epoch",
        "reference_train_accuracy",
        "reference_test_accuracy",
        "exact_band",
        "overshoot",
        "reference_checkpoint_path",
        "reference_checkpoint_sha256",
        "reference_checkpoint_size_bytes",
        "reference_geometry_path",
        "reference_geometry_semantic_sha256",
        "selection_manifest_path",
        "selection_manifest_sha256",
        "selection_semantic_sha256",
        "protocol_path",
        "protocol_sha256",
        "canonical_prompt_sha256",
        "target_epochs",
        "target_checkpoints",
        "paths",
        "analysis_params",
        "batched_canary",
    )
    for key in immutable:
        if payload.get(key) != metadata.get(key):
            raise ValueError(f"{metadata['run_id']}: existing {key} mismatch")
    processed = [int(epoch) for epoch in payload.get("epochs", [])]
    target = [int(epoch) for epoch in metadata["target_epochs"]]
    if processed != target[: len(processed)]:
        raise ValueError(
            f"{metadata['run_id']}: existing epochs are not a target prefix"
        )
    for path_name in PATHS:
        if len(payload["path_scores"].get(path_name, [])) != len(processed):
            raise ValueError(
                f"{metadata['run_id']}: partial path length mismatch"
            )
    if len(payload.get("checkpoint_elapsed_seconds", [])) != len(processed):
        raise ValueError(
            f"{metadata['run_id']}: partial timing length mismatch"
        )
    validate_trajectory_checkpoint_provenance(payload)
    return payload


def run_one(
    selection: Mapping[str, Any],
    *,
    experiment_dir: Path,
    canary: Mapping[str, Any],
    device: str,
    save_every: int,
) -> dict[str, Any]:
    geometry_path = (
        experiment_dir
        / "reference_geometries"
        / f"{selection['run_id']}.pkl"
    )
    trajectory_path = (
        experiment_dir
        / "trajectories"
        / f"{selection['run_id']}.pkl"
    )
    target_checkpoints = build_target_checkpoint_records(selection)
    validate_trajectory_checkpoint_provenance(
        {
            "target_checkpoints": target_checkpoints,
            "processed_checkpoints": [],
            "epochs": [],
        }
    )
    geometry = build_or_load_reference_geometry(
        selection,
        geometry_path=geometry_path,
        device=device,
    )
    geometry_semantic_sha = reference_geometry_semantic_sha256(geometry)
    metadata = _trajectory_metadata(
        selection,
        geometry_path=geometry_path,
        geometry_semantic_sha256=geometry_semantic_sha,
        target_checkpoints=target_checkpoints,
        canary=canary,
    )
    payload = _load_or_initialize_trajectory(trajectory_path, metadata)
    target_epochs = metadata["target_epochs"]
    if bool(payload.get("complete", False)):
        if payload["epochs"] != target_epochs:
            raise ValueError(
                f"{selection['run_id']}: complete trajectory has wrong epochs"
            )
        return {
            "run_id": selection["run_id"],
            "status": "skipped_complete",
            "n_epochs": len(target_epochs),
            "trajectory_path": str(trajectory_path.resolve()),
        }

    start_index = len(payload["epochs"])
    progress = tqdm(
        target_epochs[start_index:],
        desc=f"{selection['run_id']} near-solved GCS",
        leave=False,
    )
    for offset, epoch in enumerate(progress, start=start_index):
        started = time.perf_counter()
        checkpoint_record = target_checkpoints[offset]
        validate_file_record(
            checkpoint_record,
            expected_epoch=int(epoch),
        )
        model, _ = load_checkpoint(
            Path(checkpoint_record["path"]),
            device=device,
        )
        model = model.to(device=device, dtype=torch.float64)
        try:
            scores: dict[str, float] = {}
            for path in PATHS:
                details = _compute_path_details(
                    model=model,
                    geometry=geometry,
                    path=path,
                    device=device,
                    batched=True,
                )
                scores[path] = float(details["gcs_score"])
        finally:
            del model
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        payload["epochs"].append(int(epoch))
        payload["processed_checkpoints"].append(
            copy.deepcopy(checkpoint_record)
        )
        for path in PATHS:
            payload["path_scores"][path].append(scores[path])
        elapsed = float(time.perf_counter() - started)
        payload["checkpoint_elapsed_seconds"].append(elapsed)
        progress.set_postfix(seconds=f"{elapsed:.2f}")
        if (offset + 1) % max(int(save_every), 1) == 0:
            save_pickle_atomic(trajectory_path, payload)

    payload["complete"] = payload["epochs"] == target_epochs
    validate_trajectory_checkpoint_provenance(payload)
    payload["total_elapsed_seconds"] = float(
        sum(payload["checkpoint_elapsed_seconds"])
    )
    save_pickle_atomic(trajectory_path, payload)
    return {
        "run_id": selection["run_id"],
        "status": "completed",
        "n_epochs": len(target_epochs),
        "trajectory_path": str(trajectory_path.resolve()),
        "elapsed_seconds": payload["total_elapsed_seconds"],
    }


def _write_queue_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validate_plan(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_epochs = 0
    total_bytes = 0
    checkpoint_records: list[dict[str, Any]] = []
    odd_reference_runs: list[str] = []
    for selection in rows:
        prompts = _validate_prompt_source(selection)
        del prompts
        checkpoint_map = _checkpoint_map(Path(selection["source_exp_dir"]))
        target = [int(epoch) for epoch in selection["target_epochs"]]
        if target[-1] != int(selection["reference_epoch"]):
            raise ValueError(
                f"{selection['run_id']}: prefix does not end at reference"
            )
        missing = [epoch for epoch in target if epoch not in checkpoint_map]
        if missing:
            raise FileNotFoundError(
                f"{selection['run_id']}: missing target checkpoints {missing}"
            )
        records = build_target_checkpoint_records(selection)
        validate_trajectory_checkpoint_provenance(
            {
                "target_checkpoints": records,
                "processed_checkpoints": [],
                "epochs": [],
            }
        )
        checkpoint_records.extend(records)
        total_bytes += sum(int(record["size_bytes"]) for record in records)
        if int(selection["reference_epoch"]) % 200 != 0:
            odd_reference_runs.append(str(selection["run_id"]))
        total_epochs += len(target)
    return {
        "run_count": len(rows),
        "target_checkpoint_analyses": total_epochs,
        "target_checkpoint_files_rehashed": len(checkpoint_records),
        "target_checkpoint_total_bytes": total_bytes,
        "target_checkpoint_ledger_sha256": canonical_json_sha256(
            checkpoint_records
        ),
        "selection_manifest_sha256": rows[0][
            "selection_manifest_sha256"
        ],
        "selection_semantic_sha256": rows[0][
            "selection_semantic_sha256"
        ],
        "protocol_sha256": rows[0]["protocol_sha256"],
        "odd_100_reference_runs": odd_reference_runs,
        "formal_gpu_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--mode",
        choices=("plan", "canary", "run"),
        default="plan",
    )
    parser.add_argument(
        "--cohort",
        choices=("primary", "full"),
        default="primary",
    )
    parser.add_argument(
        "--run-ids",
        default=None,
        help="Optional comma-separated subset after cohort filtering.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--canary-run-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    experiment_dir = resolve_path(args.experiment_dir)
    rows = _load_selections(experiment_dir)
    requested = (
        {
            value.strip()
            for value in args.run_ids.split(",")
            if value.strip()
        }
        if args.run_ids
        else None
    )
    queue = _select_queue(
        rows,
        cohort=args.cohort,
        requested_run_ids=requested,
        max_runs=args.max_runs,
    )

    if args.mode == "plan":
        print(json.dumps(_validate_plan(queue), indent=2, sort_keys=True))
        return

    if args.mode == "canary":
        exact = [row for row in rows if bool(row["exact_band"])]
        if args.canary_run_id is None:
            selection = exact[0]
        else:
            matches = [
                row
                for row in exact
                if str(row["run_id"]) == args.canary_run_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    "--canary-run-id must name one primary exact-band run"
                )
            selection = matches[0]
        result = run_batched_canary(
            selection,
            experiment_dir=experiment_dir,
            device=args.device,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    canary = require_passing_canary(experiment_dir, rows)
    results: list[dict[str, Any]] = []
    for selection in queue:
        result = run_one(
            selection,
            experiment_dir=experiment_dir,
            canary=canary,
            device=args.device,
            save_every=args.save_every,
        )
        results.append(result)
        _write_queue_manifest(
            experiment_dir / "queue_manifest.csv",
            results,
        )
    save_json_atomic(
        experiment_dir / "queue_manifest.json",
        {
            "cohort": args.cohort,
            "results": results,
        },
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
