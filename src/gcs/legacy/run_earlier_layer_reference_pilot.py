#!/usr/bin/env python3
"""Run the frozen preceding-block-reference pilot on the registered 8-run cohort.

The source prompts, target block, path definitions, and checkpoint schedule are
held fixed to the canonical output-adjacent GCS analysis.  Only the k-NN graph
and tangent bases are rebuilt, from the preceding block's final-checkpoint
``hook_resid_post`` representation.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis.batched_gcs import compute_path_gcs_batched  # noqa: E402
from gcs.analysis.gmn_tsa import compute_path_gmn_tsa  # noqa: E402
from gcs.analysis.reference_geometry import (  # noqa: E402
    build_reference_geometry_from_prompts,
)
from gcs.legacy.reanalyze_gcs import load_checkpoint  # noqa: E402


DEFAULT_MANIFEST = (
    ROOT / "scripts" / "manifests" / "earlier_layer_reference_pilot_20260724.csv"
)
DEFAULT_PROTOCOL = (
    ROOT / "rebuttal_2026" / "EARLIER_LAYER_REFERENCE_PILOT_PROTOCOL_20260724.md"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "experiments" / "rebuttal_earlier_layer_reference_pilot_20260724"
)

FROZEN_N = 200
FROZEN_K = 15
FROZEN_D = 8
FROZEN_DTYPE = torch.float64
FROZEN_DTYPE_NAME = "torch.float64"
LEGACY_TOLERANCE = 1e-10
PATH_CONFIGS = {
    "overall_path": ("hook_resid_pre", "hook_resid_post"),
    "attention_path": ("hook_resid_pre", "hook_attn_out"),
    "mlp_path": ("hook_resid_mid", "hook_mlp_out"),
}
PATH_ORDER = list(PATH_CONFIGS)


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_pickle_cpu(path: Path) -> Any:
    """Load tensors serialized on any device onto CPU."""

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


def save_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha256_int64le(prompts: torch.Tensor) -> str:
    array = (
        prompts.detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
        .numpy()
        .astype("<i8", copy=False)
    )
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _update_tensor_digest(
    digest: Any,
    label: str,
    value: torch.Tensor | np.ndarray,
    *,
    integer: bool = False,
) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().to(device="cpu").contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    dtype = "<i8" if integer else "<f8"
    stable = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    digest.update(label.encode("utf-8"))
    digest.update(np.asarray(stable.shape, dtype="<i8").tobytes())
    digest.update(stable.tobytes(order="C"))


def geometry_semantic_sha256(geometry: dict[str, Any]) -> str:
    """Hash all geometry fields that can affect a GCS score."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "N": int(geometry["N"]),
                "k": int(geometry["k"]),
                "d": int(geometry["d"]),
                "layer_idx": int(geometry["layer_idx"]),
                "hook_point": str(geometry["hook_point"]),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    _update_tensor_digest(digest, "X_prompts", geometry["X_prompts"], integer=True)
    _update_tensor_digest(digest, "X_sample", geometry["X_sample"])
    _update_tensor_digest(
        digest,
        "knn_indices",
        np.asarray(geometry["knn_indices"]),
        integer=True,
    )
    for index, value in enumerate(geometry["X_embeddings"]):
        _update_tensor_digest(digest, f"X_embeddings:{index}", value)
    for index, value in enumerate(geometry["all_input_tangent_bases"]):
        _update_tensor_digest(digest, f"tangent:{index}", value)
    return digest.hexdigest()


def geometry_to_cpu(geometry: dict[str, Any]) -> dict[str, Any]:
    output = dict(geometry)
    for key in ("X_sample", "X_prompts"):
        output[key] = geometry[key].detach().cpu()
    output["X_embeddings"] = [
        value.detach().cpu() for value in geometry["X_embeddings"]
    ]
    output["all_input_tangent_bases"] = [
        value.detach().cpu() for value in geometry["all_input_tangent_bases"]
    ]
    return output


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def run_slug(row: dict[str, Any]) -> str:
    return (
        f"{str(row['activation']).lower()}_{row['task']}_s{int(row['seed'])}"
        f"_{int(row['depth'])}l"
    )


def read_manifest(path: Path) -> list[dict[str, Any]]:
    required = {
        "activation",
        "task",
        "seed",
        "depth",
        "exp_dir",
        "target_layer",
        "reference_layer",
        "anchor_geometry",
        "gcs_epochs",
        "final_acc",
        "prompt_sha256_int64le",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")

    parsed: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        for field in ("seed", "depth", "target_layer", "reference_layer", "gcs_epochs"):
            row[field] = int(row[field])
        row["final_acc"] = float(row["final_acc"])
        if row["target_layer"] != row["depth"] - 1:
            raise ValueError(f"{row['exp_dir']}: target_layer must equal depth - 1")
        if row["reference_layer"] != row["target_layer"] - 1:
            raise ValueError(
                f"{row['exp_dir']}: reference_layer must precede target_layer"
            )
        parsed.append(row)
    if len(parsed) != 8:
        raise ValueError(f"Frozen manifest must contain 8 rows, got {len(parsed)}")
    return parsed


def canonical_epochs_and_scores(
    canonical_path: Path,
) -> tuple[list[int], dict[str, Any]]:
    payload = load_pickle_cpu(canonical_path)
    epochs = [int(value) for value in payload["epochs"]]
    scores = payload["path_gcs_scores"]
    if len(epochs) != len(set(epochs)) or epochs != sorted(epochs):
        raise ValueError(f"Canonical epoch schedule is not strictly ordered: {canonical_path}")
    return epochs, scores


def preflight_row(
    row: dict[str, Any],
    *,
    verify_prompt_hash: bool = True,
) -> dict[str, Any]:
    exp_dir = resolve_path(row["exp_dir"])
    anchor_path = exp_dir / str(row["anchor_geometry"])
    canonical_path = exp_dir / "gcs_results_union.pkl"
    metrics_path = exp_dir / "training_metrics.pkl"
    checkpoint_dir = exp_dir / "checkpoints"
    for required in (exp_dir, anchor_path, canonical_path, metrics_path, checkpoint_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing frozen input: {required}")

    epochs, scores = canonical_epochs_and_scores(canonical_path)
    if len(epochs) != int(row["gcs_epochs"]):
        raise ValueError(
            f"{exp_dir}: manifest gcs_epochs={row['gcs_epochs']}, observed={len(epochs)}"
        )
    required_scores = {
        f"layer{row['target_layer']}_{path_name}" for path_name in PATH_ORDER
    }
    if not required_scores.issubset(scores):
        raise KeyError(
            f"{canonical_path}: missing canonical paths "
            f"{sorted(required_scores - set(scores))}"
        )

    checkpoint_map = {
        checkpoint_epoch(path): path for path in checkpoint_dir.glob("*.pth")
    }
    missing_epochs = [epoch for epoch in epochs if epoch not in checkpoint_map]
    if missing_epochs:
        raise FileNotFoundError(
            f"{exp_dir}: {len(missing_epochs)} canonical epochs lack checkpoints; "
            f"first={missing_epochs[:5]}"
        )
    if not checkpoint_map:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    final_epoch = max(checkpoint_map)
    final_checkpoint = checkpoint_map[final_epoch]

    prompt_hash = str(row["prompt_sha256_int64le"])
    if verify_prompt_hash:
        geometry = load_pickle_cpu(anchor_path)
        observed = prompt_sha256_int64le(geometry["X_prompts"])
        if observed != prompt_hash:
            raise ValueError(
                f"{anchor_path}: prompt hash mismatch; observed={observed}, "
                f"manifest={prompt_hash}"
            )
        observed_dims = (
            int(geometry["N"]),
            int(geometry["k"]),
            int(geometry["d"]),
        )
        if observed_dims != (FROZEN_N, FROZEN_K, FROZEN_D):
            raise ValueError(
                f"{anchor_path}: expected N/k/d="
                f"{(FROZEN_N, FROZEN_K, FROZEN_D)}, got {observed_dims}"
            )
        if int(geometry.get("layer_idx", -1)) != int(row["target_layer"]):
            raise ValueError(
                f"{anchor_path}: anchor layer does not match target_layer"
            )

    return {
        "slug": run_slug(row),
        "exp_dir": str(exp_dir),
        "anchor_path": str(anchor_path),
        "canonical_path": str(canonical_path),
        "metrics_path": str(metrics_path),
        "canonical_epoch_count": len(epochs),
        "canonical_first_epoch": epochs[0],
        "canonical_last_epoch": epochs[-1],
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_epoch": final_epoch,
        "prompt_sha256_int64le": prompt_hash,
    }


def build_paired_reference_geometry(
    final_model: Any,
    anchor_geometry: dict[str, Any],
    *,
    reference_layer: int,
    device: str,
) -> tuple[dict[str, Any], float]:
    model_dtype = next(final_model.parameters()).dtype
    if model_dtype != FROZEN_DTYPE:
        raise RuntimeError(
            f"Frozen pilot requires {FROZEN_DTYPE_NAME}, got {model_dtype}"
        )
    prompts = anchor_geometry["X_prompts"].detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    geometry = build_reference_geometry_from_prompts(
        final_model,
        prompts,
        N=FROZEN_N,
        k=FROZEN_K,
        d=FROZEN_D,
        layer_idx=int(reference_layer),
        device=device,
        hook_point="hook_resid_post",
    )

    dtype = model_dtype
    anchor_embeddings = [
        value.detach().to(device=device, dtype=dtype)
        for value in anchor_geometry["X_embeddings"]
    ]
    differences = [
        float(
            torch.max(
                torch.abs(
                    rebuilt.to(device=device, dtype=dtype) - anchor.to(device=device)
                )
            ).item()
        )
        for rebuilt, anchor in zip(geometry["X_embeddings"], anchor_embeddings)
    ]
    max_embedding_difference = max(differences, default=0.0)
    if max_embedding_difference > 1e-12:
        raise RuntimeError(
            "Anchor embeddings do not match the final checkpoint embedding table: "
            f"max_abs={max_embedding_difference:.3g}"
        )

    # Retain the exact serialized canonical inputs so the paired comparison
    # changes only the reference graph and tangent directions.
    geometry["X_prompts"] = prompts.to(device=device)
    geometry["X_embeddings"] = anchor_embeddings
    geometry["reference_construction"] = (
        "preceding_block_final_checkpoint_resid_post_on_anchor_prompts"
    )
    geometry["anchor_embedding_max_abs_difference"] = max_embedding_difference
    return geometry, max_embedding_difference


def cast_geometry(
    geometry: dict[str, Any],
    *,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    output = dict(geometry)
    output["X_sample"] = geometry["X_sample"].to(device=device, dtype=dtype)
    output["X_prompts"] = geometry["X_prompts"].to(device=device)
    output["X_embeddings"] = [
        value.to(device=device, dtype=dtype) for value in geometry["X_embeddings"]
    ]
    output["all_input_tangent_bases"] = [
        value.to(device=device, dtype=dtype)
        for value in geometry["all_input_tangent_bases"]
    ]
    return output


def compute_three_paths_batched(
    model: Any,
    geometry: dict[str, Any],
    *,
    target_layer: int,
    device: str,
) -> dict[str, float]:
    current = cast_geometry(
        geometry,
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    return {
        path_name: float(
            compute_path_gcs_batched(
                model,
                current,
                start_hook,
                end_hook,
                target_layer=int(target_layer),
                device=device,
                alignment_method="cosine",
                token_probe="final",
            )
        )
        for path_name, (start_hook, end_hook) in PATH_CONFIGS.items()
    }


def validate_batched_against_legacy(
    model: Any,
    geometry: dict[str, Any],
    *,
    target_layer: int,
    device: str,
    path_name: str = "overall_path",
    tolerance: float = LEGACY_TOLERANCE,
) -> dict[str, Any]:
    model_dtype = next(model.parameters()).dtype
    if model_dtype != FROZEN_DTYPE:
        raise RuntimeError(
            f"Legacy gate requires {FROZEN_DTYPE_NAME}, got {model_dtype}"
        )
    start_hook, end_hook = PATH_CONFIGS[path_name]
    current = cast_geometry(
        geometry,
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    batched = float(
        compute_path_gcs_batched(
            model,
            current,
            start_hook,
            end_hook,
            target_layer=int(target_layer),
            device=device,
            alignment_method="cosine",
            token_probe="final",
        )
    )
    legacy = float(
        compute_path_gmn_tsa(
            model,
            current,
            start_hook,
            end_hook,
            target_layer=int(target_layer),
            device=device,
            alignment_method="cosine",
            token_probe="final",
        )
    )
    difference = abs(batched - legacy)
    result = {
        "status": "PASS" if difference <= tolerance else "FAIL",
        "path": path_name,
        "target_layer": int(target_layer),
        "batched_score": batched,
        "legacy_score": legacy,
        "absolute_difference": difference,
        "tolerance": float(tolerance),
        "model_dtype": FROZEN_DTYPE_NAME,
        "real_multilayer_checkpoint": True,
    }
    if difference > tolerance:
        raise RuntimeError(
            "Batched GCS failed the frozen legacy agreement gate: "
            f"|{batched} - {legacy}|={difference:.3g} > {tolerance}"
        )
    return result


def validate_legacy_validation_payload(
    payload: dict[str, Any],
    *,
    gate_row: dict[str, Any],
    manifest_sha256: str,
    protocol_sha256: str,
    final_checkpoint_path: Path,
    final_checkpoint_sha256: str,
    geometry_semantic_hash: str,
) -> None:
    """Fail closed unless a saved PASS is bound to the frozen canary inputs."""
    expected = {
        "status": "PASS",
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "tolerance": LEGACY_TOLERANCE,
        "model_dtype": FROZEN_DTYPE_NAME,
        "run": dict(gate_row),
        "final_checkpoint_path": str(final_checkpoint_path),
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "reference_geometry_semantic_sha256": geometry_semantic_hash,
    }
    mismatches = [
        key for key, value in expected.items() if payload.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Saved legacy validation is stale or bound to different inputs; "
            f"mismatched fields={mismatches}"
        )
    difference = float(payload.get("absolute_difference", float("inf")))
    if difference > LEGACY_TOLERANCE:
        raise RuntimeError(
            f"Saved legacy validation difference={difference:.3g} exceeds "
            f"{LEGACY_TOLERANCE}"
        )


def load_frozen_legacy_validation(
    gate_row: dict[str, Any],
    *,
    output_dir: Path,
    manifest_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Reload and fully bind the single frozen-canary validation."""
    validation_path = output_dir / "legacy_batched_validation.json"
    if not validation_path.exists():
        raise RuntimeError(
            "The frozen canary gate is missing. Run --legacy-gate-only (or "
            "shard 0, whose first row is the canary) before non-canary shards."
        )
    gate_exp_dir = resolve_path(gate_row["exp_dir"])
    checkpoint_paths = list((gate_exp_dir / "checkpoints").glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No canary checkpoints in {gate_exp_dir / 'checkpoints'}"
        )
    final_checkpoint = max(checkpoint_paths, key=checkpoint_epoch)
    final_hash = sha256_file(final_checkpoint)
    geometry_path = (
        output_dir / "reference_geometries" / f"{run_slug(gate_row)}.pkl"
    )
    if not geometry_path.exists():
        raise FileNotFoundError(
            f"Saved gate lacks its paired reference geometry: {geometry_path}"
        )
    geometry_payload = load_pickle_cpu(geometry_path)
    if geometry_payload.get("metadata", {}).get("run") != gate_row:
        raise RuntimeError(
            f"Canary reference geometry row mismatch: {geometry_path}"
        )
    if (
        geometry_payload.get("metadata", {}).get("model_dtype")
        != FROZEN_DTYPE_NAME
    ):
        raise RuntimeError(
            f"Canary reference geometry dtype mismatch: {geometry_path}"
        )
    observed_semantic_hash = geometry_semantic_sha256(
        geometry_payload["geometry"]
    )
    if observed_semantic_hash != geometry_payload.get("geometry_semantic_sha256"):
        raise RuntimeError(
            f"Canary reference geometry semantic hash mismatch: {geometry_path}"
        )
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    validate_legacy_validation_payload(
        payload,
        gate_row=gate_row,
        manifest_sha256=manifest_sha256,
        protocol_sha256=protocol_sha256,
        final_checkpoint_path=final_checkpoint,
        final_checkpoint_sha256=final_hash,
        geometry_semantic_hash=observed_semantic_hash,
    )
    return payload


def _metadata_fields() -> tuple[str, ...]:
    return (
        "run",
        "manifest_sha256",
        "protocol_sha256",
        "source_hashes",
        "target_epochs",
        "full_canonical_epoch_count",
        "paths",
        "analysis_params",
        "reference_geometry_path",
        "reference_geometry_semantic_sha256",
        "legacy_validation_contract",
    )


def initialize_or_resume(
    output_path: Path,
    *,
    metadata: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        payload = load_pickle_cpu(output_path)
        for key in _metadata_fields():
            if payload.get(key) != metadata.get(key):
                raise RuntimeError(
                    f"Existing partial output metadata mismatch for {key}: "
                    f"{output_path}"
                )
        processed = [int(value) for value in payload.get("epochs", [])]
        target = [int(value) for value in metadata["target_epochs"]]
        if processed != target[: len(processed)]:
            raise RuntimeError(
                f"Existing partial output is not a target-schedule prefix: {output_path}"
            )
        expected_length = len(processed)
        for path_name in metadata["paths"]:
            if len(payload.get("path_scores", {}).get(path_name, [])) != expected_length:
                raise RuntimeError(
                    f"Existing partial path length mismatch for {path_name}: {output_path}"
                )
        if len(payload.get("checkpoint_sha256", [])) != expected_length:
            raise RuntimeError(
                f"Existing partial checkpoint hash length mismatch: {output_path}"
            )
        return payload

    return {
        **metadata,
        "schema_version": 1,
        "complete": False,
        "epochs": [],
        "path_scores": {path_name: [] for path_name in metadata["paths"]},
        "checkpoint_paths": [],
        "checkpoint_sha256": [],
        "checkpoint_elapsed_seconds": [],
    }


def _load_or_build_reference_geometry(
    row: dict[str, Any],
    *,
    output_dir: Path,
    device: str,
    source_hashes: dict[str, str],
) -> tuple[dict[str, Any], Path, str, Any]:
    exp_dir = resolve_path(row["exp_dir"])
    anchor_path = exp_dir / str(row["anchor_geometry"])
    checkpoint_paths = sorted(
        (exp_dir / "checkpoints").glob("*.pth"),
        key=checkpoint_epoch,
    )
    final_checkpoint = checkpoint_paths[-1]
    final_model, _ = load_checkpoint(final_checkpoint, device=device)
    if next(final_model.parameters()).dtype != FROZEN_DTYPE:
        raise RuntimeError(
            f"{final_checkpoint}: expected {FROZEN_DTYPE_NAME}, got "
            f"{next(final_model.parameters()).dtype}"
        )
    if int(getattr(final_model, "num_layers", -1)) != int(row["depth"]):
        raise ValueError(
            f"{exp_dir}: checkpoint depth does not match manifest depth={row['depth']}"
        )

    geometry_path = (
        output_dir / "reference_geometries" / f"{run_slug(row)}.pkl"
    )
    expected_metadata = {
        "schema_version": 1,
        "run": dict(row),
        "source_hashes": source_hashes,
        "final_checkpoint": str(final_checkpoint),
        "reference_layer": int(row["reference_layer"]),
        "target_layer": int(row["target_layer"]),
        "N": FROZEN_N,
        "k": FROZEN_K,
        "d": FROZEN_D,
        "model_dtype": FROZEN_DTYPE_NAME,
        "hook_point": "hook_resid_post",
        "prompt_sha256_int64le": str(row["prompt_sha256_int64le"]),
    }
    if geometry_path.exists():
        stored = load_pickle_cpu(geometry_path)
        stored_metadata = stored.get("metadata", {})
        if any(
            stored_metadata.get(key) != value
            for key, value in expected_metadata.items()
        ):
            raise RuntimeError(
                f"Existing reference geometry metadata mismatch: {geometry_path}"
            )
        geometry = stored["geometry"]
        semantic_hash = geometry_semantic_sha256(geometry)
        if semantic_hash != stored.get("geometry_semantic_sha256"):
            raise RuntimeError(
                f"Existing reference geometry semantic hash mismatch: {geometry_path}"
            )
        return geometry, geometry_path, semantic_hash, final_model

    anchor_geometry = load_pickle_cpu(anchor_path)
    geometry, embedding_difference = build_paired_reference_geometry(
        final_model,
        anchor_geometry,
        reference_layer=int(row["reference_layer"]),
        device=device,
    )
    geometry = geometry_to_cpu(geometry)
    semantic_hash = geometry_semantic_sha256(geometry)
    save_pickle_atomic(
        geometry_path,
        {
            "metadata": {
                **expected_metadata,
                "anchor_embedding_max_abs_difference": embedding_difference,
            },
            "geometry_semantic_sha256": semantic_hash,
            "geometry": geometry,
        },
    )
    # Include the checked value in every future exact metadata comparison.
    stored = load_pickle_cpu(geometry_path)
    stored_metadata = dict(stored["metadata"])
    stored_metadata.pop("anchor_embedding_max_abs_difference", None)
    if stored_metadata != expected_metadata:
        raise RuntimeError(f"Reference geometry write verification failed: {geometry_path}")
    return geometry, geometry_path, semantic_hash, final_model


def ensure_legacy_validation(
    row: dict[str, Any],
    *,
    model: Any,
    geometry: dict[str, Any],
    output_dir: Path,
    manifest_sha256: str,
    protocol_sha256: str,
    final_checkpoint_path: Path,
    final_checkpoint_sha256: str,
    geometry_semantic_hash: str,
    device: str,
) -> dict[str, Any]:
    validation_path = output_dir / "legacy_batched_validation.json"
    if validation_path.exists():
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        validate_legacy_validation_payload(
            payload,
            gate_row=row,
            manifest_sha256=manifest_sha256,
            protocol_sha256=protocol_sha256,
            final_checkpoint_path=final_checkpoint_path,
            final_checkpoint_sha256=final_checkpoint_sha256,
            geometry_semantic_hash=geometry_semantic_hash,
        )
        return payload

    validation = validate_batched_against_legacy(
        model,
        geometry,
        target_layer=int(row["target_layer"]),
        device=device,
    )
    payload = {
        **validation,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "run": dict(row),
        "final_checkpoint_path": str(final_checkpoint_path),
        "final_checkpoint_sha256": final_checkpoint_sha256,
        "reference_geometry_semantic_sha256": geometry_semantic_hash,
        "model_dtype": FROZEN_DTYPE_NAME,
    }
    save_json_atomic(validation_path, payload)
    return payload


def run_one(
    row: dict[str, Any],
    *,
    manifest_sha256: str,
    protocol_sha256: str,
    output_dir: Path,
    device: str,
    max_checkpoints: int | None,
    save_every: int,
    overwrite: bool,
    legacy_gate_row: dict[str, Any],
) -> dict[str, Any]:
    torch.set_default_dtype(FROZEN_DTYPE)
    exp_dir = resolve_path(row["exp_dir"])
    anchor_path = exp_dir / str(row["anchor_geometry"])
    canonical_path = exp_dir / "gcs_results_union.pkl"
    metrics_path = exp_dir / "training_metrics.pkl"
    epochs, _ = canonical_epochs_and_scores(canonical_path)
    checkpoint_map = {
        checkpoint_epoch(path): path
        for path in (exp_dir / "checkpoints").glob("*.pth")
    }
    missing = [epoch for epoch in epochs if epoch not in checkpoint_map]
    if missing:
        raise FileNotFoundError(f"{exp_dir}: missing checkpoints for {missing[:5]}")
    target_epochs = list(epochs)
    if max_checkpoints is not None:
        if max_checkpoints <= 0:
            raise ValueError("--max-checkpoints must be positive")
        indices = np.linspace(
            0,
            len(target_epochs) - 1,
            num=min(max_checkpoints, len(target_epochs)),
            dtype=int,
        )
        target_epochs = [target_epochs[int(index)] for index in indices]

    final_checkpoint = max(checkpoint_map.values(), key=checkpoint_epoch)
    source_hashes = {
        "anchor_geometry_sha256": sha256_file(anchor_path),
        "canonical_gcs_sha256": sha256_file(canonical_path),
        "training_metrics_sha256": sha256_file(metrics_path),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
    }
    geometry, geometry_path, geometry_hash, final_model = (
        _load_or_build_reference_geometry(
            row,
            output_dir=output_dir,
            device=device,
            source_hashes=source_hashes,
        )
    )
    if row == legacy_gate_row:
        validation = ensure_legacy_validation(
            row,
            model=final_model,
            geometry=geometry,
            output_dir=output_dir,
            manifest_sha256=manifest_sha256,
            protocol_sha256=protocol_sha256,
            final_checkpoint_path=final_checkpoint,
            final_checkpoint_sha256=source_hashes["final_checkpoint_sha256"],
            geometry_semantic_hash=geometry_hash,
            device=device,
        )
    else:
        validation = load_frozen_legacy_validation(
            legacy_gate_row,
            output_dir=output_dir,
            manifest_sha256=manifest_sha256,
            protocol_sha256=protocol_sha256,
        )
    del final_model

    analysis_params = {
        "N": FROZEN_N,
        "k": FROZEN_K,
        "d": FROZEN_D,
        "model_dtype": FROZEN_DTYPE_NAME,
        "reference_checkpoint": "final_saved_checkpoint",
        "reference_hook": (
            f"blocks.{int(row['reference_layer'])}.hook_resid_post"
        ),
        "target_layer": int(row["target_layer"]),
        "input_prompts": "exact_ordered_anchor_geometry_prompts",
        "input_embeddings": "exact_serialized_anchor_geometry_embeddings",
        "token_probe": "final",
        "alignment_method": "cosine",
        "jvp_implementation": "analysis.batched_gcs.compute_path_gcs_batched",
        "checkpoint_schedule": (
            "exact_canonical" if max_checkpoints is None else "even_smoke_subsample"
        ),
        "max_checkpoints": max_checkpoints,
    }
    metadata = {
        "run": dict(row),
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "source_hashes": source_hashes,
        "target_epochs": target_epochs,
        "full_canonical_epoch_count": len(epochs),
        "paths": PATH_ORDER,
        "analysis_params": analysis_params,
        "reference_geometry_path": str(geometry_path),
        "reference_geometry_semantic_sha256": geometry_hash,
        "legacy_validation_contract": {
            "status": validation["status"],
            "tolerance": validation["tolerance"],
            "model_dtype": validation["model_dtype"],
            "manifest_sha256": validation["manifest_sha256"],
            "protocol_sha256": validation["protocol_sha256"],
        },
    }
    suffix = "" if max_checkpoints is None else f"_smoke{len(target_epochs)}"
    output_path = output_dir / "trajectories" / f"{run_slug(row)}{suffix}.pkl"
    payload = initialize_or_resume(
        output_path,
        metadata=metadata,
        overwrite=overwrite,
    )
    if payload.get("complete", False):
        return {
            "slug": run_slug(row),
            "status": "skipped_complete",
            "output": str(output_path),
            "n_epochs": len(target_epochs),
        }

    start_index = len(payload["epochs"])
    progress = tqdm(
        target_epochs[start_index:],
        desc=f"{run_slug(row)} earlier-ref",
        leave=False,
    )
    for index, epoch in enumerate(progress, start=start_index):
        checkpoint_path = checkpoint_map[epoch]
        started = time.perf_counter()
        checkpoint_hash = sha256_file(checkpoint_path)
        model, _ = load_checkpoint(checkpoint_path, device=device)
        if next(model.parameters()).dtype != FROZEN_DTYPE:
            raise RuntimeError(
                f"{checkpoint_path}: expected {FROZEN_DTYPE_NAME}, got "
                f"{next(model.parameters()).dtype}"
            )
        if int(getattr(model, "num_layers", -1)) != int(row["depth"]):
            raise ValueError(f"{checkpoint_path}: model depth changed within run")
        scores = compute_three_paths_batched(
            model,
            geometry,
            target_layer=int(row["target_layer"]),
            device=device,
        )
        del model
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        payload["epochs"].append(int(epoch))
        payload["checkpoint_paths"].append(str(checkpoint_path))
        payload["checkpoint_sha256"].append(checkpoint_hash)
        for path_name in PATH_ORDER:
            payload["path_scores"][path_name].append(float(scores[path_name]))
        elapsed = time.perf_counter() - started
        payload["checkpoint_elapsed_seconds"].append(float(elapsed))
        progress.set_postfix(seconds=f"{elapsed:.1f}")
        if (index + 1) % max(int(save_every), 1) == 0:
            save_pickle_atomic(output_path, payload)

    payload["complete"] = len(payload["epochs"]) == len(target_epochs)
    payload["total_elapsed_seconds"] = float(
        sum(payload["checkpoint_elapsed_seconds"])
    )
    save_pickle_atomic(output_path, payload)
    return {
        "slug": run_slug(row),
        "status": "completed",
        "output": str(output_path),
        "n_epochs": len(target_epochs),
        "elapsed_seconds": payload["total_elapsed_seconds"],
    }


def run_legacy_gate_only(
    row: dict[str, Any],
    *,
    manifest_sha256: str,
    protocol_sha256: str,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Build one paired geometry and certify batched/legacy equivalence."""
    torch.set_default_dtype(FROZEN_DTYPE)
    exp_dir = resolve_path(row["exp_dir"])
    anchor_path = exp_dir / str(row["anchor_geometry"])
    canonical_path = exp_dir / "gcs_results_union.pkl"
    metrics_path = exp_dir / "training_metrics.pkl"
    checkpoint_paths = list((exp_dir / "checkpoints").glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints in {exp_dir / 'checkpoints'}")
    final_checkpoint = max(checkpoint_paths, key=checkpoint_epoch)
    source_hashes = {
        "anchor_geometry_sha256": sha256_file(anchor_path),
        "canonical_gcs_sha256": sha256_file(canonical_path),
        "training_metrics_sha256": sha256_file(metrics_path),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
    }
    geometry, geometry_path, geometry_hash, model = (
        _load_or_build_reference_geometry(
            row,
            output_dir=output_dir,
            device=device,
            source_hashes=source_hashes,
        )
    )
    validation = ensure_legacy_validation(
        row,
        model=model,
        geometry=geometry,
        output_dir=output_dir,
        manifest_sha256=manifest_sha256,
        protocol_sha256=protocol_sha256,
        final_checkpoint_path=final_checkpoint,
        final_checkpoint_sha256=source_hashes["final_checkpoint_sha256"],
        geometry_semantic_hash=geometry_hash,
        device=device,
    )
    return {
        "status": validation["status"],
        "run_slug": run_slug(row),
        "reference_geometry_path": str(geometry_path),
        "reference_geometry_semantic_sha256": geometry_hash,
        "absolute_difference": float(validation["absolute_difference"]),
        "tolerance": float(validation["tolerance"]),
        "model_dtype": validation["model_dtype"],
    }


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        help="Evenly subsample epochs for an explicitly labeled smoke artifact.",
    )
    parser.add_argument("--run-slug", action="append", default=[])
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--legacy-gate-only",
        action="store_true",
        help=(
            "Build the first selected reference geometry and certify one real "
            "batched-vs-legacy score, then exit before trajectory computation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the manifest, prompts, schedules, and artifacts without writes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(FROZEN_DTYPE)
    manifest_path = resolve_path(args.manifest)
    protocol_path = resolve_path(args.protocol)
    output_dir = resolve_path(args.output_dir)
    if not protocol_path.exists():
        raise FileNotFoundError(f"Missing frozen protocol: {protocol_path}")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    all_rows = read_manifest(manifest_path)
    selected = [
        row
        for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_index
    ]
    if args.run_slug:
        requested = set(args.run_slug)
        selected = [row for row in selected if run_slug(row) in requested]
        missing = requested - {run_slug(row) for row in selected}
        if missing:
            raise ValueError(f"Requested run slugs were not selected: {sorted(missing)}")
    if args.max_runs is not None:
        selected = selected[: args.max_runs]
    if not selected:
        raise ValueError("No manifest rows selected")

    preflight = [preflight_row(row) for row in selected]
    invocation = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "output_dir": str(output_dir),
        "device": str(args.device),
        "model_dtype": FROZEN_DTYPE_NAME,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "max_runs": args.max_runs,
        "max_checkpoints": args.max_checkpoints,
        "legacy_gate_only": bool(args.legacy_gate_only),
        "selected_run_count": len(selected),
        "preflight": preflight,
    }
    if args.dry_run:
        print(json.dumps(invocation, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(
        output_dir / f"invocation_shard{args.shard_index}.json",
        invocation,
    )
    if args.legacy_gate_only:
        result = run_legacy_gate_only(
            all_rows[0],
            manifest_sha256=invocation["manifest_sha256"],
            protocol_sha256=invocation["protocol_sha256"],
            output_dir=output_dir,
            device=args.device,
        )
        save_json_atomic(output_dir / "legacy_gate_result.json", result)
        print(json.dumps(result, indent=2))
        return

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in selected:
        try:
            results.append(
                run_one(
                    row,
                    manifest_sha256=invocation["manifest_sha256"],
                    protocol_sha256=invocation["protocol_sha256"],
                    output_dir=output_dir,
                    device=args.device,
                    max_checkpoints=args.max_checkpoints,
                    save_every=args.save_every,
                    overwrite=args.overwrite,
                    legacy_gate_row=all_rows[0],
                )
            )
        except Exception as error:
            failures.append(
                {
                    "slug": run_slug(row),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            break

    suffix = f"_shard{args.shard_index}"
    write_csv_atomic(output_dir / f"run_manifest{suffix}.csv", results)
    save_json_atomic(output_dir / f"failures{suffix}.json", failures)
    print(json.dumps({"results": results, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
