"""Content-addressed provenance helpers for the frozen near-solved queue."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_file_record(
    path: str | Path,
    *,
    epoch: int | None = None,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    record: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }
    if epoch is not None:
        record["epoch"] = int(epoch)
    return record


def validate_file_record(
    record: Mapping[str, Any],
    *,
    expected_path: str | Path | None = None,
    expected_epoch: int | None = None,
) -> None:
    required = {"path", "sha256", "size_bytes"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"File record lacks fields {sorted(missing)}")
    path = Path(str(record["path"])).resolve()
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(
            f"File-record path mismatch: observed={path}, "
            f"expected={Path(expected_path).resolve()}"
        )
    if expected_epoch is not None and int(record.get("epoch", -1)) != int(
        expected_epoch
    ):
        raise ValueError(
            f"File-record epoch mismatch: observed={record.get('epoch')}, "
            f"expected={int(expected_epoch)}"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = int(path.stat().st_size)
    if observed_size != int(record["size_bytes"]):
        raise ValueError(
            f"Stale file size for {path}: observed={observed_size}, "
            f"frozen={record['size_bytes']}"
        )
    observed_hash = sha256_file(path)
    if observed_hash != str(record["sha256"]):
        raise ValueError(
            f"Stale file hash for {path}: observed={observed_hash}, "
            f"frozen={record['sha256']}"
        )


def _array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").contiguous().numpy()
    return np.ascontiguousarray(np.asarray(value))


def _update_array(
    digest: "hashlib._Hash",
    *,
    name: str,
    value: Any,
) -> None:
    array = _array(value)
    descriptor = {
        "name": name,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    raw = array.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def reference_geometry_semantic_sha256(
    geometry: Mapping[str, Any],
) -> str:
    """Hash the semantic geometry payload, excluding provenance metadata.

    The hash covers prompts, selected-checkpoint representations, neighbor
    graph/distances, embeddings, tangent bases, and geometry conventions.  It
    intentionally excludes ``near_solved_reference_metadata`` so the digest
    can be stored inside that metadata without self-reference.
    """
    required = {
        "X_sample",
        "X_prompts",
        "knn_indices",
        "knn_distances",
        "X_embeddings",
        "all_input_tangent_bases",
        "N",
        "k",
        "d",
        "d_model",
        "layer_idx",
        "hook_point",
        "model_type",
        "embedding_type",
    }
    missing = required - set(geometry)
    if missing:
        raise ValueError(
            f"Reference geometry lacks semantic fields {sorted(missing)}"
        )

    conventions = {
        key: geometry.get(key)
        for key in (
            "N",
            "k",
            "d",
            "d_model",
            "layer_idx",
            "hook_point",
            "model_type",
            "embedding_type",
            "prompt_subset_rule",
            "knn_components",
        )
    }
    digest = hashlib.sha256()
    digest.update(b"near-solved-reference-geometry-v1\0")
    encoded = json.dumps(
        conventions,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    _update_array(digest, name="X_prompts", value=geometry["X_prompts"])
    _update_array(digest, name="X_sample", value=geometry["X_sample"])
    _update_array(
        digest,
        name="knn_indices",
        value=geometry["knn_indices"],
    )
    _update_array(
        digest,
        name="knn_distances",
        value=geometry["knn_distances"],
    )
    _update_array(
        digest,
        name="X_embeddings",
        value=np.stack(
            [_array(value) for value in geometry["X_embeddings"]],
            axis=0,
        ),
    )
    _update_array(
        digest,
        name="all_input_tangent_bases",
        value=np.stack(
            [
                _array(value)
                for value in geometry["all_input_tangent_bases"]
            ],
            axis=0,
        ),
    )
    return digest.hexdigest()
