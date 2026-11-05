#!/usr/bin/env python3
"""Run the frozen cohort-level N/k/d Attention-GCS sensitivity experiment.

The formal workflow is deliberately two phase:

1. ``--phase default_control`` recomputes (N, k, d) = (200, 15, 8).
2. The evaluator must issue ``default_control_gate.json`` with status PASS.
3. ``--phase variants`` computes the five frozen one-factor variants.

This prevents formal variants from running behind a stale or failed control.
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
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis.batched_gcs import compute_path_gcs_batched  # noqa: E402
from gcs.analysis.gmn_tsa import compute_path_gmn_tsa  # noqa: E402
from gcs.analysis.reference_geometry import rebuild_geometry_variant  # noqa: E402
from gcs.legacy.diagnostics_common import RUNS_1L, load_gcs, resolve_path  # noqa: E402
from gcs.legacy.reanalyze_gcs import load_checkpoint  # noqa: E402


PROTOCOL_PATH = (
    ROOT / "rebuttal_2026" / "SELECTOR_NKD_SENSITIVITY_PROTOCOL_20260724.md"
)
FROZEN_COHORT_MANIFEST = (
    ROOT
    / "rebuttal_2026"
    / "direct_jacobian_graph_audit_20260723"
    / "run_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "experiments" / "rebuttal_gcs_nkd_selector_sensitivity_20260724"
)

VARIANTS = {
    "default_control": {"N": 200, "k": 15, "d": 8},
    "k10": {"N": 200, "k": 10, "d": 8},
    "k20": {"N": 200, "k": 20, "d": 8},
    "d4": {"N": 200, "k": 15, "d": 4},
    "d12": {"N": 200, "k": 15, "d": 12},
    "n100": {"N": 100, "k": 15, "d": 8},
}
FORMAL_VARIANT_NAMES = ("k10", "k20", "d4", "d12", "n100")
ATTENTION_PATH = ("hook_resid_pre", "hook_attn_out")
EXPECTED_COHORT_SIZE = 23
EXPECTED_ACCURACY_RUNS = 23
EXPECTED_EQUIVARIANCE_RUNS = 11
LEGACY_BATCHED_TOLERANCE = 1e-10
METADATA_KEYS = (
    "variant",
    "act",
    "task",
    "seed",
    "source_exp_dir",
    "geometry_path",
    "geometry_hashes",
    "target_epochs",
    "target_checkpoint_parameter_sha256",
    "full_saved_epoch_count",
    "analysis_params",
)


def load_pickle_cpu(path: Path) -> Any:
    """Load tensor pickles without depending on their original CUDA device."""

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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: torch.Tensor | np.ndarray) -> str:
    if isinstance(value, torch.Tensor):
        array = value.detach().to(device="cpu").contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_slug(act: str, task: str, seed: int) -> str:
    return f"{act.lower()}_{task}_s{int(seed)}"


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Checkpoint filename lacks epoch_<integer>: {path}")
    return int(match.group(1))


def evenly_subsample(values: list[int], max_items: int | None) -> list[int]:
    if max_items is None or len(values) <= max_items:
        return list(values)
    if max_items <= 0:
        raise ValueError("--max-checkpoints must be positive")
    if max_items == 1:
        return [values[-1]]
    indices = np.linspace(0, len(values) - 1, num=max_items, dtype=int)
    return [values[int(index)] for index in indices]


def _absolute_from_root(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def load_frozen_cohort(
    manifest_path: Path,
    *,
    registered_runs: Iterable[tuple[str, str, int, str]] = RUNS_1L,
    root: Path = ROOT,
    expected_count: int = EXPECTED_COHORT_SIZE,
) -> list[tuple[str, str, int, str]]:
    """Fail closed unless the frozen manifest exactly matches registered runs."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen cohort manifest: {manifest_path}")
    source_root = None
    path_metadata = manifest_path.with_suffix(".paths.json")
    if path_metadata.exists():
        metadata = json.loads(path_metadata.read_text())
        if metadata.get("manifest_sha256") != file_sha256(manifest_path):
            raise RuntimeError(
                f"Frozen cohort manifest checksum does not match {path_metadata}"
            )
        source_root = Path(metadata["source_root"])
        if not source_root.is_absolute():
            raise RuntimeError("Frozen cohort source_root must be absolute")
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Frozen cohort manifest row count={len(rows)}, expected={expected_count}"
        )

    registered = [
        (act, task, int(seed), base)
        for act, task, seed, base in registered_runs
        if int(seed) in {123, 777}
    ]
    expected = {
        (act, task, seed): _absolute_from_root(base, root)
        for act, task, seed, base in registered
    }
    if len(expected) != expected_count:
        raise RuntimeError(
            f"Registered held-out cohort size={len(expected)}, expected={expected_count}"
        )

    observed_keys: list[tuple[str, str, int]] = []
    output: list[tuple[str, str, int, str]] = []
    for row in rows:
        key = (str(row["act"]), str(row["task"]), int(row["seed"]))
        if key in observed_keys:
            raise RuntimeError(f"Duplicate frozen cohort row: {key}")
        observed_keys.append(key)
        if key not in expected:
            raise RuntimeError(f"Unexpected frozen cohort row: {key}")
        if row.get("status") not in {"completed", "skipped_complete"}:
            raise RuntimeError(f"Frozen manifest status is not complete for {key}")
        observed_path = Path(row["exp_dir"])
        if source_root is not None and observed_path.is_absolute():
            try:
                observed_path = observed_path.relative_to(source_root)
            except ValueError as error:
                raise RuntimeError(
                    f"Frozen cohort path outside declared source_root for {key}: "
                    f"{observed_path}"
                ) from error
        observed_path = _absolute_from_root(observed_path, root)
        if observed_path != expected[key]:
            raise RuntimeError(
                f"Frozen cohort path mismatch for {key}: "
                f"observed={observed_path}, expected={expected[key]}"
            )
        output.append((*key, str(expected[key])))

    if set(observed_keys) != set(expected):
        missing = sorted(set(expected) - set(observed_keys))
        extra = sorted(set(observed_keys) - set(expected))
        raise RuntimeError(
            f"Frozen cohort key mismatch: missing={missing}, unexpected={extra}"
        )
    return output


def geometry_hashes(
    geometry: dict[str, Any],
    *,
    source_geometry_path: Path,
    variant: str,
) -> dict[str, str]:
    embeddings = torch.stack(
        [value.detach().to(device="cpu") for value in geometry["X_embeddings"]]
    )
    tangent_bases = torch.stack(
        [
            value.detach().to(device="cpu")
            for value in geometry["all_input_tangent_bases"]
        ]
    )
    params = {
        "variant": variant,
        "N": int(geometry["N"]),
        "k": int(geometry["k"]),
        "d": int(geometry["d"]),
    }
    return {
        "source_geometry_sha256": file_sha256(source_geometry_path),
        "prompt_sha256": array_sha256(geometry["X_prompts"]),
        "representation_sha256": array_sha256(geometry["X_sample"]),
        "embedding_sha256": array_sha256(embeddings),
        "neighbor_sha256": array_sha256(geometry["knn_indices"]),
        "neighbor_distance_sha256": array_sha256(geometry["knn_distances"]),
        "tangent_sha256": array_sha256(tangent_bases),
        "variant_parameter_sha256": json_sha256(params),
    }


def output_is_complete(
    path: Path,
    *,
    target_epochs: list[int],
    expected_metadata: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_pickle_cpu(path)
    except Exception:
        return False
    if not bool(payload.get("complete", False)):
        return False
    if [int(value) for value in payload.get("epochs", [])] != target_epochs:
        return False
    if [int(value) for value in payload.get("target_epochs", [])] != target_epochs:
        return False
    for key in METADATA_KEYS:
        if payload.get(key) != expected_metadata.get(key):
            return False
    expected_length = len(target_epochs)
    return (
        len(payload.get("attention_scores", [])) == expected_length
        and len(payload.get("checkpoint_parameter_sha256", [])) == expected_length
        and len(payload.get("checkpoint_elapsed_seconds", [])) == expected_length
        and payload.get("checkpoint_parameter_sha256")
        == expected_metadata.get("target_checkpoint_parameter_sha256")
    )


def initialize_or_resume(
    output_path: Path,
    *,
    metadata: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        payload = load_pickle_cpu(output_path)
        processed = [int(value) for value in payload.get("epochs", [])]
        target_epochs = [int(value) for value in metadata["target_epochs"]]
        if processed != target_epochs[: len(processed)]:
            raise RuntimeError(
                f"Existing partial output has a non-prefix epoch schedule: {output_path}"
            )
        for key in METADATA_KEYS:
            if payload.get(key) != metadata.get(key):
                raise RuntimeError(
                    f"Existing output metadata mismatch for {key}: {output_path}"
                )
        lengths = {
            len(processed),
            len(payload.get("attention_scores", [])),
            len(payload.get("checkpoint_parameter_sha256", [])),
            len(payload.get("checkpoint_elapsed_seconds", [])),
        }
        if len(lengths) != 1:
            raise RuntimeError(
                f"Existing partial output has misaligned trajectory fields: {output_path}"
            )
        expected_hash_prefix = metadata["target_checkpoint_parameter_sha256"][
            : len(processed)
        ]
        if payload["checkpoint_parameter_sha256"] != expected_hash_prefix:
            raise RuntimeError(
                f"Existing partial output checkpoint hash mismatch: {output_path}"
            )
        return payload

    return {
        **metadata,
        "schema_version": 1,
        "complete": False,
        "epochs": [],
        "attention_scores": [],
        "checkpoint_parameter_sha256": [],
        "checkpoint_elapsed_seconds": [],
    }


def verify_default_control_gate(
    output_dir: Path,
    *,
    expected_slugs: set[str],
    expected_accuracy_runs: int = EXPECTED_ACCURACY_RUNS,
    expected_equivariance_runs: int = EXPECTED_EQUIVARIANCE_RUNS,
) -> dict[str, Any]:
    gate_path = output_dir / "default_control_gate.json"
    if not gate_path.exists():
        raise RuntimeError(
            "Formal variants are locked until the evaluator creates "
            f"a PASS gate: {gate_path}"
        )
    with gate_path.open() as handle:
        gate = json.load(handle)
    if gate.get("status") != "PASS" or gate.get("formal") is not True:
        raise RuntimeError(f"Default-control gate is not a formal PASS: {gate_path}")
    if int(gate.get("expected_accuracy_runs", -1)) != expected_accuracy_runs:
        raise RuntimeError("Default-control gate accuracy denominator mismatch")
    if int(gate.get("expected_equivariance_runs", -1)) != expected_equivariance_runs:
        raise RuntimeError("Default-control gate equivariance denominator mismatch")

    recorded = gate.get("trajectory_sha256", {})
    if set(recorded) != expected_slugs:
        raise RuntimeError(
            "Default-control gate trajectory set mismatch: "
            f"observed={sorted(recorded)}, expected={sorted(expected_slugs)}"
        )
    for slug in sorted(expected_slugs):
        trajectory = (
            output_dir / "trajectories" / "default_control" / f"{slug}.pkl"
        )
        if not trajectory.exists():
            raise RuntimeError(f"Missing gated default trajectory: {trajectory}")
        observed_hash = file_sha256(trajectory)
        if observed_hash != recorded[slug]:
            raise RuntimeError(
                f"Default-control trajectory hash mismatch for {slug}: "
                f"observed={observed_hash}, gate={recorded[slug]}"
            )
    return gate


def _checkpoint_map(exp_dir: Path) -> dict[int, Path]:
    paths = sorted(
        {
            *list((exp_dir / "checkpoints").glob("*.pth")),
            *list(exp_dir.glob("*.pth")),
        }
    )
    if not paths:
        raise FileNotFoundError(
            "No checkpoints found in either supported layout: "
            f"{exp_dir / 'checkpoints'}/*.pth or {exp_dir}/*.pth"
        )
    result: dict[int, Path] = {}
    for path in paths:
        epoch = checkpoint_epoch(path)
        if epoch in result:
            raise RuntimeError(
                f"Duplicate checkpoint epoch {epoch}: {result[epoch]} and {path}"
            )
        result[epoch] = path
    return result


def prepare_run(
    act: str,
    task: str,
    seed: int,
    base: str,
    *,
    max_checkpoints: int | None,
    hash_checkpoints: bool,
) -> dict[str, Any]:
    exp_dir = resolve_path(base)
    geometry_path = exp_dir / "final_geometry.pkl"
    if not geometry_path.exists():
        raise FileNotFoundError(f"Missing submitted geometry: {geometry_path}")
    fixed_geometry = load_pickle_cpu(geometry_path)
    observed_default = tuple(
        int(fixed_geometry[key]) for key in ("N", "k", "d")
    )
    if observed_default != (200, 15, 8):
        raise RuntimeError(
            f"Submitted geometry is not (200,15,8) for {act}/{task}/s{seed}: "
            f"{observed_default}"
        )

    saved_epochs, _ = load_gcs(base)
    rounded_epochs = [int(value) for value in saved_epochs]
    if any(float(raw) != float(integer) for raw, integer in zip(saved_epochs, rounded_epochs)):
        raise RuntimeError(f"Non-integral submitted GCS epoch in {exp_dir}")
    checkpoint_map = _checkpoint_map(exp_dir)
    missing = [epoch for epoch in rounded_epochs if epoch not in checkpoint_map]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} submitted epochs lack checkpoints in {exp_dir}: "
            f"{missing[:5]}"
        )
    target_epochs = evenly_subsample(rounded_epochs, max_checkpoints)
    parameter_hashes = (
        [file_sha256(checkpoint_map[epoch]) for epoch in target_epochs]
        if hash_checkpoints
        else []
    )
    return {
        "act": act,
        "task": task,
        "seed": int(seed),
        "exp_dir": exp_dir,
        "geometry_path": geometry_path,
        "fixed_geometry": fixed_geometry,
        "checkpoint_map": checkpoint_map,
        "target_epochs": target_epochs,
        "full_saved_epoch_count": len(rounded_epochs),
        "parameter_hashes": parameter_hashes,
    }


def _analysis_params(
    variant: str,
    *,
    max_checkpoints: int | None,
    formal: bool,
) -> dict[str, Any]:
    params = VARIANTS[variant]
    return {
        **params,
        "path": "attention_path",
        "start_hook": ATTENTION_PATH[0],
        "end_hook": ATTENTION_PATH[1],
        "target_layer": 0,
        "alignment_method": "cosine",
        "alignment": "row_wise_absolute_cosine",
        "token_probe": "final",
        "aggregation": "spectral_entropy_N_minus_2_pow_H",
        "batched_jvp": True,
        "prompt_subset_rule": "canonical_saved_order_prefix",
        "max_checkpoints": max_checkpoints,
        "formal": formal,
    }


def run_one_variant(
    prepared: dict[str, Any],
    *,
    variant: str,
    output_dir: Path,
    device: str,
    save_every: int,
    max_checkpoints: int | None,
    formal: bool,
    overwrite: bool,
) -> dict[str, Any]:
    act = str(prepared["act"])
    task = str(prepared["task"])
    seed = int(prepared["seed"])
    target_epochs = list(prepared["target_epochs"])
    parameter_hashes = list(prepared["parameter_hashes"])
    if len(parameter_hashes) != len(target_epochs):
        raise RuntimeError("Checkpoint parameter hashes were not prepared")

    params = VARIANTS[variant]
    geometry = rebuild_geometry_variant(
        prepared["fixed_geometry"],
        N=params["N"],
        k=params["k"],
        d=params["d"],
        device=device,
        dtype=torch.float64,
    )
    hashes = geometry_hashes(
        geometry,
        source_geometry_path=prepared["geometry_path"],
        variant=variant,
    )
    analysis_params = _analysis_params(
        variant,
        max_checkpoints=max_checkpoints,
        formal=formal,
    )
    metadata = {
        "variant": variant,
        "act": act,
        "task": task,
        "seed": seed,
        "source_exp_dir": str(prepared["exp_dir"]),
        "geometry_path": str(prepared["geometry_path"]),
        "geometry_hashes": hashes,
        "target_epochs": target_epochs,
        "target_checkpoint_parameter_sha256": parameter_hashes,
        "full_saved_epoch_count": int(prepared["full_saved_epoch_count"]),
        "analysis_params": analysis_params,
    }
    output_path = (
        output_dir
        / "trajectories"
        / variant
        / f"{run_slug(act, task, seed)}.pkl"
    )
    if (
        output_is_complete(
            output_path,
            target_epochs=target_epochs,
            expected_metadata=metadata,
        )
        and not overwrite
    ):
        return {
            "variant": variant,
            "act": act,
            "task": task,
            "seed": seed,
            "exp_dir": str(prepared["exp_dir"]),
            "output": str(output_path),
            "status": "skipped_complete",
            "n_epochs": len(target_epochs),
        }

    payload = initialize_or_resume(
        output_path,
        metadata=metadata,
        overwrite=overwrite,
    )
    if overwrite or not output_path.exists():
        save_pickle_atomic(output_path, payload)
    start_index = len(payload["epochs"])
    progress = tqdm(
        target_epochs[start_index:],
        desc=f"{variant}/{run_slug(act, task, seed)}",
        leave=False,
    )
    for index, epoch in enumerate(progress, start=start_index):
        started = time.perf_counter()
        checkpoint_path = prepared["checkpoint_map"][epoch]
        observed_parameter_hash = file_sha256(checkpoint_path)
        expected_parameter_hash = parameter_hashes[index]
        if observed_parameter_hash != expected_parameter_hash:
            raise RuntimeError(
                f"Checkpoint changed after schedule preparation: {checkpoint_path}"
            )

        model, _ = load_checkpoint(checkpoint_path, device=device)
        model = model.to(device=device, dtype=torch.float64)
        score = compute_path_gcs_batched(
            model,
            geometry,
            ATTENTION_PATH[0],
            ATTENTION_PATH[1],
            target_layer=0,
            device=device,
            alignment_method="cosine",
            token_probe="final",
        )
        score = float(score)
        if not np.isfinite(score):
            raise RuntimeError(
                f"Non-finite Attention-GCS at {variant}/{act}/{task}/s{seed}/e{epoch}"
            )
        elapsed = time.perf_counter() - started
        payload["epochs"].append(int(epoch))
        payload["attention_scores"].append(score)
        payload["checkpoint_parameter_sha256"].append(observed_parameter_hash)
        payload["checkpoint_elapsed_seconds"].append(float(elapsed))
        progress.set_postfix(seconds=f"{elapsed:.2f}", score=f"{score:.6g}")
        del model
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        if (index + 1) % max(int(save_every), 1) == 0:
            save_pickle_atomic(output_path, payload)

    payload["complete"] = len(payload["epochs"]) == len(target_epochs)
    payload["total_elapsed_seconds"] = float(
        sum(payload["checkpoint_elapsed_seconds"])
    )
    save_pickle_atomic(output_path, payload)
    del geometry
    return {
        "variant": variant,
        "act": act,
        "task": task,
        "seed": seed,
        "exp_dir": str(prepared["exp_dir"]),
        "output": str(output_path),
        "status": "completed",
        "n_epochs": len(target_epochs),
        "elapsed_seconds": payload["total_elapsed_seconds"],
    }


def run_legacy_batched_gate(
    prepared: dict[str, Any],
    *,
    output_dir: Path,
    cohort_manifest: Path,
    device: str,
    tolerance: float = LEGACY_BATCHED_TOLERANCE,
) -> dict[str, Any]:
    """Match batched and legacy GCS on one deterministic real checkpoint."""
    target_epochs = list(prepared["target_epochs"])
    gate_index = len(target_epochs) // 2
    epoch = target_epochs[gate_index]
    checkpoint_path = prepared["checkpoint_map"][epoch]
    params = VARIANTS["default_control"]
    geometry = rebuild_geometry_variant(
        prepared["fixed_geometry"],
        N=params["N"],
        k=params["k"],
        d=params["d"],
        device=device,
        dtype=torch.float64,
    )
    hashes = geometry_hashes(
        geometry,
        source_geometry_path=prepared["geometry_path"],
        variant="default_control",
    )
    identifying = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "cohort_manifest_sha256": file_sha256(cohort_manifest),
        "cohort_path_metadata_sha256": (
            file_sha256(cohort_manifest.with_suffix(".paths.json"))
            if cohort_manifest.with_suffix(".paths.json").exists() else None
        ),
        "act": prepared["act"],
        "task": prepared["task"],
        "seed": int(prepared["seed"]),
        "source_exp_dir": str(prepared["exp_dir"]),
        "epoch": int(epoch),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_parameter_sha256": file_sha256(checkpoint_path),
        "geometry_hashes": hashes,
        "path": "attention_path",
        "target_layer": 0,
        "alignment_method": "cosine",
        "token_probe": "final",
        "tolerance": float(tolerance),
    }
    gate_path = output_dir / "legacy_batched_gate.json"
    if gate_path.exists():
        with gate_path.open() as handle:
            cached = json.load(handle)
        if (
            cached.get("status") == "PASS"
            and all(cached.get(key) == value for key, value in identifying.items())
            and float(cached.get("absolute_difference", np.inf)) <= tolerance
        ):
            return cached

    model, _ = load_checkpoint(checkpoint_path, device=device)
    model = model.to(device=device, dtype=torch.float64)
    legacy_score = float(
        compute_path_gmn_tsa(
            model,
            geometry,
            ATTENTION_PATH[0],
            ATTENTION_PATH[1],
            target_layer=0,
            device=device,
            alignment_method="cosine",
            token_probe="final",
        )
    )
    batched_score = float(
        compute_path_gcs_batched(
            model,
            geometry,
            ATTENTION_PATH[0],
            ATTENTION_PATH[1],
            target_layer=0,
            device=device,
            alignment_method="cosine",
            token_probe="final",
        )
    )
    difference = abs(legacy_score - batched_score)
    passed = (
        np.isfinite(legacy_score)
        and np.isfinite(batched_score)
        and difference <= tolerance
    )
    payload = {
        **identifying,
        "legacy_score": legacy_score,
        "batched_score": batched_score,
        "absolute_difference": difference,
        "status": "PASS" if passed else "FAIL",
    }
    write_json_atomic(gate_path, payload)
    del model, geometry
    if not passed:
        raise RuntimeError(
            "Legacy-vs-batched real-checkpoint gate failed: "
            f"legacy={legacy_score}, batched={batched_score}, "
            f"abs_diff={difference}, tolerance={tolerance}"
        )
    return payload


def _selected_variants(args: argparse.Namespace, *, formal: bool) -> list[str]:
    if args.phase == "default_control":
        return ["default_control"]
    values = [value.strip() for value in args.variants.split(",") if value.strip()]
    if len(values) != len(set(values)):
        raise ValueError("--variants contains duplicates")
    unknown = sorted(set(values) - set(FORMAL_VARIANT_NAMES))
    if unknown:
        raise ValueError(f"Unknown formal variants: {unknown}")
    if formal and tuple(values) != FORMAL_VARIANT_NAMES:
        raise RuntimeError(
            "An uncapped formal run must use exactly "
            f"{','.join(FORMAL_VARIANT_NAMES)} in frozen order"
        )
    if not values:
        raise ValueError("No variants selected")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen N/k/d Attention-GCS sensitivity trajectories. "
            "Run default_control, evaluate its gate, then run variants."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("default_control", "variants"),
        default="default_control",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=FROZEN_COHORT_MANIFEST,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--variants",
        default=",".join(FORMAL_VARIANT_NAMES),
        help="Variants phase only; formal runs require the frozen five in order.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-legacy-gate",
        action="store_true",
        help="Allowed only for capped, explicitly non-formal smoke runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate cohort/artifact schedules and print work without GPU compute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")
    if args.max_checkpoints is not None and args.max_checkpoints <= 0:
        raise ValueError("--max-checkpoints must be positive")
    formal = args.max_runs is None and args.max_checkpoints is None
    if args.skip_legacy_gate and formal:
        raise RuntimeError("--skip-legacy-gate is forbidden for formal runs")

    cohort_manifest = resolve_path(args.cohort_manifest)
    cohort = load_frozen_cohort(cohort_manifest)
    expected_slugs = {
        run_slug(act, task, seed) for act, task, seed, _ in cohort
    }
    selected_variants = _selected_variants(args, formal=formal)
    runs = cohort[: args.max_runs] if args.max_runs is not None else cohort

    prepared_runs: list[dict[str, Any]] = []
    checkpoint_count = 0
    for act, task, seed, base in runs:
        prepared = prepare_run(
            act,
            task,
            seed,
            base,
            max_checkpoints=args.max_checkpoints,
            hash_checkpoints=not args.dry_run,
        )
        prepared_runs.append(prepared)
        checkpoint_count += len(prepared["target_epochs"])

    print(
        "N/k/d sensitivity worklist: "
        f"phase={args.phase} formal={formal} runs={len(runs)} "
        f"variants={','.join(selected_variants)} "
        f"checkpoint_evaluations={checkpoint_count * len(selected_variants)}"
    )
    if args.dry_run:
        print("DRY RUN: cohort and all submitted checkpoint schedules validated.")
        return

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "variants":
        verify_default_control_gate(
            output_dir,
            expected_slugs=expected_slugs,
        )

    phase_config = {
        "schema_version": 1,
        "frozen_protocol": str(PROTOCOL_PATH),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "cohort_manifest": str(cohort_manifest),
        "cohort_manifest_sha256": file_sha256(cohort_manifest),
        "cohort_path_metadata_sha256": (
            file_sha256(cohort_manifest.with_suffix(".paths.json"))
            if cohort_manifest.with_suffix(".paths.json").exists() else None
        ),
        "phase": args.phase,
        "formal": formal,
        "device": args.device,
        "variants": selected_variants,
        "registered_run_count": len(cohort),
        "computed_run_count": len(runs),
        "expected_accuracy_runs": EXPECTED_ACCURACY_RUNS,
        "expected_equivariance_runs": EXPECTED_EQUIVARIANCE_RUNS,
        "max_runs": args.max_runs,
        "max_checkpoints": args.max_checkpoints,
        "save_every": args.save_every,
        "legacy_gate_skipped": bool(args.skip_legacy_gate),
    }
    config_path = output_dir / f"run_config_{args.phase}.json"
    if config_path.exists() and not args.overwrite:
        with config_path.open() as handle:
            existing_config = json.load(handle)
        if existing_config != phase_config:
            raise RuntimeError(
                f"Existing phase config differs; use a separate output directory "
                f"or explicit --overwrite: {config_path}"
            )
    write_json_atomic(config_path, phase_config)

    if args.phase == "default_control" and not args.skip_legacy_gate:
        gate = run_legacy_batched_gate(
            prepared_runs[0],
            output_dir=output_dir,
            cohort_manifest=cohort_manifest,
            device=args.device,
        )
        print(
            "Legacy/batched gate PASS: "
            f"abs_diff={gate['absolute_difference']:.3e}"
        )

    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    manifest_path = output_dir / f"run_manifest_{args.phase}.csv"
    failures_path = output_dir / f"failures_{args.phase}.json"
    write_json_atomic(failures_path, failures)
    total_jobs = len(prepared_runs) * len(selected_variants)
    job_index = 0
    for prepared in prepared_runs:
        for variant in selected_variants:
            job_index += 1
            label = (
                f"[{job_index}/{total_jobs}] {variant}/"
                f"{prepared['act']}/{prepared['task']}/s{prepared['seed']}"
            )
            print(f"{label} START", flush=True)
            try:
                row = run_one_variant(
                    prepared,
                    variant=variant,
                    output_dir=output_dir,
                    device=args.device,
                    save_every=args.save_every,
                    max_checkpoints=args.max_checkpoints,
                    formal=formal,
                    overwrite=args.overwrite,
                )
                manifest_rows.append(row)
                print(
                    f"{label} {row['status'].upper()} "
                    f"epochs={row['n_epochs']} "
                    f"elapsed={row.get('elapsed_seconds', 0.0):.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "variant": variant,
                    "act": prepared["act"],
                    "task": prepared["task"],
                    "seed": int(prepared["seed"]),
                    "exp_dir": str(prepared["exp_dir"]),
                    "status": "failed",
                    "error": repr(exc),
                }
                failures.append(failure)
                manifest_rows.append(failure)
                print(f"{label} FAILED: {exc!r}", flush=True)
            write_csv_atomic(manifest_path, manifest_rows)
            write_json_atomic(failures_path, failures)

    if failures:
        raise SystemExit(
            f"{len(failures)} job(s) failed; see {failures_path}"
        )
    print(
        f"All {len(manifest_rows)} {args.phase} jobs complete. Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
