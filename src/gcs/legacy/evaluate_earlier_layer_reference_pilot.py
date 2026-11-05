#!/usr/bin/env python3
"""Evaluate the frozen preceding-block-reference pilot trajectories."""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = _gcs_data_root()
pass  # Imports use the installed package.

from gcs.legacy.diagnostics_common import (  # noqa: E402
    detect_attention_peak_then_dip,
    detect_grok_epoch,
    detect_mlp_u_shape,
    stable_turning_epoch,
)
from gcs.legacy.evaluate_workflow_selection_heldout import (  # noqa: E402
    align_to_epochs,
    finite_array,
    score_window,
    select_first_prominent_post_valley,
    window_bounds,
)
from gcs.legacy.run_earlier_layer_reference_pilot import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROTOCOL,
    load_pickle_cpu,
    read_manifest,
    resolve_path,
    run_slug,
    sha256_file,
)


WINDOW_LEN = 11
COMMUTATIVE_TASKS = {"add", "mul"}
NONCOMMUTATIVE_TASKS = {"sub", "div"}
PATHS = ("overall_path", "attention_path", "mlp_path")


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def paired_spearman(canonical: np.ndarray, preceding: np.ndarray) -> float:
    if canonical.size != preceding.size or canonical.size < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    result = spearmanr(canonical, preceding)
    return float(result.statistic)


def canonical_range_nrmse(
    canonical: np.ndarray,
    preceding: np.ndarray,
) -> tuple[float, float]:
    """Return RMSE and RMSE/(canonical max-min), the frozen normalization."""
    if canonical.size != preceding.size or canonical.size == 0:
        raise ValueError("NRMSE inputs must be non-empty and equally sized")
    rmse = float(np.sqrt(np.mean(np.square(preceding - canonical))))
    scale = float(np.max(canonical) - np.min(canonical))
    return rmse, float(rmse / scale) if scale > 1e-12 else float("nan")


def canonical_path_scores(
    payload: dict[str, Any],
    target_layer: int,
) -> dict[str, np.ndarray]:
    scores = payload["path_gcs_scores"]
    output: dict[str, np.ndarray] = {}
    for path_name in PATHS:
        key = f"layer{int(target_layer)}_{path_name}"
        if key not in scores:
            raise KeyError(f"Missing canonical score key: {key}")
        output[path_name] = finite_array(scores[key])
    return output


def load_depth_task_targets(
    paths_by_depth: dict[int, Path],
) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], dict[str, str]]:
    """Load existing depth-specific task-symmetry trajectories.

    The stored readout is cosine symmetry.  We convert it to distance
    ``1-cosine_sym`` so the unchanged workflow target orientations apply:
    distance drop for commutative tasks and distance rise for noncommutative
    tasks.
    """
    index: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for depth, path in sorted(paths_by_depth.items()):
        if not path.exists():
            continue
        payload = load_pickle_cpu(path)
        hashes[str(path)] = sha256_file(path)
        if int(payload.get("depth", depth)) != int(depth):
            raise ValueError(f"Depth mismatch in task trajectory input: {path}")
        for run in payload.get("run_data", []):
            key = (
                str(run["act"]),
                str(run["task"]),
                int(run["seed"]),
                int(depth),
            )
            if key in index:
                raise ValueError(f"Duplicate task trajectory: {key}")
            trajectory = run.get("trajectory", [])
            if trajectory:
                index[key] = run
    return index, hashes


def targets_for_run(
    row: dict[str, Any],
    metrics: dict[str, Any],
    epochs: np.ndarray,
    task_index: dict[tuple[str, str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    accuracy = align_to_epochs(
        metrics.get("log_steps", []),
        metrics.get("test_accuracies", []),
        epochs,
    )
    if accuracy is not None:
        targets.append(
            {
                "target": "test_accuracy_rise",
                "mode": "rise",
                "values": accuracy,
                "source": "training_metrics.test_accuracies",
            }
        )

    key = (
        str(row["activation"]),
        str(row["task"]),
        int(row["seed"]),
        int(row["depth"]),
    )
    task_run = task_index.get(key)
    if task_run is not None:
        trajectory = task_run["trajectory"]
        source_epochs = [int(item["epoch"]) for item in trajectory]
        distance = [1.0 - float(item["cosine_sym"]) for item in trajectory]
        aligned = align_to_epochs(source_epochs, distance, epochs)
        if aligned is not None:
            if row["task"] in COMMUTATIVE_TASKS:
                target_name = "attention_equivariance_drop"
                mode = "drop"
            elif row["task"] in NONCOMMUTATIVE_TASKS:
                target_name = "attention_order_sensitivity_rise"
                mode = "rise"
            else:
                raise ValueError(f"Unknown task orientation: {row['task']}")
            targets.append(
                {
                    "target": target_name,
                    "mode": mode,
                    "values": aligned,
                    "source": (
                        f"commutativity_1C_results_{int(row['depth'])}L:"
                        "1-cosine_sym"
                    ),
                }
            )
    return targets


def trajectory_file(
    experiment_dir: Path,
    row: dict[str, Any],
    *,
    allow_smoke: bool,
) -> Path:
    formal = experiment_dir / "trajectories" / f"{run_slug(row)}.pkl"
    if formal.exists() or not allow_smoke:
        return formal
    smoke = sorted(
        (experiment_dir / "trajectories").glob(f"{run_slug(row)}_smoke*.pkl")
    )
    if len(smoke) != 1:
        raise FileNotFoundError(
            f"Expected exactly one smoke trajectory for {run_slug(row)}, got {smoke}"
        )
    return smoke[0]


def _run_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_slug": run_slug(row),
        "activation": row["activation"],
        "task": row["task"],
        "seed": int(row["seed"]),
        "depth": int(row["depth"]),
        "target_layer": int(row["target_layer"]),
        "reference_layer": int(row["reference_layer"]),
        "exp_dir": str(resolve_path(row["exp_dir"])),
    }


def evaluate_run(
    row: dict[str, Any],
    *,
    experiment_dir: Path,
    task_index: dict[tuple[str, str, int, int], dict[str, Any]],
    allow_smoke: bool,
    manifest_sha256: str,
    protocol_sha256: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, str],
]:
    identity = _run_identity(row)
    exp_dir = resolve_path(row["exp_dir"])
    trajectory_path = trajectory_file(
        experiment_dir,
        row,
        allow_smoke=allow_smoke,
    )
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing trajectory: {trajectory_path}")
    trajectory = load_pickle_cpu(trajectory_path)
    if not trajectory.get("complete", False):
        raise RuntimeError(f"Incomplete trajectory: {trajectory_path}")
    schedule_type = trajectory.get("analysis_params", {}).get("checkpoint_schedule")
    if schedule_type != "exact_canonical" and not allow_smoke:
        raise RuntimeError(
            f"Non-canonical schedule requires --allow-smoke: {trajectory_path}"
        )
    if trajectory.get("run") != row:
        raise RuntimeError(f"Trajectory manifest row mismatch: {trajectory_path}")
    if trajectory.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError(f"Trajectory manifest hash mismatch: {trajectory_path}")
    if trajectory.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError(f"Trajectory protocol hash mismatch: {trajectory_path}")
    params = trajectory.get("analysis_params", {})
    if params.get("model_dtype") != "torch.float64":
        raise RuntimeError(f"Trajectory is not frozen float64: {trajectory_path}")
    legacy_contract = trajectory.get("legacy_validation_contract", {})
    if (
        legacy_contract.get("status") != "PASS"
        or legacy_contract.get("model_dtype") != "torch.float64"
        or float(legacy_contract.get("tolerance", float("inf"))) != 1e-10
        or legacy_contract.get("manifest_sha256") != manifest_sha256
        or legacy_contract.get("protocol_sha256") != protocol_sha256
    ):
        raise RuntimeError(f"Invalid batched/legacy contract: {trajectory_path}")

    canonical_path = exp_dir / "gcs_results_union.pkl"
    metrics_path = exp_dir / "training_metrics.pkl"
    anchor_path = exp_dir / str(row["anchor_geometry"])
    checkpoint_paths = sorted((exp_dir / "checkpoints").glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints in {exp_dir / 'checkpoints'}")
    final_checkpoint = max(
        checkpoint_paths,
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    observed_source_hashes = {
        "anchor_geometry_sha256": sha256_file(anchor_path),
        "canonical_gcs_sha256": sha256_file(canonical_path),
        "training_metrics_sha256": sha256_file(metrics_path),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
    }
    if trajectory.get("source_hashes") != observed_source_hashes:
        raise RuntimeError(f"Frozen source hash mismatch: {trajectory_path}")
    reference_geometry_path = Path(trajectory["reference_geometry_path"])
    if not reference_geometry_path.exists():
        raise FileNotFoundError(
            f"Missing paired reference geometry: {reference_geometry_path}"
        )
    reference_payload = load_pickle_cpu(reference_geometry_path)
    if (
        reference_payload.get("geometry_semantic_sha256")
        != trajectory.get("reference_geometry_semantic_sha256")
        or reference_payload.get("metadata", {}).get("model_dtype")
        != "torch.float64"
    ):
        raise RuntimeError(
            f"Paired reference geometry contract mismatch: {reference_geometry_path}"
        )
    canonical_payload = load_pickle_cpu(canonical_path)
    metrics = load_pickle_cpu(metrics_path)
    canonical_epochs = np.asarray(canonical_payload["epochs"], dtype=int)
    preceding_epochs = np.asarray(trajectory["epochs"], dtype=int)
    epoch_to_canonical = {
        int(epoch): index for index, epoch in enumerate(canonical_epochs)
    }
    if any(int(epoch) not in epoch_to_canonical for epoch in preceding_epochs):
        raise RuntimeError(f"Trajectory has non-canonical epochs: {trajectory_path}")
    canonical_indices = np.asarray(
        [epoch_to_canonical[int(epoch)] for epoch in preceding_epochs],
        dtype=int,
    )
    epochs = preceding_epochs
    if epochs.size < WINDOW_LEN:
        raise ValueError(f"Trajectory shorter than W={WINDOW_LEN}: {trajectory_path}")

    canonical_all = canonical_path_scores(
        canonical_payload,
        int(row["target_layer"]),
    )
    canonical = {
        name: values[canonical_indices] for name, values in canonical_all.items()
    }
    preceding = {
        name: finite_array(trajectory["path_scores"][name]) for name in PATHS
    }
    for name in PATHS:
        if preceding[name].size != epochs.size:
            raise RuntimeError(f"{trajectory_path}: {name} length mismatch")

    path_rows: list[dict[str, Any]] = []
    for path_name in PATHS:
        rmse, nrmse = canonical_range_nrmse(
            canonical[path_name],
            preceding[path_name],
        )
        path_rows.append(
            {
                **identity,
                "path": path_name,
                "n_common_epochs": int(epochs.size),
                "spearman_rho": paired_spearman(
                    canonical[path_name],
                    preceding[path_name],
                ),
                "rmse": rmse,
                "nrmse": nrmse,
                "nrmse_denominator": "canonical_max_minus_min",
                "canonical_min": float(np.min(canonical[path_name])),
                "canonical_max": float(np.max(canonical[path_name])),
                "canonical_range": float(np.ptp(canonical[path_name])),
                "preceding_min": float(np.min(preceding[path_name])),
                "preceding_max": float(np.max(preceding[path_name])),
                "preceding_range": float(np.ptp(preceding[path_name])),
            }
        )

    canonical_center = select_first_prominent_post_valley(
        canonical["attention_path"],
        WINDOW_LEN,
    )
    preceding_center = select_first_prominent_post_valley(
        preceding["attention_path"],
        WINDOW_LEN,
    )
    canonical_start, canonical_end = window_bounds(
        canonical_center,
        len(epochs),
        WINDOW_LEN,
    )
    preceding_start, preceding_end = window_bounds(
        preceding_center,
        len(epochs),
        WINDOW_LEN,
    )
    selector_row = {
        **identity,
        "window_len": WINDOW_LEN,
        "smoothing_window": 3,
        "selector": "first_prominent_post_valley",
        "canonical_center_idx": int(canonical_center),
        "canonical_center_epoch": int(epochs[canonical_center]),
        "canonical_window_start_epoch": int(epochs[canonical_start]),
        "canonical_window_end_epoch": int(epochs[canonical_end]),
        "preceding_center_idx": int(preceding_center),
        "preceding_center_epoch": int(epochs[preceding_center]),
        "preceding_window_start_epoch": int(epochs[preceding_start]),
        "preceding_window_end_epoch": int(epochs[preceding_end]),
        "center_shift_checkpoints": int(preceding_center - canonical_center),
        "center_shift_epochs": int(
            epochs[preceding_center] - epochs[canonical_center]
        ),
    }

    target_rows: list[dict[str, Any]] = []
    for target in targets_for_run(row, metrics, epochs, task_index):
        canonical_stats = score_window(
            np.asarray(target["values"], dtype=float),
            WINDOW_LEN,
            canonical_start,
            str(target["mode"]),
        )
        preceding_stats = score_window(
            np.asarray(target["values"], dtype=float),
            WINDOW_LEN,
            preceding_start,
            str(target["mode"]),
        )
        shared = {
            **identity,
            "target": target["target"],
            "target_mode": target["mode"],
            "target_source": target["source"],
            "window_len": WINDOW_LEN,
            "center_shift_checkpoints": selector_row["center_shift_checkpoints"],
            "center_shift_epochs": selector_row["center_shift_epochs"],
            "preceding_minus_canonical_percentile": float(
                preceding_stats["percentile"] - canonical_stats["percentile"]
            ),
        }
        for reference, center, start, end, stats in (
            (
                "canonical_output_adjacent",
                canonical_center,
                canonical_start,
                canonical_end,
                canonical_stats,
            ),
            (
                "preceding_block",
                preceding_center,
                preceding_start,
                preceding_end,
                preceding_stats,
            ),
        ):
            target_rows.append(
                {
                    **shared,
                    "reference": reference,
                    "center_idx": int(center),
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
            )

    t_grok = detect_grok_epoch(metrics)
    if t_grok is None:
        raise RuntimeError(f"Frozen successful run has no solved t_grok: {exp_dir}")
    canonical_attention = detect_attention_peak_then_dip(
        canonical["attention_path"],
        epochs,
        int(t_grok),
    )
    preceding_attention = detect_attention_peak_then_dip(
        preceding["attention_path"],
        epochs,
        int(t_grok),
    )
    canonical_mlp = detect_mlp_u_shape(
        canonical["mlp_path"],
        epochs,
        int(t_grok),
    )
    preceding_mlp = detect_mlp_u_shape(
        preceding["mlp_path"],
        epochs,
        int(t_grok),
    )
    canonical_turning = stable_turning_epoch(
        canonical["overall_path"],
        epochs,
        int(t_grok),
    )
    preceding_turning = stable_turning_epoch(
        preceding["overall_path"],
        epochs,
        int(t_grok),
    )
    canonical_lead = int(t_grok) - int(canonical_turning)
    preceding_lead = int(t_grok) - int(preceding_turning)
    diagnostic_row = {
        **identity,
        "t_grok": int(t_grok),
        "attention_canonical": bool(canonical_attention[0]),
        "attention_preceding": bool(preceding_attention[0]),
        "attention_agreement": bool(
            bool(canonical_attention[0]) == bool(preceding_attention[0])
        ),
        "attention_canonical_rise": float(canonical_attention[1]),
        "attention_preceding_rise": float(preceding_attention[1]),
        "attention_canonical_dip": float(canonical_attention[2]),
        "attention_preceding_dip": float(preceding_attention[2]),
        "attention_canonical_peak_epoch": canonical_attention[3],
        "attention_preceding_peak_epoch": preceding_attention[3],
        "mlp_canonical": bool(canonical_mlp[0]),
        "mlp_preceding": bool(preceding_mlp[0]),
        "mlp_agreement": bool(bool(canonical_mlp[0]) == bool(preceding_mlp[0])),
        "mlp_canonical_depth": float(canonical_mlp[1]),
        "mlp_preceding_depth": float(preceding_mlp[1]),
        "overall_canonical_turning_epoch": int(canonical_turning),
        "overall_preceding_turning_epoch": int(preceding_turning),
        "overall_canonical_lead": int(canonical_lead),
        "overall_preceding_lead": int(preceding_lead),
        "overall_canonical_positive_lead": bool(canonical_lead > 0),
        "overall_preceding_positive_lead": bool(preceding_lead > 0),
        "overall_positive_lead_agreement": bool(
            (canonical_lead > 0) == (preceding_lead > 0)
        ),
    }
    hashes = {
        str(trajectory_path): sha256_file(trajectory_path),
        str(canonical_path): sha256_file(canonical_path),
        str(metrics_path): sha256_file(metrics_path),
        str(anchor_path): sha256_file(anchor_path),
        str(final_checkpoint): sha256_file(final_checkpoint),
        str(reference_geometry_path): sha256_file(reference_geometry_path),
    }
    return path_rows, selector_row, target_rows, diagnostic_row, hashes


def summarize_paths(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["path"])].append(row)
    output: list[dict[str, Any]] = []
    for path_name, items in sorted(grouped.items()):
        rho = np.asarray([float(item["spearman_rho"]) for item in items])
        nrmse = np.asarray([float(item["nrmse"]) for item in items])
        output.append(
            {
                "path": path_name,
                "n_runs": len(items),
                "median_spearman_rho": float(np.nanmedian(rho)),
                "mean_spearman_rho": float(np.nanmean(rho)),
                "median_nrmse": float(np.nanmedian(nrmse)),
                "mean_nrmse": float(np.nanmean(nrmse)),
                "median_canonical_range": float(
                    np.median([float(item["canonical_range"]) for item in items])
                ),
                "median_preceding_range": float(
                    np.median([float(item["preceding_range"]) for item in items])
                ),
            }
        )
    return output


def summarize_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["reference"]), str(row["target"]))].append(row)
    output: list[dict[str, Any]] = []
    for (reference, target), items in sorted(grouped.items()):
        percentiles = np.asarray([float(item["percentile"]) for item in items])
        output.append(
            {
                "reference": reference,
                "target": target,
                "n_runs": len(items),
                "median_percentile": float(np.median(percentiles)),
                "mean_percentile": float(np.mean(percentiles)),
                "top20_hit_count": int(np.sum(percentiles >= 0.8)),
                "top20_hit_rate": float(np.mean(percentiles >= 0.8)),
            }
        )
    return output


def summarize_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = (
        ("attention", "attention_canonical", "attention_preceding", "attention_agreement"),
        ("mlp", "mlp_canonical", "mlp_preceding", "mlp_agreement"),
        (
            "overall_positive_lead",
            "overall_canonical_positive_lead",
            "overall_preceding_positive_lead",
            "overall_positive_lead_agreement",
        ),
    )
    output: list[dict[str, Any]] = []
    for name, canonical_key, preceding_key, agreement_key in specs:
        output.append(
            {
                "detector": name,
                "n_runs": len(rows),
                "canonical_positive_count": int(
                    sum(bool(row[canonical_key]) for row in rows)
                ),
                "preceding_positive_count": int(
                    sum(bool(row[preceding_key]) for row in rows)
                ),
                "agreement_count": int(
                    sum(bool(row[agreement_key]) for row in rows)
                ),
                "agreement_rate": float(
                    np.mean([bool(row[agreement_key]) for row in rows])
                ),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--task-trajectories-2l",
        type=Path,
        default=Path(
            "experiments/commutativity_analysis/commutativity_1C_results_2L.pkl"
        ),
    )
    parser.add_argument(
        "--task-trajectories-3l",
        type=Path,
        default=Path(
            "experiments/commutativity_analysis/commutativity_1C_results_3L.pkl"
        ),
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    protocol_path = resolve_path(args.protocol)
    experiment_dir = resolve_path(args.experiment_dir)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else experiment_dir / "evaluation"
    )
    rows = read_manifest(manifest_path)
    task_index, task_hashes = load_depth_task_targets(
        {
            2: resolve_path(args.task_trajectories_2l),
            3: resolve_path(args.task_trajectories_3l),
        }
    )

    path_rows: list[dict[str, Any]] = []
    selector_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    input_hashes: dict[str, str] = {
        str(manifest_path): sha256_file(manifest_path),
        str(protocol_path): sha256_file(protocol_path),
        **task_hashes,
    }
    manifest_hash = input_hashes[str(manifest_path)]
    protocol_hash = input_hashes[str(protocol_path)]
    for row in rows:
        try:
            run_paths, selector, run_targets, diagnostic, hashes = evaluate_run(
                row,
                experiment_dir=experiment_dir,
                task_index=task_index,
                allow_smoke=args.allow_smoke,
                manifest_sha256=manifest_hash,
                protocol_sha256=protocol_hash,
            )
        except Exception as error:
            skipped.append(
                {
                    "run_slug": run_slug(row),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if not args.allow_partial:
                raise
            continue
        path_rows.extend(run_paths)
        selector_rows.append(selector)
        target_rows.extend(run_targets)
        diagnostic_rows.append(diagnostic)
        input_hashes.update(hashes)

    if not path_rows:
        raise RuntimeError("No complete pilot trajectories were evaluated")
    if not args.allow_partial and len(selector_rows) != len(rows):
        raise RuntimeError(
            f"Expected {len(rows)} evaluated runs, got {len(selector_rows)}"
        )

    path_summary = summarize_paths(path_rows)
    target_summary = summarize_targets(target_rows)
    diagnostic_summary = summarize_diagnostics(diagnostic_rows)
    summary = {
        "schema_version": 1,
        "status": "COMPLETE" if len(selector_rows) == len(rows) else "PARTIAL",
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "experiment_dir": str(experiment_dir),
        "evaluated_runs": len(selector_rows),
        "frozen_run_count": len(rows),
        "window_len": WINDOW_LEN,
        "selector_smoothing_window": 3,
        "nrmse_definition": "RMSE(preceding-canonical)/(canonical max-min)",
        "task_target_definition": (
            "depth-specific 1-cosine_sym; drop for add/mul, rise for sub/div"
        ),
        "path_summary": path_summary,
        "target_summary": target_summary,
        "diagnostic_summary": diagnostic_summary,
        "input_sha256": dict(sorted(input_hashes.items())),
        "skipped": skipped,
    }
    write_csv_atomic(output_dir / "per_run_path_metrics.csv", path_rows)
    write_csv_atomic(output_dir / "attention_selector.csv", selector_rows)
    if target_rows:
        write_csv_atomic(
            output_dir / "attention_target_percentiles.csv",
            target_rows,
        )
    write_csv_atomic(output_dir / "strict_diagnostics.csv", diagnostic_rows)
    write_csv_atomic(output_dir / "path_summary.csv", path_summary)
    if target_summary:
        write_csv_atomic(output_dir / "target_summary.csv", target_summary)
    write_csv_atomic(
        output_dir / "diagnostic_summary.csv",
        diagnostic_summary,
    )
    write_json_atomic(output_dir / "summary.json", summary)
    write_json_atomic(output_dir / "skipped.json", skipped)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
