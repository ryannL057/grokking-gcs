#!/usr/bin/env python3
"""
Recompute simple aggregation baselines on top of existing saved final geometries.

This keeps the original GCS trajectories intact and writes a separate payload with
alternative aggregation summaries such as mean edge coherence and participation rank.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis.gmn_tsa import compute_path_gmn_tsa
from gcs.legacy.recompute_gcs_existing_geometry import (
    PATH_CONFIGS,
    cast_geometry,
    checkpoint_epoch,
    ensure_geometries,
    evenly_subsample_epochs,
    infer_num_layers,
    load_checkpoint,
    load_existing_result,
    paper_runs,
    resolve_path,
    target_layers_for_recompute,
)


METRIC_KEYS = (
    "gcs_score",
    "gmn_tsa_score",
    "mean_edge_coherence",
    "participation_rank",
)


def output_path_for(exp_dir: Path, suffix: str) -> Path:
    return exp_dir / f"gcs_aggregation_baselines{suffix}.pkl"


def compute_path_details(
    model: Any,
    geometry: dict[str, Any],
    target_layer: int,
    device: str,
    token_probe: str,
) -> dict[str, dict[str, float]]:
    dtype = next(model.parameters()).dtype
    geo = cast_geometry(geometry, device=device, dtype=dtype)
    out: dict[str, dict[str, float]] = {}
    for path_key, (start_hook, end_hook) in PATH_CONFIGS.items():
        details = compute_path_gmn_tsa(
            model,
            geo,
            start_hook,
            end_hook,
            target_layer=target_layer,
            device=device,
            token_probe=token_probe,
            return_details=True,
        )
        out[path_key] = {metric: float(details[metric]) for metric in METRIC_KEYS}
    return out


def init_payload(exp_dir: Path, target_layer: int, token_probe: str, selected_epoch_count: int) -> dict[str, Any]:
    return {
        "exp_dir": str(exp_dir),
        "target_layer": int(target_layer),
        "token_probe": token_probe,
        "selected_epoch_count": int(selected_epoch_count),
        "epochs": [],
        "metric_scores": {
            metric: {path_key: [] for path_key in PATH_CONFIGS}
            for metric in METRIC_KEYS
        },
    }


def append_epoch(payload: dict[str, Any], epoch: int, path_details: dict[str, dict[str, float]]) -> None:
    payload["epochs"].append(int(epoch))
    for metric in METRIC_KEYS:
        for path_key in PATH_CONFIGS:
            payload["metric_scores"][metric][path_key].append(float(path_details[path_key][metric]))


def run_experiment(
    exp_dir: Path,
    *,
    device: str,
    suffix: str,
    token_probe: str,
    geometry_N: int,
    geometry_k: int,
    geometry_d: int,
    max_checkpoints: int | None,
    force_rebuild_geometry: bool,
    skip_existing: bool,
) -> tuple[Path, bool]:
    _, _, epochs, _ = load_existing_result(exp_dir)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_map = {checkpoint_epoch(p): p for p in checkpoint_dir.glob("*.pth")}
    selected_epochs = [epoch for epoch in epochs if epoch in checkpoint_map]
    selected_epochs = evenly_subsample_epochs(selected_epochs, max_checkpoints)
    if not selected_epochs:
        raise RuntimeError(f"No matching checkpoints found for {exp_dir}")

    num_layers = infer_num_layers(exp_dir, device=device)
    target_layers = target_layers_for_recompute(num_layers)
    compat_layer = target_layers[-1]
    out_path = output_path_for(exp_dir, suffix)
    if skip_existing and out_path.exists():
        return out_path, True

    geometries = ensure_geometries(
        exp_dir,
        target_layers,
        num_layers,
        device,
        N=geometry_N,
        k=geometry_k,
        d=geometry_d,
        force_rebuild=force_rebuild_geometry,
    )

    payload = init_payload(exp_dir, compat_layer, token_probe, len(selected_epochs))
    payload["geometry_params"] = {
        "N": geometry_N,
        "k": geometry_k,
        "d": geometry_d,
        "force_rebuild_geometry": force_rebuild_geometry,
    }

    for epoch in tqdm(selected_epochs, desc=exp_dir.name, leave=False):
        model, _ = load_checkpoint(checkpoint_map[epoch], device=device)
        path_details = compute_path_details(
            model,
            geometries[compat_layer],
            compat_layer,
            device,
            token_probe,
        )
        append_epoch(payload, epoch, path_details)

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return out_path, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute simple GCS aggregation baselines")
    parser.add_argument("--exp-dir", action="append", default=[], help="Experiment directory to analyze")
    parser.add_argument("--paper-main", action="store_true", help="Run the registered paper cohorts")
    parser.add_argument("--depths", type=str, default="1", help="Comma-separated depths for --paper-main")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-suffix", type=str, default="_final")
    parser.add_argument(
        "--token-probe",
        choices=("final", "operands", "all"),
        default="final",
        help="Which token positions receive the JVP tangent",
    )
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--force-rebuild-geometry", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dirs = [resolve_path(Path(p)) for p in args.exp_dir]
    if args.paper_main:
        depths = [int(x) for x in args.depths.split(",") if x.strip()]
        exp_dirs.extend(paper_runs(depths))

    if not exp_dirs:
        raise SystemExit("No experiments selected. Use --exp-dir or --paper-main.")

    print(f"Recomputing aggregation baselines for {len(exp_dirs)} experiments on {args.device}")
    for exp_dir in exp_dirs:
        out_path, skipped = run_experiment(
            exp_dir,
            device=args.device,
            suffix=args.output_suffix,
            token_probe=args.token_probe,
            geometry_N=args.N,
            geometry_k=args.k,
            geometry_d=args.d,
            max_checkpoints=args.max_checkpoints,
            force_rebuild_geometry=args.force_rebuild_geometry,
            skip_existing=args.skip_existing,
        )
        tag = "skip" if skipped else "ok"
        print(f"[{tag}] {exp_dir} -> {out_path.name}")


if __name__ == "__main__":
    main()
