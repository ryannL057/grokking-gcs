#!/usr/bin/env python3
"""
Recompute GCS trajectories using existing saved geometries.

This isolates analysis-code changes from geometry resampling: the saved
`final_geometry*.pkl` files are reused directly, and only the checkpoint-wise
GCS computation is rerun.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import copy
import io
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis import build_complete_geometric_structure
from gcs.analysis.gmn_tsa import compute_path_gmn_tsa
from gcs.core.data import create_data
from gcs.legacy.diagnostics_common import RUNS_1L, RUNS_2L, RUNS_3L, resolve_path
from gcs.legacy.reanalyze_gcs import load_checkpoint


PATH_CONFIGS = {
    "overall_path": ("hook_resid_pre", "hook_resid_post"),
    "attention_path": ("hook_resid_pre", "hook_attn_out"),
    "mlp_path": ("hook_resid_mid", "hook_mlp_out"),
}


def load_pickle(path: Path) -> Any:
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return CPUUnpickler(f).load()


def save_pickle(path: Path, payload: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def infer_num_layers(exp_dir: Path, device: str) -> int:
    ckpts = sorted((exp_dir / "checkpoints").glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {exp_dir / 'checkpoints'}")
    model, _ = load_checkpoint(ckpts[-1], device=device)
    return int(getattr(model, "num_layers", 1))


def load_existing_result(exp_dir: Path) -> tuple[str, Path, list[int], dict[str, Any]]:
    gcs_path = exp_dir / "gcs_results.pkl"
    if gcs_path.exists():
        payload = load_pickle(gcs_path)
        return "gcs_results", gcs_path, list(payload["epochs"]), payload

    final_geo_path = exp_dir / "training_metrics_final_geo.pkl"
    if final_geo_path.exists():
        payload = load_pickle(final_geo_path)
        return "training_metrics_final_geo", final_geo_path, list(payload["gcs_steps_final_geo"]), payload

    raise FileNotFoundError(f"No existing GCS result found in {exp_dir}")


def existing_path_scores(payload_type: str, payload: dict[str, Any]) -> dict[str, list[float | None]]:
    if payload_type == "gcs_results":
        return payload["path_gcs_scores"]
    return payload["path_gcs_scores_final_geo"]


def target_layers_for_recompute(num_layers: int) -> list[int]:
    """
    Match the paper-analysis contract:
    - 1L: analyze layer 0
    - 2L/3L: analyze last layer only
    """
    if num_layers <= 1:
        return [0]
    return [num_layers - 1]


def output_score_keys(num_layers: int, target_layers: list[int]) -> list[str]:
    keys: list[str] = []
    if num_layers > 1:
        for layer_idx in target_layers:
            keys.extend(
                [
                    f"layer{layer_idx}_overall_path",
                    f"layer{layer_idx}_attention_path",
                    f"layer{layer_idx}_mlp_path",
                ]
            )
    keys.extend(["overall_path", "attention_path", "mlp_path"])
    return keys


def cast_geometry(geometry: dict[str, Any], device: str, dtype: torch.dtype) -> dict[str, Any]:
    geo = copy.deepcopy(geometry)
    geo["X_sample"] = geo["X_sample"].to(device=device, dtype=dtype)
    geo["X_prompts"] = geo["X_prompts"].to(device=device)
    geo["X_embeddings"] = [x.to(device=device, dtype=dtype) for x in geo["X_embeddings"]]
    geo["all_input_tangent_bases"] = [x.to(device=device, dtype=dtype) for x in geo["all_input_tangent_bases"]]
    return geo


def infer_config(exp_dir: Path) -> dict[str, Any]:
    metrics_path = exp_dir / "training_metrics.pkl"
    metrics = load_pickle(metrics_path) if metrics_path.exists() else {}
    cfg = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
    if not cfg:
        cfg_path = exp_dir / "config.pkl"
        if cfg_path.exists():
            cfg = load_pickle(cfg_path)
    if not cfg:
        cfg = {
            "p": 113,
            "fraction": 0.3,
            "seed": 777,
            "task": metrics.get("task_name", "add"),
        }
    return cfg


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else 0


def ensure_geometries(
    exp_dir: Path,
    target_layers: list[int],
    num_layers: int,
    device: str,
    *,
    N: int,
    k: int,
    d: int,
    force_rebuild: bool,
) -> dict[int, dict[str, Any]]:
    geometries: dict[int, dict[str, Any]] = {}
    missing_layers: list[int] = []

    for layer_idx in target_layers:
        if force_rebuild:
            missing_layers.append(layer_idx)
            continue
        candidates = [exp_dir / f"final_geometry_layer{layer_idx}.pkl"]
        if layer_idx == num_layers - 1:
            candidates.append(exp_dir / "final_geometry.pkl")
        for path in candidates:
            if path.exists():
                geometries[layer_idx] = load_pickle(path)
                break
        if layer_idx not in geometries:
            missing_layers.append(layer_idx)

    if not missing_layers:
        return geometries

    cfg = infer_config(exp_dir)
    ckpts = sorted((exp_dir / "checkpoints").glob("*.pth"), key=checkpoint_epoch)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {exp_dir / 'checkpoints'}")
    model, _ = load_checkpoint(ckpts[-1], device=device)
    train_data, _ = create_data(
        p=cfg.get("p", 113),
        fraction=cfg.get("fraction", 0.3),
        seed=cfg.get("seed", 777),
        device=device,
        task=cfg.get("task", "add"),
    )

    for layer_idx in missing_layers:
        geometry = build_complete_geometric_structure(
            model=model,
            train_data=train_data,
            N=N,
            k=k,
            d=d,
            device=device,
            layer_idx=layer_idx,
            hook_point="hook_resid_post",
        )
        if not force_rebuild:
            save_pickle(exp_dir / f"final_geometry_layer{layer_idx}.pkl", geometry)
            if layer_idx == num_layers - 1:
                save_pickle(exp_dir / "final_geometry.pkl", geometry)
        geometries[layer_idx] = geometry

    return geometries


def evenly_subsample_epochs(epochs: list[int], max_items: int | None) -> list[int]:
    if max_items is None or len(epochs) <= max_items:
        return epochs
    if max_items <= 1:
        return [epochs[-1]]

    import numpy as np

    indices = np.linspace(0, len(epochs) - 1, num=max_items, dtype=int)
    return [epochs[int(idx)] for idx in indices]


def compute_three_paths(
    model: Any,
    geometry: dict[str, Any],
    target_layer: int,
    device: str,
    token_probe: str,
    alignment_method: str,
) -> dict[str, float]:
    dtype = next(model.parameters()).dtype
    geo = cast_geometry(geometry, device=device, dtype=dtype)
    out: dict[str, float] = {}
    for key, (start_hook, end_hook) in PATH_CONFIGS.items():
        out[key] = float(
            compute_path_gmn_tsa(
                model,
                geo,
                start_hook,
                end_hook,
                target_layer=target_layer,
                device=device,
                token_probe=token_probe,
                alignment_method=alignment_method,
            )
        )
    return out


def init_result_payload(
    payload_type: str,
    old_payload: dict[str, Any],
    output_keys: list[str],
    token_probe: str,
    alignment_method: str,
) -> dict[str, Any]:
    if payload_type == "gcs_results":
        payload = {
            "epochs": [],
            "path_gcs_scores": {key: [] for key in output_keys},
        }
        for key in ("geometry_type", "analysis_params"):
            if key in old_payload:
                payload[key] = old_payload[key]
        analysis_params = dict(payload.get("analysis_params", {}))
        analysis_params["token_probe"] = token_probe
        analysis_params["alignment_method"] = alignment_method
        payload["analysis_params"] = analysis_params
        return payload

    payload = dict(old_payload)
    payload["gcs_steps_final_geo"] = []
    payload["path_gcs_scores_final_geo"] = {key: [] for key in output_keys}
    payload["token_probe"] = token_probe
    payload["alignment_method"] = alignment_method
    return payload


def append_scores(
    payload_type: str,
    result: dict[str, Any],
    epoch: int,
    layer_results: dict[int, dict[str, float]],
    target_layers: list[int],
    compat_layer: int,
) -> None:
    if payload_type == "gcs_results":
        result["epochs"].append(epoch)
        dest = result["path_gcs_scores"]
    else:
        result["gcs_steps_final_geo"].append(epoch)
        dest = result["path_gcs_scores_final_geo"]

    for layer_idx in target_layers:
        layer_scores = layer_results[layer_idx]
        prefixed = {
            f"layer{layer_idx}_overall_path": layer_scores["overall_path"],
            f"layer{layer_idx}_attention_path": layer_scores["attention_path"],
            f"layer{layer_idx}_mlp_path": layer_scores["mlp_path"],
        }
        for key, value in prefixed.items():
            if key in dest:
                dest[key].append(value)

    if {"overall_path", "attention_path", "mlp_path"}.issubset(dest):
        for key, value in layer_results[compat_layer].items():
            dest[key].append(value)
    elif compat_layer == 0 and len(target_layers) == 1 and {"overall_path", "attention_path", "mlp_path"}.issubset(dest):
        for key, value in layer_results[0].items():
            dest[key].append(value)


def output_path_for(exp_dir: Path, payload_type: str, suffix: str) -> Path:
    if payload_type == "gcs_results":
        return exp_dir / f"gcs_results{suffix}.pkl"
    return exp_dir / f"training_metrics_final_geo{suffix}.pkl"


def existing_output_keys(path: Path, payload_type: str) -> set[str] | None:
    if not path.exists():
        return None
    payload = load_pickle(path)
    if payload_type == "gcs_results":
        path_scores = payload.get("path_gcs_scores")
    else:
        path_scores = payload.get("path_gcs_scores_final_geo")
    if not isinstance(path_scores, dict):
        return None
    return set(path_scores.keys())


def run_experiment(
    exp_dir: Path,
    device: str,
    suffix: str,
    *,
    token_probe: str,
    geometry_N: int,
    geometry_k: int,
    geometry_d: int,
    alignment_method: str,
    max_checkpoints: int | None,
    force_rebuild_geometry: bool,
    skip_existing_compatible: bool = False,
) -> tuple[Path, bool]:
    payload_type, _, epochs, old_payload = load_existing_result(exp_dir)
    path_scores = existing_path_scores(payload_type, old_payload)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_map = {checkpoint_epoch(p): p for p in checkpoint_dir.glob("*.pth")}
    selected_epochs = [epoch for epoch in epochs if epoch in checkpoint_map]
    selected_epochs = evenly_subsample_epochs(selected_epochs, max_checkpoints)
    if not selected_epochs:
        raise RuntimeError(f"No matching checkpoints found for saved GCS epochs in {exp_dir}")

    num_layers = infer_num_layers(exp_dir, device=device)
    target_layers = target_layers_for_recompute(num_layers)
    compat_layer = target_layers[-1]
    output_keys = output_score_keys(num_layers, target_layers)
    out_path = output_path_for(exp_dir, payload_type, suffix)
    if skip_existing_compatible:
        existing_keys = existing_output_keys(out_path, payload_type)
        if existing_keys is not None and existing_keys == set(output_keys):
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
    result = init_result_payload(payload_type, old_payload, output_keys, token_probe, alignment_method)
    result["selected_epoch_count"] = len(selected_epochs)
    result["geometry_params"] = {
        "N": geometry_N,
        "k": geometry_k,
        "d": geometry_d,
        "force_rebuild_geometry": force_rebuild_geometry,
        "alignment_method": alignment_method,
    }

    for epoch in tqdm(selected_epochs, desc=exp_dir.name, leave=False):
        model, _ = load_checkpoint(checkpoint_map[epoch], device=device)
        layer_results = {
            layer_idx: compute_three_paths(
                model,
                geometries[layer_idx],
                layer_idx,
                device,
                token_probe,
                alignment_method,
            )
            for layer_idx in target_layers
        }
        append_scores(payload_type, result, epoch, layer_results, target_layers, compat_layer)

    save_pickle(out_path, result)
    return out_path, False


def paper_runs(depths: list[int]) -> list[Path]:
    runs: list[tuple[str, str, int, str]] = []
    if 1 in depths:
        runs.extend(RUNS_1L)
    if 2 in depths:
        runs.extend(run for task_runs in RUNS_2L.values() for run in task_runs)
    if 3 in depths:
        runs.extend(run for task_runs in RUNS_3L.values() for run in task_runs)

    seen = set()
    out = []
    for _, _, _, base in runs:
        path = resolve_path(base)
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute GCS using existing saved geometries")
    parser.add_argument("--exp-dir", action="append", default=[], help="Experiment directory to reanalyze")
    parser.add_argument("--paper-main", action="store_true", help="Run the paper's registered 1L/2L/3L cohorts")
    parser.add_argument("--depths", type=str, default="1,2,3", help="Comma-separated depths for --paper-main")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-suffix", type=str, default="_union")
    parser.add_argument(
        "--token-probe",
        choices=("final", "operands", "all"),
        default="final",
        help="Which token positions receive the JVP tangent",
    )
    parser.add_argument("--N", type=int, default=200, help="Geometry sample count")
    parser.add_argument("--k", type=int, default=15, help="k-NN size for geometry")
    parser.add_argument("--d", type=int, default=8, help="Intrinsic dimension for geometry")
    parser.add_argument(
        "--alignment-method",
        choices=("cosine", "procrustes", "rank_permuted"),
        default="cosine",
        help="Basis-alignment rule used inside compute_path_gmn_tsa",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Evenly subsample the saved checkpoint schedule",
    )
    parser.add_argument(
        "--force-rebuild-geometry",
        action="store_true",
        help="Ignore saved final_geometry*.pkl and rebuild geometry with --N/--k/--d",
    )
    parser.add_argument("--skip-existing-compatible", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dirs = [Path(p) for p in args.exp_dir]
    if args.paper_main:
        depths = [int(x) for x in args.depths.split(",") if x.strip()]
        exp_dirs.extend(paper_runs(depths))

    if not exp_dirs:
        raise SystemExit("No experiments selected. Use --exp-dir or --paper-main.")

    suffix = args.output_suffix
    if args.token_probe != "final" and suffix == "_union":
        suffix = f"{suffix}_{args.token_probe}"

    print(
        f"Recomputing {len(exp_dirs)} experiments on {args.device} "
        f"(token_probe={args.token_probe}, alignment={args.alignment_method})"
    )
    for exp_dir in exp_dirs:
        out_path, skipped = run_experiment(
            exp_dir,
            device=args.device,
            suffix=suffix,
            token_probe=args.token_probe,
            geometry_N=args.N,
            geometry_k=args.k,
            geometry_d=args.d,
            alignment_method=args.alignment_method,
            max_checkpoints=args.max_checkpoints,
            force_rebuild_geometry=args.force_rebuild_geometry,
            skip_existing_compatible=args.skip_existing_compatible,
        )
        tag = "skip" if skipped else "ok"
        print(f"[{tag}] {exp_dir} -> {out_path.name}")


if __name__ == "__main__":
    main()
