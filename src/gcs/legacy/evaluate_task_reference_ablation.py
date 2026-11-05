#!/usr/bin/env python3
"""
Task-pair reference ablation for GCS.

For a small set of target runs, recompute GCS using final reference geometries
from other tasks. This probes whether the final-reference signal behaves in a
task-structured way rather than as arbitrary same-run hindsight.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

pass  # Package imports do not search the experiment-data directory.

from gcs.core.data import create_data  # noqa: E402
from gcs.legacy.cross_seed_reference import (  # noqa: E402
    build_reference_geometry,
    compute_gcs_trajectory,
    load_existing_matched_trajectory,
    trajectory_correlation,
)
from gcs.legacy.diagnostics_common import (  # noqa: E402
    RUNS_1L,
    detect_attention_peak_then_dip,
    detect_grok_epoch,
    detect_mlp_u_shape,
    load_metrics,
    resolve_path,
    stable_turning_epoch,
)


COMMUTATIVE = {"add", "mul"}
NONCOMMUTATIVE = {"sub", "div"}


def run_lookup() -> Dict[Tuple[str, str, int], str]:
    return {(act, task, int(seed)): base for act, task, seed, base in RUNS_1L}


def task_relation(target_task: str, ref_task: str) -> str:
    if target_task == ref_task:
        return "same_task"
    if (target_task in COMMUTATIVE and ref_task in COMMUTATIVE) or (
        target_task in NONCOMMUTATIVE and ref_task in NONCOMMUTATIVE
    ):
        return "same_algebra"
    return "mismatched_algebra"


def final_checkpoint(exp_dir: Path) -> Path:
    ckpts = sorted((exp_dir / "checkpoints").glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {exp_dir / 'checkpoints'}")
    return ckpts[-1]


def evaluate_trajectory_details(traj: dict, t_grok: int) -> dict:
    epochs = np.asarray(traj["epochs"], dtype=int)
    scores = traj["path_gcs_scores"]
    overall = np.asarray([np.nan if v is None else float(v) for v in scores["overall_path"]], dtype=float)
    attn = np.asarray([np.nan if v is None else float(v) for v in scores["attention_path"]], dtype=float)
    mlp = np.asarray([np.nan if v is None else float(v) for v in scores["mlp_path"]], dtype=float)
    valid = ~(np.isnan(overall) | np.isnan(attn) | np.isnan(mlp))
    epochs = epochs[valid]
    overall = overall[valid]
    attn = attn[valid]
    mlp = mlp[valid]
    if len(epochs) < 5:
        return {
            "lead": float("nan"),
            "mlp_ushape": False,
            "attn_peak_dip": False,
            "attn_peak_epoch": -1,
            "attn_peak_ratio": float("nan"),
            "attn_rise": float("nan"),
            "attn_dip": float("nan"),
        }
    t_turn = stable_turning_epoch(overall, epochs, t_grok)
    mlp_ushape, mlp_depth = detect_mlp_u_shape(mlp, epochs, t_grok)
    attn_peak_dip, attn_rise, attn_dip, attn_peak_epoch = detect_attention_peak_then_dip(
        attn,
        epochs,
        t_grok,
    )
    return {
        "lead": float(t_grok - t_turn),
        "mlp_ushape": bool(mlp_ushape),
        "mlp_depth": float(mlp_depth),
        "attn_peak_dip": bool(attn_peak_dip),
        "attn_peak_epoch": int(attn_peak_epoch) if attn_peak_epoch is not None else -1,
        "attn_peak_ratio": float(attn_peak_epoch / t_grok) if attn_peak_epoch else float("nan"),
        "attn_rise": float(attn_rise),
        "attn_dip": float(attn_dip),
    }


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate task-pair reference ablations.")
    parser.add_argument("--activation", type=str, default="SiLU")
    parser.add_argument("--seeds", type=str, default="777")
    parser.add_argument("--target-tasks", type=str, default="add,mul,sub,div")
    parser.add_argument("--ref-tasks", type=str, default="add,mul,sub,div")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint-freq", type=int, default=1000)
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="experiments/task_reference_ablation")
    parser.add_argument("--tag", type=str, default="pilot")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.set_default_dtype(torch.float64)

    lookup = run_lookup()
    seeds = [int(s) for s in parse_csv_list(args.seeds)]
    target_tasks = parse_csv_list(args.target_tasks)
    ref_tasks = parse_csv_list(args.ref_tasks)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / f"trajectories_{args.tag}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    trajectories = {}
    for seed in seeds:
        for target_task in target_tasks:
            target_key = (args.activation, target_task, seed)
            if target_key not in lookup:
                print(f"Skipping target missing from RUNS_1L: {target_key}")
                continue
            target_base = lookup[target_key]
            target_exp = resolve_path(target_base)
            metrics = load_metrics(target_base)
            t_grok = detect_grok_epoch(metrics)
            if t_grok is None:
                print(f"Skipping non-grok target: {target_key}")
                continue
            config = metrics.get("config", {})
            train_data, _ = create_data(
                p=int(config.get("p", 113)),
                fraction=float(config.get("fraction", 0.3)),
                seed=int(config.get("seed", seed)),
                device=device,
                task=target_task,
            )
            matched = load_existing_matched_trajectory(target_base, args.checkpoint_freq)
            if matched is None:
                raise FileNotFoundError(f"No matched trajectory for {target_base}")

            for ref_task in ref_tasks:
                ref_key = (args.activation, ref_task, seed)
                if ref_key not in lookup:
                    print(f"Skipping ref missing from RUNS_1L: {ref_key}")
                    continue
                relation = task_relation(target_task, ref_task)
                if relation == "same_task":
                    traj = matched
                else:
                    ref_base = lookup[ref_key]
                    ref_exp = resolve_path(ref_base)
                    print(
                        f"Computing target={args.activation}/{target_task}/s{seed} "
                        f"ref={ref_task} relation={relation}"
                    )
                    ref_geo = build_reference_geometry(
                        final_checkpoint(ref_exp),
                        train_data,
                        device=device,
                        N=args.N,
                        k=args.k,
                        d=args.d,
                    )
                    traj = compute_gcs_trajectory(
                        target_exp,
                        ref_geo,
                        checkpoint_freq=args.checkpoint_freq,
                        device=device,
                    )

                rho_overall = trajectory_correlation(matched, traj, "overall_path")
                rho_attn = trajectory_correlation(matched, traj, "attention_path")
                rho_mlp = trajectory_correlation(matched, traj, "mlp_path")
                details = evaluate_trajectory_details(traj, t_grok)
                row = {
                    "activation": args.activation,
                    "seed": seed,
                    "target_task": target_task,
                    "ref_task": ref_task,
                    "relation": relation,
                    "t_grok": int(t_grok),
                    "checkpoint_freq": int(args.checkpoint_freq),
                    "N": int(args.N),
                    "k": int(args.k),
                    "d": int(args.d),
                    "rho_overall": float(rho_overall),
                    "rho_attn": float(rho_attn),
                    "rho_mlp": float(rho_mlp),
                    **details,
                }
                rows.append(row)
                trajectories[(seed, target_task, ref_task)] = traj

    out_csv = out_dir / f"task_reference_ablation_{args.tag}.csv"
    if rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    with open(traj_dir / "trajectories.pkl", "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Saved: {out_csv}")
    print(f"Saved: {traj_dir / 'trajectories.pkl'}")


if __name__ == "__main__":
    main()
