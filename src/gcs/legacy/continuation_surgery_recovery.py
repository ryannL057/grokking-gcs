#!/usr/bin/env python3
"""Continuation-after-surgery recovery control for commutative modular runs.

The immediate checkpoint-surgery result shows that replacing final attention
parameters with their pre-symmetry values hurts more than replacing MLP
parameters.  This script continues training the surgically modified models
with the original task objective and records how quickly accuracy and
attention-output swap equivariance recover.

The experiment is intentionally a control, not a new causal proof: fast
recovery would suggest much of the immediate damage is interface shock, while
slow or incomplete recovery strengthens the formed-state dependence claim.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = _gcs_data_root()
if str(ROOT) not in sys.path:
    pass  # Package imports do not search the experiment-data directory.
if str(ROOT / "scripts") not in sys.path:
    pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer  # noqa: E402
from gcs.core.training import create_optimizer, full_loss  # noqa: E402
from gcs.legacy.checkpoint_surgery_modular import (  # noqa: E402
    choose_donor_epoch,
    condition_param_names,
    copy_selected_params,
    evaluate_model,
    extract_epoch,
    load_checkpoint,
    load_equivariance_trajectory,
    modular_dataset,
    nearest_checkpoint_epoch,
    select_surgery_epochs,
)
from gcs.legacy.diagnostics_common import RUNS_1L, detect_grok_epoch, load_metrics, resolve_path  # noqa: E402


COMMUTATIVE_TASKS = {"add", "mul"}


def checkpoint_path_for_epoch(exp_dir: Path, epoch: int) -> Path:
    matches = sorted((exp_dir / "checkpoints").glob(f"*epoch_{epoch}.pth"))
    if matches:
        return matches[-1]
    ckpts = sorted((exp_dir / "checkpoints").glob("*.pth"), key=extract_epoch)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {exp_dir / 'checkpoints'}")
    return min(ckpts, key=lambda path: abs(extract_epoch(path) - epoch))


def available_checkpoints(exp_dir: Path) -> list[Path]:
    ckpts = sorted((exp_dir / "checkpoints").glob("*.pth"), key=extract_epoch)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {exp_dir / 'checkpoints'}")
    return ckpts


def discover_commutative_runs(activations: set[str], tasks: set[str], seeds: set[int] | None) -> list[tuple[str, str, int, Path]]:
    runs: list[tuple[str, str, int, Path]] = []
    for act, task, seed, path in RUNS_1L:
        if task not in COMMUTATIVE_TASKS:
            continue
        if activations and act not in activations:
            continue
        if tasks and task not in tasks:
            continue
        if seeds is not None and seed not in seeds:
            continue
        runs.append((act, task, seed, resolve_path(path)))
    return runs


def instantiate_model(cfg: dict, state_dict: dict, device: str) -> Transformer:
    model = Transformer(
        num_layers=cfg.get("num_layers", 1),
        d_vocab=cfg.get("d_vocab", int(cfg.get("p", 113)) + 1),
        d_model=cfg.get("d_model", 128),
        d_mlp=cfg.get("d_mlp", 512),
        d_head=cfg.get("d_head", 32),
        num_heads=cfg.get("num_heads", 4),
        n_ctx=cfg.get("n_ctx", 3),
        act_type=cfg.get("act_type", "SiLU"),
        use_cache=False,
        use_ln=cfg.get("use_ln", False),
        attention_rate=cfg.get("attention_rate", 1.0) or 1.0,
    ).to(device)
    model.load_state_dict(state_dict)
    return model


def train_epochs(
    model: Transformer,
    train_data,
    *,
    optimizer: torch.optim.Optimizer,
    device: str,
    n_epochs: int,
    max_grad_norm: float | None,
) -> None:
    for _ in range(n_epochs):
        model.train()
        loss = full_loss(model, train_data, device)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def build_surgery_states(
    exp_dir: Path,
    act: str,
    task: str,
    seed: int,
    donor_mode: str,
    donor_fixed_frac: float,
    donor_train_acc_threshold: float,
    donor_test_acc_max: float,
    device: str,
) -> dict:
    checkpoints = available_checkpoints(exp_dir)
    metrics = load_metrics(exp_dir)
    t_grok = detect_grok_epoch(metrics)
    if t_grok is None:
        raise ValueError(f"Run did not grok: {exp_dir}")
    eq = load_equivariance_trajectory(exp_dir, act, task, seed)
    epochs = np.asarray(eq["epochs"], dtype=int)
    rel_equi = np.asarray(eq["equi_dist"]) / np.where(
        np.asarray(eq["norm_scale"]) > 1e-8,
        np.asarray(eq["norm_scale"]),
        1.0,
    )
    selection = select_surgery_epochs(epochs, rel_equi, t_grok=t_grok)
    donor_epoch, donor_target_epoch = choose_donor_epoch(
        selection,
        t_grok=t_grok,
        checkpoints=checkpoints,
        metrics=metrics,
        donor_mode=donor_mode,
        donor_fixed_frac=donor_fixed_frac,
        donor_train_acc_threshold=donor_train_acc_threshold,
        donor_test_acc_max=donor_test_acc_max,
    )
    final_epoch = extract_epoch(checkpoints[-1])
    _donor_model, donor_cfg, donor_state = load_checkpoint(
        checkpoint_path_for_epoch(exp_dir, donor_epoch),
        device=device,
    )
    _base_model, base_cfg, base_state = load_checkpoint(
        checkpoint_path_for_epoch(exp_dir, final_epoch),
        device=device,
    )
    reset_states = {
        "base_final": base_state,
        "reset_attn": copy_selected_params(
            base_state,
            donor_state,
            condition_param_names(base_state, "reset_attn"),
        ),
        "reset_mlp": copy_selected_params(
            base_state,
            donor_state,
            condition_param_names(base_state, "reset_mlp"),
        ),
    }
    cfg = dict(donor_cfg)
    cfg.update(base_cfg)
    return {
        "cfg": cfg,
        "states": reset_states,
        "t_grok": int(t_grok),
        "donor_epoch": int(donor_epoch),
        "donor_target_epoch": int(donor_target_epoch),
        "final_epoch": int(final_epoch),
        "selection": selection,
    }


def run_one(
    act: str,
    task: str,
    seed: int,
    exp_dir: Path,
    *,
    budgets: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    surgery = build_surgery_states(
        exp_dir,
        act,
        task,
        seed,
        donor_mode=args.donor_mode,
        donor_fixed_frac=args.donor_fixed_frac,
        donor_train_acc_threshold=args.donor_train_acc_threshold,
        donor_test_acc_max=args.donor_test_acc_max,
        device=args.device,
    )
    cfg = surgery["cfg"]
    p = int(cfg.get("p", cfg.get("d_vocab", 114) - 1))
    fraction = float(cfg.get("fraction", args.fraction))
    lr = float(cfg.get("lr", args.lr))
    weight_decay = float(cfg.get("weight_decay", args.weight_decay))
    max_grad_norm = cfg.get("max_grad_norm", args.max_grad_norm)
    if max_grad_norm is not None and isinstance(max_grad_norm, float) and math.isnan(max_grad_norm):
        max_grad_norm = None

    train_data, test_data = modular_dataset(task=task, seed=seed, p=p, fraction=fraction, device=args.device)
    rows: list[dict] = []
    base_model = instantiate_model(cfg, surgery["states"]["base_final"], args.device)
    base_scores = evaluate_model(base_model, train_data, test_data, task=task, p=p, device=args.device)
    base_test_acc = float(base_scores["test_acc"])
    base_attn = float(base_scores["attn_rel_equi"])
    del base_model

    for condition in ["reset_attn", "reset_mlp"]:
        model = instantiate_model(cfg, surgery["states"][condition], args.device)
        optimizer = create_optimizer(
            model,
            lr=lr,
            weight_decay=weight_decay,
            wd_attn=cfg.get("wd_attn"),
            wd_mlp=cfg.get("wd_mlp"),
        )
        previous_budget = 0
        for budget in budgets:
            delta = int(budget - previous_budget)
            if delta < 0:
                raise ValueError("Budgets must be sorted")
            if delta:
                train_epochs(
                    model,
                    train_data,
                    optimizer=optimizer,
                    device=args.device,
                    n_epochs=delta,
                    max_grad_norm=max_grad_norm,
                )
            scores = evaluate_model(model, train_data, test_data, task=task, p=p, device=args.device)
            initial_drop = base_test_acc - float(scores["test_acc"]) if budget == 0 else None
            rows.append(
                {
                    "exp_dir": str(exp_dir),
                    "act": act,
                    "task": task,
                    "seed": seed,
                    "p": p,
                    "condition": condition,
                    "budget_epochs": int(budget),
                    "t_grok": surgery["t_grok"],
                    "donor_epoch": surgery["donor_epoch"],
                    "donor_target_epoch": surgery["donor_target_epoch"],
                    "final_epoch": surgery["final_epoch"],
                    "base_test_acc": base_test_acc,
                    "base_attn_rel_equi": base_attn,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    **scores,
                    "test_acc_gap_to_base": base_test_acc - float(scores["test_acc"]),
                    "attn_rel_gap_to_base": float(scores["attn_rel_equi"]) - base_attn,
                    "initial_drop_at_this_row": initial_drop if initial_drop is not None else "",
                }
            )
            previous_budget = int(budget)
        del model

    meta = {
        "exp_dir": str(exp_dir),
        "act": act,
        "task": task,
        "seed": seed,
        "p": p,
        "base_test_acc": base_test_acc,
        "base_attn_rel_equi": base_attn,
        "t_grok": surgery["t_grok"],
        "donor_epoch": surgery["donor_epoch"],
        "final_epoch": surgery["final_epoch"],
    }
    return rows, meta


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], int(row["budget_epochs"])), []).append(row)

    out: list[dict] = []
    for (condition, budget), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        gaps = np.array([float(r["test_acc_gap_to_base"]) for r in items], dtype=float)
        attn_gaps = np.array([float(r["attn_rel_gap_to_base"]) for r in items], dtype=float)
        accs = np.array([float(r["test_acc"]) for r in items], dtype=float)
        out.append(
            {
                "condition": condition,
                "budget_epochs": budget,
                "n": len(items),
                "mean_test_acc": float(np.mean(accs)),
                "median_test_acc": float(np.median(accs)),
                "mean_test_acc_gap_to_base": float(np.mean(gaps)),
                "median_test_acc_gap_to_base": float(np.median(gaps)),
                "mean_attn_rel_gap_to_base": float(np.mean(attn_gaps)),
                "median_attn_rel_gap_to_base": float(np.median(attn_gaps)),
                "n_recovered_99pct": int(np.sum(accs >= 0.99)),
                "n_recovered_95pct": int(np.sum(accs >= 0.95)),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activations", default="SiLU,ReLU,GeLU")
    parser.add_argument("--tasks", default="add,mul")
    parser.add_argument("--seeds", default="777,42,123")
    parser.add_argument("--budgets", default="0,50,100,200,500,1000")
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--donor-mode", default="equiv_drop")
    parser.add_argument("--donor-fixed-frac", type=float, default=0.8)
    parser.add_argument("--donor-train-acc-threshold", type=float, default=0.999)
    parser.add_argument("--donor-test-acc-max", type=float, default=0.1)
    parser.add_argument("--output-dir", default="experiments/continuation_surgery_recovery_20260429")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    activations = {item.strip() for item in args.activations.split(",") if item.strip()}
    tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    seeds = {int(item) for item in args.seeds.split(",") if item.strip()} if args.seeds else None
    budgets = sorted({int(item) for item in args.budgets.split(",") if item.strip()})
    if not budgets or budgets[0] != 0:
        budgets = [0] + budgets

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    meta_rows: list[dict] = []
    failures: list[dict] = []
    runs = discover_commutative_runs(activations, tasks, seeds)
    for act, task, seed, exp_dir in runs:
        try:
            print(f"Running continuation surgery: {act}/{task}/s{seed} ({exp_dir})", flush=True)
            run_rows, meta = run_one(act, task, seed, exp_dir, budgets=budgets, args=args)
            rows.extend(run_rows)
            meta_rows.append(meta)
            write_csv(output_dir / "continuation_recovery_rows.partial.csv", rows)
        except Exception as exc:
            failures.append({"act": act, "task": task, "seed": seed, "exp_dir": str(exp_dir), "error": repr(exc)})
            print(f"FAIL {act}/{task}/s{seed}: {exc}", flush=True)

    write_csv(output_dir / "continuation_recovery_rows.csv", rows)
    write_csv(output_dir / "continuation_recovery_run_meta.csv", meta_rows)
    write_csv(output_dir / "continuation_recovery_summary.csv", summarize(rows))
    if failures:
        write_csv(output_dir / "failures.csv", failures)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "n_runs_requested": len(runs),
                "n_runs_completed": len(meta_rows),
                "n_failures": len(failures),
                "budgets": budgets,
                "output_rows": str(output_dir / "continuation_recovery_rows.csv"),
                "output_summary": str(output_dir / "continuation_recovery_summary.csv"),
            },
            f,
            indent=2,
        )
    print(f"Saved {output_dir / 'continuation_recovery_rows.csv'}")
    print(f"Saved {output_dir / 'continuation_recovery_summary.csv'}")


if __name__ == "__main__":
    main()
