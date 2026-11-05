#!/usr/bin/env python3
"""
Activation-level patching for commutative modular-arithmetic runs.

For each add/mul 1-layer run, compare parameter-free activation patches:
  - final model with donor-pre attention output / MLP output patched in
  - donor-pre model with final attention output / MLP output patched in

This tests whether the final computation depends immediately on the formed
attention-side activation state, without replacing any parameters.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer  # noqa: E402
from gcs.legacy.checkpoint_surgery_modular import (  # noqa: E402
    available_checkpoints,
    checkpoint_path_for_epoch,
    choose_donor_epoch,
    extract_epoch,
    load_checkpoint,
    load_equivariance_trajectory,
    lookup_run,
    modular_dataset,
    nearest_checkpoint_epoch,
    select_surgery_epochs,
)
from gcs.legacy.diagnostics_common import detect_grok_epoch, load_metrics, resolve_path  # noqa: E402
from gcs.legacy.measure_equivariance import build_swap_pairs  # noqa: E402


COMMUTATIVE_RUNS = [
    ("ReLU", "add", 777),
    ("ReLU", "add", 42),
    ("ReLU", "add", 123),
    ("GeLU", "add", 777),
    ("GeLU", "add", 42),
    ("GeLU", "add", 123),
    ("SiLU", "add", 777),
    ("SiLU", "add", 42),
    ("ReLU", "mul", 777),
    ("ReLU", "mul", 42),
    ("ReLU", "mul", 123),
    ("GeLU", "mul", 777),
    ("GeLU", "mul", 42),
    ("GeLU", "mul", 123),
    ("SiLU", "mul", 777),
    ("SiLU", "mul", 42),
    ("SiLU", "mul", 123),
]

HOOKS = {
    "attn_out": "blocks.0.hook_attn_out",
    "resid_mid": "blocks.0.hook_resid_mid",
    "mlp_out": "blocks.0.hook_mlp_out",
}


def condition_specs() -> List[Dict[str, str]]:
    return [
        {
            "condition": "final_base",
            "target": "final",
            "source": "",
            "hook_key": "",
            "direction": "base",
        },
        {
            "condition": "donor_pre",
            "target": "donor",
            "source": "",
            "hook_key": "",
            "direction": "base",
        },
        {
            "condition": "final_with_donor_attn_out",
            "target": "final",
            "source": "donor",
            "hook_key": "attn_out",
            "direction": "damage",
        },
        {
            "condition": "final_with_donor_resid_mid",
            "target": "final",
            "source": "donor",
            "hook_key": "resid_mid",
            "direction": "damage",
        },
        {
            "condition": "final_with_donor_mlp_out",
            "target": "final",
            "source": "donor",
            "hook_key": "mlp_out",
            "direction": "damage",
        },
        {
            "condition": "donor_with_final_attn_out",
            "target": "donor",
            "source": "final",
            "hook_key": "attn_out",
            "direction": "rescue",
        },
        {
            "condition": "donor_with_final_resid_mid",
            "target": "donor",
            "source": "final",
            "hook_key": "resid_mid",
            "direction": "rescue",
        },
        {
            "condition": "donor_with_final_mlp_out",
            "target": "donor",
            "source": "final",
            "hook_key": "mlp_out",
            "direction": "rescue",
        },
    ]


def batch_iter(dataset, batch_size: int):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x, y in loader:
        yield x, y


def build_model_from_cfg(cfg: Dict[str, Any], device: str) -> Transformer:
    return Transformer(
        num_layers=cfg.get("num_layers", 1),
        d_vocab=cfg.get("d_vocab", 114),
        d_model=cfg.get("d_model", 128),
        d_mlp=cfg.get("d_mlp", 512),
        d_head=cfg.get("d_head", 32),
        num_heads=cfg.get("num_heads", 4),
        n_ctx=cfg.get("n_ctx", 3),
        act_type=cfg.get("act_type", "SiLU"),
        use_cache=False,
        use_ln=cfg.get("use_ln", False),
        attention_rate=cfg.get("attention_rate", 1.0),
    ).to(device)


def collect_activation(model: Transformer, x: torch.Tensor, hook_name: str) -> torch.Tensor:
    cache: Dict[str, torch.Tensor] = {}
    model.cache_all(cache)
    with torch.no_grad():
        model(x)
    model.remove_all_hooks()
    return cache[hook_name].detach()


def run_model_with_optional_patch(
    target_model: Transformer,
    source_model: Transformer | None,
    x: torch.Tensor,
    *,
    hook_name: str | None,
) -> torch.Tensor:
    if source_model is None or hook_name is None:
        with torch.no_grad():
            return target_model(x)

    source_activation = collect_activation(source_model, x, hook_name)

    def patch_hook(_tensor, name):
        if name != hook_name:
            raise RuntimeError(f"Patch hook attached to unexpected hook {name}")
        return source_activation

    hook_point = target_model.hook_dict[hook_name]
    hook_point.add_hook(patch_hook, "fwd")
    try:
        with torch.no_grad():
            logits = target_model(x)
    finally:
        hook_point.remove_hooks("fwd")
    return logits


def evaluate_condition_on_dataset(
    target_model: Transformer,
    source_model: Transformer | None,
    dataset,
    *,
    hook_name: str | None,
    device: str,
    batch_size: int,
) -> Dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total = 0
    target_model.eval()
    if source_model is not None:
        source_model.eval()

    for x, labels in batch_iter(dataset, batch_size):
        x = x.to(device)
        labels = labels.to(device)
        logits = run_model_with_optional_patch(target_model, source_model, x, hook_name=hook_name)
        if logits.dim() == 3:
            logits = logits[:, -1]
        loss = F.cross_entropy(logits, labels, reduction="sum")
        preds = torch.argmax(logits, dim=-1)
        total_loss += float(loss.item())
        total_correct += int((preds == labels).sum().item())
        total += int(labels.shape[0])

    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
    }


def evaluate_patched_attention_equivariance(
    target_model: Transformer,
    source_model: Transformer | None,
    *,
    hook_name: str | None,
    p: int,
    device: str,
    batch_size: int,
) -> Dict[str, float]:
    prompts_ab, prompts_ba = build_swap_pairs(p, device=device)
    n = int(prompts_ab.shape[0])
    total_diff = 0.0
    total_norm = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ab = prompts_ab[start:end]
        ba = prompts_ba[start:end]

        # We need the attention-output activation that is actually consumed by
        # the downstream computation. For attention-output patches, this is the
        # source activation. For other conditions, it is the target activation
        # under the same optional patch.
        if hook_name == HOOKS["attn_out"] and source_model is not None:
            attn_ab = collect_activation(source_model, ab, HOOKS["attn_out"])
            attn_ba = collect_activation(source_model, ba, HOOKS["attn_out"])
        else:
            cache_ab: Dict[str, torch.Tensor] = {}
            cache_ba: Dict[str, torch.Tensor] = {}

            def cache_attn_ab(tensor, name):
                cache_ab["attn"] = tensor.detach()
                return None

            def cache_attn_ba(tensor, name):
                cache_ba["attn"] = tensor.detach()
                return None

            attn_hook = target_model.hook_dict[HOOKS["attn_out"]]
            attn_hook.add_hook(cache_attn_ab, "fwd")
            try:
                run_model_with_optional_patch(target_model, source_model, ab, hook_name=hook_name)
            finally:
                attn_hook.remove_hooks("fwd")

            attn_hook = target_model.hook_dict[HOOKS["attn_out"]]
            attn_hook.add_hook(cache_attn_ba, "fwd")
            try:
                run_model_with_optional_patch(target_model, source_model, ba, hook_name=hook_name)
            finally:
                attn_hook.remove_hooks("fwd")

            attn_ab = cache_ab["attn"]
            attn_ba = cache_ba["attn"]

        attn_ab_last = attn_ab[:, 2, :]
        attn_ba_last = attn_ba[:, 2, :]
        total_diff += float((attn_ab_last - attn_ba_last).norm(dim=-1).sum().item())
        total_norm += float(attn_ab_last.norm(dim=-1).sum().item())

    mean_diff = total_diff / max(n, 1)
    mean_norm = total_norm / max(n, 1)
    return {
        "attn_equi_dist": mean_diff,
        "attn_norm_scale": mean_norm,
        "attn_rel_equi": mean_diff / max(mean_norm, 1e-10),
    }


def get_models_and_metadata(
    act: str,
    task: str,
    seed: int,
    *,
    exp_dir_arg: str | None,
    device: str,
    donor_mode: str,
    donor_fixed_frac: float,
    donor_train_acc_threshold: float,
    donor_test_acc_max: float,
):
    exp_dir = resolve_path(exp_dir_arg) if exp_dir_arg else lookup_run(act, task, seed)
    checkpoints = available_checkpoints(exp_dir)
    metrics = load_metrics(exp_dir)
    t_grok = detect_grok_epoch(metrics)
    if t_grok is None:
        raise ValueError(f"Run did not grok: {exp_dir}")

    eq = load_equivariance_trajectory(exp_dir, act, task, seed)
    epochs = np.asarray(eq["epochs"], dtype=int)
    rel_equi = np.asarray(eq["equi_dist"], dtype=float) / np.maximum(
        np.asarray(eq["norm_scale"], dtype=float),
        1e-8,
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
    t_sym_epoch = nearest_checkpoint_epoch(selection["t_sym"], checkpoints)
    t_minus_epoch = nearest_checkpoint_epoch(selection["t_minus"], checkpoints)

    final_model, final_cfg, final_state = load_checkpoint(
        checkpoint_path_for_epoch(exp_dir, final_epoch),
        device=device,
    )
    donor_model, donor_cfg, donor_state = load_checkpoint(
        checkpoint_path_for_epoch(exp_dir, donor_epoch),
        device=device,
    )
    p = int(final_cfg.get("p", final_cfg.get("d_vocab", 114) - 1))
    fraction = float(final_cfg.get("fraction", 0.3))

    return {
        "exp_dir": exp_dir,
        "metrics": metrics,
        "selection": selection,
        "t_grok": int(t_grok),
        "t_minus": int(t_minus_epoch),
        "t_sym": int(t_sym_epoch),
        "donor_epoch": int(donor_epoch),
        "donor_target_epoch": int(donor_target_epoch),
        "final_epoch": int(final_epoch),
        "final_model": final_model,
        "donor_model": donor_model,
        "final_cfg": final_cfg,
        "donor_cfg": donor_cfg,
        "p": p,
        "fraction": fraction,
    }


def evaluate_run(
    act: str,
    task: str,
    seed: int,
    *,
    exp_dir_arg: str | None,
    device: str,
    batch_size: int,
    donor_mode: str,
    donor_fixed_frac: float,
    donor_train_acc_threshold: float,
    donor_test_acc_max: float,
    conditions: Iterable[str],
) -> List[Dict[str, Any]]:
    info = get_models_and_metadata(
        act,
        task,
        seed,
        exp_dir_arg=exp_dir_arg,
        device=device,
        donor_mode=donor_mode,
        donor_fixed_frac=donor_fixed_frac,
        donor_train_acc_threshold=donor_train_acc_threshold,
        donor_test_acc_max=donor_test_acc_max,
    )

    train_data, test_data = modular_dataset(
        task=task,
        seed=seed,
        p=info["p"],
        fraction=info["fraction"],
        device=device,
    )
    models = {
        "final": info["final_model"],
        "donor": info["donor_model"],
    }

    wanted = set(conditions)
    rows: List[Dict[str, Any]] = []
    for spec in condition_specs():
        if spec["condition"] not in wanted:
            continue
        target_model = models[spec["target"]]
        source_model = models[spec["source"]] if spec["source"] else None
        hook_name = HOOKS[spec["hook_key"]] if spec["hook_key"] else None

        train_scores = evaluate_condition_on_dataset(
            target_model,
            source_model,
            train_data,
            hook_name=hook_name,
            device=device,
            batch_size=batch_size,
        )
        test_scores = evaluate_condition_on_dataset(
            target_model,
            source_model,
            test_data,
            hook_name=hook_name,
            device=device,
            batch_size=batch_size,
        )
        eq_scores = evaluate_patched_attention_equivariance(
            target_model,
            source_model,
            hook_name=hook_name,
            p=info["p"],
            device=device,
            batch_size=batch_size,
        )

        rows.append({
            "act": act,
            "task": task,
            "seed": seed,
            "exp_dir": str(info["exp_dir"]),
            "condition": spec["condition"],
            "direction": spec["direction"],
            "target_model": spec["target"],
            "source_model": spec["source"],
            "hook_key": spec["hook_key"],
            "hook_name": hook_name or "",
            "t_grok": info["t_grok"],
            "t_minus": info["t_minus"],
            "t_sym": info["t_sym"],
            "donor_epoch": info["donor_epoch"],
            "donor_target_epoch": info["donor_target_epoch"],
            "final_epoch": info["final_epoch"],
            "train_loss": train_scores["loss"],
            "train_acc": train_scores["acc"],
            "test_loss": test_scores["loss"],
            "test_acc": test_scores["acc"],
            **eq_scores,
        })

    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key = {(r["act"], r["task"], int(r["seed"]), r["condition"]): r for r in rows}
    run_keys = sorted({(r["act"], r["task"], int(r["seed"])) for r in rows})
    per_run = []
    damage_attn_gt_mlp = 0
    damage_attn_gt_resid_mid = 0
    damage_resid_mid_gt_mlp = 0
    rescue_attn_gt_mlp = 0
    rescue_resid_mid_gt_attn = 0
    rescue_resid_mid_gt_mlp = 0
    for key in run_keys:
        final_base = by_key.get((*key, "final_base"))
        donor_pre = by_key.get((*key, "donor_pre"))
        f_d_attn = by_key.get((*key, "final_with_donor_attn_out"))
        f_d_resid = by_key.get((*key, "final_with_donor_resid_mid"))
        f_d_mlp = by_key.get((*key, "final_with_donor_mlp_out"))
        d_f_attn = by_key.get((*key, "donor_with_final_attn_out"))
        d_f_resid = by_key.get((*key, "donor_with_final_resid_mid"))
        d_f_mlp = by_key.get((*key, "donor_with_final_mlp_out"))
        if final_base and f_d_attn and f_d_mlp:
            attn_damage = float(final_base["test_acc"] - f_d_attn["test_acc"])
            mlp_damage = float(final_base["test_acc"] - f_d_mlp["test_acc"])
            damage_attn_gt_mlp += int(attn_damage > mlp_damage)
        else:
            attn_damage = float("nan")
            mlp_damage = float("nan")
        if final_base and f_d_resid:
            resid_mid_damage = float(final_base["test_acc"] - f_d_resid["test_acc"])
            if np.isfinite(attn_damage):
                damage_attn_gt_resid_mid += int(attn_damage > resid_mid_damage)
            if np.isfinite(mlp_damage):
                damage_resid_mid_gt_mlp += int(resid_mid_damage > mlp_damage)
        else:
            resid_mid_damage = float("nan")
        if donor_pre and d_f_attn and d_f_mlp:
            attn_rescue = float(d_f_attn["test_acc"] - donor_pre["test_acc"])
            mlp_rescue = float(d_f_mlp["test_acc"] - donor_pre["test_acc"])
            rescue_attn_gt_mlp += int(attn_rescue > mlp_rescue)
        else:
            attn_rescue = float("nan")
            mlp_rescue = float("nan")
        if donor_pre and d_f_resid:
            resid_mid_rescue = float(d_f_resid["test_acc"] - donor_pre["test_acc"])
            if np.isfinite(attn_rescue):
                rescue_resid_mid_gt_attn += int(resid_mid_rescue > attn_rescue)
            if np.isfinite(mlp_rescue):
                rescue_resid_mid_gt_mlp += int(resid_mid_rescue > mlp_rescue)
        else:
            resid_mid_rescue = float("nan")
        per_run.append({
            "act": key[0],
            "task": key[1],
            "seed": key[2],
            "attn_damage": attn_damage,
            "resid_mid_damage": resid_mid_damage,
            "mlp_damage": mlp_damage,
            "attn_rescue": attn_rescue,
            "resid_mid_rescue": resid_mid_rescue,
            "mlp_rescue": mlp_rescue,
        })

    def finite_mean(name: str) -> float:
        vals = np.asarray([r[name] for r in per_run], dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else float("nan")

    n_runs = len(run_keys)
    return {
        "n_runs": n_runs,
        "damage_attn_gt_mlp_count": damage_attn_gt_mlp,
        "damage_attn_gt_resid_mid_count": damage_attn_gt_resid_mid,
        "damage_resid_mid_gt_mlp_count": damage_resid_mid_gt_mlp,
        "rescue_attn_gt_mlp_count": rescue_attn_gt_mlp,
        "rescue_resid_mid_gt_attn_count": rescue_resid_mid_gt_attn,
        "rescue_resid_mid_gt_mlp_count": rescue_resid_mid_gt_mlp,
        "mean_attn_damage": finite_mean("attn_damage"),
        "mean_resid_mid_damage": finite_mean("resid_mid_damage"),
        "mean_mlp_damage": finite_mean("mlp_damage"),
        "mean_attn_rescue": finite_mean("attn_rescue"),
        "mean_resid_mid_rescue": finite_mean("resid_mid_rescue"),
        "mean_mlp_rescue": finite_mean("mlp_rescue"),
        "per_run": per_run,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_run_specs(specs: str) -> List[Tuple[str, str, int]]:
    if specs == "commutative":
        return COMMUTATIVE_RUNS
    out = []
    for chunk in specs.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        act, task, seed_s = chunk.split(":")
        out.append((act, task, int(seed_s)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Activation patching for commutative modular runs.")
    parser.add_argument("--runs", type=str, default="commutative",
                        help="'commutative' or comma-separated Act:task:seed specs.")
    parser.add_argument("--act", type=str, default=None, help="Single-run activation override.")
    parser.add_argument("--task", type=str, default=None, help="Single-run task override.")
    parser.add_argument("--seed", type=int, default=None, help="Single-run seed override.")
    parser.add_argument("--exp-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=str, default="experiments/activation_patch_commutative")
    parser.add_argument("--conditions", type=str, default=",".join(s["condition"] for s in condition_specs()))
    parser.add_argument("--donor-mode", type=str, default="equiv_drop",
                        choices=["equiv_drop", "fixed_frac_tgrok", "train_acc_threshold", "memorization_plateau"])
    parser.add_argument("--donor-fixed-frac", type=float, default=0.8)
    parser.add_argument("--donor-train-acc-threshold", type=float, default=0.999)
    parser.add_argument("--donor-test-acc-max", type=float, default=0.1)
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if args.act and args.task and args.seed is not None:
        run_specs = [(args.act, args.task, int(args.seed))]
    else:
        run_specs = parse_run_specs(args.runs)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    out_dir = resolve_path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for act, task, seed in run_specs:
        print(f"Activation patching: {act}/{task}/s{seed}")
        try:
            rows = evaluate_run(
                act,
                task,
                seed,
                exp_dir_arg=args.exp_dir,
                device=device,
                batch_size=args.batch_size,
                donor_mode=args.donor_mode,
                donor_fixed_frac=args.donor_fixed_frac,
                donor_train_acc_threshold=args.donor_train_acc_threshold,
                donor_test_acc_max=args.donor_test_acc_max,
                conditions=conditions,
            )
            all_rows.extend(rows)
            run_csv = out_dir / f"{act.lower()}_{task}_s{seed}_activation_patch.csv"
            write_csv(run_csv, rows)
            for row in rows:
                print(
                    f"  {row['condition']:<30} "
                    f"test_acc={row['test_acc']:.4f} "
                    f"attn_rel={row['attn_rel_equi']:.4f}"
                )
        except Exception as exc:
            failures.append({"act": act, "task": task, "seed": seed, "error": str(exc)})
            print(f"  FAILED: {exc}")

    write_csv(out_dir / "patch_results.csv", all_rows)
    if failures:
        write_csv(out_dir / "failures.csv", failures)
    summary = summarize(all_rows)
    summary["failures"] = failures
    write_csv(out_dir / "patch_summary_per_run.csv", summary.get("per_run", []))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {out_dir / 'patch_results.csv'}")
    print(f"Saved: {out_dir / 'patch_summary_per_run.csv'}")
    print(f"Saved: {out_dir / 'summary.json'}")
    print(
        "Summary: "
        f"damage attn>mlp {summary['damage_attn_gt_mlp_count']}/{summary['n_runs']}, "
        f"damage resid_mid>mlp {summary['damage_resid_mid_gt_mlp_count']}/{summary['n_runs']}, "
        f"rescue attn>mlp {summary['rescue_attn_gt_mlp_count']}/{summary['n_runs']}, "
        f"rescue resid_mid>mlp {summary['rescue_resid_mid_gt_mlp_count']}/{summary['n_runs']}"
    )


if __name__ == "__main__":
    main()
