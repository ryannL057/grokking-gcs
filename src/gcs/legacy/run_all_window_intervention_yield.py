#!/usr/bin/env python3
"""Evaluate final-base attention/MLP donor surgery at every legal GCS center.

The runner reuses submitted checkpoints and writes one resumable CSV per run.
It does not aggregate effect sizes; the frozen analysis lives in
``analyze_all_window_intervention_yield.py``.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = _gcs_data_root()
pass  # Imports use the installed package.

from gcs.core.models import Transformer  # noqa: E402
from gcs.core.training import full_accuracy  # noqa: E402
from gcs.legacy.checkpoint_surgery_modular import (  # noqa: E402
    available_checkpoints,
    condition_param_names,
    copy_selected_params,
    extract_epoch,
    lookup_run,
    modular_dataset,
)
from gcs.legacy.diagnostics_common import (  # noqa: E402
    detect_grok_epoch,
    load_gcs,
    load_metrics,
)


PROTOCOL_VERSION = "all_window_intervention_v1_20260712"
DEFAULT_COHORT = ROOT / "experiments/mechanistic_yield_cohort_core_heads_20260503/per_window.csv"
DEFAULT_OUTPUT = ROOT / "experiments/rebuttal_all_window_intervention_yield_20260712/raw"

FIELDNAMES = [
    "protocol_version",
    "run_id",
    "act",
    "task",
    "seed",
    "window_len",
    "t_grok",
    "base_epoch",
    "donor_epoch",
    "normalized_donor_epoch",
    "donor_test_acc",
    "base_test_acc",
    "reset_attn_test_acc",
    "reset_mlp_test_acc",
    "delta_attn",
    "delta_mlp",
    "attn_minus_mlp_delta",
    "donor_eval_sec",
]


def read_cohort_keys(path: Path, split: str) -> list[tuple[str, str, int]]:
    requested = {item.strip() for item in split.split(",") if item.strip()}
    use_all = not requested or "all" in requested
    keys: set[tuple[str, str, int]] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("selector") != "gcs_attention":
                continue
            if not use_all and row.get("split") not in requested:
                continue
            keys.add((row["act"], row["task"], int(row["seed"])))
    if not keys:
        raise SystemExit(f"No attention-GCS cohort rows found in {path}")
    return sorted(keys)


def parse_run_filter(text: str) -> set[str]:
    return {item.strip().lower() for item in text.split(",") if item.strip()}


def run_id(act: str, task: str, seed: int) -> str:
    return f"{act.lower()}_{task}_s{seed}"


def build_model(cfg: dict, device: str) -> Transformer:
    model = Transformer(
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
        attention_rate=cfg.get("attention_rate", 1.0) or 1.0,
    ).to(device)
    model.eval()
    return model


def load_payload(path: Path, device: str) -> tuple[dict, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("config") or payload.get("model_config", {})
    return cfg, payload["model_state_dict"]


def metric_accuracy_by_epoch(metrics: dict) -> dict[int, float]:
    steps = np.asarray(metrics["log_steps"], dtype=int)
    values = np.asarray(metrics["test_accuracies"], dtype=float)
    return {int(epoch): float(value) for epoch, value in zip(steps, values)}


def existing_epochs(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    versions = {row.get("protocol_version", "") for row in rows}
    if versions - {PROTOCOL_VERSION}:
        raise ValueError(f"Protocol mismatch in {path}: {sorted(versions)}")
    return {int(row["donor_epoch"]) for row in rows}


def initialize_csv(path: Path, force: bool) -> set[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return existing_epochs(path)
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
    return set()


def append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def eligible_center_epochs(
    gcs_epochs: np.ndarray,
    checkpoint_epochs: set[int],
    window_len: int,
) -> tuple[list[int], list[int]]:
    if window_len <= 0 or window_len % 2 == 0:
        raise ValueError("--window-len must be a positive odd integer")
    if gcs_epochs.size < window_len:
        return [], []
    half = window_len // 2
    centers = [int(epoch) for epoch in gcs_epochs[half : gcs_epochs.size - half]]
    eligible = [epoch for epoch in centers if epoch in checkpoint_epochs]
    missing = [epoch for epoch in centers if epoch not in checkpoint_epochs]
    return eligible, missing


def evaluate_accuracy(model: Transformer, state: dict, test_data, device: str) -> float:
    model.load_state_dict(state)
    model.eval()
    return float(full_accuracy(model, test_data, device))


def run_one(
    *,
    act: str,
    task: str,
    seed: int,
    window_len: int,
    device: str,
    fraction: float,
    output_dir: Path,
    max_donors: int,
    force: bool,
    dry_run: bool,
) -> dict[str, object]:
    rid = run_id(act, task, seed)
    exp_dir = lookup_run(act, task, seed)
    checkpoints = available_checkpoints(exp_dir)
    checkpoint_by_epoch = {extract_epoch(path): path for path in checkpoints}
    metrics = load_metrics(exp_dir)
    t_grok = detect_grok_epoch(metrics)
    if t_grok is None:
        raise ValueError(f"Run has no grok epoch: {rid}")
    gcs_epochs, _ = load_gcs(exp_dir)
    gcs_epochs = np.asarray(gcs_epochs, dtype=int)
    eligible, missing = eligible_center_epochs(
        gcs_epochs,
        set(checkpoint_by_epoch),
        window_len,
    )
    if max_donors > 0:
        eligible = eligible[:max_donors]
    out_path = output_dir / f"{rid}.csv"
    completed = initialize_csv(out_path, force=force) if not dry_run else existing_epochs(out_path)
    pending = [epoch for epoch in eligible if epoch not in completed]

    print(
        f"[manifest] {rid} legal={len(eligible)} pending={len(pending)} "
        f"missing_exact_checkpoint={len(missing)}",
        flush=True,
    )
    if dry_run or not pending:
        return {
            "run_id": rid,
            "legal": len(eligible),
            "pending": len(pending),
            "missing": len(missing),
        }

    base_epoch = max(checkpoint_by_epoch)
    base_cfg, base_state = load_payload(checkpoint_by_epoch[base_epoch], device)
    p = int(base_cfg.get("p", base_cfg.get("d_vocab", 114) - 1))
    _, test_data = modular_dataset(
        task=task,
        seed=seed,
        p=p,
        fraction=fraction,
        device=device,
    )
    model = build_model(base_cfg, device)
    base_acc = evaluate_accuracy(model, base_state, test_data, device)
    attn_names = condition_param_names(base_state, "reset_attn")
    mlp_names = condition_param_names(base_state, "reset_mlp")
    accuracy_by_epoch = metric_accuracy_by_epoch(metrics)

    for index, donor_epoch in enumerate(pending, start=1):
        started = time.perf_counter()
        _, donor_state = load_payload(checkpoint_by_epoch[donor_epoch], device)
        attn_state = copy_selected_params(base_state, donor_state, attn_names)
        reset_attn_acc = evaluate_accuracy(model, attn_state, test_data, device)
        del attn_state
        mlp_state = copy_selected_params(base_state, donor_state, mlp_names)
        reset_mlp_acc = evaluate_accuracy(model, mlp_state, test_data, device)
        del mlp_state, donor_state
        delta_attn = base_acc - reset_attn_acc
        delta_mlp = base_acc - reset_mlp_acc
        elapsed = time.perf_counter() - started
        append_row(
            out_path,
            {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": rid,
                "act": act,
                "task": task,
                "seed": seed,
                "window_len": window_len,
                "t_grok": int(t_grok),
                "base_epoch": base_epoch,
                "donor_epoch": donor_epoch,
                "normalized_donor_epoch": donor_epoch / max(base_epoch, 1),
                "donor_test_acc": accuracy_by_epoch.get(donor_epoch, float("nan")),
                "base_test_acc": base_acc,
                "reset_attn_test_acc": reset_attn_acc,
                "reset_mlp_test_acc": reset_mlp_acc,
                "delta_attn": delta_attn,
                "delta_mlp": delta_mlp,
                "attn_minus_mlp_delta": delta_attn - delta_mlp,
                "donor_eval_sec": elapsed,
            },
        )
        print(
            f"[done] {rid} donor={index}/{len(pending)} epoch={donor_epoch} "
            f"elapsed_sec={elapsed:.2f}",
            flush=True,
        )

    return {
        "run_id": rid,
        "legal": len(eligible),
        "pending": 0,
        "missing": len(missing),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--split", default="all")
    parser.add_argument("--window-len", type=int, default=11)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-filter",
        default="",
        help="Comma-separated run IDs such as gelu_add_s42.",
    )
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--max-donors", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    keys = read_cohort_keys(args.cohort, split=args.split)
    filters = parse_run_filter(args.run_filter)
    if filters:
        keys = [key for key in keys if run_id(*key).lower() in filters]
    if args.max_runs > 0:
        keys = keys[: args.max_runs]
    if not keys:
        raise SystemExit("No runs remain after filtering")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for act, task, seed in keys:
        manifests.append(
            run_one(
                act=act,
                task=task,
                seed=seed,
                window_len=args.window_len,
                device=device,
                fraction=args.fraction,
                output_dir=args.output_dir,
                max_donors=args.max_donors,
                force=args.force,
                dry_run=args.dry_run,
            )
        )
    print(
        f"[summary] runs={len(manifests)} legal_donors="
        f"{sum(int(row['legal']) for row in manifests)} missing_exact_checkpoint="
        f"{sum(int(row['missing']) for row in manifests)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

