#!/usr/bin/env python3
"""Site-wise temporal operand-swap symmetry acquisition for commutative runs.

For each checkpoint in the primary p=113 add/mul runs, this script measures
normalized operand-swap distance at attention output, post-attention residual,
MLP preactivation, MLP output, final residual, centered logits, and
probabilities.  It then reports when each site has recovered most of its
pre-grok order-sensitivity peak toward its final symmetric endpoint value.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

pass  # Package imports.

from gcs.core.models import Transformer  # noqa: E402
from gcs.legacy.checkpoint_surgery_modular import (  # noqa: E402
    available_checkpoints,
    extract_epoch,
    lookup_run,
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


SITE_KEYS = {
    "attn_out": "blocks.0.hook_attn_out",
    "resid_mid": "blocks.0.hook_resid_mid",
    "mlp_pre": "blocks.0.mlp.hook_pre",
    "mlp_post": "blocks.0.mlp.hook_post",
    "mlp_out": "blocks.0.hook_mlp_out",
    "resid_post": "blocks.0.hook_resid_post",
}

SITE_ORDER = [
    "attn_out",
    "resid_mid",
    "mlp_pre",
    "mlp_post",
    "mlp_out",
    "resid_post",
    "logits_centered",
    "probabilities",
]


def build_model_from_cfg(cfg: dict[str, Any], device: str) -> Transformer:
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
        attention_rate=cfg.get("attention_rate", 1.0) or 1.0,
    ).to(device)


def load_model_template(first_checkpoint: Path, device: str) -> tuple[Transformer, dict[str, Any]]:
    payload = torch.load(first_checkpoint, map_location=device, weights_only=False)
    cfg = payload.get("config") or payload.get("model_config", {})
    model = build_model_from_cfg(cfg, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def select_checkpoints(checkpoints: list[Path], stride: int) -> list[Path]:
    if stride <= 1:
        return checkpoints
    selected = [path for idx, path in enumerate(checkpoints) if idx % stride == 0]
    if checkpoints[-1] not in selected:
        selected.append(checkpoints[-1])
    return selected


def site_values(model: Transformer, prompts: torch.Tensor, p: int) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    model.cache_all(cache)
    logits = model(prompts)
    model.remove_all_hooks()

    values: dict[str, torch.Tensor] = {}
    for site, hook_name in SITE_KEYS.items():
        if hook_name in cache:
            values[site] = cache[hook_name][:, -1, :]

    logits_p = logits[:, -1, :p]
    values["logits_centered"] = logits_p - logits_p.mean(dim=-1, keepdim=True)
    values["probabilities"] = torch.softmax(logits_p, dim=-1)
    return values


def measure_sites_at_checkpoint(
    model: Transformer,
    prompts_ab: torch.Tensor,
    prompts_ba: torch.Tensor,
    *,
    p: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    n = int(prompts_ab.shape[0])
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            values_ab = site_values(model, prompts_ab[start:end], p)
            values_ba = site_values(model, prompts_ba[start:end], p)
            for site, x_ab in values_ab.items():
                if site not in values_ba:
                    continue
                x_ba = values_ba[site]
                diff = (x_ab - x_ba).norm(dim=-1)
                norm = x_ab.norm(dim=-1)
                item = totals.setdefault(site, {"diff_sum": 0.0, "norm_sum": 0.0, "count": 0.0})
                item["diff_sum"] += float(diff.sum().item())
                item["norm_sum"] += float(norm.sum().item())
                item["count"] += float(diff.numel())

    out: dict[str, dict[str, float]] = {}
    for site, item in totals.items():
        count = max(float(item["count"]), 1.0)
        mean_diff = float(item["diff_sum"] / count)
        mean_norm = float(item["norm_sum"] / count)
        out[site] = {
            "mean_diff": mean_diff,
            "mean_norm": mean_norm,
            "relative_dequiv": mean_diff / mean_norm if mean_norm > 1e-12 else float("nan"),
        }
    return out


def first_sustained_crossing(
    values: np.ndarray,
    threshold: float,
    start_idx: int,
    sustain: int,
) -> int | None:
    for idx in range(start_idx, values.size):
        end = min(values.size, idx + max(1, sustain))
        if np.all(values[idx:end] <= threshold):
            return int(idx)
    return None


def acquisition_stats(
    epochs: np.ndarray,
    values: np.ndarray,
    *,
    t_grok: int,
    recovery_fraction: float,
    sustain: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    epochs = np.asarray(epochs, dtype=int)
    finite = np.isfinite(values) & np.isfinite(epochs.astype(float)) & (epochs > 0)
    if finite.sum() < 4:
        return {"status": "insufficient"}

    e = epochs[finite]
    v = values[finite]
    pre = np.where(e <= int(t_grok))[0]
    if pre.size < 2:
        pre = np.arange(v.size)
    peak_local = int(pre[int(np.argmax(v[pre]))])
    peak_val = float(v[peak_local])
    final_val = float(np.median(v[-min(5, v.size):]))
    drop = peak_val - final_val
    if not np.isfinite(drop) or drop <= 1e-10:
        return {
            "status": "no_peak_drop",
            "peak_epoch": int(e[peak_local]),
            "peak_relative_dequiv": peak_val,
            "final_relative_dequiv": final_val,
            "threshold": float("nan"),
        }

    threshold = final_val + (1.0 - recovery_fraction) * drop
    crossing = first_sustained_crossing(v, threshold, peak_local + 1, sustain)
    if crossing is None:
        return {
            "status": "not_acquired",
            "peak_epoch": int(e[peak_local]),
            "peak_relative_dequiv": peak_val,
            "final_relative_dequiv": final_val,
            "threshold": float(threshold),
        }

    return {
        "status": "acquired",
        "peak_epoch": int(e[peak_local]),
        "peak_relative_dequiv": peak_val,
        "final_relative_dequiv": final_val,
        "threshold": float(threshold),
        "acquisition_epoch": int(e[crossing]),
        "acquisition_index": int(np.where(finite)[0][crossing]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_run_specs(specs: str) -> list[tuple[str, str, int]]:
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


def process_run(
    act: str,
    task: str,
    seed: int,
    *,
    device: str,
    batch_size: int,
    checkpoint_stride: int,
    recovery_fraction: float,
    sustain: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exp_dir = lookup_run(act, task, seed)
    metrics = load_metrics(exp_dir)
    t_grok = detect_grok_epoch(metrics)
    if t_grok is None:
        raise ValueError(f"Run did not grok: {act}/{task}/s{seed}")

    checkpoints = select_checkpoints(available_checkpoints(exp_dir), checkpoint_stride)
    model, cfg = load_model_template(checkpoints[0], device)
    p = int(cfg.get("p", cfg.get("d_vocab", 114) - 1))
    prompts_ab, prompts_ba = build_swap_pairs(p, device=device)

    time_rows: list[dict[str, Any]] = []
    for ckpt_path in checkpoints:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        epoch = int(payload.get("epoch", extract_epoch(ckpt_path)))
        site_stats = measure_sites_at_checkpoint(
            model,
            prompts_ab,
            prompts_ba,
            p=p,
            batch_size=batch_size,
        )
        for site in SITE_ORDER:
            if site not in site_stats:
                continue
            time_rows.append({
                "act": act,
                "task": task,
                "seed": int(seed),
                "exp_dir": str(exp_dir),
                "epoch": epoch,
                "t_grok": int(t_grok),
                "site": site,
                **site_stats[site],
            })

    by_site: dict[str, list[tuple[int, float]]] = {site: [] for site in SITE_ORDER}
    for row in time_rows:
        by_site[row["site"]].append((int(row["epoch"]), float(row["relative_dequiv"])))

    acquisition_rows: list[dict[str, Any]] = []
    for site in SITE_ORDER:
        series = sorted(by_site.get(site, []), key=lambda item: item[0])
        if not series:
            continue
        epochs = np.asarray([item[0] for item in series], dtype=int)
        values = np.asarray([item[1] for item in series], dtype=float)
        stats = acquisition_stats(
            epochs,
            values,
            t_grok=int(t_grok),
            recovery_fraction=recovery_fraction,
            sustain=sustain,
        )
        acquisition_rows.append({
            "act": act,
            "task": task,
            "seed": int(seed),
            "exp_dir": str(exp_dir),
            "site": site,
            "t_grok": int(t_grok),
            "recovery_fraction": float(recovery_fraction),
            "sustain_checkpoints": int(sustain),
            "n_checkpoints": int(epochs.size),
            **stats,
        })

    attn_row = next((row for row in acquisition_rows if row["site"] == "attn_out"), None)
    attn_epoch = None
    if attn_row and attn_row.get("status") == "acquired":
        attn_epoch = int(attn_row["acquisition_epoch"])
    for row in acquisition_rows:
        acq = row.get("acquisition_epoch")
        if row.get("status") == "acquired" and acq is not None:
            row["acquisition_minus_t_grok"] = int(acq) - int(t_grok)
            row["acquisition_over_t_grok"] = float(acq) / max(float(t_grok), 1.0)
            row["acquisition_minus_attn_out"] = (
                int(acq) - int(attn_epoch) if attn_epoch is not None else float("nan")
            )
        else:
            row["acquisition_minus_t_grok"] = float("nan")
            row["acquisition_over_t_grok"] = float("nan")
            row["acquisition_minus_attn_out"] = float("nan")

    return time_rows, acquisition_rows


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=float)
    return vals[np.isfinite(vals)]


def summarize_acquisition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for site in SITE_ORDER:
        items = [row for row in rows if row["site"] == site]
        if not items:
            continue
        acquired = [row for row in items if row.get("status") == "acquired"]
        acq_epochs = finite_values(acquired, "acquisition_epoch")
        delta_tgrok = finite_values(acquired, "acquisition_minus_t_grok")
        ratio = finite_values(acquired, "acquisition_over_t_grok")
        delta_attn = finite_values(acquired, "acquisition_minus_attn_out")
        peak = finite_values(items, "peak_relative_dequiv")
        final = finite_values(items, "final_relative_dequiv")
        out.append({
            "site": site,
            "n_runs": len(items),
            "n_acquired": len(acquired),
            "median_acquisition_epoch": float(np.median(acq_epochs)) if acq_epochs.size else float("nan"),
            "median_acquisition_minus_t_grok": float(np.median(delta_tgrok)) if delta_tgrok.size else float("nan"),
            "median_acquisition_over_t_grok": float(np.median(ratio)) if ratio.size else float("nan"),
            "median_acquisition_minus_attn_out": float(np.median(delta_attn)) if delta_attn.size else float("nan"),
            "count_no_later_than_attn_out": int(np.sum(delta_attn <= 0.0)) if delta_attn.size else 0,
            "count_later_than_attn_out": int(np.sum(delta_attn > 0.0)) if delta_attn.size else 0,
            "median_peak_relative_dequiv": float(np.median(peak)) if peak.size else float("nan"),
            "median_final_relative_dequiv": float(np.median(final)) if final.size else float("nan"),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="commutative")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--checkpoint-stride", type=int, default=1)
    parser.add_argument("--recovery-fraction", type=float, default=0.8)
    parser.add_argument("--sustain-checkpoints", type=int, default=2)
    parser.add_argument("--output-dir", default="experiments/sitewise_temporal_symmetry_20260502")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_time_rows: list[dict[str, Any]] = []
    all_acq_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for act, task, seed in parse_run_specs(args.runs):
        print(f"Site-wise temporal symmetry: {act}/{task}/s{seed}")
        try:
            time_rows, acq_rows = process_run(
                act,
                task,
                seed,
                device=device,
                batch_size=args.batch_size,
                checkpoint_stride=args.checkpoint_stride,
                recovery_fraction=args.recovery_fraction,
                sustain=args.sustain_checkpoints,
            )
            all_time_rows.extend(time_rows)
            all_acq_rows.extend(acq_rows)
            run_csv = out_dir / f"{act.lower()}_{task}_s{seed}_sitewise_timeseries.csv"
            write_csv(run_csv, time_rows)
            for row in acq_rows:
                acq = row.get("acquisition_epoch", "")
                print(f"  {row['site']:<16s} {row['status']:<12s} acq={acq}")
        except Exception as exc:
            failures.append({"act": act, "task": task, "seed": int(seed), "error": repr(exc)})
            print(f"  FAILED: {exc}")

    write_csv(out_dir / "sitewise_timeseries.csv", all_time_rows)
    write_csv(out_dir / "sitewise_acquisition_epochs.csv", all_acq_rows)
    summary = summarize_acquisition(all_acq_rows)
    write_csv(out_dir / "sitewise_acquisition_summary.csv", summary)
    if failures:
        write_csv(out_dir / "failures.csv", failures)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "n_runs": len({(r["act"], r["task"], int(r["seed"])) for r in all_acq_rows}),
            "n_time_rows": len(all_time_rows),
            "n_acquisition_rows": len(all_acq_rows),
            "recovery_fraction": float(args.recovery_fraction),
            "sustain_checkpoints": int(args.sustain_checkpoints),
            "checkpoint_stride": int(args.checkpoint_stride),
            "sites": SITE_ORDER,
            "summary": summary,
            "failures": failures,
        }, f, indent=2)

    print(f"Saved: {out_dir / 'sitewise_timeseries.csv'}")
    print(f"Saved: {out_dir / 'sitewise_acquisition_epochs.csv'}")
    print(f"Saved: {out_dir / 'sitewise_acquisition_summary.csv'}")
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
