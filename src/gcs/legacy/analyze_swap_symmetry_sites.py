#!/usr/bin/env python3
"""Measure operand-swap symmetry at multiple endpoint sites.

This script is a small calibration check for cross-prime commutative outliers:
attention-output swap equivariance need not be the only way a successful
commutative solution can become symmetric.  For selected final checkpoints it
reports normalized swap distance at the attention output, residual/MLP stages,
and logits.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = _gcs_data_root()
if str(ROOT) not in sys.path:
    pass  # Package imports.
if str(ROOT / "scripts") not in sys.path:
    pass  # Package imports.

from gcs.core.data import create_data  # noqa: E402
from gcs.legacy.measure_equivariance import build_swap_pairs  # noqa: E402
from gcs.legacy.summarize_cross_prime_equivariance import (  # noqa: E402
    final_checkpoint,
    infer_from_path,
    load_config,
    load_model,
)
from gcs.legacy.diagnostics_common import RUNS_1L, resolve_path  # noqa: E402


SITE_KEYS = {
    "attn_out": "blocks.0.hook_attn_out",
    "resid_mid": "blocks.0.hook_resid_mid",
    "mlp_pre": "blocks.0.mlp.hook_pre",
    "mlp_post": "blocks.0.mlp.hook_post",
    "mlp_out": "blocks.0.hook_mlp_out",
    "resid_post": "blocks.0.hook_resid_post",
}


def discover_default_runs(include_p113_mul: bool) -> list[Path]:
    runs: list[Path] = []
    for pattern in [
        "experiments/cross_prime_comm_silu_20260426/*_1layer_mul_p97_*",
        "experiments/cross_prime_comm_silu_20260426/*_1layer_mul_p127_*",
    ]:
        runs.extend(ROOT.glob(pattern))
    if include_p113_mul:
        for act, task, _seed, path in RUNS_1L:
            if act == "SiLU" and task == "mul":
                runs.append(resolve_path(path))

    seen: set[Path] = set()
    unique: list[Path] = []
    for run in runs:
        run = run.resolve()
        if run in seen or not (run / "checkpoints").exists():
            continue
        seen.add(run)
        unique.append(run)
    return sorted(unique, key=lambda p: str(p))


def evaluate_accuracy(model, cfg: dict, device: str, batch_size: int) -> tuple[float, float]:
    train_data, test_data = create_data(
        p=int(cfg["p"]),
        fraction=float(cfg.get("fraction", 0.3)),
        seed=int(cfg.get("seed", 777)),
        device=device,
        task=str(cfg["task"]),
    )

    def acc(dataset) -> float:
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                prompts = torch.stack([dataset[i][0] for i in range(start, end)]).to(device)
                labels = torch.stack([dataset[i][1] for i in range(start, end)]).to(device)
                logits = model(prompts)[:, -1, : int(cfg["p"])]
                correct += int((logits.argmax(dim=-1) == labels).sum().item())
                total += int(labels.numel())
        return correct / total if total else float("nan")

    return acc(train_data), acc(test_data)


def batch_site_values(model, prompts: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    model.cache_all(cache)
    logits = model(prompts)
    model.remove_all_hooks()
    values = {site: cache[key][:, -1, :] for site, key in SITE_KEYS.items() if key in cache}
    values["logits"] = logits[:, -1, :]
    centered = values["logits"] - values["logits"].mean(dim=-1, keepdim=True)
    values["logits_centered"] = centered
    values["probabilities"] = torch.softmax(values["logits"], dim=-1)
    return values, logits


def measure_sites(model, p: int, device: str, batch_size: int) -> dict[str, dict[str, float]]:
    prompts_ab, prompts_ba = build_swap_pairs(p, device=device)
    totals: dict[str, dict[str, float]] = {}
    n = prompts_ab.shape[0]
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            values_ab, _ = batch_site_values(model, prompts_ab[start:end])
            values_ba, _ = batch_site_values(model, prompts_ba[start:end])
            for site, x_ab in values_ab.items():
                if site not in values_ba:
                    continue
                x_ba = values_ba[site]
                if site.startswith("logits") or site == "probabilities":
                    x_ab = x_ab[:, :p]
                    x_ba = x_ba[:, :p]
                diff = (x_ab - x_ba).norm(dim=-1)
                norm = x_ab.norm(dim=-1)
                item = totals.setdefault(site, {"diff_sum": 0.0, "norm_sum": 0.0, "count": 0.0})
                item["diff_sum"] += float(diff.sum().item())
                item["norm_sum"] += float(norm.sum().item())
                item["count"] += float(diff.numel())

    out: dict[str, dict[str, float]] = {}
    for site, item in totals.items():
        count = max(item["count"], 1.0)
        mean_diff = item["diff_sum"] / count
        mean_norm = item["norm_sum"] / count
        out[site] = {
            "mean_diff": mean_diff,
            "mean_norm": mean_norm,
            "relative_dequiv": mean_diff / mean_norm if mean_norm > 1e-12 else float("nan"),
        }
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--include-p113-mul", action="store_true")
    parser.add_argument("--run-dirs", default="", help="Comma-separated run directories")
    parser.add_argument("--output-dir", default="experiments/swap_symmetry_sites_20260429")
    args = parser.parse_args()

    explicit = [resolve_path(item) for item in args.run_dirs.split(",") if item]
    runs = explicit or discover_default_runs(include_p113_mul=args.include_p113_mul)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[dict] = []
    for run_dir in runs:
        try:
            cfg = infer_from_path(run_dir, load_config(run_dir))
            if str(cfg.get("task")) != "mul":
                continue
            model, epoch = load_model(run_dir, cfg, args.device)
            train_acc, test_acc = evaluate_accuracy(model, cfg, args.device, args.batch_size)
            site_stats = measure_sites(model, int(cfg["p"]), args.device, args.batch_size)
            ckpt_path = final_checkpoint(run_dir)
            print(
                f"p={int(cfg['p'])} seed={int(cfg.get('seed', -1))} "
                f"acc={test_acc:.3f} epoch={epoch}"
            )
            for site, stats in sorted(site_stats.items()):
                row = {
                    "run_dir": str(run_dir),
                    "checkpoint": str(ckpt_path),
                    "p": int(cfg["p"]),
                    "task": str(cfg["task"]),
                    "act": str(cfg.get("act_type", "")),
                    "seed": int(cfg.get("seed", -1)),
                    "final_epoch": int(epoch),
                    "train_acc": float(train_acc),
                    "test_acc": float(test_acc),
                    "site": site,
                    **stats,
                }
                rows.append(row)
                print(f"  {site:16s} rel={stats['relative_dequiv']:.4f}")
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "error": repr(exc)})
            print(f"FAIL {run_dir}: {exc}")

    write_csv(out_dir / "site_swap_symmetry.csv", rows)
    if failures:
        write_csv(out_dir / "failures.csv", failures)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "n_runs": len({row["run_dir"] for row in rows}),
                "n_rows": len(rows),
                "n_failures": len(failures),
                "sites": sorted({row["site"] for row in rows}),
                "output_csv": str(out_dir / "site_swap_symmetry.csv"),
            },
            f,
            indent=2,
        )
    print(f"Saved {out_dir / 'site_swap_symmetry.csv'}")


if __name__ == "__main__":
    main()
