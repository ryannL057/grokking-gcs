#!/usr/bin/env python3
"""Single-run circuit-level audit for the SiLU/addition workflow example.

The goal is not endpoint circuit reconstruction.  It is a compact audit of the
window nominated by GCS: along the selected trajectory checkpoints, measure
readout-position attention-head swap symmetry, head-output symmetry,
stage-level swap distances, and the existing Fourier-progress readout.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import numpy as np
import torch

ROOT = _gcs_data_root()
if str(ROOT) not in sys.path:
    pass  # Package imports do not search the experiment-data directory.
if str(ROOT / "scripts") not in sys.path:
    pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer  # noqa: E402
from gcs.legacy.measure_equivariance import build_swap_pairs  # noqa: E402
from gcs.legacy.diagnostics_common import load_gcs, load_metrics, resolve_path  # noqa: E402


SITE_KEYS = {
    "attn_out": "blocks.0.hook_attn_out",
    "mlp_out": "blocks.0.hook_mlp_out",
    "resid_post": "blocks.0.hook_resid_post",
}


def checkpoint_epoch(path: Path) -> int:
    m = re.search(r"epoch_(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse epoch from {path}")
    return int(m.group(1))


def nearest_checkpoint(run_dir: Path, target_epoch: int) -> Path:
    ckpts = sorted((run_dir / "checkpoints").glob("*.pth"), key=checkpoint_epoch)
    epochs = np.asarray([checkpoint_epoch(path) for path in ckpts])
    idx = int(np.argmin(np.abs(epochs - target_epoch)))
    return ckpts[idx]


def load_model(ckpt_path: Path, device: str) -> tuple[Transformer, dict, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", ckpt.get("config", {}))
    model = Transformer(
        num_layers=int(cfg.get("num_layers", 1)),
        d_vocab=int(cfg.get("d_vocab", 114)),
        d_model=int(cfg.get("d_model", 128)),
        d_mlp=int(cfg.get("d_mlp", 512)),
        d_head=int(cfg.get("d_head", 32)),
        num_heads=int(cfg.get("num_heads", 4)),
        n_ctx=int(cfg.get("n_ctx", 3)),
        act_type=str(cfg.get("act_type", "SiLU")),
        use_cache=False,
        use_ln=bool(cfg.get("use_ln", False)),
        attention_rate=float(cfg.get("attention_rate", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, int(ckpt.get("epoch", checkpoint_epoch(ckpt_path)))


def relative_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    diff = (x - y).norm(dim=-1)
    denom = x.norm(dim=-1) + y.norm(dim=-1) + 1e-12
    return float((diff / denom).mean().item())


def similarity_from_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    return 1.0 - relative_distance(x, y)


def run_with_cache(model: Transformer, prompts: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    model.cache_all(cache, incl_bwd=False)
    with torch.no_grad():
        logits = model(prompts)
    model.remove_all_hooks()
    return cache, logits


def measure_epoch(model: Transformer, p: int, device: str, batch_size: int) -> dict:
    prompts_ab, prompts_ba = build_swap_pairs(p, device=device)
    n = prompts_ab.shape[0]
    site_diff_sum: dict[str, float] = {}
    site_count: dict[str, int] = {}
    row_sim_sums: np.ndarray | None = None
    head_out_dist_sums: np.ndarray | None = None
    batches = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        cache_ab, logits_ab = run_with_cache(model, prompts_ab[start:end])
        cache_ba, logits_ba = run_with_cache(model, prompts_ba[start:end])

        for site, key in SITE_KEYS.items():
            if key not in cache_ab or key not in cache_ba:
                continue
            value_ab = cache_ab[key][:, -1, :]
            value_ba = cache_ba[key][:, -1, :]
            site_diff_sum[site] = site_diff_sum.get(site, 0.0) + relative_distance(value_ab, value_ba) * (end - start)
            site_count[site] = site_count.get(site, 0) + (end - start)

        centered_ab = logits_ab[:, -1, :p] - logits_ab[:, -1, :p].mean(dim=-1, keepdim=True)
        centered_ba = logits_ba[:, -1, :p] - logits_ba[:, -1, :p].mean(dim=-1, keepdim=True)
        probs_ab = torch.softmax(logits_ab[:, -1, :p], dim=-1)
        probs_ba = torch.softmax(logits_ba[:, -1, :p], dim=-1)
        for site, value_ab, value_ba in [
            ("logits_centered", centered_ab, centered_ba),
            ("probabilities", probs_ab, probs_ba),
        ]:
            site_diff_sum[site] = site_diff_sum.get(site, 0.0) + relative_distance(value_ab, value_ba) * (end - start)
            site_count[site] = site_count.get(site, 0) + (end - start)

        attn_ab = cache_ab["blocks.0.attn.hook_attn"][:, :, -1, :]  # [B, H, key]
        attn_ba = cache_ba["blocks.0.attn.hook_attn"][:, :, -1, :]
        attn_ab_swapped = attn_ab[:, :, [1, 0, 2]]
        n_heads = attn_ab.shape[1]
        row_sims = []
        for h in range(n_heads):
            row_sims.append(similarity_from_distance(attn_ab_swapped[:, h, :], attn_ba[:, h, :]))
        row_sims = np.asarray(row_sims, dtype=float)
        row_sim_sums = row_sims if row_sim_sums is None else row_sim_sums + row_sims

        z_ab = cache_ab["blocks.0.attn.hook_z"][:, :, -1, :]  # [B, H, d_head]
        z_ba = cache_ba["blocks.0.attn.hook_z"][:, :, -1, :]
        W_O = model.blocks[0].attn.W_O.detach()
        d_head = z_ab.shape[-1]
        head_dists = []
        for h in range(n_heads):
            lo, hi = h * d_head, (h + 1) * d_head
            W = W_O[:, lo:hi]
            out_ab = torch.einsum("bh,dh->bd", z_ab[:, h, :], W)
            out_ba = torch.einsum("bh,dh->bd", z_ba[:, h, :], W)
            head_dists.append(relative_distance(out_ab, out_ba))
        head_dists = np.asarray(head_dists, dtype=float)
        head_out_dist_sums = head_dists if head_out_dist_sums is None else head_out_dist_sums + head_dists
        batches += 1

    site_dist = {site: site_diff_sum[site] / max(site_count[site], 1) for site in site_diff_sum}
    return {
        "site_relative_dequiv": site_dist,
        "head_readout_attn_swap_similarity": (row_sim_sums / max(batches, 1)).tolist(),
        "head_output_relative_dequiv": (head_out_dist_sums / max(batches, 1)).tolist(),
    }


def load_fourier_timeseries(run_dir: Path) -> dict[int, dict]:
    path = run_dir / "nanda_progress_comparison" / "comparison_timeseries.csv"
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(float(row["epoch"]))
            out[epoch] = row
    return out


def nearest_row(rows: dict[int, dict], epoch: int) -> dict:
    if not rows:
        return {}
    keys = np.asarray(sorted(rows))
    nearest = int(keys[int(np.argmin(np.abs(keys - epoch)))])
    return rows[nearest]


def metric_at(metrics: dict, epoch: int, name: str) -> float:
    steps = np.asarray(metrics["log_steps"], dtype=int)
    values = np.asarray(metrics[name], dtype=float)
    idx = int(np.argmin(np.abs(steps - epoch)))
    return float(values[idx])


def gcs_at(gcs_epochs: np.ndarray, scores: dict, epoch: int, key: str) -> float:
    idx = int(np.argmin(np.abs(gcs_epochs.astype(int) - epoch)))
    return float(np.asarray(scores[key], dtype=float)[idx])


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(out_path: Path, rows: list[dict], head_rows: list[dict], full_ts: dict[str, np.ndarray]) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9))
    ax = axes[0, 0]
    epochs = full_ts["epochs"]
    ax.plot(epochs, full_ts["attn_gcs_norm"], color="#E69F00", lw=1.6, label="Attention GCS")
    ax.plot(epochs, full_ts["mlp_gcs_norm"], color="#009E73", lw=1.6, label="MLP GCS")
    ax2 = ax.twinx()
    ax2.plot(epochs, full_ts["test_acc"], color="0.35", lw=1.1, ls="--", label="Test acc")
    for row in rows:
        ax.axvline(row["epoch"], color="0.65", lw=0.8, ls=":")
    ax.set_title("(a) GCS-nominated window")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized GCS")
    ax2.set_ylabel("Test acc")
    ax.set_ylim(-0.04, 1.04)
    ax2.set_ylim(-0.04, 1.04)
    ax.legend(loc="lower left", frameon=False)

    ax = axes[0, 1]
    x = np.arange(len(rows))
    labels = [row["label"] for row in rows]
    for site, color in [
        ("attn_out_dequiv", "#E69F00"),
        ("mlp_out_dequiv", "#009E73"),
        ("logits_centered_dequiv", "#0072B2"),
        ("probabilities_dequiv", "#CC79A7"),
    ]:
        ax.plot(x, [row[site] for row in rows], marker="o", lw=1.5, color=color, label=site.replace("_dequiv", ""))
    ax.set_title("(b) Swap distance moves downstream")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Relative swap distance")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 0]
    head_mat = np.array([[hr[f"head{h}_readout_attn_sim"] for h in range(4)] for hr in head_rows]).T
    im = ax.imshow(head_mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_title("(c) Readout attention pattern symmetry")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"H{h}" for h in range(4)])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Swap similarity")

    ax = axes[1, 1]
    for h, color in enumerate(["#E69F00", "#56B4E9", "#009E73", "#D55E00"]):
        ax.plot(x, [hr[f"head{h}_output_dequiv"] for hr in head_rows], marker="o", lw=1.3, color=color, label=f"H{h}")
    ax2 = ax.twinx()
    ax2.plot(x, [row["restricted_progress"] for row in rows], color="0.25", marker="s", ls="--", lw=1.2, label="Fourier progress")
    ax.set_title("(d) Head-output distance and Fourier progress")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Head output swap distance")
    ax2.set_ylabel("Restricted Fourier progress")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax2.legend(frameon=False, loc="lower right")

    for axis in axes.ravel():
        axis.grid(alpha=0.18, lw=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="checkpoints/1layer/SiLU/add")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", default="experiments/circuit_case_study_addition_20260429")
    parser.add_argument("--figure-out", default="latex_paper/Formatting_Instructions_For_NeurIPS_2026/figures/circuit_case_study_addition.png")
    parser.add_argument("--epochs", default="400,10000,10600,11200,19999")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch.set_default_dtype(torch.float64)

    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_epochs = [int(item) for item in args.epochs.split(",") if item]
    labels = ["valley", "window-start", "peak", "window-end", "final"][: len(target_epochs)]

    metrics = load_metrics(run_dir)
    gcs_epochs, gcs_scores = load_gcs(run_dir)
    fourier_rows = load_fourier_timeseries(run_dir)

    rows: list[dict] = []
    head_rows: list[dict] = []
    for label, target_epoch in zip(labels, target_epochs):
        ckpt_path = nearest_checkpoint(run_dir, target_epoch)
        model, cfg, epoch = load_model(ckpt_path, device)
        p = int(cfg.get("d_vocab", 114)) - 1
        stats = measure_epoch(model, p, device, args.batch_size)
        frow = nearest_row(fourier_rows, epoch)
        site = stats["site_relative_dequiv"]
        row = {
            "label": label,
            "epoch": epoch,
            "test_accuracy": metric_at(metrics, epoch, "test_accuracies"),
            "train_accuracy": metric_at(metrics, epoch, "train_accuracies"),
            "attention_gcs": gcs_at(gcs_epochs, gcs_scores, epoch, "attention_path"),
            "mlp_gcs": gcs_at(gcs_epochs, gcs_scores, epoch, "mlp_path"),
            "restricted_progress": float(frow.get("restricted_progress", "nan")),
            "key_freq_count": float(frow.get("key_freq_count", "nan")),
            "attn_out_dequiv": site.get("attn_out", float("nan")),
            "mlp_out_dequiv": site.get("mlp_out", float("nan")),
            "resid_post_dequiv": site.get("resid_post", float("nan")),
            "logits_centered_dequiv": site.get("logits_centered", float("nan")),
            "probabilities_dequiv": site.get("probabilities", float("nan")),
        }
        rows.append(row)
        hrow = {"label": label, "epoch": epoch}
        for h, value in enumerate(stats["head_readout_attn_swap_similarity"]):
            hrow[f"head{h}_readout_attn_sim"] = float(value)
        for h, value in enumerate(stats["head_output_relative_dequiv"]):
            hrow[f"head{h}_output_dequiv"] = float(value)
        head_rows.append(hrow)
        print(
            f"{label:12s} epoch={epoch:5d} acc={row['test_accuracy']:.3f} "
            f"attn_dequiv={row['attn_out_dequiv']:.4f} logits={row['logits_centered_dequiv']:.4f}"
        )

    write_csv(out_dir / "case_study_sites.csv", rows)
    write_csv(out_dir / "case_study_heads.csv", head_rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "target_epochs": target_epochs,
                "actual_epochs": [row["epoch"] for row in rows],
                "rows_csv": str(out_dir / "case_study_sites.csv"),
                "heads_csv": str(out_dir / "case_study_heads.csv"),
            },
            f,
            indent=2,
        )

    ts_epochs = np.asarray(gcs_epochs, dtype=float)
    attn = np.asarray(gcs_scores["attention_path"], dtype=float)
    mlp = np.asarray(gcs_scores["mlp_path"], dtype=float)
    def norm(v: np.ndarray) -> np.ndarray:
        return (v - v.min()) / max(v.max() - v.min(), 1e-12)
    metric_steps = np.asarray(metrics["log_steps"], dtype=float)
    metric_acc = np.asarray(metrics["test_accuracies"], dtype=float)
    full_ts = {
        "epochs": ts_epochs,
        "attn_gcs_norm": norm(attn),
        "mlp_gcs_norm": norm(mlp),
        "test_acc": np.interp(ts_epochs, metric_steps, metric_acc),
    }
    make_figure(resolve_path(args.figure_out), rows, head_rows, full_ts)
    print(f"Saved {out_dir / 'case_study_sites.csv'}")
    print(f"Saved {out_dir / 'case_study_heads.csv'}")
    print(f"Saved {resolve_path(args.figure_out)}")


if __name__ == "__main__":
    main()
