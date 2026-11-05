#!/usr/bin/env python3
"""
Compare GCS trajectories against Neel Nanda's Fourier progress measures.

This script is intentionally scoped to the 1-layer modular-arithmetic
Transformer setting, where the task-specific Fourier probes are well defined.

Outputs per experiment:
  - comparison_timeseries.csv
  - comparison_summary.csv
  - comparison_summary.json
  - comparison_overview.png

Usage:
    python scripts/compare_gcs_nanda_measures.py \
        --experiment experiments/silu_1layer_add_s42_20260213_000145 \
        --device cuda:0

    # Smoke test on a subset of checkpoints
    python scripts/compare_gcs_nanda_measures.py \
        --experiment experiments/silu_1layer_add_s42_20260213_000145 \
        --max-checkpoints 5 \
        --device cuda:0
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

pass  # Package imports do not search the experiment-data directory.

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from gcs.core.data import create_data
from gcs.core.models import Transformer
from gcs.analysis.fourier import (
    calculate_key_freqs, sum_sq_weights, restricted_loss, excluded_loss,
    mean_cosine_coefficient, MEASURE_SOURCES,
)
from gcs.legacy.diagnostics_common import detect_grok_epoch, load_gcs, resolve_path
from gcs.legacy.extract_diagnostics import find_turning_point
from gcs.vis.config import PATH_COLORS, PLOT_CONFIG, setup_style


MEASURE_COLORS = {
    "overall_gcs": PATH_COLORS["overall"],
    "attention_gcs": PATH_COLORS["attention"],
    "mlp_gcs": PATH_COLORS["mlp"],
    "restricted": "#C44E52",
    "excluded": "#7A68A6",
    "weights": "#666666",
    "test_acc": "#111111",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GCS with canonical Neal Nanda progress measures.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment directory containing checkpoints/, training_metrics.pkl, and GCS results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device used for checkpoint analysis.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=200,
        help="Only analyze checkpoints at this epoch frequency.",
    )
    parser.add_argument(
        "--epoch-limit",
        type=int,
        default=None,
        help="Optional upper epoch bound for partial runs / smoke tests.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Optional cap on analyzed checkpoints after filtering.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="nanda_progress_comparison",
        help="Directory name created under the experiment root for outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Compute metrics and summaries without generating PNGs.",
    )
    return parser.parse_args()


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def infer_seed(name: str) -> int:
    match = re.search(r"_s(\d+)", name)
    return int(match.group(1)) if match else 777


def infer_attention_rate(name: str) -> float:
    match = re.search(r"_a([0-9]+(?:\.[0-9]+)?)", name)
    if match:
        return float(match.group(1))
    if "pizza" in name.lower():
        return 0.0
    return 1.0


def load_experiment_context(exp_dir: Path) -> Tuple[dict, dict]:
    metrics_path = exp_dir / "training_metrics.pkl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing training metrics: {metrics_path}")

    metrics = load_pickle(metrics_path)
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected dict metrics at {metrics_path}, got {type(metrics)}")

    config = metrics.get("config")
    if isinstance(config, dict):
        return config, metrics

    config_path = exp_dir / "config.pkl"
    if config_path.exists():
        config = load_pickle(config_path)
        if isinstance(config, dict):
            return config, metrics

    # Fallback for legacy checkpoint directories without config.pkl.
    exp_name = exp_dir.name
    config = {
        "p": 113,
        "fraction": 0.3,
        "seed": infer_seed(exp_name),
        "task": metrics.get("task_name", "add"),
        "num_layers": int(metrics.get("num_layers", 1)),
        "d_model": 128,
        "d_mlp": 512,
        "d_head": 32,
        "num_heads": 4,
        "n_ctx": 3,
        "act_type": metrics.get("activation_function", "SiLU"),
        "use_cache": False,
        "use_ln": False,
        "attention_rate": infer_attention_rate(exp_name),
        "record_frequency": 200,
    }
    return config, metrics


def extract_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def load_model_from_checkpoint(checkpoint_path: Path, config: dict, device: str) -> Transformer:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)

    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")

    d_model, d_vocab = state["embed.W_E"].shape
    mlp_key = next(k for k in state if k.endswith("mlp.W_in"))
    attn_key = next(k for k in state if k.endswith("attn.W_Q"))
    d_mlp, _ = state[mlp_key].shape
    num_heads, d_head, _ = state[attn_key].shape

    layer_indices = []
    for key in state:
        match = re.search(r"blocks\.(\d+)\.attn\.W_Q", key)
        if match:
            layer_indices.append(int(match.group(1)))
    num_layers = (max(layer_indices) + 1) if layer_indices else int(config.get("num_layers", 1))

    model = Transformer(
        num_layers=num_layers,
        d_vocab=d_vocab,
        d_model=d_model,
        d_mlp=d_mlp,
        d_head=d_head,
        num_heads=num_heads,
        n_ctx=int(config.get("n_ctx", 3)),
        act_type=config.get("act_type", "SiLU"),
        use_cache=False,
        use_ln=bool(config.get("use_ln", False)),
        attention_rate=float(config.get("attention_rate", 1.0)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def build_full_dataset(config: dict) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    train_data, _ = create_data(
        p=int(config.get("p", 113)),
        fraction=float(config.get("fraction", 0.3)),
        seed=int(config.get("seed", 777)),
        device="cpu",
        task=config.get("task", "add"),
    )

    base_dataset = train_data.dataset
    prompts, labels = base_dataset.tensors
    train_mask = np.zeros(len(base_dataset), dtype=bool)
    train_mask[np.asarray(train_data.indices, dtype=int)] = True
    test_mask = ~train_mask
    return prompts, labels, train_mask, test_mask


def select_checkpoints(
    checkpoint_dir: Path,
    checkpoint_freq: int,
    epoch_limit: Optional[int],
    max_checkpoints: Optional[int],
) -> List[Path]:
    all_ckpts = sorted(checkpoint_dir.glob("*.pth"), key=extract_epoch)
    selected = []
    for path in all_ckpts:
        epoch = extract_epoch(path)
        if epoch < 0:
            continue
        if checkpoint_freq and (epoch % checkpoint_freq != 0):
            continue
        if epoch_limit is not None and epoch > epoch_limit:
            continue
        selected.append(path)

    if max_checkpoints is not None:
        selected = selected[:max_checkpoints]
    return selected


def minmax_progress(values: Sequence[float], invert: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out

    valid_arr = arr[valid]
    mn = float(np.min(valid_arr))
    mx = float(np.max(valid_arr))
    if mx - mn < 1e-12:
        out[valid] = 0.0
    else:
        out[valid] = (valid_arr - mn) / (mx - mn)

    if invert:
        out[valid] = 1.0 - out[valid]
    return out


def align_to_epochs(epochs: Sequence[int], epoch_to_value: Dict[int, float]) -> np.ndarray:
    return np.asarray([epoch_to_value.get(int(epoch), np.nan) for epoch in epochs], dtype=float)


def extract_metric_map(metrics: dict, key: str) -> Dict[int, float]:
    steps = metrics.get("log_steps", [])
    values = metrics.get(key, [])
    return {
        int(step): float(value)
        for step, value in zip(steps, values)
    }


def build_config_namespace(config: dict, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        p=int(config.get("p", 113)),
        d_mlp=int(config.get("d_mlp", 512)),
        device=device,
    )


def compute_nanda_trajectory(
    model: Transformer,
    checkpoints: Sequence[Path],
    config: dict,
    all_data_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    device: str,
) -> Dict[str, List[float]]:
    all_data = all_data_cpu.to(device)
    labels = labels_cpu.to(device)
    cfg = build_config_namespace(config, device)

    epochs: List[int] = []
    restricted_losses: List[float] = []
    excluded_losses: List[float] = []
    sumsq_weights: List[float] = []
    coeff_means: List[float] = []
    key_freq_counts: List[int] = []
    failed_epochs: List[int] = []

    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        epoch = extract_epoch(checkpoint_path)
        print(f"[{idx:03d}/{len(checkpoints):03d}] epoch={epoch}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state)
            model.eval()

            with torch.inference_mode():
                key_freqs = np.asarray(calculate_key_freqs(cfg, model, all_data, device), dtype=int)
                logits = model(all_data)[:, -1, :-1]

                restricted_value = restricted_loss(logits, cfg.p, key_freqs,
                    train_mask, test_mask, labels, mode="all")
                excluded_value = excluded_loss(logits, cfg.p, key_freqs,
                    train_mask, test_mask, labels)
                coeff_mean = mean_cosine_coefficient(logits, cfg.p).item()
                sumsq = sum_sq_weights(model)

            epochs.append(epoch)
            restricted_losses.append(
                float(restricted_value.item())
            )
            excluded_losses.append(
                float(excluded_value.item())
            )
            sumsq_weights.append(float(sumsq))
            coeff_means.append(float(coeff_mean))
            key_freq_counts.append(int(len(key_freqs)))
        except Exception as exc:
            print(f"  FAILED epoch {epoch}: {exc}")
            failed_epochs.append(epoch)

    return {
        "epochs": epochs,
        "restricted_loss": restricted_losses,
        "excluded_loss_2d_full": excluded_losses,
        "sum_sq_weights": sumsq_weights,
        "coeff_mean": coeff_means,
        "key_freq_count": key_freq_counts,
        "failed_epochs": failed_epochs,
        "sources": MEASURE_SOURCES,
    }


def build_aligned_timeseries(
    metrics: dict,
    gcs_epochs: np.ndarray,
    gcs_scores: dict,
    nanda: dict,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    metric_maps = {
        "test_accuracy": extract_metric_map(metrics, "test_accuracies"),
        "train_accuracy": extract_metric_map(metrics, "train_accuracies"),
        "test_loss": extract_metric_map(metrics, "test_losses"),
        "train_loss": extract_metric_map(metrics, "train_losses"),
    }
    gcs_maps = {
        "overall_gcs_raw": {
            int(epoch): float(score)
            for epoch, score in zip(gcs_epochs, gcs_scores["overall_path"])
        },
        "attention_gcs_raw": {
            int(epoch): float(score)
            for epoch, score in zip(gcs_epochs, gcs_scores["attention_path"])
        },
        "mlp_gcs_raw": {
            int(epoch): float(score)
            for epoch, score in zip(gcs_epochs, gcs_scores["mlp_path"])
        },
    }
    nanda_maps = {
        "restricted_loss_raw": {
            int(epoch): float(value)
            for epoch, value in zip(nanda["epochs"], nanda["restricted_loss"])
        },
        "trig_loss_raw": {
            int(epoch): float(value)
            for epoch, value in zip(nanda["epochs"], nanda["restricted_loss"])
        },
        "excluded_loss_2d_full_raw": {
            int(epoch): float(value)
            for epoch, value in zip(nanda["epochs"], nanda["excluded_loss_2d_full"])
        },
        "excluded_loss_raw": {
            int(epoch): float(value)
            for epoch, value in zip(nanda["epochs"], nanda["excluded_loss_2d_full"])
        },
        "sum_sq_weights_raw": {
            int(epoch): float(value)
            for epoch, value in zip(nanda["epochs"], nanda["sum_sq_weights"])
        },
        "coeff_mean_raw": {
            int(epoch): float(value)
            for epoch, value in zip(nanda["epochs"], nanda["coeff_mean"])
        },
        "key_freq_count": {
            int(epoch): float(value)
            for epoch, value in zip(nanda["epochs"], nanda["key_freq_count"])
        },
    }

    common_epochs = sorted(
        set(int(x) for x in gcs_epochs)
        & set(metric_maps["test_accuracy"].keys())
        & set(int(x) for x in nanda["epochs"])
    )
    epochs = np.asarray(common_epochs, dtype=int)
    if epochs.size == 0:
        raise ValueError("No common epochs across GCS, training metrics, and Nanda measures.")

    series: Dict[str, np.ndarray] = {}
    for name, mapping in metric_maps.items():
        series[name] = align_to_epochs(epochs, mapping)
    for name, mapping in gcs_maps.items():
        series[name] = align_to_epochs(epochs, mapping)
    for name, mapping in nanda_maps.items():
        series[name] = align_to_epochs(epochs, mapping)

    series["overall_gcs_progress"] = minmax_progress(series["overall_gcs_raw"])
    series["attention_gcs_progress"] = minmax_progress(series["attention_gcs_raw"])
    series["mlp_gcs_progress"] = minmax_progress(series["mlp_gcs_raw"])
    series["restricted_progress"] = minmax_progress(series["restricted_loss_raw"], invert=True)
    series["excluded_progress"] = minmax_progress(series["excluded_loss_2d_full_raw"])
    series["weight_progress"] = minmax_progress(series["sum_sq_weights_raw"], invert=True)
    return epochs, series


def to_optional_int(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def compute_turns(epochs: np.ndarray, series: Dict[str, np.ndarray]) -> Dict[str, Optional[int]]:
    turns: Dict[str, Optional[int]] = {}
    reasons: Dict[str, Optional[str]] = {}
    for name, values in (
        ("overall_gcs", series["overall_gcs_progress"]),
        ("attention_gcs", series["attention_gcs_progress"]),
        ("mlp_gcs", series["mlp_gcs_progress"]),
        ("restricted", series["restricted_progress"]),
        ("excluded", series["excluded_progress"]),
        ("weights", series["weight_progress"]),
    ):
        epoch, reason = find_turning_point(epochs.tolist(), values.tolist())
        turns[name] = to_optional_int(epoch)
        reasons[f"{name}_reason"] = reason
    turns.update(reasons)
    return turns


def save_timeseries_csv(path: Path, epochs: np.ndarray, series: Dict[str, np.ndarray]) -> None:
    fieldnames = ["epoch"] + list(series.keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, epoch in enumerate(epochs):
            row = {"epoch": int(epoch)}
            for key, values in series.items():
                value = values[idx]
                row[key] = "" if not np.isfinite(value) else float(value)
            writer.writerow(row)


def save_summary_csv(path: Path, summary: dict) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def plot_overview(
    save_path: Path,
    exp_name: str,
    epochs: np.ndarray,
    series: Dict[str, np.ndarray],
    summary: dict,
    dpi: int,
) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel A: overall GCS vs canonical task-specific measures.
    ax = axes[0]
    ax.plot(
        epochs,
        series["overall_gcs_progress"],
        color=MEASURE_COLORS["overall_gcs"],
        linewidth=PLOT_CONFIG["linewidth"],
        label="Overall GCS",
    )
    ax.plot(
        epochs,
        series["restricted_progress"],
        color=MEASURE_COLORS["restricted"],
        linewidth=PLOT_CONFIG["linewidth_thin"],
        label="Restricted Progress",
    )
    ax.plot(
        epochs,
        series["excluded_progress"],
        color=MEASURE_COLORS["excluded"],
        linewidth=PLOT_CONFIG["linewidth_thin"],
        label="Excluded Progress",
    )
    ax.plot(
        epochs,
        series["weight_progress"],
        color=MEASURE_COLORS["weights"],
        linewidth=PLOT_CONFIG["linewidth_thin"],
        label="Weight Progress",
    )
    ax.set_ylabel("Normalized progress")
    ax.set_ylim(-0.05, 1.05)

    ax_acc = ax.twinx()
    ax_acc.plot(
        epochs,
        series["test_accuracy"],
        color=MEASURE_COLORS["test_acc"],
        linewidth=PLOT_CONFIG["linewidth_thin"],
        linestyle="--",
        label="Test accuracy",
    )
    ax_acc.set_ylabel("Test accuracy")
    ax_acc.set_ylim(-0.05, 1.05)

    # Panel B: 3-path GCS view.
    ax = axes[1]
    ax.plot(
        epochs,
        series["overall_gcs_progress"],
        color=MEASURE_COLORS["overall_gcs"],
        linewidth=PLOT_CONFIG["linewidth"],
        label="Overall GCS",
    )
    ax.plot(
        epochs,
        series["attention_gcs_progress"],
        color=MEASURE_COLORS["attention_gcs"],
        linewidth=PLOT_CONFIG["linewidth_thin"],
        label="Attention GCS",
    )
    ax.plot(
        epochs,
        series["mlp_gcs_progress"],
        color=MEASURE_COLORS["mlp_gcs"],
        linewidth=PLOT_CONFIG["linewidth_thin"],
        label="MLP GCS",
    )
    ax.set_ylabel("Normalized progress")
    ax.set_xlabel("Epoch")
    ax.set_ylim(-0.05, 1.05)

    ax_acc2 = ax.twinx()
    ax_acc2.plot(
        epochs,
        series["test_accuracy"],
        color=MEASURE_COLORS["test_acc"],
        linewidth=PLOT_CONFIG["linewidth_thin"],
        linestyle="--",
        label="Test accuracy",
    )
    ax_acc2.set_ylabel("Test accuracy")
    ax_acc2.set_ylim(-0.05, 1.05)

    grok_epoch = summary.get("grok_epoch")
    marker_specs = [
        ("overall_turn_epoch", MEASURE_COLORS["overall_gcs"]),
        ("restricted_turn_epoch", MEASURE_COLORS["restricted"]),
        ("excluded_turn_epoch", MEASURE_COLORS["excluded"]),
    ]
    for marker_key, color in marker_specs:
        marker_epoch = summary.get(marker_key)
        if marker_epoch is not None:
            axes[0].axvline(marker_epoch, color=color, linestyle=":", alpha=0.6)

    path_marker_specs = [
        ("overall_turn_epoch", MEASURE_COLORS["overall_gcs"]),
        ("attention_turn_epoch", MEASURE_COLORS["attention_gcs"]),
        ("mlp_turn_epoch", MEASURE_COLORS["mlp_gcs"]),
    ]
    for marker_key, color in path_marker_specs:
        marker_epoch = summary.get(marker_key)
        if marker_epoch is not None:
            axes[1].axvline(marker_epoch, color=color, linestyle=":", alpha=0.6)

    if grok_epoch is not None:
        for panel in axes:
            panel.axvline(grok_epoch, color="black", linestyle="--", alpha=0.7)

    lines_a, labels_a = axes[0].get_legend_handles_labels()
    lines_b, labels_b = ax_acc.get_legend_handles_labels()
    axes[0].legend(lines_a + lines_b, labels_a + labels_b, loc="best")

    lines_c, labels_c = axes[1].get_legend_handles_labels()
    lines_d, labels_d = ax_acc2.get_legend_handles_labels()
    axes[1].legend(lines_c + lines_d, labels_c + labels_d, loc="best")

    axes[0].set_title("Overall GCS vs canonical task-specific measures")
    axes[1].set_title("GCS path decomposition")
    fig.suptitle(exp_name, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    exp_dir: Path,
    config: dict,
    epochs: np.ndarray,
    series: Dict[str, np.ndarray],
    nanda: dict,
    turns: Dict[str, Optional[int]],
    metrics: dict,
) -> dict:
    grok_epoch = detect_grok_epoch(metrics)
    summary = {
        "experiment": exp_dir.name,
        "exp_dir": str(exp_dir),
        "task": config.get("task", "add"),
        "activation": config.get("act_type", "unknown"),
        "seed": int(config.get("seed", infer_seed(exp_dir.name))),
        "p": int(config.get("p", 113)),
        "attention_rate": float(config.get("attention_rate", infer_attention_rate(exp_dir.name))),
        "num_layers": int(config.get("num_layers", 1)),
        "n_common_epochs": int(len(epochs)),
        "grok_epoch": to_optional_int(grok_epoch),
        "overall_turn_epoch": turns.get("overall_gcs"),
        "attention_turn_epoch": turns.get("attention_gcs"),
        "mlp_turn_epoch": turns.get("mlp_gcs"),
        "restricted_turn_epoch": turns.get("restricted"),
        "excluded_turn_epoch": turns.get("excluded"),
        "weight_turn_epoch": turns.get("weights"),
        "overall_turn_reason": turns.get("overall_gcs_reason"),
        "attention_turn_reason": turns.get("attention_gcs_reason"),
        "mlp_turn_reason": turns.get("mlp_gcs_reason"),
        "restricted_turn_reason": turns.get("restricted_reason"),
        "excluded_turn_reason": turns.get("excluded_reason"),
        "weight_turn_reason": turns.get("weights_reason"),
        "lead_overall_vs_grok": None,
        "lead_restricted_vs_grok": None,
        "lead_excluded_vs_grok": None,
        "attn_vs_mlp_gap": None,
        "failed_epochs": ",".join(str(x) for x in nanda.get("failed_epochs", [])),
        "avg_key_freq_count": float(np.nanmean(series["key_freq_count"])),
        "final_test_accuracy": float(series["test_accuracy"][-1]),
        "restricted_measure_source": nanda["sources"]["restricted_loss"],
        "excluded_measure_source": nanda["sources"]["excluded_loss_2d_full"],
    }

    if summary["grok_epoch"] is not None and summary["overall_turn_epoch"] is not None:
        summary["lead_overall_vs_grok"] = int(summary["grok_epoch"] - summary["overall_turn_epoch"])
    if summary["grok_epoch"] is not None and summary["restricted_turn_epoch"] is not None:
        summary["lead_restricted_vs_grok"] = int(summary["grok_epoch"] - summary["restricted_turn_epoch"])
    if summary["grok_epoch"] is not None and summary["excluded_turn_epoch"] is not None:
        summary["lead_excluded_vs_grok"] = int(summary["grok_epoch"] - summary["excluded_turn_epoch"])
    if summary["attention_turn_epoch"] is not None and summary["mlp_turn_epoch"] is not None:
        summary["attn_vs_mlp_gap"] = int(summary["attention_turn_epoch"] - summary["mlp_turn_epoch"])
    return summary


def main() -> int:
    args = parse_args()
    exp_dir = resolve_path(args.experiment)
    output_dir = exp_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    config, metrics = load_experiment_context(exp_dir)
    if int(config.get("num_layers", 1)) != 1:
        raise ValueError("This comparison is only defined for the canonical 1-layer setting.")
    if config.get("task", "add") != "add":
        raise ValueError("This comparison currently expects modular addition experiments.")

    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    checkpoints = select_checkpoints(
        checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        epoch_limit=args.epoch_limit,
        max_checkpoints=args.max_checkpoints,
    )
    if not checkpoints:
        raise ValueError("No checkpoints selected for analysis.")

    print(f"Experiment: {exp_dir}")
    print(f"Selected checkpoints: {len(checkpoints)}")
    print(f"Output dir: {output_dir}")

    model = load_model_from_checkpoint(checkpoints[0], config, args.device)
    all_data_cpu, labels_cpu, train_mask, test_mask = build_full_dataset(config)
    gcs_epochs, gcs_scores = load_gcs(exp_dir)

    nanda = compute_nanda_trajectory(
        model=model,
        checkpoints=checkpoints,
        config=config,
        all_data_cpu=all_data_cpu,
        labels_cpu=labels_cpu,
        train_mask=train_mask,
        test_mask=test_mask,
        device=args.device,
    )

    epochs, series = build_aligned_timeseries(metrics, gcs_epochs, gcs_scores, nanda)
    turns = compute_turns(epochs, series)
    summary = build_summary(exp_dir, config, epochs, series, nanda, turns, metrics)

    save_timeseries_csv(output_dir / "comparison_timeseries.csv", epochs, series)
    save_summary_csv(output_dir / "comparison_summary.csv", summary)
    with open(output_dir / "comparison_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if not args.skip_plots:
        plot_overview(
            save_path=output_dir / "comparison_overview.png",
            exp_name=exp_dir.name,
            epochs=epochs,
            series=series,
            summary=summary,
            dpi=args.dpi,
        )

    print("\nSummary:")
    for key in (
        "grok_epoch",
        "overall_turn_epoch",
        "restricted_turn_epoch",
        "excluded_turn_epoch",
        "attention_turn_epoch",
        "mlp_turn_epoch",
        "lead_overall_vs_grok",
        "lead_restricted_vs_grok",
        "lead_excluded_vs_grok",
        "attn_vs_mlp_gap",
    ):
        print(f"  {key}: {summary.get(key)}")

    print(f"\nSaved outputs under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
