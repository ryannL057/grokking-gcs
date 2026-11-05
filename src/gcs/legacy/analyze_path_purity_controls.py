#!/usr/bin/env python3
"""
Analyze path-purity controls for Transformer GCS without modifying core scripts.

This script reuses existing checkpoints and final-geometry data, but defines a
small family of local control paths in-script:
  - attention_path
  - mlp_path
  - overall_path
  - mlp_pure_output_control
  - residual_post_detached_attn_control
  - mlp_zero_attn_control (optional)

The goal is to test whether the visible non-monotonic path shapes survive after
cutting attention-side gradient flow inside the target block.
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
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.analysis.geometry import build_complete_geometric_structure
from gcs.analysis.gmn_tsa import (
    compute_alignment_score_cosine,
    compute_alignment_score_procrustes,
)
from gcs.core.constants import EIGENVALUE_THRESHOLD
from gcs.core.data import create_data
from gcs.core.models import Transformer


PATH_VARIANTS = {
    "attention_path": "Original attention path: embedding -> attn_out",
    "mlp_path": "Original MLP path: embedding -> mlp_out through attention then MLP",
    "overall_path": "Original whole-block path: embedding -> resid_post",
    "mlp_pure_output_control": "MLP output with attention output detached before resid_mid",
    "residual_post_detached_attn_control": "Whole-block resid_post with attention output detached before resid_mid",
    "mlp_zero_attn_control": "MLP output with attention contribution zeroed before resid_mid",
}


class CPUUnpickler(pickle.Unpickler):
    """Allow loading GPU-created pickles onto CPU/GPU transparently."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return CPUUnpickler(handle).load()


def save_pickle(path: Path, payload: Any) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def extract_epoch(path_or_name: str | Path) -> int:
    match = re.search(r"epoch_(\d+)", str(path_or_name))
    return int(match.group(1)) if match else 0


def infer_attention_rate(config: dict[str, Any], checkpoint_path: Path) -> float:
    if "attention_rate" in config:
        return float(config["attention_rate"])

    path_str = str(checkpoint_path).lower()
    if "pizza" in path_str or "a0.0" in path_str:
        return 0.0
    if "a1.0" in path_str:
        return 1.0
    return 1.0


def infer_act_type(config: dict[str, Any], checkpoint_path: Path) -> str:
    if "act_type" in config:
        return str(config["act_type"])

    path_str = str(checkpoint_path).lower()
    if "relu" in path_str:
        return "ReLU"
    if "gelu" in path_str:
        return "GeLU"
    if "silu" in path_str:
        return "SiLU"
    if "selu" in path_str:
        return "SeLU"
    return "SiLU"


def load_checkpoint(checkpoint_path: Path, device: str = "cuda:0") -> tuple[Transformer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        model_state = checkpoint
        config = {}

    d_model, d_vocab = model_state["embed.W_E"].shape
    d_mlp, _ = model_state["blocks.0.mlp.W_in"].shape
    n_heads, d_head, _ = model_state["blocks.0.attn.W_Q"].shape
    num_layers = sum(1 for key in model_state if ".attn.W_Q" in key)

    model = Transformer(
        num_layers=num_layers,
        d_vocab=d_vocab,
        d_model=d_model,
        d_mlp=d_mlp,
        d_head=d_head,
        num_heads=n_heads,
        n_ctx=3,
        act_type=infer_act_type(config, checkpoint_path),
        use_cache=False,
        use_ln=False,
        attention_rate=infer_attention_rate(config, checkpoint_path),
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model, config


def cast_geometry(geometry: dict[str, Any], device: str, dtype: torch.dtype) -> dict[str, Any]:
    geo = copy.deepcopy(geometry)
    geo["X_sample"] = geo["X_sample"].to(device=device, dtype=dtype)
    geo["X_prompts"] = geo["X_prompts"].to(device=device)
    geo["X_embeddings"] = [x.to(device=device, dtype=dtype) for x in geo["X_embeddings"]]
    geo["all_input_tangent_bases"] = [
        x.to(device=device, dtype=dtype) for x in geo["all_input_tangent_bases"]
    ]
    return geo


def iter_union_edges(knn_indices: np.ndarray | torch.Tensor) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(len(knn_indices)):
        for j in knn_indices[i]:
            j_int = int(j)
            if i == j_int:
                continue
            edges.add((min(i, j_int), max(i, j_int)))
    return sorted(edges)


def detect_grok_epoch(metrics: dict[str, Any], threshold: float = 0.99) -> int | None:
    steps = np.asarray(metrics.get("log_steps", []), dtype=int)
    accs = np.asarray(metrics.get("test_accuracies", []), dtype=float)
    if steps.size < 4 or accs.size < 4 or accs.max() < threshold:
        return None

    first_idx = int(np.argmax(accs >= threshold))
    best_epoch = None
    best_slope = float("-inf")
    for idx in range(1, first_idx + 1):
        dt = steps[idx] - steps[idx - 1]
        if dt <= 0:
            continue
        slope = (accs[idx] - accs[idx - 1]) / dt
        if slope > best_slope:
            best_slope = slope
            best_epoch = int(steps[idx])
    return best_epoch


def evenly_subsample(items: list[tuple[int, Path]], max_items: int | None) -> list[tuple[int, Path]]:
    if max_items is None or len(items) <= max_items:
        return items

    if max_items <= 1:
        return [items[-1]]

    indices = np.linspace(0, len(items) - 1, num=max_items, dtype=int)
    return [items[int(idx)] for idx in indices]


def load_or_build_geometry(
    exp_dir: Path,
    train_data: Any,
    model: Transformer,
    *,
    target_layer: int,
    device: str,
    n_samples: int,
    k_neighbors: int,
    intrinsic_dim: int,
) -> tuple[dict[str, Any], str]:
    candidates = [exp_dir / f"final_geometry_layer{target_layer}.pkl"]
    if target_layer == 0:
        candidates.append(exp_dir / "final_geometry.pkl")

    for candidate in candidates:
        if candidate.exists():
            return load_pickle(candidate), f"loaded:{candidate.name}"

    geometry = build_complete_geometric_structure(
        model=model,
        train_data=train_data,
        N=n_samples,
        k=k_neighbors,
        d=intrinsic_dim,
        device=device,
        layer_idx=target_layer,
        hook_point="hook_resid_post",
    )
    return geometry, "built:final_model"


def create_control_path_forward(
    model: Transformer,
    *,
    variant: str,
    target_layer: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if variant not in PATH_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    def path_forward(x_embedded: torch.Tensor) -> torch.Tensor:
        if x_embedded.dim() == 2:
            x = x_embedded.unsqueeze(0)
        elif x_embedded.dim() == 3:
            x = x_embedded
        else:
            raise ValueError(f"Expected dim 2 or 3, got {x_embedded.dim()}")

        x = model.pos_embed(x)

        for layer_idx in range(target_layer):
            x = model.blocks[layer_idx](x)

        block = model.blocks[target_layer]
        x_start = block.hook_resid_pre(x)
        attn_out = block.hook_attn_out(block.attn(x_start))

        if variant == "attention_path":
            return attn_out[:, -1, :].squeeze(0)

        if variant == "mlp_zero_attn_control":
            x_mid_in = x_start + torch.zeros_like(attn_out)
        elif variant in {"mlp_pure_output_control", "residual_post_detached_attn_control"}:
            x_mid_in = x_start + attn_out.detach()
        else:
            x_mid_in = x_start + attn_out

        x_mid = block.hook_resid_mid(x_mid_in)
        mlp_out = block.hook_mlp_out(block.mlp(x_mid))

        if variant in {"mlp_path", "mlp_pure_output_control", "mlp_zero_attn_control"}:
            return mlp_out[:, -1, :].squeeze(0)

        x_final = block.hook_resid_post(x_mid + mlp_out)

        if variant in {"overall_path", "residual_post_detached_attn_control"}:
            return x_final[:, -1, :].squeeze(0)

        raise ValueError(f"Unhandled variant: {variant}")

    return path_forward


def compute_custom_path_gcs(
    model: Transformer,
    geometric_data: dict[str, Any],
    path_forward: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: str,
    alignment_method: str,
) -> float:
    model.to(device)
    model.eval()

    n_points = geometric_data["N"]
    x_embeddings = geometric_data["X_embeddings"]
    input_bases = geometric_data["all_input_tangent_bases"]
    knn_indices = geometric_data["knn_indices"]

    transformed_bases = []
    for idx in range(n_points):
        basis = input_bases[idx]
        transformed_basis = []

        for vector in basis:
            direction = torch.zeros_like(x_embeddings[idx])
            direction[-1, :] = vector

            _, transformed_v = torch.autograd.functional.jvp(
                path_forward,
                x_embeddings[idx],
                direction,
            )

            if transformed_v.dim() > 1:
                transformed_v = transformed_v.squeeze()
            transformed_basis.append(transformed_v)

        transformed_bases.append(torch.stack(transformed_basis))

    coherence = torch.eye(n_points, dtype=torch.float64, device=device)
    for i, j in iter_union_edges(knn_indices):
        if alignment_method == "procrustes":
            score = compute_alignment_score_procrustes(
                input_bases[i],
                input_bases[j],
                transformed_bases[i],
                transformed_bases[j],
            )
        else:
            score = compute_alignment_score_cosine(
                transformed_bases[i],
                transformed_bases[j],
            )
        coherence[i, j] = float(score)
        coherence[j, i] = float(score)

    eigenvalues = np.abs(np.linalg.eigvals(coherence.cpu().numpy()))
    total = float(np.sum(eigenvalues))
    if total > 0.0:
        probs = eigenvalues / total
    else:
        probs = np.ones(len(eigenvalues), dtype=np.float64) / float(len(eigenvalues))

    probs_nonzero = probs[probs > EIGENVALUE_THRESHOLD]
    spectral_entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))
    gmn_tsa = 2 ** spectral_entropy
    return float(n_points - gmn_tsa)


def compute_variant_scores(
    model: Transformer,
    geometry: dict[str, Any],
    *,
    target_layer: int,
    device: str,
    alignment_method: str,
    include_zero_attn: bool,
) -> dict[str, float]:
    variants = [
        "attention_path",
        "mlp_path",
        "overall_path",
        "mlp_pure_output_control",
        "residual_post_detached_attn_control",
    ]
    if include_zero_attn:
        variants.append("mlp_zero_attn_control")

    dtype = next(model.parameters()).dtype
    geometry_typed = cast_geometry(geometry, device=device, dtype=dtype)

    scores: dict[str, float] = {}
    for variant in variants:
        path_forward = create_control_path_forward(
            model,
            variant=variant,
            target_layer=target_layer,
        )
        scores[variant] = compute_custom_path_gcs(
            model,
            geometry_typed,
            path_forward,
            device=device,
            alignment_method=alignment_method,
        )
    return scores


def plot_path_purity_controls(
    metrics: dict[str, Any],
    results: dict[str, Any],
    *,
    save_path: Path,
    include_zero_attn: bool,
) -> None:
    epochs = np.asarray(results["epochs"], dtype=float)
    scores = results["path_gcs_scores"]
    grok_epoch = detect_grok_epoch(metrics)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    def add_marker(ax: plt.Axes) -> None:
        if grok_epoch is not None:
            ax.axvline(grok_epoch, color="0.35", linestyle="--", linewidth=1.2, alpha=0.8)

    def plot_series(ax: plt.Axes, key: str, *, color: str, label: str, linewidth: float = 1.8) -> None:
        valid = [(epoch, value) for epoch, value in zip(epochs, scores[key]) if value is not None]
        if not valid:
            return
        xs, ys = zip(*valid)
        ax.plot(xs, ys, color=color, linewidth=linewidth, label=label)

    # Panel A: original paths
    ax = axes[0]
    for key, color, label in [
        ("overall_path", "#1f77b4", "Overall"),
        ("attention_path", "#ff7f0e", "Attention"),
        ("mlp_path", "#2ca02c", "MLP"),
    ]:
        plot_series(ax, key, color=color, label=label)
    add_marker(ax)
    ax.set_title("Original Paths")
    ax.set_ylabel("GCS")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # Panel B: MLP purity controls
    ax = axes[1]
    plot_series(ax, "mlp_path", color="#2ca02c", label="MLP")
    plot_series(ax, "mlp_pure_output_control", color="#d62728", label="MLP (attn detached)")
    if include_zero_attn and "mlp_zero_attn_control" in scores:
        plot_series(ax, "mlp_zero_attn_control", color="#9467bd", label="MLP (zero attn)", linewidth=1.6)
    add_marker(ax)
    ax.set_title("MLP Control Comparison")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # Panel C: overall vs detached overall
    ax = axes[2]
    plot_series(ax, "overall_path", color="#1f77b4", label="Overall")
    plot_series(
        ax,
        "residual_post_detached_attn_control",
        color="#8c564b",
        label="Overall (attn detached)",
    )
    add_marker(ax)
    ax.set_title("Whole-Block Control Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("GCS")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # Panel D: accuracy
    ax = axes[3]
    train_steps = np.asarray(metrics.get("log_steps", []), dtype=float)
    train_acc = np.asarray(metrics.get("train_accuracies", []), dtype=float)
    test_acc = np.asarray(metrics.get("test_accuracies", []), dtype=float)
    if train_steps.size and train_acc.size:
        ax.plot(train_steps, train_acc, color="#4c78a8", linewidth=1.6, label="Train acc")
    if train_steps.size and test_acc.size:
        ax.plot(train_steps, test_acc, color="#59a14f", linewidth=1.6, label="Test acc")
    add_marker(ax)
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    fig.suptitle(results["title"], fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_experiment(
    exp_dir: Path,
    *,
    target_layer: int,
    n_samples: int,
    k_neighbors: int,
    intrinsic_dim: int,
    checkpoint_freq: int,
    device: str,
    alignment_method: str,
    include_zero_attn: bool,
    max_checkpoints: int | None,
) -> dict[str, Any]:
    exp_dir = exp_dir.resolve()
    metrics_path = exp_dir / "training_metrics.pkl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing training_metrics.pkl in {exp_dir}")

    metrics = load_pickle(metrics_path)
    config = metrics.get("config", {})
    task = config.get("task", "add")
    p = int(config.get("p", 113))
    fraction = float(config.get("fraction", 0.3))
    seed = int(config.get("seed", 777))

    train_data, _ = create_data(
        p=p,
        fraction=fraction,
        seed=seed,
        device=device,
        task=task,
    )

    checkpoint_dir = exp_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=extract_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    final_model, final_config = load_checkpoint(checkpoints[-1], device=device)
    if not (0 <= target_layer < final_model.num_layers):
        raise ValueError(
            f"target_layer={target_layer} out of range for model with {final_model.num_layers} layers"
        )
    geometry, geometry_source = load_or_build_geometry(
        exp_dir,
        train_data,
        final_model,
        target_layer=target_layer,
        device=device,
        n_samples=n_samples,
        k_neighbors=k_neighbors,
        intrinsic_dim=intrinsic_dim,
    )

    selected = [(extract_epoch(path), path) for path in checkpoints if extract_epoch(path) % checkpoint_freq == 0]
    selected = evenly_subsample(selected, max_checkpoints)

    results = {
        "title": f"{exp_dir.name} - Path Purity Controls (layer {target_layer})",
        "experiment_dir": str(exp_dir),
        "epochs": [],
        "path_gcs_scores": {key: [] for key in PATH_VARIANTS if include_zero_attn or key != "mlp_zero_attn_control"},
        "variant_descriptions": {
            key: value
            for key, value in PATH_VARIANTS.items()
            if include_zero_attn or key != "mlp_zero_attn_control"
        },
        "analysis_params": {
            "target_layer": target_layer,
            "N": n_samples,
            "k": k_neighbors,
            "d": intrinsic_dim,
            "checkpoint_freq": checkpoint_freq,
            "alignment_method": alignment_method,
            "include_zero_attn": include_zero_attn,
            "max_checkpoints": max_checkpoints,
        },
        "model_config": final_config,
        "geometry_source": geometry_source,
    }

    for epoch, checkpoint_path in tqdm(selected, desc="Computing path-purity controls"):
        try:
            model, _ = load_checkpoint(checkpoint_path, device=device)
            variant_scores = compute_variant_scores(
                model,
                geometry,
                target_layer=target_layer,
                device=device,
                alignment_method=alignment_method,
                include_zero_attn=include_zero_attn,
            )
            results["epochs"].append(epoch)
            for key in results["path_gcs_scores"]:
                results["path_gcs_scores"][key].append(variant_scores.get(key))
        except Exception as exc:
            print(f"  Epoch {epoch} failed: {exc}")
            results["epochs"].append(epoch)
            for key in results["path_gcs_scores"]:
                results["path_gcs_scores"][key].append(None)

    out_path = exp_dir / "path_purity_controls.pkl"
    save_pickle(out_path, results)

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path_purity_controls(
        metrics,
        results,
        save_path=plots_dir / "path_purity_controls.png",
        include_zero_attn=include_zero_attn,
    )

    print(f"Saved results: {out_path}")
    print(f"Saved plot: {plots_dir / 'path_purity_controls.png'}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze path-purity controls for Transformer GCS")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory")
    parser.add_argument("--target-layer", type=int, default=0, help="Target Transformer layer")
    parser.add_argument("--N", type=int, default=200, help="Number of geometry samples")
    parser.add_argument("--k", type=int, default=15, help="k-NN neighborhood size")
    parser.add_argument("--d", type=int, default=8, help="Intrinsic dimension")
    parser.add_argument("--checkpoint-freq", type=int, default=200, help="Checkpoint stride")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    parser.add_argument(
        "--alignment-method",
        type=str,
        default="cosine",
        choices=["cosine", "procrustes"],
        help="Alignment method for GCS",
    )
    parser.add_argument(
        "--include-zero-attn",
        action="store_true",
        help="Also compute the zero-attention stress-test control",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Evenly subsample checkpoints for a fast smoke run",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    analyze_experiment(
        Path(args.exp_dir),
        target_layer=args.target_layer,
        n_samples=args.N,
        k_neighbors=args.k,
        intrinsic_dim=args.d,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
        alignment_method=args.alignment_method,
        include_zero_attn=args.include_zero_attn,
        max_checkpoints=args.max_checkpoints,
    )


if __name__ == "__main__":
    main()
