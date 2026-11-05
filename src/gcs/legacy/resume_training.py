#!/usr/bin/env python3
"""
Resume training from existing experiment directory.

Loads the last checkpoint, continues training to --epochs total,
saves new checkpoints, then recomputes GCS from scratch with
the new final model as reference geometry.

Usage:
  python scripts/resume_training.py experiments/relu_2layer_sub_p113_s42_... --epochs 40000
"""

import argparse
import pickle
import sys
from pathlib import Path

import torch

# Add project root to path

from gcs.core.models import Transformer
from gcs.core.data import create_data, get_model_config_from_data
from gcs.core.training import basic_training_loop, setup_training_environment
from gcs.analysis import build_complete_geometric_structure, analyze_3path_gmn_tsa


def find_last_checkpoint(ckpt_dir: Path):
    """Find the last checkpoint file and return (path, epoch)."""
    files = sorted(ckpt_dir.glob("*.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    last = files[-1]
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    return last, ckpt


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resume training from checkpoint")
    parser.add_argument("exp_dir", type=Path, help="Experiment directory")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Total epochs (not additional)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gcs-freq", type=int, default=200)
    parser.add_argument("--gcs-epochs", help="Explicit comma-separated analysis epochs")
    args = parser.parse_args(argv)
    selected_epochs = set(map(int,args.gcs_epochs.split(","))) if args.gcs_epochs else None

    exp_dir = args.exp_dir
    assert exp_dir.exists(), f"{exp_dir} does not exist"

    # Load config
    with open(exp_dir / "config.pkl", "rb") as f:
        config = pickle.load(f)

    num_layers = config.get("num_layers", 1)
    act_type = config["act_type"]
    task = config.get("task", "add")
    seed = config.get("seed", 777)
    weight_decay = config.get("weight_decay", 1.0)

    print(f"Resuming: {exp_dir}")
    print(f"  {num_layers}L {act_type} {task} seed={seed} wd={weight_decay}")

    # Setup
    device = setup_training_environment(seed=seed, device=args.device)

    # Create data (same seed → same split)
    train_data, test_data = create_data(
        p=config["p"],
        fraction=config["fraction"],
        seed=config["seed"],
        device=device,
        task=task,
    )

    # Create model
    model_config = get_model_config_from_data(config)
    model = Transformer(**model_config).to(device)

    # Load last checkpoint
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_path, ckpt = find_last_checkpoint(ckpt_dir)
    start_epoch = ckpt["epoch"] + 1
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded checkpoint: epoch {ckpt['epoch']} from {ckpt_path.name}")

    if start_epoch >= args.epochs:
        print(f"  Already trained to epoch {start_epoch - 1} >= {args.epochs}. Nothing to do.")
        return

    print(f"  Continuing from epoch {start_epoch} to {args.epochs - 1}")

    # Checkpoint callback
    checkpoint_freq = config.get("checkpoint_freq", 200)

    def checkpoint_callback(model, epoch, metrics):
        if epoch % checkpoint_freq == 0:
            ckpt_path = ckpt_dir / f"{act_type.lower()}_epoch_{epoch:05d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, ckpt_path)

    # Resume training
    metrics = basic_training_loop(
        model=model,
        train_data=train_data,
        test_data=test_data,
        total_epochs=args.epochs,
        lr=config.get("lr", 1e-3),
        weight_decay=weight_decay,
        record_frequency=config.get("record_frequency", 200),
        device=device,
        verbose=True,
        checkpoint_callback=checkpoint_callback,
        start_epoch=start_epoch,
        max_grad_norm=config.get("grad_clip", None),
    )

    # Save final checkpoint
    final_ckpt = ckpt_dir / f"{act_type.lower()}_epoch_{args.epochs - 1:05d}.pth"
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, final_ckpt)

    # Merge old + new metrics
    old_metrics_path = exp_dir / "training_metrics.pkl"
    with open(old_metrics_path, "rb") as f:
        old_m = pickle.load(f)

    new_m = metrics.to_dict()
    for key in ["log_steps", "train_accuracies", "test_accuracies",
                "train_losses", "test_losses"]:
        if key in old_m and key in new_m:
            old_m[key] = list(old_m[key]) + list(new_m[key])
    if "norms" in old_m and "norms" in new_m:
        old_m["norms"] = list(old_m["norms"]) + list(new_m["norms"])

    with open(old_metrics_path, "wb") as f:
        pickle.dump(old_m, f)
    print(f"  Merged metrics saved ({len(old_m['log_steps'])} entries)")

    # Recompute GCS with new final geometry
    print("\nRecomputing GCS with new final geometry...")
    last_layer = num_layers - 1
    gcs_last_layer_only = True  # Always last-layer-only for multi-layer

    if num_layers > 1:
        layers_to_compute = [last_layer]
        final_geometries = {}
        for layer_idx in layers_to_compute:
            final_geometries[layer_idx] = build_complete_geometric_structure(
                model=model,
                train_data=train_data,
                N=config.get("analysis_N", 200),
                k=config.get("analysis_k", 15),
                d=config.get("analysis_d", 10),
                device=device,
                layer_idx=layer_idx,
                hook_point="hook_resid_post",
            )
            with open(exp_dir / f"final_geometry_layer{layer_idx}.pkl", "wb") as f:
                pickle.dump(final_geometries[layer_idx], f)
        with open(exp_dir / "final_geometry.pkl", "wb") as f:
            pickle.dump(final_geometries[last_layer], f)
        compat_layer = last_layer
    else:
        final_geometry = build_complete_geometric_structure(
            model=model,
            train_data=train_data,
            N=config.get("analysis_N", 200),
            k=config.get("analysis_k", 15),
            d=config.get("analysis_d", 10),
            device=device,
        )
        final_geometries = {0: final_geometry}
        with open(exp_dir / "final_geometry.pkl", "wb") as f:
            pickle.dump(final_geometry, f)
        compat_layer = 0

    # Build GCS results over ALL checkpoints
    path_scores = {
        "overall_path": [],
        "attention_path": [],
        "mlp_path": [],
    }
    if num_layers > 1:
        for layer_idx in layers_to_compute:
            path_scores[f"layer{layer_idx}_overall_path"] = []
            path_scores[f"layer{layer_idx}_attention_path"] = []
            path_scores[f"layer{layer_idx}_mlp_path"] = []

    gcs_results = {
        "epochs": [],
        "path_gcs_scores": path_scores,
    }

    checkpoint_files = sorted(ckpt_dir.glob("*.pth"))
    print(f"  Computing GCS for {len(checkpoint_files)} checkpoints...")
    for ckpt_file in checkpoint_files:
        try:
            ckpt_data = torch.load(ckpt_file, map_location=device, weights_only=False)
            epoch = ckpt_data["epoch"]
            if not (epoch in selected_epochs if selected_epochs is not None else epoch % args.gcs_freq == 0):
                continue
            model.load_state_dict(ckpt_data["model_state_dict"])

            if num_layers > 1:
                gcs_results["epochs"].append(epoch)
                for layer_idx in layers_to_compute:
                    pr = analyze_3path_gmn_tsa(
                        model=model,
                        complete_geometry=final_geometries[layer_idx],
                        target_layer=layer_idx,
                        device=device,
                    )
                    gcs_results["path_gcs_scores"][f"layer{layer_idx}_overall_path"].append(pr.get("overall_path"))
                    gcs_results["path_gcs_scores"][f"layer{layer_idx}_attention_path"].append(pr.get("attention_path"))
                    gcs_results["path_gcs_scores"][f"layer{layer_idx}_mlp_path"].append(pr.get("mlp_path"))
                    if layer_idx == compat_layer:
                        gcs_results["path_gcs_scores"]["overall_path"].append(pr.get("overall_path"))
                        gcs_results["path_gcs_scores"]["attention_path"].append(pr.get("attention_path"))
                        gcs_results["path_gcs_scores"]["mlp_path"].append(pr.get("mlp_path"))
            else:
                pr = analyze_3path_gmn_tsa(
                    model=model,
                    complete_geometry=final_geometries[0],
                    target_layer=0,
                    device=device,
                )
                gcs_results["epochs"].append(epoch)
                gcs_results["path_gcs_scores"]["overall_path"].append(pr.get("overall_path"))
                gcs_results["path_gcs_scores"]["attention_path"].append(pr.get("attention_path"))
                gcs_results["path_gcs_scores"]["mlp_path"].append(pr.get("mlp_path"))

        except Exception as e:
            print(f"    Skip {ckpt_file.name}: {e}")

    with open(exp_dir / "gcs_results.pkl", "wb") as f:
        pickle.dump(gcs_results, f)
    print(f"  GCS saved: {len(gcs_results['epochs'])} epochs")
    print("Done.")


if __name__ == "__main__":
    main()
