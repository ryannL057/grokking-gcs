
from gcs.runtime import data_root as _gcs_data_root
#!/usr/bin/env python3
"""
D ablation study: Test GCS sensitivity to intrinsic dimension d.

Uses fixed geometry (same sample points and k-NN graph) and only varies d.

Usage:
    python ablation_d.py --experiment pretrained/SiLU/silu_* --d-values 4 6 8 10 12
"""

import argparse
import sys
import pickle
import re
import copy
from pathlib import Path

import torch
import numpy as np

pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer
from gcs.analysis.geometry import compute_tangent_space
from gcs.analysis import analyze_3path_gmn_tsa
from gcs.vis import plot_ablation_3path


def load_checkpoint(checkpoint_path, device='cuda:0'):
    """Load checkpoint and create model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint

    d_model, d_vocab = model_state['embed.W_E'].shape
    d_mlp, _ = model_state['blocks.0.mlp.W_in'].shape
    n_heads, d_head, _ = model_state['blocks.0.attn.W_Q'].shape
    num_layers = sum(1 for k in model_state.keys() if '.attn.W_Q' in k)

    act_type = 'SiLU'
    path_str = str(checkpoint_path).lower()
    if 'relu' in path_str:
        act_type = 'ReLU'
    elif 'gelu' in path_str:
        act_type = 'GeLU'

    model = Transformer(
        num_layers=num_layers,
        d_vocab=d_vocab,
        d_model=d_model,
        d_mlp=d_mlp,
        d_head=d_head,
        num_heads=n_heads,
        n_ctx=3,
        act_type=act_type,
        use_cache=False,
        use_ln=False
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()

    return model


def extract_epoch(filename):
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0


def recompute_tangent_bases(fixed_geometry, new_d, device):
    """
    Recompute tangent bases with new d value, keeping everything else fixed.

    Args:
        fixed_geometry: Original geometry from final model
        new_d: New intrinsic dimension
        device: Computing device

    Returns:
        Updated geometry with new tangent bases
    """
    new_geometry = copy.deepcopy(fixed_geometry)

    X_sample = new_geometry['X_sample'].to(device=device, dtype=torch.float64)
    X_embeddings = [emb.to(device=device, dtype=torch.float64)
                    for emb in new_geometry['X_embeddings']]
    knn_indices = new_geometry['knn_indices']
    N = new_geometry['N']

    new_geometry['X_sample'] = X_sample
    new_geometry['X_embeddings'] = X_embeddings

    # Recompute tangent bases with new d
    new_tangent_bases = []
    for i in range(N):
        tangent_basis = compute_tangent_space(
            i=i,
            X_sample=X_sample,
            knn_graph=knn_indices,
            d=new_d
        )
        new_tangent_bases.append(tangent_basis)

    new_geometry['all_input_tangent_bases'] = new_tangent_bases
    new_geometry['d'] = new_d

    return new_geometry


def run_d_ablation(experiment_dir, d_values, checkpoint_freq=200, device='cuda:0'):
    """
    Run d ablation study using fixed geometry.

    Args:
        experiment_dir: Experiment directory
        d_values: List of d values to test
        checkpoint_freq: Analyze every N epochs
        device: Computing device
    """
    torch.set_default_dtype(torch.float64)
    exp_dir = Path(experiment_dir)

    print(f"{'='*60}")
    print(f"D Ablation Study (Fixed Geometry)")
    print(f"{'='*60}")
    print(f"Experiment: {exp_dir.name}")
    print(f"D values: {d_values}")

    # Load fixed geometry
    geometry_file = exp_dir / 'final_geometry.pkl'
    if not geometry_file.exists():
        print(f"Error: No final_geometry.pkl found")
        print("Run training with --compute-gcs first")
        return None

    with open(geometry_file, 'rb') as f:
        fixed_geometry = pickle.load(f)

    print(f"Loaded geometry: N={fixed_geometry['N']}, k={fixed_geometry.get('k', 'N/A')}")
    print(f"Original d={fixed_geometry['d']}")

    # Get checkpoints
    checkpoint_dir = exp_dir / 'checkpoints'
    all_ckpts = sorted(checkpoint_dir.glob('*.pth'), key=lambda x: extract_epoch(x.name))
    selected = [(extract_epoch(c.name), c) for c in all_ckpts
                if extract_epoch(c.name) % checkpoint_freq == 0]

    print(f"Checkpoints: {len(selected)}")

    all_results = {}

    for d in d_values:
        print(f"\n{'='*60}")
        print(f"Testing d={d}")
        print(f"{'='*60}")

        results = {
            'epochs': [],
            'overall_gcs': [],
            'attention_gcs': [],
            'mlp_gcs': []
        }

        for epoch, ckpt_path in selected:
            try:
                model = load_checkpoint(ckpt_path, device)

                # Recompute tangent bases with new d
                geometry_d = recompute_tangent_bases(fixed_geometry, d, device)

                path_results = analyze_3path_gmn_tsa(
                    model=model,
                    complete_geometry=geometry_d,
                    target_layer=0,
                    device=device
                )

                results['epochs'].append(epoch)
                results['overall_gcs'].append(path_results.get('overall_path'))
                results['attention_gcs'].append(path_results.get('attention_path'))
                results['mlp_gcs'].append(path_results.get('mlp_path'))

            except Exception as e:
                print(f"  Epoch {epoch} failed: {e}")
                import traceback
                traceback.print_exc()

        all_results[d] = results
        print(f"  Completed: {len(results['epochs'])} epochs")

    # Save results
    output_dir = exp_dir / 'd_ablation_study'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump({
            'd_values': d_values,
            'results': all_results
        }, f)

    # Plot
    plot_ablation_3path(
        all_results,
        param_name='d',
        title='D Ablation Study (Fixed Geometry)',
        save_path=output_dir / 'd_ablation.png'
    )

    print(f"\n{'='*60}")
    print(f"D Ablation Complete!")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='D Ablation Study')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment directory (must have final_geometry.pkl)')
    parser.add_argument('--d-values', nargs='+', type=int,
                        default=[4, 6, 8, 10, 12],
                        help='D values to test')
    parser.add_argument('--checkpoint-freq', type=int, default=200,
                        help='Checkpoint sampling frequency')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computing device')

    args = parser.parse_args()

    run_d_ablation(
        experiment_dir=args.experiment,
        d_values=args.d_values,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device
    )


if __name__ == '__main__':
    main()
