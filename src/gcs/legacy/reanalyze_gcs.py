
from gcs.runtime import data_root as _gcs_data_root
#!/usr/bin/env python3
"""
Reanalyze existing experiments using final geometry.

This script loads trained models and recomputes GCS with consistent
final geometry for reliable comparison.

Usage:
    python reanalyze_gcs.py --exp-dir pretrained/SiLU/silu_*
    python reanalyze_gcs.py --exp-dir pretrained/ --recursive
"""

import argparse
import sys
import pickle
import re
from pathlib import Path

import torch
from tqdm.auto import tqdm

pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer
from gcs.core.data import create_data
from gcs.analysis import build_complete_geometric_structure, analyze_3path_gmn_tsa
from gcs.vis import plot_gcs_3path, plot_gcs_with_accuracy


def load_checkpoint(checkpoint_path, device='cuda:0'):
    """Load checkpoint and create model."""
    torch.set_default_dtype(torch.float64)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    else:
        model_state = checkpoint
        config = {}

    d_model, d_vocab = model_state['embed.W_E'].shape
    d_mlp, _ = model_state['blocks.0.mlp.W_in'].shape
    n_heads, d_head, _ = model_state['blocks.0.attn.W_Q'].shape
    num_layers = sum(1 for k in model_state.keys() if '.attn.W_Q' in k)

    act_type = config.get('act_type', 'SiLU')
    path_str = str(checkpoint_path).lower()
    if 'relu' in path_str:
        act_type = 'ReLU'
    elif 'gelu' in path_str:
        act_type = 'GeLU'
    elif 'silu' in path_str:
        act_type = 'SiLU'

    use_ln = bool(config.get('use_ln', False)) or any('.ln1.' in key or '.ln2.' in key for key in model_state)

    model = Transformer(
        num_layers=num_layers,
        d_vocab=d_vocab,
        d_model=d_model,
        d_mlp=d_mlp,
        d_head=d_head,
        num_heads=n_heads,
        n_ctx=int(config.get('n_ctx', 3)),
        act_type=act_type,
        use_cache=False,
        use_ln=use_ln,
    ).to(device=device,dtype=torch.float64)

    model.load_state_dict(model_state)
    model.eval()

    return model, config


def extract_epoch(filename):
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0


def reanalyze_experiment(exp_dir, N=200, k=15, d=8,
                         checkpoint_freq=200, device='cuda:0',
                         alignment_method='cosine',
                         reference_epoch=None):
    """
    Reanalyze a single experiment with reference geometry.

    Args:
        exp_dir: Experiment directory
        N, k, d: GMN-TSA parameters
        checkpoint_freq: Analysis frequency
        device: Computing device
        reference_epoch: Epoch to use as reference geometry (None = final checkpoint)

    Returns:
        dict: Reanalyzed metrics
    """
    exp_dir = Path(exp_dir)
    print(f"\n{'='*70}")
    print(f"Reanalyzing: {exp_dir.name}")
    print(f"{'='*70}")

    # Load original metrics if available
    metrics_path = exp_dir / 'training_metrics.pkl'
    original_metrics = {}
    if metrics_path.exists():
        with open(metrics_path, 'rb') as f:
            original_metrics = pickle.load(f)

    config = original_metrics.get('config', {})
    task = config.get('task', 'add')

    # Load training data
    print("[1/4] Loading data...")
    train_data, _ = create_data(
        p=config.get('p', 113),
        fraction=config.get('fraction', 0.3),
        seed=config.get('seed', 777),
        device=device,
        task=task
    )

    # Get checkpoints
    checkpoint_dir = exp_dir / 'checkpoints'
    all_ckpts = sorted(checkpoint_dir.glob('*.pth'), key=lambda x: extract_epoch(x.name))

    if not all_ckpts:
        print("  No checkpoints found")
        return None

    # Select reference checkpoint
    if reference_epoch is not None:
        ref_ckpt = None
        for c in all_ckpts:
            if extract_epoch(c.name) == reference_epoch:
                ref_ckpt = c
                break
        if ref_ckpt is None:
            available = [extract_epoch(c.name) for c in all_ckpts]
            raise ValueError(
                f"Reference epoch {reference_epoch} not found. "
                f"Available epochs: {available[:5]}...{available[-5:]}"
            )
        ref_label = f"ref{reference_epoch}"
    else:
        ref_ckpt = all_ckpts[-1]
        ref_label = "final"

    # Load reference model and build geometry
    print(f"[2/4] Building {ref_label} geometry (epoch {extract_epoch(ref_ckpt.name)})...")
    model, _ = load_checkpoint(ref_ckpt, device)
    num_layers = getattr(model, 'num_layers', 1)

    if num_layers > 1:
        geometries = {}
        for layer_idx in range(num_layers):
            geometries[layer_idx] = build_complete_geometric_structure(
                model=model,
                train_data=train_data,
                N=N, k=k, d=d,
                device=device,
                layer_idx=layer_idx,
                hook_point='hook_resid_post'
            )

            geo_path = exp_dir / f'{ref_label}_geometry_layer{layer_idx}.pkl'
            with open(geo_path, 'wb') as f:
                pickle.dump(geometries[layer_idx], f)

        geometry = geometries[0]
        geometry_path = exp_dir / f'{ref_label}_geometry.pkl'
        with open(geometry_path, 'wb') as f:
            pickle.dump(geometry, f)
        print(f"  Saved per-layer geometries + {geometry_path}")
    else:
        geometry = build_complete_geometric_structure(
            model=model,
            train_data=train_data,
            N=N, k=k, d=d,
            device=device
        )
        geometries = {0: geometry}

        geometry_path = exp_dir / f'{ref_label}_geometry.pkl'
        with open(geometry_path, 'wb') as f:
            pickle.dump(geometry, f)
        print(f"  Saved: {geometry_path}")

    # Select checkpoints
    selected = [(extract_epoch(c.name), c) for c in all_ckpts
                if extract_epoch(c.name) % checkpoint_freq == 0]

    print(f"[3/4] Computing GCS for {len(selected)} checkpoints...")

    path_scores = {
        'overall_path': [],
        'attention_path': [],
        'mlp_path': []
    }
    if num_layers > 1:
        for layer_idx in range(num_layers):
            path_scores[f'layer{layer_idx}_overall_path'] = []
            path_scores[f'layer{layer_idx}_attention_path'] = []
            path_scores[f'layer{layer_idx}_mlp_path'] = []

    gcs_results = {
        'gcs_steps': [],
        'path_gcs_scores': path_scores,
        'geometry_type': 'final_geometry_per_layer' if num_layers > 1 else 'final_geometry',
        'analysis_params': {'N': N, 'k': k, 'd': d, 'num_layers': num_layers}
    }

    for epoch, ckpt_path in tqdm(selected, desc="Computing GCS"):
        try:
            model, _ = load_checkpoint(ckpt_path, device)

            if num_layers > 1:
                layer_path_results = {}
                for layer_idx in range(num_layers):
                    with torch.no_grad():
                        model.eval()
                        path_results = analyze_3path_gmn_tsa(
                            model, geometries[layer_idx], target_layer=layer_idx, device=device,
                            alignment_method=alignment_method
                        )
                    layer_path_results[layer_idx] = path_results

                gcs_results['gcs_steps'].append(epoch)

                for layer_idx in range(num_layers):
                    path_results = layer_path_results[layer_idx]

                    gcs_results['path_gcs_scores'][f'layer{layer_idx}_overall_path'].append(
                        path_results.get('overall_path')
                    )
                    gcs_results['path_gcs_scores'][f'layer{layer_idx}_attention_path'].append(
                        path_results.get('attention_path')
                    )
                    gcs_results['path_gcs_scores'][f'layer{layer_idx}_mlp_path'].append(
                        path_results.get('mlp_path')
                    )

                    if layer_idx == 0:
                        gcs_results['path_gcs_scores']['overall_path'].append(path_results.get('overall_path'))
                        gcs_results['path_gcs_scores']['attention_path'].append(path_results.get('attention_path'))
                        gcs_results['path_gcs_scores']['mlp_path'].append(path_results.get('mlp_path'))
            else:
                with torch.no_grad():
                    model.eval()
                    path_results = analyze_3path_gmn_tsa(
                        model, geometry, target_layer=0, device=device,
                        alignment_method=alignment_method
                    )

                gcs_results['gcs_steps'].append(epoch)

                for path in ['overall_path', 'attention_path', 'mlp_path']:
                    gcs_results['path_gcs_scores'][path].append(path_results.get(path))

        except Exception as e:
            print(f"  Epoch {epoch} failed: {e}")
            gcs_results['gcs_steps'].append(epoch)
            for key in gcs_results['path_gcs_scores']:
                gcs_results['path_gcs_scores'][key].append(None)

    # Combine with original metrics
    print("[4/4] Saving results...")
    combined = original_metrics.copy()
    combined[f'gcs_steps_{ref_label}_geo'] = gcs_results['gcs_steps']
    combined[f'path_gcs_scores_{ref_label}_geo'] = gcs_results['path_gcs_scores']
    combined['geometry_type'] = gcs_results['geometry_type']
    combined['analysis_params'] = gcs_results['analysis_params']
    combined['reference_epoch'] = reference_epoch
    combined['reference_label'] = ref_label

    # Save
    output_path = exp_dir / f'training_metrics_{ref_label}_geo.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(combined, f)
    print(f"  Saved: {output_path}")

    # Generate plots
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # GCS evolution
    plot_gcs_3path(
        gcs_results,
        title=f'{exp_dir.name} - GCS ({ref_label} Geometry)',
        save_path=plots_dir / f'gcs_evolution_{ref_label}.png'
    )

    # GCS vs accuracy if metrics available
    if 'test_accuracies' in original_metrics:
        plot_gcs_with_accuracy(
            original_metrics,
            gcs_results,
            title=f'{exp_dir.name} - GCS vs Accuracy ({ref_label})',
            save_path=plots_dir / f'gcs_vs_accuracy_{ref_label}.png'
        )

    print(f"\nReanalysis complete: {exp_dir.name}")
    return combined


def main():
    parser = argparse.ArgumentParser(description='Reanalyze experiments with final geometry')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment directory or parent directory')
    parser.add_argument('--recursive', action='store_true',
                        help='Process all experiments in subdirectories')
    parser.add_argument('--N', type=int, default=200,
                        help='Number of sample points')
    parser.add_argument('--k', type=int, default=15,
                        help='Number of neighbors')
    parser.add_argument('--d', type=int, default=8,
                        help='Intrinsic dimension')
    parser.add_argument('--checkpoint-freq', type=int, default=200,
                        help='Analysis frequency')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computing device')
    parser.add_argument('--alignment-method', type=str, default='cosine',
                        choices=['cosine', 'procrustes'],
                        help='Alignment method: cosine (original) or procrustes')
    parser.add_argument('--reference-epoch', type=int, default=None,
                        help='Epoch to use as reference geometry (default: final checkpoint)')

    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    exp_path = Path(args.exp_dir)

    # Find experiment directories
    if args.recursive:
        exp_dirs = [p.parent for p in exp_path.rglob('training_metrics.pkl')]
        print(f"Found {len(exp_dirs)} experiments")
    else:
        exp_dirs = [exp_path]

    # Process each
    results = {}
    for exp_dir in exp_dirs:
        try:
            result = reanalyze_experiment(
                exp_dir,
                N=args.N,
                k=args.k,
                d=args.d,
                checkpoint_freq=args.checkpoint_freq,
                device=args.device,
                alignment_method=args.alignment_method,
                reference_epoch=args.reference_epoch
            )
            results[str(exp_dir)] = result
        except Exception as e:
            print(f"Failed: {exp_dir} - {e}")
            results[str(exp_dir)] = {'error': str(e)}

    # Summary
    print(f"\n{'='*70}")
    print("REANALYSIS COMPLETE")
    print(f"{'='*70}")
    success = sum(1 for r in results.values() if r and 'error' not in r)
    print(f"Successful: {success}/{len(results)}")


if __name__ == '__main__':
    main()
