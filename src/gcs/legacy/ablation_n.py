
from gcs.runtime import data_root as _gcs_data_root
#!/usr/bin/env python3
"""
N ablation study with one fixed final-model geometry for each sample size.

Usage:
    python ablation_n.py --experiment pretrained/SiLU/silu_* --n-values 100 200 500 1000
"""

import argparse
import sys
import pickle
import re
from pathlib import Path

import torch
import numpy as np

pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer
from gcs.core.data import create_data
from gcs.analysis import build_complete_geometric_structure, analyze_3path_gmn_tsa
from gcs.vis import plot_ablation_3path


def load_checkpoint(checkpoint_path, device='cuda:0'):
    from .reanalyze_gcs import load_checkpoint as load
    return load(checkpoint_path,device)[0]


def extract_epoch(filename):
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0


def run_n_ablation(experiment_dir, n_values, k=15, d=8,
                   checkpoint_freq=200, device='cuda:0', seed=777):
    """
    Run N ablation study.

    Args:
        experiment_dir: Experiment directory with checkpoints
        n_values: List of N values to test
        k: Number of neighbors
        d: Intrinsic dimension
        checkpoint_freq: Analyze every N epochs
        device: Computing device
        seed: Random seed
    """
    exp_dir = Path(experiment_dir)

    print(f"{'='*60}")
    print(f"N Ablation Study")
    print(f"{'='*60}")
    print(f"Experiment: {exp_dir.name}")
    print(f"N values: {n_values}")
    print(f"k={k}, d={d}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    checkpoint_dir = exp_dir / 'checkpoints'
    all_ckpts = sorted(checkpoint_dir.glob('*.pth'), key=lambda x: extract_epoch(x.name))
    if not all_ckpts: raise FileNotFoundError(f'No checkpoints in {checkpoint_dir}')
    from .reanalyze_gcs import load_checkpoint as load
    from gcs.runner import save_geometry
    from gcs.artifacts import sha256
    final_model,config=load(all_ckpts[-1],device)
    train_data,_=create_data(p=config.get('p',final_model.embed.W_E.shape[1]-1),
        fraction=config.get('fraction',.3),seed=config.get('seed',seed),
        device=device,task=config.get('task','add'))
    layer=config.get('num_layers',len(final_model.blocks))-1
    selected = [(extract_epoch(c.name), c) for c in all_ckpts
                if extract_epoch(c.name) % checkpoint_freq == 0]
    output_dir=exp_dir/'n_ablation_study'
    output_dir.mkdir(exist_ok=False)

    print(f"Checkpoints: {len(selected)}")

    all_results = {}

    for N in n_values:
        print(f"\n{'='*60}")
        print(f"Testing N={N}")
        print(f"{'='*60}")

        if k >= N:
            print(f"  Skipping: k={k} >= N={N}")
            continue

        if N > len(train_data):
            print(f"  Skipping: N > training data size")
            continue

        results = {
            'epochs': [],
            'overall_gcs': [],
            'attention_gcs': [],
            'mlp_gcs': []
        }

        torch.manual_seed(seed)
        np.random.seed(seed)
        geometry=build_complete_geometric_structure(final_model,train_data,N=N,k=k,d=d,
                                                    device=device,layer_idx=layer)
        geometry_path=output_dir/f'geometry_N{N}.npz'
        save_geometry(geometry,geometry_path)
        results.update(reference_epoch=extract_epoch(all_ckpts[-1].name),
                       reference_checkpoint_sha256=sha256(all_ckpts[-1]),
                       geometry_sha256=sha256(geometry_path),measurement_layer=layer)
        for epoch,ckpt_path in selected:
            model=load_checkpoint(ckpt_path,device)
            path_results=analyze_3path_gmn_tsa(model,geometry,target_layer=layer,device=device)
            if not all(np.isfinite(path_results[key]) for key in ['overall_path','attention_path','mlp_path']):
                raise RuntimeError(f'Nonfinite GCS at epoch {epoch}')
            results['epochs'].append(epoch)
            for short in ['overall','attention','mlp']:
                results[short+'_gcs'].append(path_results[short+'_path'])

        all_results[N] = results
        print(f"  Completed: {len(results['epochs'])} epochs")

    # Save results

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump({
            'n_values': n_values,
            'k': k,
            'd': d,
            'seed': seed,
            'results': all_results
        }, f)

    # Plot
    plot_ablation_3path(
        all_results,
        param_name='N',
        title='N Ablation Study',
        save_path=output_dir / 'n_ablation.png'
    )

    print(f"\n{'='*60}")
    print(f"N Ablation Complete!")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='N Ablation Study')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment directory')
    parser.add_argument('--n-values', nargs='+', type=int,
                        default=[100, 200, 500, 1000, 1500],
                        help='N values to test')
    parser.add_argument('--k', type=int, default=15,
                        help='Number of neighbors')
    parser.add_argument('--d', type=int, default=8,
                        help='Intrinsic dimension')
    parser.add_argument('--checkpoint-freq', type=int, default=200,
                        help='Checkpoint sampling frequency')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computing device')
    parser.add_argument('--seed', type=int, default=777,
                        help='Random seed')

    args = parser.parse_args()

    run_n_ablation(
        experiment_dir=args.experiment,
        n_values=args.n_values,
        k=args.k,
        d=args.d,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
