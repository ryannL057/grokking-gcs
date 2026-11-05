
from gcs.runtime import data_root as _gcs_data_root
#!/usr/bin/env python3
"""
K ablation study: Test GCS sensitivity to neighborhood size k.

Rebuilds k-NN graph and tangent spaces for each k value,
keeping the same sample points (X_sample) fixed.

Usage:
    python scripts/ablation_k.py --experiment checkpoints/1layer/SiLU/add --k-values 5 10 15 20 30
"""

import argparse
import io
import sys
import pickle
import re
import copy
from pathlib import Path

import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer
from gcs.analysis.geometry import build_knn_graph, compute_tangent_space
from gcs.analysis import analyze_3path_gmn_tsa
from gcs.vis import plot_ablation_3path


def load_geometry_pkl(path, device='cuda:0'):
    """Load a pickle file containing torch tensors, remapping CUDA devices."""
    import torch.storage as storage
    original = storage._load_from_bytes

    def patched_load(b):
        return torch.load(io.BytesIO(b), map_location=device, weights_only=False)

    storage._load_from_bytes = patched_load
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    finally:
        storage._load_from_bytes = original


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


def check_knn_connectivity(knn_indices):
    """Check whether the symmetrised k-NN graph is connected."""
    N = knn_indices.shape[0]
    rows, cols = [], []
    for i in range(N):
        for j in knn_indices[i]:
            rows.append(i)
            cols.append(int(j))
            rows.append(int(j))
            cols.append(i)
    data = np.ones(len(rows))
    graph = csr_matrix((data, (rows, cols)), shape=(N, N))
    n_components, _ = connected_components(graph, directed=False)
    return n_components == 1, int(n_components)


def rebuild_geometry_for_k(fixed_geometry, new_k, device):
    """
    Rebuild k-NN graph and tangent bases with a new k value,
    keeping the same sample points fixed.

    Args:
        fixed_geometry: Original geometry from final model
        new_k: New neighborhood size
        device: Computing device

    Returns:
        Updated geometry with new k-NN graph and tangent bases
    """
    new_geometry = copy.deepcopy(fixed_geometry)

    X_sample = new_geometry['X_sample'].to(device=device, dtype=torch.float64)
    X_embeddings = [emb.to(device=device, dtype=torch.float64)
                    for emb in new_geometry['X_embeddings']]
    N = new_geometry['N']
    d = new_geometry['d']

    new_geometry['X_sample'] = X_sample
    new_geometry['X_embeddings'] = X_embeddings

    # Rebuild k-NN graph with new k
    knn_indices, knn_distances = build_knn_graph(X_sample, k=new_k)
    new_geometry['knn_indices'] = knn_indices
    new_geometry['knn_distances'] = knn_distances
    new_geometry['k'] = new_k

    # Check connectivity
    is_connected, n_components = check_knn_connectivity(knn_indices)
    print(f"    k={new_k}: connected={is_connected} (components={n_components})")

    # Recompute tangent bases with new k-NN graph
    new_tangent_bases = []
    for i in range(N):
        tangent_basis = compute_tangent_space(
            i=i,
            X_sample=X_sample,
            knn_graph=knn_indices,
            d=d
        )
        new_tangent_bases.append(tangent_basis)

    new_geometry['all_input_tangent_bases'] = new_tangent_bases

    return new_geometry, is_connected, n_components


def run_k_ablation(experiment_dir, k_values, checkpoint_freq=200, device='cuda:0'):
    """
    Run k ablation study using fixed sample points.

    Args:
        experiment_dir: Experiment directory
        k_values: List of k values to test
        checkpoint_freq: Analyze every N epochs
        device: Computing device
    """
    torch.set_default_dtype(torch.float64)
    exp_dir = Path(experiment_dir)

    print(f"{'='*60}")
    print(f"K Ablation Study (Fixed Sample Points)")
    print(f"{'='*60}")
    print(f"Experiment: {exp_dir.name}")
    print(f"K values: {k_values}")

    # Load fixed geometry
    geometry_file = exp_dir / 'final_geometry.pkl'
    if not geometry_file.exists():
        print(f"Error: No final_geometry.pkl found")
        print("Run training with --compute-gcs first")
        return None

    fixed_geometry = load_geometry_pkl(geometry_file, device=device)

    print(f"Loaded geometry: N={fixed_geometry['N']}, d={fixed_geometry['d']}")
    print(f"Original k={fixed_geometry.get('k', 'N/A')}")

    # Get checkpoints
    checkpoint_dir = exp_dir / 'checkpoints'
    all_ckpts = sorted(checkpoint_dir.glob('*.pth'), key=lambda x: extract_epoch(x.name))
    selected = [(extract_epoch(c.name), c) for c in all_ckpts
                if extract_epoch(c.name) % checkpoint_freq == 0]

    print(f"Checkpoints: {len(selected)}")

    all_results = {}
    connectivity_info = {}

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Testing k={k}")
        print(f"{'='*60}")

        results = {
            'epochs': [],
            'overall_gcs': [],
            'attention_gcs': [],
            'mlp_gcs': []
        }

        # Rebuild geometry once per k value
        geometry_k, is_connected, n_components = rebuild_geometry_for_k(
            fixed_geometry, k, device)
        connectivity_info[k] = {
            'is_connected': is_connected,
            'n_components': n_components
        }

        if not is_connected:
            print(f"  WARNING: k-NN graph is disconnected ({n_components} components) at k={k}")

        for epoch, ckpt_path in selected:
            try:
                model = load_checkpoint(ckpt_path, device)

                path_results = analyze_3path_gmn_tsa(
                    model=model,
                    complete_geometry=geometry_k,
                    target_layer=0,
                    device=device
                )

                results['epochs'].append(epoch)
                results['overall_gcs'].append(path_results.get('overall_path'))
                results['attention_gcs'].append(path_results.get('attention_path'))
                results['mlp_gcs'].append(path_results.get('mlp_path'))

                if epoch % 2000 == 0:
                    print(f"  Epoch {epoch}: overall={path_results.get('overall_path', 0):.2f}, "
                          f"attn={path_results.get('attention_path', 0):.2f}, "
                          f"mlp={path_results.get('mlp_path', 0):.2f}")

            except Exception as e:
                print(f"  Epoch {epoch} failed: {e}")
                import traceback
                traceback.print_exc()

        all_results[k] = results
        print(f"  Completed: {len(results['epochs'])} epochs")

    # Compute pairwise correlations
    print(f"\n{'='*60}")
    print(f"Pairwise Spearman Correlations")
    print(f"{'='*60}")

    from scipy.stats import spearmanr
    for path_key in ['overall_gcs', 'attention_gcs', 'mlp_gcs']:
        print(f"\n  {path_key}:")
        sorted_k = sorted(all_results.keys())
        for i, k1 in enumerate(sorted_k):
            for k2 in sorted_k[i+1:]:
                s1 = np.array(all_results[k1][path_key], dtype=float)
                s2 = np.array(all_results[k2][path_key], dtype=float)
                # Handle potential length mismatches
                min_len = min(len(s1), len(s2))
                rho, pval = spearmanr(s1[:min_len], s2[:min_len])
                print(f"    k={k1} vs k={k2}: ρ={rho:.4f} (p={pval:.2e})")

    # Save results
    output_dir = exp_dir / 'k_ablation_study'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump({
            'k_values': k_values,
            'results': all_results,
            'connectivity': connectivity_info
        }, f)

    # Plot
    plot_ablation_3path(
        all_results,
        param_name='k',
        title='K Ablation Study (Fixed Sample Points)',
        save_path=output_dir / 'k_ablation.png'
    )

    print(f"\n{'='*60}")
    print(f"K Ablation Complete!")
    print(f"Connectivity: {connectivity_info}")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='K (Neighborhood Size) Ablation Study')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment directory (must have final_geometry.pkl)')
    parser.add_argument('--k-values', nargs='+', type=int,
                        default=[5, 10, 15, 20, 30],
                        help='K values to test (default: 5 10 15 20 30)')
    parser.add_argument('--checkpoint-freq', type=int, default=200,
                        help='Checkpoint sampling frequency')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computing device')

    args = parser.parse_args()

    run_k_ablation(
        experiment_dir=args.experiment,
        k_values=args.k_values,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device
    )


if __name__ == '__main__':
    main()
