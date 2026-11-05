
from gcs.runtime import data_root as _gcs_data_root
#!/usr/bin/env python3
"""
Reference ablation experiment: verify GCS conclusions are stable
across different reference epochs in the converged regime.

Computes GCS trajectories using multiple reference geometries and
reports k-NN diagnostics (Jaccard overlap, connectivity, spectral
separation) to demonstrate that the choice of converged reference
does not affect qualitative conclusions (U-shape, path ordering).

Usage:
    python scripts/ablation_reference.py \
        --exp-dir checkpoints/1layer/SiLU/add \
        --ref-epochs 12000,16000,19999 \
        --device cuda:0
"""

import argparse
import sys
import pickle
import numpy as np
from pathlib import Path

import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

pass  # Package imports do not search the experiment-data directory.

from gcs.legacy.reanalyze_gcs import reanalyze_experiment


# ---------------------------------------------------------------------------
# k-NN diagnostics
# ---------------------------------------------------------------------------

def compute_jaccard_overlap(knn_a, knn_b):
    """
    Per-point Jaccard overlap between two k-NN index arrays.

    Args:
        knn_a: [N, k] neighbor indices from geometry A
        knn_b: [N, k] neighbor indices from geometry B

    Returns:
        (mean_jaccard, std_jaccard) across N points
    """
    N = knn_a.shape[0]
    jaccards = np.empty(N)
    for i in range(N):
        set_a = set(knn_a[i])
        set_b = set(knn_b[i])
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccards[i] = intersection / union if union > 0 else 0.0
    return float(jaccards.mean()), float(jaccards.std())


def check_knn_connectivity(knn_indices):
    """
    Check whether the k-NN graph (symmetrised / undirected) is connected.

    Args:
        knn_indices: [N, k] neighbor indices

    Returns:
        (is_connected, num_components)
    """
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


def compute_spectral_separation(geometry):
    """
    Local spectral separation sigma_1 / sigma_d for each point.

    Re-performs SVD on the centred k-NN neighbourhoods stored in the
    geometry dict and returns the mean and std of the gap ratio.

    Args:
        geometry: dict with X_sample, knn_indices, d

    Returns:
        (mean_ratio, std_ratio)
    """
    X_sample = geometry['X_sample']
    knn_indices = geometry['knn_indices']
    d = geometry['d']
    N = geometry['N']

    ratios = []
    for i in range(N):
        neighborhood = X_sample[knn_indices[i]]
        centered_np = (neighborhood - X_sample[i]).cpu().numpy()
        _, S, _ = np.linalg.svd(centered_np, full_matrices=False)
        if len(S) >= d and S[d - 1] > 1e-12:
            ratios.append(float(S[0] / S[d - 1]))

    ratios = np.array(ratios)
    return float(ratios.mean()), float(ratios.std())


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_reference_ablation(exp_dir, ref_epochs, N=200, k=15, d=8,
                           checkpoint_freq=200, device='cuda:0',
                           alignment_method='cosine', seed=42):
    """
    Run reference ablation: compute GCS for each reference epoch and
    report k-NN diagnostics across reference geometries.

    Args:
        exp_dir: experiment directory
        ref_epochs: list of reference epoch numbers
        N, k, d: GMN-TSA parameters
        checkpoint_freq: how often to sample checkpoints for GCS
        device: compute device
        alignment_method: 'cosine' or 'procrustes'
        seed: random seed (fixed so every ref samples the same N points)

    Returns:
        diagnostics dict (also saved to disk)
    """
    exp_dir = Path(exp_dir)
    print(f"\n{'='*70}")
    print(f"Reference Ablation: {exp_dir}")
    print(f"Reference epochs: {ref_epochs}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # Step 1: Compute GCS trajectory for each reference epoch
    # ------------------------------------------------------------------
    ref_results = {}
    for ref_epoch in ref_epochs:
        print(f"\n--- Reference epoch: {ref_epoch} ---")
        # Fix seed so the same N training points are sampled each time.
        # Different reference models will still produce different hidden
        # representations, which is exactly what we want to compare.
        torch.manual_seed(seed)
        np.random.seed(seed)
        result = reanalyze_experiment(
            exp_dir,
            N=N, k=k, d=d,
            checkpoint_freq=checkpoint_freq,
            device=device,
            alignment_method=alignment_method,
            reference_epoch=ref_epoch,
        )
        ref_results[ref_epoch] = result

    # ------------------------------------------------------------------
    # Step 2: Load saved geometries and compute k-NN diagnostics
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Computing k-NN diagnostics...")
    print(f"{'='*70}")

    geometries = {}
    for ref_epoch in ref_epochs:
        ref_label = f"ref{ref_epoch}"
        geo_path = exp_dir / f'{ref_label}_geometry.pkl'
        with open(geo_path, 'rb') as f:
            geometries[ref_epoch] = pickle.load(f)

    # 2a. Pairwise Jaccard overlap
    jaccard_results = {}
    for i, ref_i in enumerate(ref_epochs):
        for j, ref_j in enumerate(ref_epochs):
            if j <= i:
                continue
            knn_a = geometries[ref_i]['knn_indices']
            knn_b = geometries[ref_j]['knn_indices']
            mean_j, std_j = compute_jaccard_overlap(knn_a, knn_b)
            jaccard_results[(ref_i, ref_j)] = {'mean': mean_j, 'std': std_j}
            print(f"  Jaccard({ref_i}, {ref_j}): {mean_j:.4f} +/- {std_j:.4f}")

    # 2b. Connectivity
    connectivity_results = {}
    for ref_epoch in ref_epochs:
        knn = geometries[ref_epoch]['knn_indices']
        is_conn, n_comp = check_knn_connectivity(knn)
        connectivity_results[ref_epoch] = {
            'connected': is_conn,
            'num_components': n_comp,
        }
        status = "CONNECTED" if is_conn else f"DISCONNECTED ({n_comp} components)"
        print(f"  k-NN graph (ref={ref_epoch}): {status}")

    # 2c. Spectral separation sigma_1 / sigma_d
    spectral_results = {}
    for ref_epoch in ref_epochs:
        mean_r, std_r = compute_spectral_separation(geometries[ref_epoch])
        spectral_results[ref_epoch] = {'mean': mean_r, 'std': std_r}
        print(f"  Spectral separation (ref={ref_epoch}): {mean_r:.2f} +/- {std_r:.2f}")

    # ------------------------------------------------------------------
    # Step 3: Save diagnostics
    # ------------------------------------------------------------------
    diagnostics = {
        'ref_epochs': ref_epochs,
        'jaccard_overlap': jaccard_results,
        'connectivity': connectivity_results,
        'spectral_separation': spectral_results,
        'params': {'N': N, 'k': k, 'd': d, 'seed': seed},
    }

    diag_path = exp_dir / 'reference_ablation_diagnostics.pkl'
    with open(diag_path, 'wb') as f:
        pickle.dump(diagnostics, f)
    print(f"\nSaved diagnostics: {diag_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_connected = all(v['connected'] for v in connectivity_results.values())
    all_jaccard_ok = all(v['mean'] > 0.7 for v in jaccard_results.values())
    print(f"  All k-NN graphs connected: {all_connected}")
    print(f"  All Jaccard overlaps > 0.7: {all_jaccard_ok}")
    for (ri, rj), val in jaccard_results.items():
        print(f"    ({ri}, {rj}): {val['mean']:.4f}")

    return diagnostics


def main():
    parser = argparse.ArgumentParser(
        description='Reference ablation: verify GCS stability across reference epochs')
    parser.add_argument('--exp-dir', type=str, required=True,
                        help='Experiment directory')
    parser.add_argument('--ref-epochs', type=str, required=True,
                        help='Comma-separated reference epochs (e.g. 12000,16000,19999)')
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
                        help='Alignment method')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sampling')

    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    ref_epochs = [int(e.strip()) for e in args.ref_epochs.split(',')]

    run_reference_ablation(
        exp_dir=args.exp_dir,
        ref_epochs=ref_epochs,
        N=args.N,
        k=args.k,
        d=args.d,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
        alignment_method=args.alignment_method,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
