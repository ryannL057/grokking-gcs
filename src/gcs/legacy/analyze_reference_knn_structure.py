#!/usr/bin/env python3
"""
Experiment 1: Verify that the reference model's k-NN graph encodes Fourier structure.

For the converged model, extract representations, build k-NN graph,
and check if neighbors share the same modular arithmetic answer.

Also analyzes: Fourier basis alignment of representations, and
how the k-NN neighborhood structure relates to task-relevant groupings.

Usage:
  python scripts/analyze_reference_knn_structure.py \
    --source-dir checkpoints/1layer/SiLU/add \
    --device cuda:0
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

pass  # Imports use the installed package.

from gcs.core.models import Transformer
from gcs.core.data import create_data
from gcs.analysis.geometry import sample_data_points, build_knn_graph


def load_model_and_data(source_dir, device, checkpoint_epoch=None):
    """Load model from source directory, return (model, train_data, test_data, config)."""
    source_dir = Path(source_dir)
    ckpt_dir = source_dir / 'checkpoints'

    # Find checkpoint
    if checkpoint_epoch is not None:
        ckpt_path = list(ckpt_dir.glob(f"*epoch_{checkpoint_epoch:05d}.pth"))
        if not ckpt_path:
            raise FileNotFoundError(f"No checkpoint for epoch {checkpoint_epoch}")
        ckpt_path = ckpt_path[0]
    else:
        # Use final checkpoint
        ckpts = sorted(ckpt_dir.glob("*.pth"), key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit() else 0)
        ckpt_path = ckpts[-1]

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = ckpt.get('model_config', {})

    # Infer config
    act_type = model_config.get('act_type', 'SiLU')
    task = source_dir.name.lower()
    if task not in ('add', 'sub', 'mul', 'div'):
        task = 'add'

    config = {
        'p': model_config.get('d_vocab', 114) - 1,
        'seed': 777,
        'fraction': 0.3,
        'task': task,
        'act_type': act_type,
        'num_layers': model_config.get('num_layers', 1),
        'd_vocab': model_config.get('d_vocab', 114),
        'd_model': model_config.get('d_model', 128),
        'd_mlp': model_config.get('d_mlp', 512),
        'd_head': model_config.get('d_head', 32),
        'num_heads': model_config.get('num_heads', 4),
        'n_ctx': model_config.get('n_ctx', 3),
        'use_ln': model_config.get('use_ln', False),
        'attention_rate': model_config.get('attention_rate', 1.0),
    }

    # Setup environment
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    torch.set_default_dtype(torch.float64)

    # Create data
    train_data, test_data = create_data(
        p=config['p'], fraction=config['fraction'],
        seed=config['seed'], device=device, task=config['task'],
    )

    # Build model
    model = Transformer(
        num_layers=config['num_layers'], d_vocab=config['d_vocab'],
        d_model=config['d_model'], d_mlp=config['d_mlp'],
        d_head=config['d_head'], num_heads=config['num_heads'],
        n_ctx=config['n_ctx'], act_type=config['act_type'],
        use_cache=False, use_ln=config['use_ln'],
        attention_rate=config.get('attention_rate', 1.0),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, train_data, test_data, config, ckpt_path


def compute_answers(prompts, p, task):
    """Compute (a op b) mod p for each prompt."""
    a = prompts[:, 0].cpu().long()
    b = prompts[:, 1].cpu().long()
    if task == 'add':
        return ((a + b) % p).numpy()
    elif task == 'sub':
        return ((a - b) % p).numpy()
    elif task == 'mul':
        return ((a * b) % p).numpy()
    elif task == 'div':
        results = np.zeros(len(a), dtype=np.int64)
        for i in range(len(a)):
            if b[i] == 0:
                results[i] = 0
            else:
                b_inv = pow(int(b[i]), p - 2, p)
                results[i] = (int(a[i]) * b_inv) % p
        return results
    else:
        raise ValueError(f"Unknown task: {task}")


def analyze_knn_answer_consistency(knn_indices, answers):
    """For each point, check what fraction of its k-NN neighbors share the same answer."""
    N, k = knn_indices.shape
    consistencies = np.zeros(N)
    for i in range(N):
        neighbor_answers = answers[knn_indices[i]]
        consistencies[i] = np.mean(neighbor_answers == answers[i])
    return consistencies


def analyze_knn_operand_structure(knn_indices, prompts_np):
    """Check if neighbors share operands (a or b values)."""
    N, k = knn_indices.shape
    a_vals = prompts_np[:, 0]
    b_vals = prompts_np[:, 1]

    a_consist = np.zeros(N)
    b_consist = np.zeros(N)
    for i in range(N):
        a_consist[i] = np.mean(a_vals[knn_indices[i]] == a_vals[i])
        b_consist[i] = np.mean(b_vals[knn_indices[i]] == b_vals[i])
    return a_consist, b_consist


def compute_fourier_content(X_sample, p):
    """Compute how much of the representation lies in Fourier modes.

    For modular arithmetic, the key Fourier basis consists of
    sin(2*pi*k*n/p) and cos(2*pi*k*n/p) for frequencies k.

    We check if the k-NN structure clusters points by their
    position on Fourier circles.
    """
    X_np = X_sample.cpu().numpy()  # [N, d_model]

    # SVD to get principal directions
    X_centered = X_np - X_np.mean(axis=0)
    U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)

    # Effective dimensionality
    S_norm = S / S.sum()
    eff_dim = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-12)))

    # Energy concentration: fraction in top components
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)

    return {
        'singular_values': S,
        'effective_dim': eff_dim,
        'cumulative_variance': cumvar,
        'top5_energy': cumvar[4] if len(cumvar) >= 5 else cumvar[-1],
        'top10_energy': cumvar[9] if len(cumvar) >= 10 else cumvar[-1],
    }


def random_baseline_consistency(answers, k, n_trials=1000):
    """Compute expected answer consistency for random k-NN graphs."""
    N = len(answers)
    consistencies = np.zeros(n_trials)
    for t in range(n_trials):
        i = np.random.randint(N)
        neighbors = np.random.choice(N, k, replace=False)
        consistencies[t] = np.mean(answers[neighbors] == answers[i])
    return consistencies.mean(), consistencies.std()


def main():
    parser = argparse.ArgumentParser(description="Analyze reference k-NN Fourier structure")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--N", type=int, default=200, help="Number of sample points")
    parser.add_argument("--k", type=int, default=15, help="k for k-NN")
    parser.add_argument("--all-train", action="store_true",
                        help="Use ALL training points instead of N samples")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    print("=" * 70)
    print("Reference k-NN Fourier Structure Analysis")
    print("=" * 70)

    model, train_data, test_data, config, ckpt_path = load_model_and_data(source_dir, device)
    p = config['p']
    task = config['task']
    print(f"Model: {config['act_type']}/{task}, p={p}, checkpoint={ckpt_path.name}")

    # --- Extract representations ---
    if args.all_train:
        # Use all training data
        N_use = len(train_data)
        all_prompts = []
        for idx in range(N_use):
            prompt, _ = train_data[idx]
            all_prompts.append(prompt)
        X_prompts = torch.stack(all_prompts).to(device).to(torch.float64)

        with torch.no_grad():
            model.eval()
            cache = {}
            model.cache_all(cache, incl_bwd=False)
            _ = model(X_prompts.to(torch.int64))
            layer_idx = len(model.blocks) - 1
            hook_name = f'blocks.{layer_idx}.hook_resid_post'
            X_sample = cache[hook_name][:, -1, :]
            model.remove_all_hooks()
        print(f"Extracted ALL {N_use} training representations: {X_sample.shape}")
    else:
        # Use standard sampling (same as GCS)
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        X_sample, X_prompts = sample_data_points(model, train_data, N=args.N, device=device)
        N_use = args.N
        print(f"Sampled {N_use} representations: {X_sample.shape}")

    # --- Compute answers ---
    answers = compute_answers(X_prompts, p, task)
    prompts_np = X_prompts.cpu().numpy().astype(int)
    n_unique_answers = len(np.unique(answers))
    print(f"Unique answers in sample: {n_unique_answers} / {p}")

    # --- Build k-NN graph ---
    knn_indices, knn_distances = build_knn_graph(X_sample, k=args.k)
    print(f"Built k-NN graph: k={args.k}")

    # --- Answer consistency ---
    consistencies = analyze_knn_answer_consistency(knn_indices, answers)
    mean_consist = consistencies.mean()
    rand_mean, rand_std = random_baseline_consistency(answers, args.k)

    print(f"\n{'='*50}")
    print(f"ANSWER CONSISTENCY IN k-NN NEIGHBORHOODS")
    print(f"{'='*50}")
    print(f"  Mean answer consistency: {mean_consist:.4f}")
    print(f"  Random baseline:         {rand_mean:.4f} +/- {rand_std:.4f}")
    print(f"  Ratio (actual/random):   {mean_consist/rand_mean:.1f}x")
    print(f"  Points with >50% same answer neighbors: {(consistencies > 0.5).sum()}/{N_use}")
    print(f"  Points with >80% same answer neighbors: {(consistencies > 0.8).sum()}/{N_use}")
    print(f"  Points with 100% same answer neighbors: {(consistencies == 1.0).sum()}/{N_use}")

    # --- Operand structure ---
    a_consist, b_consist = analyze_knn_operand_structure(knn_indices, prompts_np)
    print(f"\n  Operand a consistency: {a_consist.mean():.4f}")
    print(f"  Operand b consistency: {b_consist.mean():.4f}")

    # --- Representation structure ---
    fourier_info = compute_fourier_content(X_sample, p)
    print(f"\nREPRESENTATION STRUCTURE")
    print(f"  Effective dimensionality: {fourier_info['effective_dim']:.1f}")
    print(f"  Top-5 components energy:  {fourier_info['top5_energy']:.4f}")
    print(f"  Top-10 components energy: {fourier_info['top10_energy']:.4f}")

    # --- Compare with intermediate checkpoints ---
    print(f"\n{'='*50}")
    print(f"k-NN CONSISTENCY ACROSS TRAINING")
    print(f"{'='*50}")
    ckpt_dir = source_dir / 'checkpoints'
    import re
    checkpoints_to_test = [0, 400, 1000, 2000, 5000, 8000, 10000, 10400, 12000, 15000, 19800]

    epoch_results = []
    for ep in checkpoints_to_test:
        ckpt_files = list(ckpt_dir.glob(f"*epoch_{ep:05d}.pth"))
        if not ckpt_files:
            continue
        ckpt = torch.load(ckpt_files[0], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # Re-seed for consistent sampling
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        np.random.seed(config['seed'])

        if args.all_train:
            with torch.no_grad():
                cache = {}
                model.cache_all(cache, incl_bwd=False)
                _ = model(X_prompts.to(torch.int64))
                layer_idx = len(model.blocks) - 1
                X_ep = cache[f'blocks.{layer_idx}.hook_resid_post'][:, -1, :]
                model.remove_all_hooks()
        else:
            X_ep, _ = sample_data_points(model, train_data, N=args.N, device=device)

        knn_ep, _ = build_knn_graph(X_ep, k=args.k)
        consist_ep = analyze_knn_answer_consistency(knn_ep, answers)
        fc = compute_fourier_content(X_ep, p)

        print(f"  Epoch {ep:>6}: answer_consist={consist_ep.mean():.4f}  "
              f"eff_dim={fc['effective_dim']:.1f}  "
              f"top5_energy={fc['top5_energy']:.4f}")
        epoch_results.append({
            'epoch': ep,
            'answer_consistency': consist_ep.mean(),
            'effective_dim': fc['effective_dim'],
            'top5_energy': fc['top5_energy'],
        })

    # --- Generate figure ---
    out_path = Path(args.output) if args.output else source_dir / 'knn_fourier_structure.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Answer consistency across training
    if epoch_results:
        eps = [r['epoch'] for r in epoch_results]
        consist_vals = [r['answer_consistency'] for r in epoch_results]
        axes[0].plot(eps, consist_vals, 'o-', color='#2ca02c', linewidth=2, markersize=5)
        axes[0].axhline(rand_mean, color='gray', linestyle='--', label=f'random baseline ({rand_mean:.3f})')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Answer consistency in k-NN')
        axes[0].set_title('(a) k-NN answer consistency across training')
        axes[0].legend(fontsize=8)
        axes[0].set_ylim(0, 1.05)

    # (b) Effective dimensionality across training
    if epoch_results:
        eff_dims = [r['effective_dim'] for r in epoch_results]
        axes[1].plot(eps, eff_dims, 's-', color='#d62728', linewidth=2, markersize=5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Effective dimensionality')
        axes[1].set_title('(b) Representation effective dim')

    # (c) Histogram of answer consistency for final model
    axes[2].hist(consistencies, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
    axes[2].axvline(rand_mean, color='red', linestyle='--', label=f'random ({rand_mean:.3f})')
    axes[2].axvline(mean_consist, color='green', linestyle='-', linewidth=2,
                     label=f'actual mean ({mean_consist:.3f})')
    axes[2].set_xlabel('Fraction of k-NN with same answer')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f'(c) Final model k-NN answer consistency')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved figure to {out_path}")
    plt.close()

    # --- Save results ---
    results = {
        'config': config,
        'N': N_use,
        'k': args.k,
        'mean_answer_consistency': float(mean_consist),
        'random_baseline': float(rand_mean),
        'ratio': float(mean_consist / rand_mean),
        'consistencies': consistencies,
        'epoch_results': epoch_results,
        'fourier_info': fourier_info,
    }
    results_path = out_path.with_suffix('.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
