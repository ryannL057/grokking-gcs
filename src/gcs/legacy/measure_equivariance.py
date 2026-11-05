#!/usr/bin/env python3
"""
Route D: Measure attention-output operand-swap equivariance across training.

For each checkpoint, computes:
  E_{a,b}[ ||attn_out(a,b,=) - attn_out(b,a,=)|| ] / E_{a,b}[ ||attn_out(a,b,=)|| ]

This normalized metric → 0 means the attention output is perfectly equivariant
to operand swap; → large means it distinguishes operand order.

Prediction: commutative tasks (add/mul) develop equivariance during grokking;
non-commutative tasks (sub/div) do not.

Usage:
    python scripts/measure_equivariance.py
    python scripts/measure_equivariance.py --tasks add,mul,sub,div --seeds 777,42
    python scripts/measure_equivariance.py --activation ReLU
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer
from gcs.core.data import create_data
from gcs.legacy.diagnostics_common import RUNS_1L, resolve_path, detect_grok_epoch, load_metrics


def build_swap_pairs(p: int, device: str = 'cpu'):
    """Build (a,b,=) and (b,a,=) prompt tensors for all pairs."""
    equals_token = p
    a_vals, b_vals = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    a = a_vals.flatten()
    b = b_vals.flatten()
    eq = torch.full_like(a, equals_token)

    prompts_ab = torch.stack([a, b, eq], dim=1).to(device)  # [p*p, 3]
    prompts_ba = torch.stack([b, a, eq], dim=1).to(device)  # [p*p, 3]
    return prompts_ab, prompts_ba


def measure_attn_equivariance(model, prompts_ab, prompts_ba, device='cpu', batch_size=4096):
    """Compute normalized attention-output equivariance.

    Returns (equivariance_distance, norm_scale):
        equivariance_distance = mean ||attn_out(a,b) - attn_out(b,a)|| at position 2
        norm_scale = mean ||attn_out(a,b)|| at position 2
    """
    model.eval()
    total_diff = 0.0
    total_norm = 0.0
    n = prompts_ab.shape[0]

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ab = prompts_ab[start:end]
            ba = prompts_ba[start:end]

            # Forward with cache
            cache_ab = {}
            model.cache_all(cache_ab)
            model(ab)
            model.remove_all_hooks()
            attn_out_ab = cache_ab['blocks.0.hook_attn_out'][:, 2, :]  # [batch, d_model]

            cache_ba = {}
            model.cache_all(cache_ba)
            model(ba)
            model.remove_all_hooks()
            attn_out_ba = cache_ba['blocks.0.hook_attn_out'][:, 2, :]

            diff = (attn_out_ab - attn_out_ba).norm(dim=-1)  # [batch]
            total_diff += diff.sum().item()
            total_norm += attn_out_ab.norm(dim=-1).sum().item()

    mean_diff = total_diff / n
    mean_norm = total_norm / n
    return mean_diff, mean_norm


def process_run(run_path: str, p: int = 113, device: str = 'cpu',
                checkpoint_stride: int = 1):
    """Process all checkpoints for a single run.

    Returns dict with epochs, equivariance_distance, norm_scale arrays.
    """
    run_path = resolve_path(run_path)
    ckpt_dir = run_path / 'checkpoints'
    if not ckpt_dir.exists():
        print(f"  No checkpoints dir: {run_path}")
        return None

    ckpt_files = sorted(ckpt_dir.glob('*.pth'))
    if not ckpt_files:
        print(f"  No checkpoint files: {ckpt_dir}")
        return None

    # Load first checkpoint to get config — handle both old and new formats
    first_ckpt = torch.load(ckpt_files[0], map_location=device, weights_only=False)
    # Old format: 'model_config'; new format: 'config'
    cfg = first_ckpt.get('config') or first_ckpt.get('model_config', {})
    p_cfg = cfg.get('p', p)
    d_vocab = cfg.get('d_vocab', p_cfg + 1)
    p = p_cfg

    model = Transformer(
        num_layers=cfg.get('num_layers', 1),
        d_vocab=d_vocab,
        d_model=cfg.get('d_model', 128),
        d_mlp=cfg.get('d_mlp', 512),
        d_head=cfg.get('d_head', 32),
        num_heads=cfg.get('num_heads', 4),
        n_ctx=cfg.get('n_ctx', 3),
        act_type=cfg.get('act_type', 'SiLU'),
    ).to(device)

    prompts_ab, prompts_ba = build_swap_pairs(p, device=device)

    epochs = []
    equi_dists = []
    norm_scales = []

    for i, ckpt_file in enumerate(ckpt_files):
        if i % checkpoint_stride != 0 and i != len(ckpt_files) - 1:
            continue
        try:
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            epoch = ckpt['epoch']
            model.load_state_dict(ckpt['model_state_dict'])
            diff, norm = measure_attn_equivariance(model, prompts_ab, prompts_ba, device=device)
            epochs.append(epoch)
            equi_dists.append(diff)
            norm_scales.append(norm)
        except Exception as e:
            print(f"  Failed {ckpt_file.name}: {e}")
            continue

    return {
        'epochs': np.array(epochs),
        'equi_dist': np.array(equi_dists),
        'norm_scale': np.array(norm_scales),
    }


def main():
    parser = argparse.ArgumentParser(description='Measure attention equivariance')
    parser.add_argument('--tasks', type=str, default='add,mul,sub,div')
    parser.add_argument('--seeds', type=str, default='777,42,123')
    parser.add_argument('--activation', type=str, default='SiLU')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--stride', type=int, default=1,
                        help='Process every Nth checkpoint (1=all)')
    parser.add_argument('--output', type=str, default='experiments/equivariance_analysis')
    args = parser.parse_args()

    tasks = args.tasks.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build run lookup
    run_lookup = {}
    for act, task, seed, path in RUNS_1L:
        if act == args.activation:
            run_lookup[(task, seed)] = path

    # Collect results
    all_results = {}
    for task in tasks:
        for seed in seeds:
            key = (task, seed)
            if key not in run_lookup:
                print(f"Skipping {args.activation}/{task}/s{seed}: not in RUNS_1L")
                continue
            run_path = run_lookup[key]
            print(f"Processing {args.activation}/{task}/s{seed} ...")
            result = process_run(run_path, device=args.device,
                                 checkpoint_stride=args.stride)
            if result is not None:
                # Get t_grok
                try:
                    metrics = load_metrics(run_path)
                    t_grok = detect_grok_epoch(metrics)
                except Exception:
                    t_grok = None
                result['t_grok'] = t_grok
                all_results[key] = result
                n_rel = result['equi_dist'] / np.where(result['norm_scale'] > 1e-8,
                                                        result['norm_scale'], 1.0)
                print(f"  {len(result['epochs'])} checkpoints, "
                      f"t_grok={t_grok}, "
                      f"final relative equi={n_rel[-1]:.4f}")

    # Save raw results
    with open(output_dir / 'equivariance_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved: {output_dir / 'equivariance_results.pkl'}")

    # --- Visualization ---
    COMMUTATIVE = {'add', 'mul'}
    comm_color = '#2196F3'
    noncomm_color = '#FF5722'

    # Figure 1: Relative equivariance over training, grouped by commutativity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (group_name, group_tasks) in zip(axes, [
        ('Commutative (add, mul)', ['add', 'mul']),
        ('Non-commutative (sub, div)', ['sub', 'div']),
    ]):
        color = comm_color if group_tasks[0] in COMMUTATIVE else noncomm_color
        for task in group_tasks:
            for seed in seeds:
                key = (task, seed)
                if key not in all_results:
                    continue
                r = all_results[key]
                rel_equi = r['equi_dist'] / np.where(r['norm_scale'] > 1e-8,
                                                      r['norm_scale'], 1.0)
                alpha = 0.8 if seed == 777 else 0.4
                ls = '-' if task == group_tasks[0] else '--'
                label = f"{task}/s{seed}" if seed == 777 else None
                ax.plot(r['epochs'], rel_equi, color=color, alpha=alpha,
                        linestyle=ls, linewidth=1.5, label=label)

                # Mark t_grok
                if r['t_grok'] is not None:
                    idx = np.argmin(np.abs(r['epochs'] - r['t_grok']))
                    ax.axvline(r['t_grok'], color=color, alpha=0.15, linestyle=':')

        ax.set_title(group_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Relative equivariance distance\n'
                       r'$\Vert A(a,b) - A(b,a) \Vert / \Vert A(a,b) \Vert$'
                       if ax == axes[0] else '')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(f'{args.activation} 1-Layer: Attention Operand-Swap Equivariance',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = output_dir / f'equivariance_trajectory_{args.activation}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")

    # Figure 2: Pre-grok vs post-grok equivariance comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    pre_comm, post_comm = [], []
    pre_nonc, post_nonc = [], []

    for (task, seed), r in all_results.items():
        if r['t_grok'] is None:
            continue
        rel_equi = r['equi_dist'] / np.where(r['norm_scale'] > 1e-8,
                                              r['norm_scale'], 1.0)
        # Pre-grok: average over first half of epochs before t_grok
        pre_mask = r['epochs'] < r['t_grok'] * 0.5
        post_mask = r['epochs'] > r['t_grok'] * 1.5
        if pre_mask.sum() < 2 or post_mask.sum() < 2:
            continue
        pre_val = np.mean(rel_equi[pre_mask])
        post_val = np.mean(rel_equi[post_mask])
        if task in COMMUTATIVE:
            pre_comm.append(pre_val)
            post_comm.append(post_val)
        else:
            pre_nonc.append(pre_val)
            post_nonc.append(post_val)

    x = [0, 1, 3, 4]
    heights = [
        np.mean(pre_comm) if pre_comm else 0,
        np.mean(post_comm) if post_comm else 0,
        np.mean(pre_nonc) if pre_nonc else 0,
        np.mean(post_nonc) if post_nonc else 0,
    ]
    errs = [
        np.std(pre_comm)/max(np.sqrt(len(pre_comm)),1) if pre_comm else 0,
        np.std(post_comm)/max(np.sqrt(len(post_comm)),1) if post_comm else 0,
        np.std(pre_nonc)/max(np.sqrt(len(pre_nonc)),1) if pre_nonc else 0,
        np.std(post_nonc)/max(np.sqrt(len(post_nonc)),1) if post_nonc else 0,
    ]
    colors = [comm_color, comm_color, noncomm_color, noncomm_color]
    hatches = ['///', '', '///', '']
    labels_x = ['Pre-grok', 'Post-grok', 'Pre-grok', 'Post-grok']

    bars = ax.bar(x, heights, yerr=errs, color=colors, alpha=0.7,
                  edgecolor='black', linewidth=0.5, capsize=4)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylabel('Relative equivariance distance')
    ax.text(0.5, -0.12, 'Commutative (add/mul)', transform=ax.transAxes,
            ha='left', fontsize=11, color=comm_color, fontweight='bold')
    ax.text(3.5/5, -0.12, 'Non-commutative (sub/div)', transform=ax.transAxes,
            ha='left', fontsize=11, color=noncomm_color, fontweight='bold')
    ax.set_title(f'{args.activation} 1L: Equivariance Before vs After Grokking',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path2 = output_dir / f'equivariance_pre_post_{args.activation}.png'
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path2}")

    # Print summary stats
    print("\n" + "=" * 60)
    print("Summary: Pre-grok vs Post-grok Equivariance Distance")
    print("=" * 60)
    if pre_comm and post_comm:
        drop_comm = (np.mean(pre_comm) - np.mean(post_comm)) / np.mean(pre_comm) * 100
        print(f"  Commutative:     pre={np.mean(pre_comm):.4f} → post={np.mean(post_comm):.4f} "
              f"({drop_comm:+.1f}%)")
    if pre_nonc and post_nonc:
        drop_nonc = (np.mean(pre_nonc) - np.mean(post_nonc)) / np.mean(pre_nonc) * 100
        print(f"  Non-commutative: pre={np.mean(pre_nonc):.4f} → post={np.mean(post_nonc):.4f} "
              f"({drop_nonc:+.1f}%)")
    print()
    print("Prediction: commutative tasks show larger drop (more equivariant post-grok)")


if __name__ == '__main__':
    main()
