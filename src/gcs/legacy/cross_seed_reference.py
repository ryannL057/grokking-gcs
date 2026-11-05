#!/usr/bin/env python3
"""
Cross-seed reference control for GCS trajectory robustness.

For each (activation, task) combination with multiple seeds, computes GCS
trajectories using another seed's final checkpoint as the reference geometry.
Compares "matched reference" (own final) vs "cross-seed reference" (other's final).

Phase 1: 1L add × 3 activations = 9 matched + 18 cross-seed pairs.

Usage:
    python scripts/cross_seed_reference.py --device cuda:0
    python scripts/cross_seed_reference.py --device cuda:0 --tasks add,mul
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm.auto import tqdm
import torch

pass  # Package imports do not search the experiment-data directory.
pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer
from gcs.core.data import create_data
from gcs.analysis import build_complete_geometric_structure, analyze_3path_gmn_tsa
from gcs.legacy.diagnostics_common import (
    RUNS_1L, detect_grok_epoch, load_gcs, load_metrics,
    resolve_path, stable_turning_epoch,
    detect_mlp_u_shape, detect_attention_peak_then_dip,
)


# ---------------------------------------------------------------------------
# Checkpoint utilities (from reanalyze_gcs.py)
# ---------------------------------------------------------------------------

def extract_epoch(filename):
    match = re.search(r'epoch_(\d+)', str(filename))
    return int(match.group(1)) if match else 0


def load_checkpoint(checkpoint_path, device='cuda:0'):
    """Load checkpoint and create model."""
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
    return model, config


# ---------------------------------------------------------------------------
# Core: compute GCS trajectory with a given reference geometry
# ---------------------------------------------------------------------------

def compute_gcs_trajectory(
    target_dir: Path,
    ref_geometry: dict,
    checkpoint_freq: int = 200,
    device: str = 'cuda:0',
) -> dict:
    """Compute 3-path GCS trajectory for target checkpoints using ref_geometry."""
    checkpoint_dir = target_dir / 'checkpoints'
    all_ckpts = sorted(checkpoint_dir.glob('*.pth'), key=lambda x: extract_epoch(x.name))

    selected = [(extract_epoch(c.name), c) for c in all_ckpts
                if extract_epoch(c.name) % checkpoint_freq == 0]

    epochs = []
    scores = {'overall_path': [], 'attention_path': [], 'mlp_path': []}

    for epoch, ckpt_path in tqdm(selected, desc=f'  GCS {target_dir.name}', leave=False):
        try:
            model, _ = load_checkpoint(ckpt_path, device)
            with torch.no_grad():
                model.eval()
                path_results = analyze_3path_gmn_tsa(
                    model, ref_geometry, target_layer=0, device=device,
                    alignment_method='cosine'
                )
            epochs.append(epoch)
            for p in scores:
                scores[p].append(path_results.get(p))
        except Exception as e:
            print(f'    Epoch {epoch} failed: {e}')
            epochs.append(epoch)
            for p in scores:
                scores[p].append(None)

    return {'epochs': np.array(epochs), 'path_gcs_scores': scores}


# ---------------------------------------------------------------------------
# Build reference geometry from a model + data
# ---------------------------------------------------------------------------

def build_reference_geometry(
    model_path: Path,
    train_data,
    device: str = 'cuda:0',
    N: int = 200, k: int = 15, d: int = 8,
) -> dict:
    """Build reference geometry from a checkpoint."""
    model, _ = load_checkpoint(model_path, device)
    geometry = build_complete_geometric_structure(
        model=model,
        train_data=train_data,
        N=N, k=k, d=d,
        device=device,
        layer_idx=0,
        hook_point='hook_resid_post'
    )
    return geometry


# ---------------------------------------------------------------------------
# Group runs by (activation, task)
# ---------------------------------------------------------------------------

def group_runs(tasks: list[str]) -> dict[tuple[str, str], list[tuple[int, str]]]:
    """Group 1L runs by (act, task) → [(seed, base_path), ...]"""
    groups = defaultdict(list)
    for act, task, seed, base in RUNS_1L:
        if task not in tasks:
            continue
        exp_path = resolve_path(base)
        if not exp_path.exists():
            continue
        # Check grokking
        try:
            metrics = load_metrics(base)
            t_grok = detect_grok_epoch(metrics)
            if t_grok is None:
                continue
        except Exception:
            continue
        groups[(act, task)].append((seed, base))
    return dict(groups)


def load_existing_matched_trajectory(base: str | Path, checkpoint_freq: int) -> dict | None:
    """Load a cached matched-reference trajectory if one already exists."""
    try:
        epochs, scores = load_gcs(base)
    except FileNotFoundError:
        return None

    epochs = np.asarray(epochs, dtype=int)
    keep_mask = (epochs % checkpoint_freq) == 0
    if not keep_mask.any():
        return None

    filtered_scores = {}
    for path in ('overall_path', 'attention_path', 'mlp_path'):
        values = list(scores[path])
        filtered_scores[path] = [value for value, keep in zip(values, keep_mask) if keep]

    return {
        'epochs': epochs[keep_mask],
        'path_gcs_scores': filtered_scores,
    }


# ---------------------------------------------------------------------------
# Spearman correlation between two GCS trajectories
# ---------------------------------------------------------------------------

def trajectory_correlation(traj_a: dict, traj_b: dict, path: str = 'overall_path') -> float:
    """Spearman ρ between two GCS trajectories on shared epochs."""
    ep_a = set(traj_a['epochs'].tolist())
    ep_b = set(traj_b['epochs'].tolist())
    shared = sorted(ep_a & ep_b)
    if len(shared) < 5:
        return np.nan

    idx_a = {int(e): i for i, e in enumerate(traj_a['epochs'])}
    idx_b = {int(e): i for i, e in enumerate(traj_b['epochs'])}

    vals_a, vals_b = [], []
    for e in shared:
        va = traj_a['path_gcs_scores'][path][idx_a[e]]
        vb = traj_b['path_gcs_scores'][path][idx_b[e]]
        if va is not None and vb is not None:
            vals_a.append(va)
            vals_b.append(vb)

    if len(vals_a) < 5:
        return np.nan
    rho, _ = stats.spearmanr(vals_a, vals_b)
    return rho


# ---------------------------------------------------------------------------
# Evaluate diagnostics on a trajectory
# ---------------------------------------------------------------------------

def evaluate_trajectory(traj: dict, t_grok: int) -> dict:
    """Check if standard diagnostics hold on a GCS trajectory."""
    epochs = traj['epochs']
    overall = np.array([v if v is not None else np.nan
                       for v in traj['path_gcs_scores']['overall_path']])
    attn = np.array([v if v is not None else np.nan
                    for v in traj['path_gcs_scores']['attention_path']])
    mlp = np.array([v if v is not None else np.nan
                   for v in traj['path_gcs_scores']['mlp_path']])

    # Remove NaN entries
    valid = ~(np.isnan(overall) | np.isnan(attn) | np.isnan(mlp))
    if valid.sum() < 5:
        return dict(lead=np.nan, mlp_ushape=False, attn_peak_dip=False)

    epochs_v = epochs[valid]
    overall_v = overall[valid]
    attn_v = attn[valid]
    mlp_v = mlp[valid]

    t_turn = stable_turning_epoch(overall_v, epochs_v, t_grok)
    lead = t_grok - t_turn
    mlp_ushape, _ = detect_mlp_u_shape(mlp_v, epochs_v, t_grok)
    attn_peak_dip, _, _, _ = detect_attention_peak_then_dip(attn_v, epochs_v, t_grok)

    return dict(lead=lead, mlp_ushape=mlp_ushape, attn_peak_dip=attn_peak_dip)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PATH_COLORS = {
    'overall_path': '#0173B2',
    'attention_path': '#DE8F05',
    'mlp_path': '#029E73',
}


def setup_style():
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.24,
        'grid.linestyle': '--',
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })


def plot_overlay(matched_traj, cross_trajs, act, task, target_seed, out_dir):
    """Overlay matched vs cross-seed trajectories for one target run."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, path in zip(axes, ['overall_path', 'attention_path', 'mlp_path']):
        color = PATH_COLORS[path]
        # Matched
        vals = [v for v in matched_traj['path_gcs_scores'][path] if v is not None]
        ep = [e for e, v in zip(matched_traj['epochs'],
              matched_traj['path_gcs_scores'][path]) if v is not None]
        ax.plot(ep, vals, color=color, lw=2.5, label='Matched ref', zorder=3)

        # Cross-seed
        for ref_seed, cross_traj in cross_trajs:
            vals_c = [v for v in cross_traj['path_gcs_scores'][path] if v is not None]
            ep_c = [e for e, v in zip(cross_traj['epochs'],
                    cross_traj['path_gcs_scores'][path]) if v is not None]
            ax.plot(ep_c, vals_c, color=color, lw=1.2, ls='--', alpha=0.6,
                    label=f'Ref s{ref_seed}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('GCS')
        ax.set_title(path.replace('_', ' ').title())
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'{act} {task} seed={target_seed}: matched vs cross-seed reference',
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / f'{act}_{task}_s{target_seed}_overlay.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cohort_figure(summary_rows, out_path: Path):
    """Bar chart of Spearman ρ distribution."""
    rhos = [r['spearman_rho'] for r in summary_rows if not np.isnan(r['spearman_rho'])]
    if not rhos:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(rhos, bins=15, color='#0173B2', alpha=0.75, edgecolor='white', linewidth=0.8)
    med = np.median(rhos)
    ax.axvline(med, color='#DE8F05', lw=2, label=f'Median ρ = {med:.3f}')
    ax.set_xlabel('Spearman ρ (matched vs cross-seed)')
    ax.set_ylabel('Count')
    ax.set_title(f'Cross-seed reference stability (n={len(rhos)} pairs)')
    ax.legend(fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path.name}  (median ρ = {med:.3f})')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Cross-seed GCS reference control')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tasks', type=str, default='add',
                        help='Comma-separated tasks (default: add)')
    parser.add_argument(
        '--tag',
        type=str,
        default='',
        help='Optional suffix for top-level summary outputs to avoid filename collisions',
    )
    parser.add_argument('--checkpoint-freq', type=int, default=200)
    parser.add_argument('--N', type=int, default=200)
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--d', type=int, default=8)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    setup_style()

    tasks = [t.strip() for t in args.tasks.split(',')]
    out_dir = _gcs_data_root() / 'experiments' / 'cross_seed_reference'
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = group_runs(tasks)
    print(f'Found {len(groups)} (act, task) groups with multiple seeds')
    for key, runs in groups.items():
        print(f'  {key[0]}/{key[1]}: seeds {[s for s, _ in runs]}')

    tag_suffix = f'_{args.tag}' if args.tag else ''
    summary_rows = []

    for (act, task), runs in sorted(groups.items()):
        if len(runs) < 2:
            print(f'\nSkipping {act}/{task}: only 1 seed')
            continue

        print(f'\n{"="*60}')
        print(f'{act} / {task} ({len(runs)} seeds)')
        print(f'{"="*60}')

        group_dir = out_dir / f'{act}_{task}'
        group_dir.mkdir(exist_ok=True)

        # Compute trajectories: matched + cross-seed
        all_trajectories = {}  # (target_seed, ref_seed) → trajectory

        for target_seed, target_base in runs:
            target_path = resolve_path(target_base)
            metrics = load_metrics(target_base)
            t_grok = detect_grok_epoch(metrics)
            config = metrics.get('config', {})
            train_data, _ = create_data(
                p=config.get('p', 113),
                fraction=config.get('fraction', 0.3),
                seed=config.get('seed', target_seed),
                device=args.device,
                task=task,
            )

            matched_traj = load_existing_matched_trajectory(target_base, args.checkpoint_freq)
            if matched_traj is not None:
                print(f'  Reusing matched GCS: target=s{target_seed}, ref=s{target_seed}')
            else:
                target_ckpts = sorted((target_path / 'checkpoints').glob('*.pth'),
                                      key=lambda x: extract_epoch(x.name))
                target_final = target_ckpts[-1]

                print(f'  Computing GCS: target=s{target_seed}, ref=s{target_seed}...')
                ref_geo = build_reference_geometry(
                    target_final, train_data, device=args.device,
                    N=args.N, k=args.k, d=args.d
                )
                matched_traj = compute_gcs_trajectory(
                    target_path, ref_geo,
                    checkpoint_freq=args.checkpoint_freq,
                    device=args.device,
                )

            all_trajectories[(target_seed, target_seed)] = matched_traj
            diags = evaluate_trajectory(matched_traj, t_grok)
            summary_rows.append(dict(
                act=act, task=task,
                target_seed=target_seed, ref_seed=target_seed,
                matched=True,
                spearman_rho=np.nan,
                lead=diags['lead'],
                mlp_ushape=diags['mlp_ushape'],
                attn_peak_dip=diags['attn_peak_dip'],
            ))

            for ref_seed, ref_base in runs:
                if ref_seed == target_seed:
                    continue

                ref_base_path = resolve_path(ref_base)
                ref_ckpts = sorted((ref_base_path / 'checkpoints').glob('*.pth'),
                                   key=lambda x: extract_epoch(x.name))
                ref_final = ref_ckpts[-1]

                print(f'  Computing GCS: target=s{target_seed}, ref=s{ref_seed}...')
                ref_geo = build_reference_geometry(
                    ref_final, train_data, device=args.device,
                    N=args.N, k=args.k, d=args.d
                )

                traj = compute_gcs_trajectory(
                    target_path, ref_geo,
                    checkpoint_freq=args.checkpoint_freq,
                    device=args.device,
                )
                all_trajectories[(target_seed, ref_seed)] = traj

                rho = trajectory_correlation(matched_traj, traj)
                diags = evaluate_trajectory(traj, t_grok)

                summary_rows.append(dict(
                    act=act, task=task,
                    target_seed=target_seed, ref_seed=ref_seed,
                    matched=False,
                    spearman_rho=rho,
                    lead=diags['lead'],
                    mlp_ushape=diags['mlp_ushape'],
                    attn_peak_dip=diags['attn_peak_dip'],
                ))

            # Plot overlay for this target
            cross_trajs = [(rs, all_trajectories[(target_seed, rs)])
                          for rs, _ in runs if rs != target_seed]
            plot_overlay(matched_traj, cross_trajs, act, task, target_seed, group_dir)

        # Save trajectories
        with open(group_dir / 'trajectories.pkl', 'wb') as f:
            pickle.dump(all_trajectories, f)
        print(f'  Saved {group_dir / "trajectories.pkl"}')

    # Save cohort summary
    out_csv = out_dir / f'cohort_summary{tag_suffix}.csv'
    if summary_rows:
        fieldnames = ['act', 'task', 'target_seed', 'ref_seed', 'matched',
                      'spearman_rho', 'lead', 'mlp_ushape', 'attn_peak_dip']
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'\nSaved {out_csv}')

    # Cohort figure
    cross_rows = [r for r in summary_rows if not r['matched']]
    if cross_rows:
        cohort_figure_path = out_dir / f'cohort_figure{tag_suffix}.png'
        plot_cohort_figure(cross_rows, cohort_figure_path)

    # Summary statistics
    cross_rhos = [r['spearman_rho'] for r in cross_rows
                  if not np.isnan(r['spearman_rho'])]
    if cross_rhos:
        med = np.median(cross_rhos)
        print(f'\n{"="*60}')
        print(f'CROSS-SEED REFERENCE SUMMARY')
        print(f'{"="*60}')
        print(f'  Pairs: {len(cross_rhos)}')
        print(f'  Spearman ρ: median={med:.3f}, min={min(cross_rhos):.3f}, max={max(cross_rhos):.3f}')
        cross_diag = [r for r in cross_rows]
        n_lead = sum(1 for r in cross_diag if r['lead'] > 0)
        n_ushape = sum(1 for r in cross_diag if r['mlp_ushape'])
        n_peakdip = sum(1 for r in cross_diag if r['attn_peak_dip'])
        n_total = len(cross_diag)
        print(f'  Diagnostics under cross-seed ref:')
        print(f'    Lead > 0:        {n_lead}/{n_total}')
        print(f'    MLP U-shape:     {n_ushape}/{n_total}')
        print(f'    Attn peak-dip:   {n_peakdip}/{n_total}')

    print(f'\nAll outputs in: {out_dir}')


if __name__ == '__main__':
    main()
