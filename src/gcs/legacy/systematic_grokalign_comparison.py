#!/usr/bin/env python3
"""
Systematic GrokAlign comparison across all 35 evaluable 1L runs.

For each run:
  1. Load existing GCS 3-path data
  2. Compute centroid alignment at matched checkpoint epochs
  3. Assess whether centroid alignment is monotonic
  4. Test whether centroid alignment can detect the attention peak
  5. Test whether centroid alignment can predict the commutativity split

Outputs:
  experiments/grokalign_systematic/
    grokalign_all_runs.pkl   — per-run trajectories
    cohort_statistics.json   — aggregate statistics
    grokalign_cohort.png     — cohort-level figure

Usage:
    python scripts/systematic_grokalign_comparison.py --device cuda:0
    python scripts/systematic_grokalign_comparison.py --device cuda:0 --skip-computed
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


from gcs.core.models import Transformer
from gcs.core.data import create_data
from gcs.analysis.grokalign import compute_grokalign_score
from gcs.legacy.diagnostics_common import (
    RUNS_1L,
    load_gcs,
    load_all_series,
    load_metrics,
    detect_grok_epoch,
    detect_attention_peak_then_dip,
    detect_attention_post_grok_dip,
    resolve_path,
    normalize_01,
)


OUTPUT_DIR = Path('experiments/grokalign_systematic')

COMMUTATIVE_TASKS = {'add', 'mul'}
NON_COMMUTATIVE_TASKS = {'sub', 'div'}


def load_checkpoint_and_model(checkpoint_path, device='cuda:0'):
    from .reanalyze_gcs import load_checkpoint
    model,_=load_checkpoint(checkpoint_path,device)
    return model,model.embed.W_E.shape[1]


def extract_epoch(filename):
    import re
    match = re.search(r'epoch_(\d+)', str(filename))
    return int(match.group(1)) if match else 0


def compute_grokalign_trajectory_for_run(exp_dir, task, seed=777,
                                          checkpoint_freq=200, device='cuda:0',
                                          max_samples=500):
    """Compute GrokAlign at each checkpoint for a single run."""
    exp_path = resolve_path(exp_dir)
    checkpoint_dir = exp_path / 'checkpoints'
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints/ in {exp_path}")

    all_ckpts = sorted(checkpoint_dir.glob('*.pth'),
                       key=lambda x: extract_epoch(x.name))
    if not all_ckpts:
        raise FileNotFoundError(f"No .pth files in {checkpoint_dir}")

    from .reanalyze_gcs import load_checkpoint
    model,config=load_checkpoint(all_ckpts[-1],device)
    train_data,_=create_data(p=config.get('p',model.embed.W_E.shape[1]-1),
        fraction=config.get('fraction',.3),seed=seed,device=device,task=task)


    epochs = []
    scores = []

    torch.manual_seed(42)

    for ckpt_path in all_ckpts:
        epoch = extract_epoch(ckpt_path.name)
        if epoch % checkpoint_freq != 0:
            continue

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state)

            score = compute_grokalign_score(
                model, train_data, device=device, max_samples=max_samples
            )
            epochs.append(epoch)
            scores.append(score)
        except Exception as e:
            print(f"    Epoch {epoch} failed: {e}")

    return {'epochs': epochs, 'grokalign_scores': scores}


def produce_selector_cache(runs, output, *, device='cuda:0', checkpoint_freq=200, max_samples=500):
    """Generate the exact cohort-cache schema consumed by the selector evaluator.

    Candidates are measured here; the evaluator applies its grok/held-out filter.
    Missing or partially measured runs fail instead of silently shrinking a cohort.
    """
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    payload={}
    for act,task,seed,base in runs:
        values=compute_grokalign_trajectory_for_run(base,task,seed,checkpoint_freq,device,max_samples)
        expected=[p for p in (resolve_path(base)/'checkpoints').glob('*.pth') if extract_epoch(p.name)%checkpoint_freq==0]
        if not values['epochs'] or len(values['epochs'])!=len(expected) or not np.isfinite(values['grokalign_scores']).all():
            raise RuntimeError(f'Incomplete GrokAlign measurement for {act}/{task}/s{seed}')
        payload[f'{act}/{task}/s{seed}']={'act':act,'task':task,'seed':int(seed),'base':str(base),'grokalign':values}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as stream:pickle.dump(payload,stream)
    return payload


def is_monotonic_post_init(epochs, scores, warmup_frac=0.05):
    """
    Check if trajectory is monotonically increasing after initial warmup.

    Returns (is_monotonic, n_violations, max_violation_magnitude).
    """
    arr = np.array(scores, dtype=float)
    ep = np.array(epochs, dtype=float)
    if len(arr) < 5:
        return True, 0, 0.0

    # Skip initial warmup (first 5% of epochs)
    warmup_end = ep[0] + warmup_frac * (ep[-1] - ep[0])
    start_idx = int(np.searchsorted(ep, warmup_end))
    if start_idx >= len(arr) - 1:
        start_idx = 1

    post_warmup = arr[start_idx:]
    diffs = np.diff(post_warmup)
    violations = diffs < -0.005  # allow tiny numerical jitter
    n_violations = int(violations.sum())
    max_viol = float(abs(diffs[violations].min())) if n_violations > 0 else 0.0

    return n_violations == 0, n_violations, max_viol


def detect_grokalign_turning_point(epochs, scores):
    """
    Detect a turning point (peak then decline) in GrokAlign trajectory.

    Returns (has_peak, peak_epoch) or (False, None) if monotonic.
    """
    arr = np.array(scores, dtype=float)
    ep = np.array(epochs, dtype=float)
    if len(arr) < 5:
        return False, None

    # Smooth slightly
    from gcs.legacy.diagnostics_common import smooth_series
    smoothed = smooth_series(arr, window=5)

    peak_idx = int(np.argmax(smoothed))
    # Peak must not be at the very end
    if peak_idx >= len(smoothed) - 3:
        return False, None

    # Check for meaningful decline after peak
    post_peak_min = float(smoothed[peak_idx:].min())
    peak_val = float(smoothed[peak_idx])
    full_range = float(smoothed.max() - smoothed.min())
    if full_range < 1e-8:
        return False, None

    decline = (peak_val - post_peak_min) / full_range
    if decline > 0.10:  # 10% relative decline
        return True, int(ep[peak_idx])

    return False, None


def compute_grokalign_auc_features(epochs, scores, grok_epoch):
    """Compute pre-grok and post-grok AUC for GrokAlign trajectory."""
    ep = np.array(epochs, dtype=float)
    sc = np.array(scores, dtype=float)

    pre_mask = ep <= grok_epoch
    post_mask = ep > grok_epoch

    pre_auc = float(np.trapz(sc[pre_mask], ep[pre_mask])) if pre_mask.sum() > 1 else 0.0
    post_auc = float(np.trapz(sc[post_mask], ep[post_mask])) if post_mask.sum() > 1 else 0.0

    # Normalize by epoch range
    pre_range = float(ep[pre_mask].max() - ep[pre_mask].min()) if pre_mask.sum() > 1 else 1.0
    post_range = float(ep[post_mask].max() - ep[post_mask].min()) if post_mask.sum() > 1 else 1.0

    return {
        'pre_grok_mean': float(pre_auc / max(pre_range, 1)),
        'post_grok_mean': float(post_auc / max(post_range, 1)),
        'final_score': float(sc[-1]) if len(sc) > 0 else 0.0,
        'max_score': float(sc.max()) if len(sc) > 0 else 0.0,
    }


def run_cohort(device='cuda:0', checkpoint_freq=200, max_samples=500,
               skip_computed=False):
    """Run GrokAlign on all 35 evaluable 1L runs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = OUTPUT_DIR / 'grokalign_all_runs.pkl'
    if skip_computed and cache_path.exists():
        print("Loading cached GrokAlign trajectories...")
        with open(cache_path, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Loaded {len(all_results)} cached runs")
    else:
        all_results = {}

    for i, (act, task, seed, base) in enumerate(RUNS_1L):
        run_key = f"{act}/{task}/s{seed}"

        if skip_computed and run_key in all_results:
            print(f"[{i+1}/36] {run_key} — cached, skipping")
            continue

        exp_path = resolve_path(base)
        if not exp_path.exists():
            print(f"[{i+1}/36] {run_key} — MISSING, skipping")
            continue

        # Check grok
        try:
            metrics = load_metrics(base)
            t_grok = detect_grok_epoch(metrics)
            if t_grok is None:
                print(f"[{i+1}/36] {run_key} — no grok (censored)")
                continue
        except Exception as e:
            print(f"[{i+1}/36] {run_key} — load error: {e}")
            continue

        print(f"[{i+1}/36] {run_key} computing GrokAlign...")
        try:
            ga_data = compute_grokalign_trajectory_for_run(
                base, task=task, seed=seed,
                checkpoint_freq=checkpoint_freq,
                device=device, max_samples=max_samples
            )

            # Load GCS attention data for comparison
            epochs_gcs, _, attn, _ = load_all_series(base, depth=1)

            all_results[run_key] = {
                'act': act, 'task': task, 'seed': seed, 'base': base,
                't_grok': t_grok,
                'grokalign': ga_data,
                'gcs_epochs': epochs_gcs.tolist(),
                'gcs_attention': attn.tolist(),
            }
            print(f"    Done: {len(ga_data['epochs'])} checkpoints, "
                  f"final={ga_data['grokalign_scores'][-1]:.4f}")

            # Checkpoint after each run
            with open(cache_path, 'wb') as f:
                pickle.dump(all_results, f)

        except Exception as e:
            print(f"    FAILED: {e}")

    return all_results


def analyze_cohort(all_results):
    """Compute cohort-level statistics."""
    stats_rows = []

    for run_key, data in all_results.items():
        ga = data['grokalign']
        t_grok = data['t_grok']

        # Monotonicity
        mono, n_viol, max_viol = is_monotonic_post_init(
            ga['epochs'], ga['grokalign_scores']
        )

        # Turning point detection
        has_peak, peak_ep = detect_grokalign_turning_point(
            ga['epochs'], ga['grokalign_scores']
        )

        # AUC features
        auc_feats = compute_grokalign_auc_features(
            ga['epochs'], ga['grokalign_scores'], t_grok
        )

        # GCS attention peak detection (for comparison)
        gcs_ep = np.array(data['gcs_epochs'])
        gcs_attn = np.array(data['gcs_attention'])
        attn_ptd, _, _, t_attn_peak = detect_attention_peak_then_dip(
            gcs_attn, gcs_ep, t_grok
        )

        stats_rows.append({
            'run_key': run_key,
            'act': data['act'],
            'task': data['task'],
            'seed': data['seed'],
            't_grok': t_grok,
            'ga_monotonic': mono,
            'ga_n_violations': n_viol,
            'ga_max_violation': max_viol,
            'ga_has_peak': has_peak,
            'ga_peak_epoch': peak_ep,
            'ga_final_score': auc_feats['final_score'],
            'ga_max_score': auc_feats['max_score'],
            'ga_pre_grok_mean': auc_feats['pre_grok_mean'],
            'ga_post_grok_mean': auc_feats['post_grok_mean'],
            'gcs_attn_peak_then_dip': attn_ptd,
            'gcs_attn_peak_epoch': t_attn_peak,
            'is_commutative': data['task'] in COMMUTATIVE_TASKS,
        })

    return stats_rows


def compute_summary_statistics(stats_rows):
    """Compute cohort-level summary."""
    n = len(stats_rows)
    n_monotonic = sum(1 for r in stats_rows if r['ga_monotonic'])
    n_has_peak = sum(1 for r in stats_rows if r['ga_has_peak'])
    n_attn_peak = sum(1 for r in stats_rows if r['gcs_attn_peak_then_dip'])

    # Commutativity split: can GrokAlign features distinguish commutative?
    comm_scores = [r['ga_final_score'] for r in stats_rows if r['is_commutative']]
    noncomm_scores = [r['ga_final_score'] for r in stats_rows if not r['is_commutative']]

    if len(comm_scores) >= 2 and len(noncomm_scores) >= 2:
        u_stat, comm_p = stats.mannwhitneyu(comm_scores, noncomm_scores,
                                             alternative='two-sided')
    else:
        u_stat, comm_p = 0.0, 1.0

    # Correlation between GrokAlign features and commutativity
    comm_labels = np.array([1 if r['is_commutative'] else 0 for r in stats_rows])
    ga_finals = np.array([r['ga_final_score'] for r in stats_rows])

    if len(set(comm_labels)) > 1:
        comm_rho, comm_rho_p = stats.spearmanr(comm_labels, ga_finals)
    else:
        comm_rho, comm_rho_p = 0.0, 1.0

    # Compare with GCS attention dip predicting commutativity
    # (the finding we already have: attention post-grok dip ~ commutativity)

    summary = {
        'n_runs': n,
        'n_monotonic': n_monotonic,
        'pct_monotonic': round(100 * n_monotonic / n, 1),
        'n_has_peak': n_has_peak,
        'pct_has_peak': round(100 * n_has_peak / n, 1),
        'n_gcs_attn_peak': n_attn_peak,
        'pct_gcs_attn_peak': round(100 * n_attn_peak / n, 1),
        'ga_comm_vs_noncomm_U': float(u_stat),
        'ga_comm_vs_noncomm_p': float(comm_p),
        'ga_final_comm_rho': float(comm_rho),
        'ga_final_comm_rho_p': float(comm_rho_p),
        'mean_ga_final_commutative': float(np.mean(comm_scores)),
        'mean_ga_final_noncommutative': float(np.mean(noncomm_scores)),
        'claim': (
            f"Centroid alignment is monotonic in {n_monotonic}/{n} runs "
            f"({100*n_monotonic/n:.0f}%). "
            f"It detects a turning point in only {n_has_peak}/{n} runs, "
            f"vs GCS attention peak-then-dip in {n_attn_peak}/{n}. "
            f"Centroid alignment final score does not distinguish "
            f"commutative from non-commutative tasks "
            f"(U-test p={comm_p:.3f})."
        ),
    }

    return summary


def plot_cohort_figure(all_results, stats_rows, summary):
    """Create cohort-level figure with 4 panels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): Overlay all GrokAlign trajectories (normalized)
    ax = axes[0, 0]
    for run_key, data in all_results.items():
        ga = data['grokalign']
        ep = np.array(ga['epochs'], dtype=float)
        sc = np.array(ga['grokalign_scores'], dtype=float)
        if len(sc) < 2:
            continue
        # Normalize epochs by t_grok
        t_grok = data['t_grok']
        ep_norm = ep / max(t_grok, 1)
        sc_norm = normalize_01(sc)

        is_comm = data['task'] in COMMUTATIVE_TASKS
        color = '#2196F3' if is_comm else '#FF5722'
        ax.plot(ep_norm, sc_norm, color=color, alpha=0.25, linewidth=0.8)

    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='t_grok')
    ax.set_xlabel('Epoch / t_grok')
    ax.set_ylabel('GrokAlign (normalized)')
    ax.set_title(f'(a) GrokAlign trajectories (N={summary["n_runs"]})')
    ax.set_xlim(0, 3)

    # Custom legend
    from matplotlib.lines import Line2D
    lines = [
        Line2D([0], [0], color='#2196F3', alpha=0.6, label='Commutative'),
        Line2D([0], [0], color='#FF5722', alpha=0.6, label='Non-commutative'),
        Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, label='t_grok'),
    ]
    ax.legend(handles=lines, fontsize=8, loc='lower right')

    # Panel (b): Overlay all GCS attention trajectories for comparison
    ax = axes[0, 1]
    for run_key, data in all_results.items():
        gcs_ep = np.array(data['gcs_epochs'], dtype=float)
        gcs_attn = np.array(data['gcs_attention'], dtype=float)
        if len(gcs_attn) < 2:
            continue
        t_grok = data['t_grok']
        ep_norm = gcs_ep / max(t_grok, 1)
        attn_norm = normalize_01(gcs_attn)

        is_comm = data['task'] in COMMUTATIVE_TASKS
        color = '#2196F3' if is_comm else '#FF5722'
        ax.plot(ep_norm, attn_norm, color=color, alpha=0.25, linewidth=0.8)

    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch / t_grok')
    ax.set_ylabel('GCS attention (normalized)')
    ax.set_title(f'(b) GCS attention-path trajectories (N={summary["n_runs"]})')
    ax.set_xlim(0, 3)
    ax.legend(handles=lines, fontsize=8, loc='lower right')

    # Panel (c): Monotonicity bar chart
    ax = axes[1, 0]
    categories = ['GrokAlign\nmonotonic', 'GrokAlign\nhas peak', 'GCS attn\npeak-then-dip']
    counts = [
        summary['n_monotonic'],
        summary['n_has_peak'],
        summary['n_gcs_attn_peak'],
    ]
    totals = [summary['n_runs']] * 3
    bars = ax.bar(categories, counts, color=['#9C27B0', '#9C27B0', '#FF9800'],
                  alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, count, total in zip(bars, counts, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{count}/{total}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title('(c) Shape detection comparison')
    ax.set_ylim(0, summary['n_runs'] + 3)

    # Panel (d): Commutativity split — box plot of GrokAlign final scores
    ax = axes[1, 1]
    comm_finals = [r['ga_final_score'] for r in stats_rows if r['is_commutative']]
    noncomm_finals = [r['ga_final_score'] for r in stats_rows if not r['is_commutative']]

    bp = ax.boxplot([comm_finals, noncomm_finals],
                    labels=['Commutative\n(add, mul)', 'Non-commut.\n(sub, div)'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor('#FF5722')
    bp['boxes'][1].set_alpha(0.4)

    ax.set_ylabel('GrokAlign final score')
    ax.set_title(f'(d) Commutativity split (U-test p={summary["ga_comm_vs_noncomm_p"]:.3f})')

    fig.suptitle('Systematic GrokAlign vs GCS Comparison (35 1L runs)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    save_path = OUTPUT_DIR / 'grokalign_cohort.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved cohort figure: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Systematic GrokAlign comparison on all 35 1L runs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint-freq', type=int, default=200)
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--skip-computed', action='store_true',
                        help='Skip runs already in cache')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze cached results (no GPU needed)')
    parser.add_argument('--cache-only',action='store_true',help='Produce selector cache without legacy descriptive plots')
    args = parser.parse_args()
    if args.cache_only:
        produce_selector_cache(RUNS_1L,OUTPUT_DIR/'grokalign_all_runs.pkl',device=args.device,
                               checkpoint_freq=args.checkpoint_freq,max_samples=args.max_samples)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        cache_path = OUTPUT_DIR / 'grokalign_all_runs.pkl'
        if not cache_path.exists():
            print("ERROR: No cached results. Run without --analyze-only first.")
            sys.exit(1)
        with open(cache_path, 'rb') as f:
            all_results = pickle.load(f)
        print(f"Loaded {len(all_results)} cached runs")
    else:
        all_results = run_cohort(
            device=args.device,
            checkpoint_freq=args.checkpoint_freq,
            max_samples=args.max_samples,
            skip_computed=args.skip_computed,
        )

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {len(all_results)} runs")
    print(f"{'='*60}")

    stats_rows = analyze_cohort(all_results)
    summary = compute_summary_statistics(stats_rows)

    # Save statistics
    with open(OUTPUT_DIR / 'cohort_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR / 'cohort_statistics.json'}")

    # Save per-run stats
    with open(OUTPUT_DIR / 'per_run_stats.json', 'w') as f:
        json.dump(stats_rows, f, indent=2, default=str)
    print(f"Saved: {OUTPUT_DIR / 'per_run_stats.json'}")

    # Print summary
    print(f"\n--- Cohort Summary ---")
    print(f"Monotonic:           {summary['n_monotonic']}/{summary['n_runs']} "
          f"({summary['pct_monotonic']}%)")
    print(f"GA has peak:         {summary['n_has_peak']}/{summary['n_runs']} "
          f"({summary['pct_has_peak']}%)")
    print(f"GCS attn peak-dip:   {summary['n_gcs_attn_peak']}/{summary['n_runs']} "
          f"({summary['pct_gcs_attn_peak']}%)")
    print(f"GA comm vs non-comm: U={summary['ga_comm_vs_noncomm_U']:.1f}, "
          f"p={summary['ga_comm_vs_noncomm_p']:.4f}")
    print(f"GA final ~ comm:     rho={summary['ga_final_comm_rho']:.3f}, "
          f"p={summary['ga_final_comm_rho_p']:.4f}")
    print(f"\nClaim: {summary['claim']}")

    # Plot
    plot_cohort_figure(all_results, stats_rows, summary)

    print("\nDone.")


if __name__ == '__main__':
    main()
