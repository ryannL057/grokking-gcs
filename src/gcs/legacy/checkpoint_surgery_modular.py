#!/usr/bin/env python3
"""
Immediate checkpoint surgery for modular commutative runs.

For a selected 1-layer Transformer run, this script:
  1. loads the attention swap-equivariance trajectory,
  2. picks a pre-symmetry checkpoint t_minus and a symmetry-onset checkpoint t_sym,
  3. constructs base / donor / reset-* models,
  4. evaluates train/test loss, train/test accuracy, and attention equivariance.

The goal is to test whether removing already-formed attention or MLP parameters
from a post-symmetry model immediately damages generalization and symmetry.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pass  # Package imports do not search the experiment-data directory.

from gcs.core.data import create_data
from gcs.core.models import Transformer
from gcs.core.training import full_accuracy, full_loss
from gcs.legacy.diagnostics_common import RUNS_1L, detect_grok_epoch, load_metrics, resolve_path
from gcs.legacy.measure_equivariance import build_swap_pairs, measure_attn_equivariance


CONDITION_SUFFIXES = {
    'reset_attn': ['.attn.W_Q', '.attn.W_K', '.attn.W_V', '.attn.W_O'],
    'reset_mlp': ['.mlp.W_in', '.mlp.b_in', '.mlp.W_out', '.mlp.b_out'],
    'reset_q': ['.attn.W_Q'],
    'reset_k': ['.attn.W_K'],
    'reset_v': ['.attn.W_V'],
    'reset_o': ['.attn.W_O'],
    'reset_qk': ['.attn.W_Q', '.attn.W_K'],
    'reset_qkv': ['.attn.W_Q', '.attn.W_K', '.attn.W_V'],
    'reset_vo': ['.attn.W_V', '.attn.W_O'],
    'reset_mlp_in': ['.mlp.W_in', '.mlp.b_in'],
    'reset_mlp_out': ['.mlp.W_out', '.mlp.b_out'],
}

DEFAULT_CONDITIONS = ['donor_pre', 'base', 'reset_attn', 'reset_mlp']

CONDITION_COLORS = {
    'donor_pre': '#7f7f7f',
    'base': '#1f77b4',
    'reset_attn': '#d62728',
    'reset_mlp': '#2ca02c',
    'reset_q': '#ff7f0e',
    'reset_k': '#9467bd',
    'reset_v': '#8c564b',
    'reset_o': '#bcbd22',
    'reset_qk': '#e377c2',
    'reset_qkv': '#17becf',
    'reset_vo': '#7f7f00',
    'reset_mlp_in': '#4daf4a',
    'reset_mlp_out': '#377eb8',
}


def stable_seed_from_parts(*parts: object) -> int:
    text = '|'.join(str(part) for part in parts)
    seed = 0
    for ch in text:
        seed = (seed * 131 + ord(ch)) % (2**32)
    return seed


def lookup_run(act: str, task: str, seed: int) -> Path:
    for run_act, run_task, run_seed, path in RUNS_1L:
        if run_act == act and run_task == task and run_seed == seed:
            return resolve_path(path)
    raise ValueError(f'Run not found for {act}/{task}/s{seed}')


def extract_epoch(path: Path) -> int:
    match = re.search(r'epoch_(\d+)', path.name)
    return int(match.group(1)) if match else -1


def available_checkpoints(exp_dir: Path) -> list[Path]:
    checkpoints = sorted((exp_dir / 'checkpoints').glob('*.pth'), key=extract_epoch)
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints in {exp_dir / "checkpoints"}')
    return checkpoints


def load_equivariance_cache(act: str):
    candidates = [
        Path('experiments/equivariance_analysis/equivariance_all_activations.pkl'),
        Path(f'experiments/equivariance_analysis_{act.lower()}/equivariance_results.pkl'),
        Path('experiments/equivariance_analysis/equivariance_results.pkl'),
    ]
    for path in candidates:
        full = resolve_path(path)
        if full.exists():
            with open(full, 'rb') as f:
                return pickle.load(f)
    return None


def load_equivariance_trajectory(
    exp_dir: Path,
    act: str,
    task: str,
    seed: int,
    cache_path: str | None = None,
):
    if cache_path:
        with open(resolve_path(cache_path), 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = load_equivariance_cache(act)
    if cache is None:
        raise FileNotFoundError('No equivariance cache found.')

    keyed = cache.get((act, task, seed))
    if keyed is not None:
        return keyed

    keyed = cache.get((task, seed))
    if keyed is not None:
        return keyed

    keyed = cache.get(str(exp_dir))
    if keyed is not None:
        return keyed

    if {'epochs', 'equi_dist', 'norm_scale'}.issubset(cache.keys()):
        return cache

    raise KeyError(f'No equivariance trajectory for {act}/{task}/s{seed}')


def select_surgery_epochs(epochs: np.ndarray, rel_equi: np.ndarray, t_grok: int):
    """Pick t_minus/t_sym from the steepest pre-grok drop in rel. equivariance."""
    if epochs.size < 4:
        raise ValueError('Too few checkpoints for surgery epoch selection')

    valid = epochs > 0
    if not np.any(valid):
        raise ValueError('No post-initialization checkpoints')

    epochs_v = epochs[valid]
    rel_v = rel_equi[valid]
    pre_mask = epochs_v <= t_grok
    if not np.any(pre_mask):
        pre_mask = np.ones_like(epochs_v, dtype=bool)

    pre_epochs = epochs_v[pre_mask]
    pre_rel = rel_v[pre_mask]
    if pre_epochs.size < 2:
        raise ValueError('Need at least two pre-grok checkpoints')

    finite_diffs = np.diff(pre_rel)
    drop_idx = int(np.argmin(finite_diffs))
    t_minus = int(pre_epochs[drop_idx])
    t_sym = int(pre_epochs[drop_idx + 1])

    peak_idx = int(np.argmax(pre_rel))
    peak_epoch = int(pre_epochs[peak_idx])
    peak_val = float(pre_rel[peak_idx])
    final_val = float(np.median(rel_v[-5:])) if rel_v.size >= 5 else float(rel_v[-1])

    return {
        'peak_epoch': peak_epoch,
        'peak_rel_equi': peak_val,
        'final_rel_equi': final_val,
        'largest_negative_delta': float(finite_diffs[drop_idx]),
        't_minus': t_minus,
        't_sym': t_sym,
        't_plus': t_sym,  # Backward-compatible alias for earlier outputs.
    }


def nearest_checkpoint_epoch_before(target: int, checkpoints: list[Path], upper_bound: int | None = None) -> int:
    epochs = [extract_epoch(path) for path in checkpoints]
    eligible = [epoch for epoch in epochs if epoch > 0 and (upper_bound is None or epoch < upper_bound)]
    if not eligible:
        raise ValueError('No eligible checkpoints before the requested bound')
    return min(eligible, key=lambda value: (abs(value - target), -value))


def first_metric_epoch_at_or_above(metrics: dict, key: str, threshold: float) -> int:
    steps = np.asarray(metrics['log_steps'], dtype=int)
    values = np.asarray(metrics[key], dtype=float)
    idx = np.where(values >= threshold)[0]
    if idx.size == 0:
        raise ValueError(f'No epoch found where {key} >= {threshold}')
    return int(steps[int(idx[0])])


def latest_memorization_plateau_epoch(
    metrics: dict,
    *,
    train_acc_threshold: float,
    test_acc_max: float,
) -> int:
    steps = np.asarray(metrics['log_steps'], dtype=int)
    train_acc = np.asarray(metrics['train_accuracies'], dtype=float)
    test_acc = np.asarray(metrics['test_accuracies'], dtype=float)
    mask = (train_acc >= train_acc_threshold) & (test_acc <= test_acc_max)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(
            'No epoch found for memorization plateau donor: '
            f'train_acc >= {train_acc_threshold} and test_acc <= {test_acc_max}'
        )
    return int(steps[int(idx[-1])])


def choose_donor_epoch(
    selection: dict,
    t_grok: int,
    checkpoints: list[Path],
    metrics: dict,
    donor_mode: str,
    donor_fixed_frac: float,
    donor_train_acc_threshold: float,
    donor_test_acc_max: float,
    donor_explicit_epoch: int | None = None,
) -> tuple[int, int]:
    if donor_mode == 'equiv_drop':
        target_epoch = int(selection['t_minus'])
        donor_epoch = nearest_checkpoint_epoch(target_epoch, checkpoints)
        return donor_epoch, target_epoch

    if donor_mode == 'explicit_epoch':
        if donor_explicit_epoch is None:
            raise ValueError('--donor-epoch is required when --donor-mode=explicit_epoch')
        target_epoch = int(donor_explicit_epoch)
        donor_epoch = nearest_checkpoint_epoch(target_epoch, checkpoints)
        return donor_epoch, target_epoch

    if donor_mode == 'fixed_frac_tgrok':
        target_epoch = int(round(donor_fixed_frac * t_grok))
        donor_epoch = nearest_checkpoint_epoch_before(target_epoch, checkpoints, upper_bound=t_grok)
        return donor_epoch, target_epoch

    if donor_mode == 'train_acc_threshold':
        target_epoch = first_metric_epoch_at_or_above(
            metrics,
            key='train_accuracies',
            threshold=donor_train_acc_threshold,
        )
        donor_epoch = nearest_checkpoint_epoch(target_epoch, checkpoints)
        return donor_epoch, target_epoch

    if donor_mode == 'memorization_plateau':
        target_epoch = latest_memorization_plateau_epoch(
            metrics,
            train_acc_threshold=donor_train_acc_threshold,
            test_acc_max=donor_test_acc_max,
        )
        donor_epoch = nearest_checkpoint_epoch(target_epoch, checkpoints)
        return donor_epoch, target_epoch

    raise ValueError(f'Unknown donor mode: {donor_mode}')


def nearest_checkpoint_epoch(target: int, checkpoints: list[Path]) -> int:
    epochs = [extract_epoch(path) for path in checkpoints]
    return min(epochs, key=lambda value: abs(value - target))


def checkpoint_path_for_epoch(exp_dir: Path, epoch: int) -> Path:
    matches = list((exp_dir / 'checkpoints').glob(f'*epoch_{epoch:05d}.pth'))
    if not matches:
        raise FileNotFoundError(f'No checkpoint for epoch {epoch} in {exp_dir}')
    return matches[0]


def load_checkpoint(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get('config') or payload.get('model_config', {})
    model = Transformer(
        num_layers=cfg.get('num_layers', 1),
        d_vocab=cfg.get('d_vocab', 114),
        d_model=cfg.get('d_model', 128),
        d_mlp=cfg.get('d_mlp', 512),
        d_head=cfg.get('d_head', 32),
        num_heads=cfg.get('num_heads', 4),
        n_ctx=cfg.get('n_ctx', 3),
        act_type=cfg.get('act_type', 'SiLU'),
        use_cache=False,
        use_ln=cfg.get('use_ln', False),
        attention_rate=cfg.get('attention_rate', 1.0) or 1.0,
    ).to(device)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    return model, cfg, payload['model_state_dict']


def modular_dataset(task: str, seed: int, p: int, fraction: float, device: str):
    return create_data(
        p=p,
        fraction=fraction,
        seed=seed,
        device=device,
        task=task,
    )


def attention_equivariance_relative(model: Transformer, p: int, device: str) -> float:
    prompts_ab, prompts_ba = build_swap_pairs(p, device=device)
    diff, norm = measure_attn_equivariance(model, prompts_ab, prompts_ba, device=device)
    return float(diff / max(norm, 1e-10)), float(norm)


def matching_param_names(state_dict: dict, suffixes: list[str]) -> list[str]:
    matches = [name for name in state_dict if any(name.endswith(suffix) for suffix in suffixes)]
    if not matches:
        raise ValueError(f'No parameters matched suffixes {suffixes}')
    return matches


def copy_selected_params(base_state: dict, donor_state: dict, selected_names: list[str]) -> dict:
    merged = {name: tensor.clone() for name, tensor in base_state.items()}
    for name in selected_names:
        merged[name] = donor_state[name].clone()
    return merged


def flat_param_numel(state_dict: dict, param_names: list[str]) -> int:
    return sum(int(state_dict[name].numel()) for name in param_names)


def param_delta_l2_sq(base_state: dict, donor_state: dict, param_names: list[str]) -> float:
    total = 0.0
    for name in param_names:
        diff = (base_state[name] - donor_state[name]).detach().float()
        total += float(torch.sum(diff * diff).item())
    return total


def flatten_param_delta_sq(base_state: dict, donor_state: dict, param_names: list[str]) -> np.ndarray:
    chunks = []
    for name in param_names:
        diff = (base_state[name] - donor_state[name]).detach().cpu().float().reshape(-1)
        chunks.append((diff * diff).numpy())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def copy_selected_flat_params(
    base_state: dict,
    donor_state: dict,
    param_names: list[str],
    flat_indices: np.ndarray,
) -> dict:
    merged = {name: tensor.clone() for name, tensor in base_state.items()}
    if flat_indices.size == 0:
        return merged

    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    flat_indices.sort()

    offset = 0
    ptr = 0
    for name in param_names:
        base_flat = merged[name].view(-1)
        donor_flat = donor_state[name].view(-1)
        numel = int(base_flat.numel())
        next_offset = offset + numel

        start = ptr
        while ptr < flat_indices.size and flat_indices[ptr] < next_offset:
            ptr += 1
        if ptr > start:
            local = flat_indices[start:ptr] - offset
            idx = torch.as_tensor(local, dtype=torch.long, device=base_flat.device)
            base_flat[idx] = donor_flat[idx]
        offset = next_offset

    return merged


def mlp_neuron_layout(base_state: dict) -> tuple[str, str, str]:
    required = ['blocks.0.mlp.W_in', 'blocks.0.mlp.b_in', 'blocks.0.mlp.W_out']
    missing = [name for name in required if name not in base_state]
    if missing:
        raise ValueError(f'MLP neuron matching requires parameters {required}; missing {missing}')
    return required[0], required[1], required[2]


def mlp_neuron_param_count(base_state: dict) -> int:
    w_in_name, b_in_name, w_out_name = mlp_neuron_layout(base_state)
    w_in = base_state[w_in_name]
    b_in = base_state[b_in_name]
    w_out = base_state[w_out_name]
    return int(w_in.shape[1] + 1 + w_out.shape[0])


def mlp_neuron_delta_sq(base_state: dict, donor_state: dict) -> np.ndarray:
    w_in_name, b_in_name, w_out_name = mlp_neuron_layout(base_state)
    w_in_diff = (base_state[w_in_name] - donor_state[w_in_name]).detach().cpu().float()
    b_in_diff = (base_state[b_in_name] - donor_state[b_in_name]).detach().cpu().float()
    w_out_diff = (base_state[w_out_name] - donor_state[w_out_name]).detach().cpu().float()
    per_neuron = (
        torch.sum(w_in_diff * w_in_diff, dim=1)
        + b_in_diff * b_in_diff
        + torch.sum(w_out_diff * w_out_diff, dim=0)
    )
    return per_neuron.numpy()


def copy_selected_mlp_neurons(base_state: dict, donor_state: dict, neuron_indices: np.ndarray) -> dict:
    merged = {name: tensor.clone() for name, tensor in base_state.items()}
    if neuron_indices.size == 0:
        return merged
    idx = torch.as_tensor(np.asarray(neuron_indices, dtype=np.int64), dtype=torch.long)
    w_in_name, b_in_name, w_out_name = mlp_neuron_layout(base_state)
    merged[w_in_name][idx, :] = donor_state[w_in_name][idx, :]
    merged[b_in_name][idx] = donor_state[b_in_name][idx]
    merged[w_out_name][:, idx] = donor_state[w_out_name][:, idx]
    return merged


def matched_mlp_subset_configs(
    base_state: dict,
    donor_state: dict,
    *,
    rng: np.random.Generator,
    n_candidates: int,
    n_norm_topk: int,
    mode: str,
) -> list[dict]:
    if n_candidates <= 0:
        return []

    attn_names = condition_param_names(base_state, 'reset_attn')
    target_count = flat_param_numel(base_state, attn_names)
    target_delta_l2_sq = param_delta_l2_sq(base_state, donor_state, attn_names)
    target_delta_l2 = float(np.sqrt(max(target_delta_l2_sq, 0.0)))

    if mode == 'element':
        mlp_names = condition_param_names(base_state, 'reset_mlp')
        total_mlp = flat_param_numel(base_state, mlp_names)
        if target_count > total_mlp:
            raise ValueError(
                f'Cannot sample matched MLP subset of size {target_count}; '
                f'only {total_mlp} MLP parameters available.'
            )
        sample_pool_size = total_mlp
        subset_size = target_count
        subset_param_count = target_count
        mlp_delta_sq = flatten_param_delta_sq(base_state, donor_state, mlp_names)
    elif mode == 'neuron':
        w_in_name, _, _ = mlp_neuron_layout(base_state)
        d_mlp = int(base_state[w_in_name].shape[0])
        params_per_neuron = mlp_neuron_param_count(base_state)
        subset_size = min(target_count // params_per_neuron, d_mlp)
        if subset_size <= 0:
            raise ValueError(
                f'Attention target count {target_count} is too small for neuron-level matching '
                f'with {params_per_neuron} parameters per neuron.'
            )
        sample_pool_size = d_mlp
        subset_param_count = subset_size * params_per_neuron
        mlp_delta_sq = mlp_neuron_delta_sq(base_state, donor_state)
    else:
        raise ValueError(f'Unknown matched MLP mode: {mode}')

    candidate_records = []
    for trial_idx in range(n_candidates):
        subset_indices = rng.choice(sample_pool_size, size=subset_size, replace=False)
        subset_delta_l2_sq = float(mlp_delta_sq[subset_indices].sum())
        candidate_records.append(
            {
                'trial_idx': int(trial_idx),
                'subset_indices': subset_indices,
                'subset_delta_l2_sq': subset_delta_l2_sq,
                'subset_delta_l2': float(np.sqrt(max(subset_delta_l2_sq, 0.0))),
            }
        )

    configs = []
    for record in candidate_records:
        if mode == 'element':
            state_dict = copy_selected_flat_params(
                base_state,
                donor_state,
                mlp_names,
                record['subset_indices'],
            )
        else:
            state_dict = copy_selected_mlp_neurons(
                base_state,
                donor_state,
                record['subset_indices'],
            )
        configs.append(
            {
                'condition': f'mlp_count_match_{record["trial_idx"]:02d}',
                'state_dict': state_dict,
                'reset_param_count': int(subset_param_count),
                'match_family': 'mlp_count_match',
                'match_sampling_unit': mode,
                'match_trial_idx': int(record['trial_idx']),
                'match_group_rank': int(record['trial_idx']),
                'match_target_param_count': int(target_count),
                'match_target_delta_l2': target_delta_l2,
                'match_subset_delta_l2': float(record['subset_delta_l2']),
                'match_subset_delta_l2_ratio': (
                    float(record['subset_delta_l2'] / target_delta_l2)
                    if target_delta_l2 > 1e-12 else float('nan')
                ),
                'match_subset_delta_l2_abs_error': float(
                    abs(record['subset_delta_l2'] - target_delta_l2)
                ),
            }
        )

    if n_norm_topk > 0:
        top_records = sorted(
            candidate_records,
            key=lambda rec: (abs(rec['subset_delta_l2_sq'] - target_delta_l2_sq), rec['trial_idx']),
        )[: min(n_norm_topk, len(candidate_records))]
        for norm_rank, record in enumerate(top_records):
            if mode == 'element':
                state_dict = copy_selected_flat_params(
                    base_state,
                    donor_state,
                    mlp_names,
                    record['subset_indices'],
                )
            else:
                state_dict = copy_selected_mlp_neurons(
                    base_state,
                    donor_state,
                    record['subset_indices'],
                )
            configs.append(
                {
                    'condition': f'mlp_norm_match_{norm_rank:02d}',
                    'state_dict': state_dict,
                    'reset_param_count': int(subset_param_count),
                    'match_family': 'mlp_norm_match',
                    'match_sampling_unit': mode,
                    'match_trial_idx': int(record['trial_idx']),
                    'match_group_rank': int(norm_rank),
                    'match_target_param_count': int(target_count),
                    'match_target_delta_l2': target_delta_l2,
                    'match_subset_delta_l2': float(record['subset_delta_l2']),
                    'match_subset_delta_l2_ratio': (
                        float(record['subset_delta_l2'] / target_delta_l2)
                        if target_delta_l2 > 1e-12 else float('nan')
                    ),
                    'match_subset_delta_l2_abs_error': float(
                        abs(record['subset_delta_l2'] - target_delta_l2)
                    ),
                }
            )

    return configs


def condition_param_names(base_state: dict, condition: str) -> list[str]:
    if condition in {'donor_pre', 'base'}:
        return []
    suffixes = CONDITION_SUFFIXES.get(condition)
    if suffixes is None:
        raise ValueError(
            f'Unknown condition {condition}. Available reset conditions: '
            f'{", ".join(sorted(CONDITION_SUFFIXES))}'
        )
    return matching_param_names(base_state, suffixes)


def evaluate_model(
    model: Transformer,
    train_data,
    test_data,
    *,
    task: str,
    p: int,
    device: str,
):
    train_loss = float(full_loss(model, train_data, device).item())
    test_loss = float(full_loss(model, test_data, device).item())
    train_acc = float(full_accuracy(model, train_data, device))
    test_acc = float(full_accuracy(model, test_data, device))
    attn_rel, attn_norm = attention_equivariance_relative(model, p, device=device)
    return {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'attn_rel_equi': attn_rel,
        'attn_norm_scale': attn_norm,
    }


def plot_results(rows: list[dict], output_path: Path, title: str):
    if not rows:
        return

    base_labels = []
    for row in rows:
        if row['base_label'] not in base_labels:
            base_labels.append(row['base_label'])
    conditions = []
    for row in rows:
        if row['condition'] not in conditions:
            conditions.append(row['condition'])
    metrics = [
        ('test_acc', 'Test accuracy'),
        ('attn_rel_equi', 'Attention rel. equivariance'),
        ('attn_norm_scale', 'Attention output norm'),
    ]
    fallback_colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#72b7b2', '#b279a2']

    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 4.2))
    if len(metrics) == 1:
        axes = [axes]

    width = min(0.76 / max(len(conditions), 1), 0.18)
    x = np.arange(len(base_labels))
    for ax, (metric, ylabel) in zip(axes, metrics):
        offsets = (np.arange(len(conditions)) - (len(conditions) - 1) / 2.0) * width
        for j, condition in enumerate(conditions):
            vals = []
            for base_label in base_labels:
                match = next(
                    row for row in rows
                    if row['base_label'] == base_label and row['condition'] == condition
                )
                vals.append(match[metric])
            ax.bar(
                x + offsets[j],
                vals,
                width=width,
                color=CONDITION_COLORS.get(condition, fallback_colors[j % len(fallback_colors)]),
                alpha=0.85,
                label=condition,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(base_labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.2)
    axes[0].legend(loc='best', fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Immediate checkpoint surgery for modular runs')
    parser.add_argument('--exp-dir', type=str, default=None)
    parser.add_argument(
        '--equivariance-cache',
        type=str,
        default=None,
        help='Optional pickle containing the equivariance trajectory for --exp-dir.',
    )
    parser.add_argument('--act', type=str, default='SiLU')
    parser.add_argument('--task', type=str, default='add')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--fraction', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--base-labels', type=str, default='t_plus,t_grok,final')
    parser.add_argument(
        '--donor-mode',
        type=str,
        default='equiv_drop',
        choices=['equiv_drop', 'explicit_epoch', 'fixed_frac_tgrok', 'train_acc_threshold', 'memorization_plateau'],
        help='How to choose the donor checkpoint used for donor_pre and reset_* conditions.',
    )
    parser.add_argument(
        '--donor-epoch',
        type=int,
        default=None,
        help='Explicit donor target epoch used when --donor-mode=explicit_epoch.',
    )
    parser.add_argument(
        '--donor-fixed-frac',
        type=float,
        default=0.8,
        help='Fraction of t_grok used when --donor-mode=fixed_frac_tgrok.',
    )
    parser.add_argument(
        '--donor-train-acc-threshold',
        type=float,
        default=0.999,
        help='Train-accuracy threshold used when --donor-mode=train_acc_threshold.',
    )
    parser.add_argument(
        '--donor-test-acc-max',
        type=float,
        default=0.1,
        help='Maximum test accuracy used when --donor-mode=memorization_plateau.',
    )
    parser.add_argument(
        '--conditions',
        type=str,
        default=','.join(DEFAULT_CONDITIONS),
        help='Comma-separated list from donor_pre, base, and reset_* conditions.',
    )
    parser.add_argument(
        '--matched-mlp-candidates',
        type=int,
        default=0,
        help='Number of random parameter-count-matched MLP subsets to evaluate.',
    )
    parser.add_argument(
        '--matched-mlp-norm-topk',
        type=int,
        default=0,
        help=(
            'From the random count-matched subsets, additionally evaluate the '
            'top-k whose donor/base delta L2 norm is closest to the attention reset.'
        ),
    )
    parser.add_argument(
        '--matched-mlp-seed',
        type=int,
        default=0,
        help='Base RNG seed for matched-MLP subset sampling.',
    )
    parser.add_argument(
        '--matched-mlp-mode',
        type=str,
        default='neuron',
        choices=['element', 'neuron'],
        help='Sampling unit for matched MLP subsets.',
    )
    parser.add_argument(
        '--skip-plot',
        action='store_true',
        help='Skip the per-run summary plot (useful when many matched-null conditions are present).',
    )
    parser.add_argument('--output', type=str, default='experiments/checkpoint_surgery_modular')
    args = parser.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    exp_dir = resolve_path(args.exp_dir) if args.exp_dir else lookup_run(args.act, args.task, args.seed)
    checkpoints = available_checkpoints(exp_dir)
    metrics = load_metrics(exp_dir)
    t_grok = detect_grok_epoch(metrics)
    if t_grok is None:
        raise ValueError(f'Run did not grok: {exp_dir}')

    eq = load_equivariance_trajectory(
        exp_dir,
        args.act,
        args.task,
        args.seed,
        cache_path=args.equivariance_cache,
    )
    epochs = np.asarray(eq['epochs'], dtype=int)
    rel_equi = eq['equi_dist'] / np.where(np.asarray(eq['norm_scale']) > 1e-8, np.asarray(eq['norm_scale']), 1.0)
    selection = select_surgery_epochs(epochs, rel_equi, t_grok=t_grok)

    t_sym_epoch = nearest_checkpoint_epoch(selection['t_sym'], checkpoints)
    t_minus_epoch = nearest_checkpoint_epoch(selection['t_minus'], checkpoints)
    donor_epoch, donor_target_epoch = choose_donor_epoch(
        selection,
        t_grok=t_grok,
        checkpoints=checkpoints,
        metrics=metrics,
        donor_mode=args.donor_mode,
        donor_fixed_frac=args.donor_fixed_frac,
        donor_train_acc_threshold=args.donor_train_acc_threshold,
        donor_test_acc_max=args.donor_test_acc_max,
        donor_explicit_epoch=args.donor_epoch,
    )
    t_grok_epoch = nearest_checkpoint_epoch(t_grok, checkpoints)
    final_epoch = extract_epoch(checkpoints[-1])

    base_epoch_map = {
        't_plus': t_sym_epoch,
        't_sym': t_sym_epoch,
        't_grok': t_grok_epoch,
        'final': final_epoch,
    }
    base_labels = [label.strip() for label in args.base_labels.split(',') if label.strip()]
    selected_conditions = [label.strip() for label in args.conditions.split(',') if label.strip()]

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    donor_model, donor_cfg, donor_state = load_checkpoint(
        checkpoint_path_for_epoch(exp_dir, donor_epoch),
        device=device,
    )
    p = int(donor_cfg.get('p', donor_cfg.get('d_vocab', 114) - 1))
    train_data, test_data = modular_dataset(
        task=args.task,
        seed=args.seed,
        p=p,
        fraction=args.fraction,
        device=device,
    )

    rows = []
    for base_label in base_labels:
        if base_label not in base_epoch_map:
            raise ValueError(f'Unknown base label: {base_label}')
        base_epoch = base_epoch_map[base_label]
        base_model, base_cfg, base_state = load_checkpoint(
            checkpoint_path_for_epoch(exp_dir, base_epoch),
            device=device,
        )

        configs = []
        for condition in selected_conditions:
            if condition == 'donor_pre':
                configs.append(
                    {
                        'condition': condition,
                        'state_dict': donor_state,
                        'reset_param_count': 0,
                        'match_family': '',
                        'match_sampling_unit': '',
                        'match_trial_idx': -1,
                        'match_group_rank': -1,
                        'match_target_param_count': 0,
                        'match_target_delta_l2': float('nan'),
                        'match_subset_delta_l2': float('nan'),
                        'match_subset_delta_l2_ratio': float('nan'),
                        'match_subset_delta_l2_abs_error': float('nan'),
                    }
                )
                continue
            if condition == 'base':
                configs.append(
                    {
                        'condition': condition,
                        'state_dict': base_state,
                        'reset_param_count': 0,
                        'match_family': '',
                        'match_sampling_unit': '',
                        'match_trial_idx': -1,
                        'match_group_rank': -1,
                        'match_target_param_count': 0,
                        'match_target_delta_l2': float('nan'),
                        'match_subset_delta_l2': float('nan'),
                        'match_subset_delta_l2_ratio': float('nan'),
                        'match_subset_delta_l2_abs_error': float('nan'),
                    }
                )
                continue
            param_names = condition_param_names(base_state, condition)
            reset_count = sum(int(base_state[name].numel()) for name in param_names)
            configs.append(
                {
                    'condition': condition,
                    'state_dict': copy_selected_params(base_state, donor_state, param_names),
                    'reset_param_count': reset_count,
                    'match_family': '',
                    'match_sampling_unit': '',
                    'match_trial_idx': -1,
                    'match_group_rank': -1,
                    'match_target_param_count': 0,
                    'match_target_delta_l2': float('nan'),
                    'match_subset_delta_l2': float('nan'),
                    'match_subset_delta_l2_ratio': float('nan'),
                    'match_subset_delta_l2_abs_error': float('nan'),
                }
            )

        if args.matched_mlp_candidates > 0:
            match_seed = stable_seed_from_parts(
                args.matched_mlp_seed,
                args.act,
                args.task,
                args.seed,
                base_label,
                base_epoch,
            )
            match_rng = np.random.default_rng(match_seed)
            configs.extend(
                matched_mlp_subset_configs(
                    base_state,
                    donor_state,
                    rng=match_rng,
                    n_candidates=args.matched_mlp_candidates,
                    n_norm_topk=args.matched_mlp_norm_topk,
                    mode=args.matched_mlp_mode,
                )
            )

        for config in configs:
            condition = config['condition']
            state_dict = config['state_dict']
            reset_param_count = config['reset_param_count']
            model = Transformer(
                num_layers=base_cfg.get('num_layers', 1),
                d_vocab=base_cfg.get('d_vocab', 114),
                d_model=base_cfg.get('d_model', 128),
                d_mlp=base_cfg.get('d_mlp', 512),
                d_head=base_cfg.get('d_head', 32),
                num_heads=base_cfg.get('num_heads', 4),
                n_ctx=base_cfg.get('n_ctx', 3),
                act_type=base_cfg.get('act_type', 'SiLU'),
                use_cache=False,
                use_ln=base_cfg.get('use_ln', False),
                attention_rate=base_cfg.get('attention_rate', 1.0) or 1.0,
            ).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            scores = evaluate_model(
                model,
                train_data,
                test_data,
                task=args.task,
                p=p,
                device=device,
            )
            rows.append(
                {
                    'exp_dir': str(exp_dir),
                    'act': args.act,
                    'task': args.task,
                    'seed': args.seed,
                    't_grok': int(t_grok),
                    'peak_epoch': int(selection['peak_epoch']),
                    'largest_negative_delta': float(selection['largest_negative_delta']),
                    't_minus': int(t_minus_epoch),
                    't_sym': int(t_sym_epoch),
                    't_plus': int(t_sym_epoch),
                    'donor_mode': args.donor_mode,
                    'donor_target_epoch': int(donor_target_epoch),
                    'donor_epoch': int(donor_epoch),
                    'base_label': base_label,
                    'base_epoch': int(base_epoch),
                    'condition': condition,
                    'reset_param_count': int(reset_param_count),
                    'match_family': config['match_family'],
                    'match_sampling_unit': config['match_sampling_unit'],
                    'match_trial_idx': int(config['match_trial_idx']),
                    'match_group_rank': int(config['match_group_rank']),
                    'match_target_param_count': int(config['match_target_param_count']),
                    'match_target_delta_l2': float(config['match_target_delta_l2']),
                    'match_subset_delta_l2': float(config['match_subset_delta_l2']),
                    'match_subset_delta_l2_ratio': float(config['match_subset_delta_l2_ratio']),
                    'match_subset_delta_l2_abs_error': float(config['match_subset_delta_l2_abs_error']),
                    **scores,
                }
            )

    csv_path = output_dir / f'{args.act.lower()}_{args.task}_s{args.seed}_surgery.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pkl_path = output_dir / f'{args.act.lower()}_{args.task}_s{args.seed}_surgery.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(
            {
                'rows': rows,
                'selection': selection,
                'equivariance_epochs': epochs,
                'equivariance_rel': rel_equi,
            },
            f,
        )

    if not args.skip_plot:
        plot_results(
            rows,
            output_path=output_dir / f'{args.act.lower()}_{args.task}_s{args.seed}_surgery.png',
            title=f'{args.act}/{args.task}/s{args.seed} checkpoint surgery',
        )

    print(f'Run: {exp_dir}')
    print(
        f'Selected epochs: peak={selection["peak_epoch"]}, '
        f't_minus={t_minus_epoch}, donor={donor_epoch} ({args.donor_mode}), '
        f't_sym={t_sym_epoch}, t_grok={t_grok_epoch}, final={final_epoch}'
    )
    print(
        f'Equivariance: peak={selection["peak_rel_equi"]:.4f}, '
        f'final={selection["final_rel_equi"]:.4f}, '
        f'largest pre-grok diff={selection["largest_negative_delta"]:.4f}'
    )
    if args.matched_mlp_candidates > 0:
        print(
            f'Matched MLP nulls: count-matched={args.matched_mlp_candidates}, '
            f'norm-topk={args.matched_mlp_norm_topk}, mode={args.matched_mlp_mode}'
        )
    print('\nSummary')
    print('-' * 88)
    print(
        f'{"base":<8} {"condition":<14} {"test_acc":>9} {"attn_rel":>10} '
        f'{"attn_norm":>10} {"train_acc":>10} {"n_reset":>10}'
    )
    for row in rows:
        print(
            f'{row["base_label"]:<8} {row["condition"]:<14} '
            f'{row["test_acc"]:>9.4f} {row["attn_rel_equi"]:>10.4f} '
            f'{row["attn_norm_scale"]:>10.4f} {row["train_acc"]:>10.4f} '
            f'{row["reset_param_count"]:>10d}'
        )
    print(f'\nSaved: {csv_path}')
    print(f'Saved: {pkl_path}')


if __name__ == '__main__':
    main()
