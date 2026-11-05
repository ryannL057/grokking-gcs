
from gcs.runtime import data_root as _gcs_data_root
"""
Unified figure generation module for GCS visualization.

This module generates figures for figures_unified/ with pixel-level consistency
to the original scripts in scripts/. It consolidates:

| Original Script                           | Function in this module           |
|-------------------------------------------|-----------------------------------|
| generate_unified_figures.py               | plot_1layer_gcs_with_training()   |
| generate_1layer_mod_add_comparison.py*    | plot_activation_comparison()      |
| generate_1layer_multitask_gcs_comparison* | plot_multitask_comparison()       |
| generate_silu_with_markers.py*            | plot_1layer_with_markers()        |
| generate_2layer_gelu_stable_combined.py   | plot_2layer_combined()            |
| generate_3layer_gelu_combined_2row.py     | plot_3layer_combined()            |

* These scripts may have been renamed or consolidated in the codebase.

Style parameters are extracted directly from the original scripts to ensure
pixel-perfect consistency.
"""

import re
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import PATH_COLORS, LAYER_COLORS, TRAINING_COLORS

# =============================================================================
# Checkpoints Directory Constants
# =============================================================================

GCS_ROOT = _gcs_data_root()
CHECKPOINTS_DIR = GCS_ROOT / 'checkpoints'

# =============================================================================
# Style Constants (extracted from scripts/ for pixel-level consistency)
# =============================================================================

UNIFIED_STYLE = {
    # 2x1 layout (1-layer Transformer: GCS + Training)
    'paper_2x1': {
        'figsize': (6, 5),
        'title_fontsize': 12,
        'label_fontsize': 9,
        'legend_fontsize': 6,
        'tick_fontsize': 8,
        'linewidth': 2.5,
    },
    # 2x2 layout (activation comparison, multitask)
    'paper_2x2': {
        'figsize': (8, 6),
        'title_fontsize': 12,
        'label_fontsize': 11,
        'legend_fontsize': 9,
        'tick_fontsize': 10,
        'linewidth': 2,
    },
    # 2x3 layout (multi-layer combined)
    'paper_2x3': {
        'figsize': (10, 6),
        'title_fontsize': 9,
        'label_fontsize': 9,
        'legend_fontsize': 5,
        'tick_fontsize': 7,
        'linewidth': 2.5,
    },
    # 2x3 layout for 2-layer stable (from generate_2layer_gelu_stable_combined.py)
    'paper_2x3_stable': {
        'figsize': (9, 5.5),
        'title_fontsize': 9,
        'label_fontsize': 9,
        'legend_fontsize': 5,
        'tick_fontsize': 7,
        'linewidth': 2,
        'hspace': 0.35,
        'wspace': 0.3,
    },
}

# Grokking epochs for different experiments (from scripts/)
GROKKING_EPOCHS = {
    # 1-layer Transformer mod-add
    'relu_add': 5900,
    'gelu_add': 12700,
    'silu_add': 10400,
    # 1-layer Transformer mod-sub
    'relu_sub': 6000,
    'gelu_sub': 12000,
    'silu_sub': 10000,
    # 1-layer Transformer mod-mul
    'relu_mul': 8000,
    'gelu_mul': 14000,
    'silu_mul': 12000,
    # 1-layer Transformer mod-div
    'relu_div': 9000,
    'gelu_div': 13000,
    'silu_div': 11000,
    # 2-layer Transformer
    '2layer_gelu_higher_wd_clip': 12000,
    '2layer_gelu_higher_wd': 12000,
    # 3-layer Transformer
    '3layer_gelu_add': 11000,
}

# Train accuracy = 1 epoch (for SiLU with dual markers)
# From generate_silu_with_markers.py
TRAIN_ACC_1_EPOCHS = {
    'silu_add': 600,  # Epoch where train acc first reaches 1.0
}

# Grokking marker color
GROK_COLOR = '#E74C3C'

# Marker shapes for activation comparison
GROK_MARKERS = {
    'ReLU': 'o',
    'GeLU': 's',
    'SiLU': '^',
}

# Per-task epoch truncation limits for multitask figures
# None = auto-scale; integer = truncate at that epoch
MULTITASK_EPOCH_LIMITS = {
    'GeLU': {'div': 40000, 'sub': 40000},  # add/mul auto-scale
    'ReLU': {'add': 10000, 'mul': 10000, 'div': 30000, 'sub': 30000},
    'SiLU': {},  # All tasks auto-scale
}


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_to_01(scores: Union[List, np.ndarray]) -> np.ndarray:
    """
    Normalize scores to [0, 1] range using min-max normalization.

    Exact implementation from scripts/generate_unified_figures.py to ensure
    pixel-level consistency.
    """
    scores = np.array(scores)
    if len(scores) == 0:
        return scores

    min_val = np.min(scores)
    max_val = np.max(scores)

    if max_val - min_val < 1e-10:
        return np.zeros_like(scores)

    return (scores - min_val) / (max_val - min_val)


def parse_experiment_info(exp_path: Path) -> Dict[str, Any]:
    """
    Extract experiment metadata from path.

    From scripts/generate_unified_figures.py.
    """
    exp_path = Path(exp_path)
    parts = exp_path.parts

    info = {
        'path': exp_path,
        'layers': None,
        'architecture': None,
        'activation': None,
        'task': 'add',
        'timestamp': None,
        'variant': None,
    }

    dir_name = exp_path.name
    timestamp_match = re.search(r'(\d{8}_\d{6})', dir_name)
    if timestamp_match:
        info['timestamp'] = timestamp_match.group(1)

    if 'higher_wd_clip' in dir_name:
        info['variant'] = 'higher_wd_clip'
    elif 'higher_wd' in dir_name:
        info['variant'] = 'higher_wd'

    if 'pretrained_tasks' in parts:
        info['layers'] = 1
        info['architecture'] = 'Transformer'

        for part in parts:
            if part.startswith('mod-'):
                info['task'] = part.replace('mod-', '')
                break

        if 'relu' in dir_name.lower():
            info['activation'] = 'ReLU'
        elif 'gelu' in dir_name.lower():
            info['activation'] = 'GeLU'
        elif 'silu' in dir_name.lower():
            info['activation'] = 'SiLU'

    elif '2layer_MLP' in parts or '3layer_MLP' in parts:
        info['architecture'] = 'MLP'
        info['activation'] = 'ReLU'
        info['layers'] = 2 if '2layer_MLP' in parts else 3

        for part in parts:
            if part.startswith('mod-'):
                info['task'] = part.replace('mod-', '')
                break

    elif any('2layer_' in part for part in parts) or any('3layer_' in part for part in parts):
        info['architecture'] = 'Transformer'

        for part in parts:
            if '2layer_' in part:
                info['layers'] = 2
                act = part.replace('2layer_', '')
                if act.startswith('GeLU'):
                    info['activation'] = 'GeLU'
                elif act.startswith('ReLU'):
                    info['activation'] = 'ReLU'
                elif act.startswith('SiLU'):
                    info['activation'] = 'SiLU'
                elif act.startswith('MLP'):
                    info['activation'] = 'ReLU'
                    info['architecture'] = 'MLP'
                else:
                    info['activation'] = act
                break
            elif '3layer_' in part:
                info['layers'] = 3
                act = part.replace('3layer_', '')
                if act.startswith('GeLU'):
                    info['activation'] = 'GeLU'
                elif act.startswith('ReLU'):
                    info['activation'] = 'ReLU'
                elif act.startswith('SiLU'):
                    info['activation'] = 'SiLU'
                elif act.startswith('MLP'):
                    info['activation'] = 'ReLU'
                    info['architecture'] = 'MLP'
                else:
                    info['activation'] = act
                break

    elif any('mlp_' in part for part in parts):
        # MLP experiment: mlp_onehot, mlp_embed_concat, mlp_embed_sum
        for part in parts:
            if part.startswith('mlp_'):
                embed_type = part.replace('mlp_', '')
                info['architecture'] = f'MLP_{embed_type}'
                break

        # Parse directory name: {act}_{layers}layer_{task}_h{hidden_dim}_{timestamp}
        dir_match = re.match(
            r'(\w+)_(\d+)layer_(\w+)_h(\d+)',
            dir_name
        )
        if dir_match:
            info['activation'] = dir_match.group(1).upper()
            # Normalize activation name
            act_map = {'RELU': 'ReLU', 'GELU': 'GeLU', 'SILU': 'SiLU', 'SELU': 'SeLU'}
            info['activation'] = act_map.get(info['activation'], info['activation'])
            info['layers'] = int(dir_match.group(2))
            info['task'] = dir_match.group(3)

    else:
        info['layers'] = 1
        info['architecture'] = 'Transformer'

        for part in parts:
            if part in ['ReLU', 'GeLU', 'SiLU']:
                info['activation'] = part
                break

    return info


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics from pickle file."""
    with open(metrics_path, 'rb') as f:
        return pickle.load(f)


def _apply_epoch_truncation(
    steps: np.ndarray,
    scores: np.ndarray,
    max_epoch: Optional[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply epoch truncation to data arrays."""
    if max_epoch is not None:
        mask = steps <= max_epoch
        return steps[mask], scores[mask]
    return steps, scores


# =============================================================================
# Plot Functions - 1-Layer Transformer
# =============================================================================

def plot_1layer_gcs_with_training(
    metrics_path: Union[str, Path],
    mode: str = 'normalized',
    save_path: Optional[Union[str, Path]] = None,
    show_grokking_marker: bool = True,
    max_epoch: Optional[int] = None,
) -> plt.Figure:
    """
    Generate 2×1 figure for 1-layer Transformer: GCS subplot + Training subplot.

    Matches the style from generate_unified_figures.py for 1-layer experiments.

    Args:
        metrics_path: Path to training_metrics_final_geo.pkl
        mode: 'raw' or 'normalized'
        save_path: Optional path to save figure
        show_grokking_marker: Whether to show grokking epoch marker
        max_epoch: Optional max epoch for truncation

    Returns:
        matplotlib Figure object
    """
    metrics_path = Path(metrics_path)
    metrics = load_metrics(metrics_path)

    # Extract info from path
    info = parse_experiment_info(metrics_path.parent)

    # Extract data
    log_steps = np.array(metrics.get('log_steps', []))
    train_accs = metrics.get('train_accuracies', metrics.get('train_accs', []))
    test_accs = metrics.get('test_accuracies', metrics.get('test_accs', []))
    test_losses = metrics.get('test_losses', [])

    gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', [])))
    path_gcs = metrics.get('path_gcs_scores_final_geo', {})
    gcs_overall = path_gcs.get('overall_path', [])
    gcs_attn = path_gcs.get('attention_path', [])
    gcs_mlp = path_gcs.get('mlp_path', [])

    # Style
    style = UNIFIED_STYLE['paper_2x1']

    # Color palette
    path_colors = {
        'overall': '#0173B2',
        'attention': '#DE8F05',
        'mlp': '#CC3311',  # Red for MLP (from generate_silu_with_markers.py style)
    }

    # Create figure
    fig = plt.figure(figsize=style['figsize'])
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.3)

    # === Top: GCS curves ===
    ax_gcs = fig.add_subplot(gs[0])

    ylabel = 'Normalized GCS' if mode == 'normalized' else 'GCS'

    if len(gcs_overall) > 0:
        steps_plot = gcs_steps[:len(gcs_overall)]
        scores = gcs_overall if mode == 'raw' else normalize_to_01(gcs_overall)
        steps_plot, scores = _apply_epoch_truncation(np.array(steps_plot), np.array(scores), max_epoch)
        ax_gcs.plot(steps_plot, scores, '-', color=path_colors['overall'],
                   linewidth=style['linewidth'], label='Overall Flow', alpha=0.9)

    if len(gcs_attn) > 0:
        steps_plot = gcs_steps[:len(gcs_attn)]
        scores = gcs_attn if mode == 'raw' else normalize_to_01(gcs_attn)
        steps_plot, scores = _apply_epoch_truncation(np.array(steps_plot), np.array(scores), max_epoch)
        ax_gcs.plot(steps_plot, scores, '-', color=path_colors['attention'],
                   linewidth=style['linewidth'], label='Attention Flow', alpha=0.9)

    if len(gcs_mlp) > 0:
        steps_plot = gcs_steps[:len(gcs_mlp)]
        scores = gcs_mlp if mode == 'raw' else normalize_to_01(gcs_mlp)
        steps_plot, scores = _apply_epoch_truncation(np.array(steps_plot), np.array(scores), max_epoch)
        ax_gcs.plot(steps_plot, scores, '-', color=path_colors['mlp'],
                   linewidth=style['linewidth'], label='MLP Flow', alpha=0.9)

    ax_gcs.set_xlabel('Training Epoch', fontsize=style['label_fontsize'], weight='bold')
    ax_gcs.set_ylabel(ylabel, fontsize=style['label_fontsize'], weight='bold')

    title = f"{info['layers']}-Layer {info['architecture']} ({info['activation']}) - Task: mod-{info['task']}"
    ax_gcs.set_title(title, fontsize=style['title_fontsize'], weight='bold', pad=10)

    ax_gcs.legend(loc='best', fontsize=style['legend_fontsize'], framealpha=0.95)
    ax_gcs.grid(True, alpha=0.3, linestyle='--')
    ax_gcs.tick_params(labelsize=style['tick_fontsize'])

    if mode == 'raw':
        ax_gcs.relim()
        ax_gcs.autoscale_view()
        y_min, y_max = ax_gcs.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax_gcs.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_gcs.set_ylim(-0.05, 1.05)

    # === Bottom: Training dynamics ===
    ax_acc = fig.add_subplot(gs[1])

    if len(train_accs) > 0:
        steps_plot = log_steps[:len(train_accs)]
        ax_acc.plot(steps_plot, train_accs, '-', color='#0173B2',
                   linewidth=style['linewidth'], label='Train Acc', alpha=0.9)

    if len(test_accs) > 0:
        steps_plot = log_steps[:len(test_accs)]
        ax_acc.plot(steps_plot, test_accs, '-', color='#029E73',
                   linewidth=style['linewidth'], label='Test Acc', alpha=0.9)

    ax_acc.set_xlabel('Training Epoch', fontsize=style['label_fontsize'], weight='bold')
    ax_acc.set_ylabel('Accuracy', fontsize=style['label_fontsize'], weight='bold', color='#0173B2')
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.tick_params(axis='y', labelcolor='#0173B2', labelsize=style['tick_fontsize'])
    ax_acc.tick_params(axis='x', labelsize=style['tick_fontsize'])
    ax_acc.grid(True, alpha=0.3, linestyle='--')

    # Right y-axis: Loss
    ax_loss = ax_acc.twinx()
    if len(test_losses) > 0:
        steps_plot = log_steps[:len(test_losses)]
        ax_loss.plot(steps_plot, test_losses, '--', color='#CC3311',
                    linewidth=style['linewidth'], label='Test Loss', alpha=0.9)

    ax_loss.set_ylabel('Loss (log scale)', fontsize=style['label_fontsize'], weight='bold', color='#CC3311')
    ax_loss.set_yscale('log')
    ax_loss.tick_params(axis='y', labelcolor='#CC3311', labelsize=style['tick_fontsize'])

    # Combined legend
    lines1, labels1 = ax_acc.get_legend_handles_labels()
    lines2, labels2 = ax_loss.get_legend_handles_labels()
    ax_acc.legend(lines1 + lines2, labels1 + labels2,
                 loc='center right', fontsize=style['legend_fontsize'], framealpha=0.95)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_1layer_with_markers(
    metrics_path: Union[str, Path],
    mode: str = 'normalized',
    save_path: Optional[Union[str, Path]] = None,
    grokking_epoch: Optional[int] = None,
    train_acc_1_epoch: Optional[int] = None,
) -> plt.Figure:
    """
    Generate 2×1 figure with dual markers (grokking + train_acc=1).

    Exactly matches generate_silu_with_markers.py style for pixel-level consistency.

    Args:
        metrics_path: Path to training_metrics_final_geo.pkl
        mode: 'raw' or 'normalized'
        save_path: Optional path to save figure
        grokking_epoch: Grokking epoch (auto-detected if None)
        train_acc_1_epoch: Train accuracy = 1 epoch (auto-detected if None)

    Returns:
        matplotlib Figure object
    """
    metrics_path = Path(metrics_path)
    metrics = load_metrics(metrics_path)
    info = parse_experiment_info(metrics_path.parent)

    # Auto-detect epochs if not provided
    activation_task_key = f"{info['activation'].lower()}_{info['task']}"
    if grokking_epoch is None:
        grokking_epoch = GROKKING_EPOCHS.get(activation_task_key)
    if train_acc_1_epoch is None:
        train_acc_1_epoch = TRAIN_ACC_1_EPOCHS.get(activation_task_key)

    # Extract data
    log_steps = np.array(metrics.get('log_steps', []))
    train_accs = metrics.get('train_accuracies', metrics.get('train_accs', []))
    test_accs = metrics.get('test_accuracies', metrics.get('test_accs', []))
    test_losses = metrics.get('test_losses', [])

    gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', [])))
    path_gcs = metrics.get('path_gcs_scores_final_geo', {})
    gcs_overall = path_gcs.get('overall_path', [])
    gcs_attn = path_gcs.get('attention_path', [])
    gcs_mlp = path_gcs.get('mlp_path', [])

    # Exact style from generate_silu_with_markers.py
    # figsize=(6, 5), fontsize 9/6/8
    path_colors = {
        'overall': '#0173B2',    # Blue
        'attention': '#DE8F05',  # Orange
        'mlp': '#029E73',        # Green (used in original, NOT #CC3311)
    }

    # Marker colors (from generate_silu_with_markers.py)
    TRAIN_ACC_COLOR = '#9B59B6'  # Purple for train_acc=1

    # Create figure - exactly matching generate_silu_with_markers.py
    fig = plt.figure(figsize=(6, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.3)

    # === Top: GCS curves ===
    ax_gcs = fig.add_subplot(gs[0, :])

    # Plot 3-path GCS
    if len(gcs_overall) > 0:
        steps_plot = gcs_steps[:len(gcs_overall)]
        scores = gcs_overall if mode == 'raw' else normalize_to_01(gcs_overall)
        ax_gcs.plot(steps_plot, scores, '-', color=path_colors['overall'],
                   linewidth=2.5, label='Overall Flow', alpha=0.9)

    if len(gcs_attn) > 0:
        steps_plot = gcs_steps[:len(gcs_attn)]
        scores = gcs_attn if mode == 'raw' else normalize_to_01(gcs_attn)
        ax_gcs.plot(steps_plot, scores, '-', color=path_colors['attention'],
                   linewidth=2.5, label='Attention Flow', alpha=0.9)

    if len(gcs_mlp) > 0:
        steps_plot = gcs_steps[:len(gcs_mlp)]
        scores = gcs_mlp if mode == 'raw' else normalize_to_01(gcs_mlp)
        # MLP uses #CC3311 in generate_silu_with_markers.py
        ax_gcs.plot(steps_plot, scores, '-', color='#CC3311',
                   linewidth=2.5, label='MLP Flow', alpha=0.9)

    # Add vertical markers on GCS plot
    if train_acc_1_epoch is not None:
        ax_gcs.axvline(x=train_acc_1_epoch, color=TRAIN_ACC_COLOR, linestyle='--',
                      linewidth=2, alpha=0.8, label=f'Train Acc=1.0 (epoch {train_acc_1_epoch})')

    if grokking_epoch is not None:
        ax_gcs.axvline(x=grokking_epoch, color=GROK_COLOR, linestyle='--',
                      linewidth=2, alpha=0.8, label=f'Grokking (epoch {grokking_epoch})')

    # Styling for GCS plot - exact match
    ax_gcs.set_xlabel('Training Epoch', fontsize=9)
    ylabel = 'Normalized GCS' if mode == 'normalized' else 'GCS'
    ax_gcs.set_ylabel(ylabel, fontsize=9)
    # No title - will be added in paper (from original script comment)
    ax_gcs.legend(loc='lower right', fontsize=6, framealpha=0.9)
    ax_gcs.grid(True, alpha=0.3, linestyle='--')
    ax_gcs.tick_params(labelsize=8)

    if mode == 'raw':
        ax_gcs.relim()
        ax_gcs.autoscale_view()
        y_min, y_max = ax_gcs.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax_gcs.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_gcs.set_ylim(-0.05, 1.05)

    # === Bottom: Training dynamics (accuracy only) ===
    ax_acc = fig.add_subplot(gs[1, :])

    # Plot accuracies
    if len(train_accs) > 0:
        steps_plot = log_steps[:len(train_accs)]
        ax_acc.plot(steps_plot, train_accs, '-', color='#0173B2',
                   linewidth=2.5, label='Train Acc', alpha=0.9)

    if len(test_accs) > 0:
        steps_plot = log_steps[:len(test_accs)]
        ax_acc.plot(steps_plot, test_accs, '-', color='#029E73',
                   linewidth=2.5, label='Test Acc', alpha=0.9)

    ax_acc.set_xlabel('Training Epoch', fontsize=9)
    ax_acc.set_ylabel('Accuracy', fontsize=9)
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.tick_params(axis='both', labelsize=8)
    ax_acc.grid(True, alpha=0.3, linestyle='--')

    # Add vertical markers on accuracy plot
    if train_acc_1_epoch is not None:
        ax_acc.axvline(x=train_acc_1_epoch, color=TRAIN_ACC_COLOR, linestyle='--',
                      linewidth=2, alpha=0.8, label=f'Train Acc=1.0 (epoch {train_acc_1_epoch})')

    if grokking_epoch is not None:
        ax_acc.axvline(x=grokking_epoch, color=GROK_COLOR, linestyle='--',
                      linewidth=2, alpha=0.8, label=f'Grokking (epoch {grokking_epoch})')

    # Legend
    ax_acc.legend(loc='center right', fontsize=6, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_activation_comparison(
    metrics_dict: Dict[str, Union[Path, Any]],
    task: str = 'add',
    mode: str = 'normalized',
    save_path: Optional[Union[str, Path]] = None,
    show_grokking_markers: bool = True,
) -> plt.Figure:
    """
    Generate 2×2 activation comparison figure for mod-add task.

    Exactly matches generate_plots/generate_1layer_mod_add_comparison.py::plot_gcs_comparison().

    Layout:
    - Top-left: Test Accuracy (all activations)
    - Top-right: Overall Flow GCS (Normalized)
    - Bottom-left: Attention Flow GCS (Normalized)
    - Bottom-right: MLP Flow GCS (Normalized)

    Args:
        metrics_dict: Dict mapping activation name to metrics path or loaded metrics dict
            e.g., {'ReLU': Path(...), 'GeLU': Path(...), 'SiLU': Path(...)}
        task: Task name (default 'add')
        mode: 'raw' or 'normalized' (only 'normalized' matches the original output)
        save_path: Optional path to save figure
        show_grokking_markers: Whether to show grokking markers

    Returns:
        matplotlib Figure object
    """
    # Style settings - exact match from generate_1layer_mod_add_comparison.py
    title_fontsize = 12
    label_fontsize = 11
    legend_fontsize = 9
    tick_fontsize = 10
    linewidth = 2

    activation_functions = ['ReLU', 'GeLU', 'SiLU']

    # Color palette matching generate_unified_figures.py
    color_map = {
        'ReLU': '#0173B2',    # Blue
        'GeLU': '#DE8F05',    # Orange
        'SiLU': '#029E73',    # Green
    }

    # Grokking points
    grokking_epochs = {
        'ReLU': GROKKING_EPOCHS.get(f'relu_{task}', 5900),
        'GeLU': GROKKING_EPOCHS.get(f'gelu_{task}', 12700),
        'SiLU': GROKKING_EPOCHS.get(f'silu_{task}', 10400),
    }

    # Marker styles for grokking points
    grok_markers = {
        'ReLU': 'o',      # circle
        'GeLU': 's',      # square
        'SiLU': '^',      # triangle
    }

    # Load all data
    all_results = {}
    for activation, metrics_source in metrics_dict.items():
        if isinstance(metrics_source, dict):
            all_results[activation] = metrics_source
        else:
            all_results[activation] = load_metrics(metrics_source)

    # Normalization function (exact match from original)
    def normalize_data(data):
        data = np.array([x for x in data if x is not None])
        if len(data) > 0 and np.max(data) != np.min(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        return data

    # Helper function to add grokking point markers on a subplot
    def add_grokking_markers_fn(ax, show_legend=False):
        if not show_grokking_markers:
            return
        for act_func in activation_functions:
            if act_func not in all_results:
                continue
            epoch = grokking_epochs.get(act_func)
            if epoch is None:
                continue
            ax.axvline(x=epoch, color=color_map[act_func], linestyle='--',
                      linewidth=1.5, alpha=0.7,
                      label=f'{act_func} grok' if show_legend else None)
            # Add a small marker at grokking point
            ylim = ax.get_ylim()
            y_pos = ylim[0] + 0.92 * (ylim[1] - ylim[0])  # Near top
            ax.plot(epoch, y_pos, marker=grok_markers[act_func],
                   color=color_map[act_func], markersize=8, zorder=10)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

    # Subplot 1 (left-top): Test Accuracy
    ax1.set_title('Test Accuracy', fontsize=title_fontsize)
    for act_func in activation_functions:
        if act_func in all_results and all_results[act_func] is not None:
            metrics = all_results[act_func]
            steps = metrics.get('log_steps', [])
            test_accs = metrics.get('test_accuracies', [])
            if steps and test_accs:
                ax1.plot(steps, test_accs,
                        color=color_map.get(act_func, 'gray'),
                        label=act_func,
                        linewidth=linewidth,
                        alpha=0.9)

    ax1.set_xlabel('Training Epoch', fontsize=label_fontsize)
    ax1.set_ylabel('Test Accuracy', fontsize=label_fontsize)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, None)
    add_grokking_markers_fn(ax1, show_legend=True)
    ax1.legend(fontsize=legend_fontsize, framealpha=0.95, loc='lower right')

    # Subplot 2 (right-top): Overall Flow GCS Score
    ax2.set_title('Overall Flow GCS' + (' (Normalized)' if mode == 'normalized' else ''), fontsize=title_fontsize)
    for act_func in activation_functions:
        if act_func in all_results and all_results[act_func] is not None:
            metrics = all_results[act_func]
            gcs_steps = metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', []))
            overall_scores = metrics.get('path_gcs_scores_final_geo', metrics.get('path_gcs_scores', {})).get('overall_path', [])
            if gcs_steps and overall_scores:
                valid_steps = []
                valid_scores = []
                for i, score in enumerate(overall_scores):
                    if score is not None and i < len(gcs_steps):
                        valid_steps.append(gcs_steps[i])
                        valid_scores.append(score)
                if valid_scores:
                    normalized_scores = normalize_data(valid_scores) if mode == 'normalized' else np.array(valid_scores)
                    if len(normalized_scores) > 0:
                        ax2.plot(valid_steps, normalized_scores,
                               color=color_map.get(act_func, 'gray'),
                               label=act_func,
                               linewidth=linewidth,
                               alpha=0.9)

    ax2.set_xlabel('Training Epoch', fontsize=label_fontsize)
    ax2.set_ylabel('Normalized GCS' if mode == 'normalized' else 'GCS', fontsize=label_fontsize)
    ax2.legend(fontsize=legend_fontsize, framealpha=0.95, loc='lower right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if mode == 'normalized':
        ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(0, None)
    add_grokking_markers_fn(ax2, show_legend=False)

    # Subplot 3 (left-bottom): Attention Flow GCS Score
    ax3.set_title('Attention Flow GCS' + (' (Normalized)' if mode == 'normalized' else ''), fontsize=title_fontsize)
    for act_func in activation_functions:
        if act_func in all_results and all_results[act_func] is not None:
            metrics = all_results[act_func]
            gcs_steps = metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', []))
            attention_scores = metrics.get('path_gcs_scores_final_geo', metrics.get('path_gcs_scores', {})).get('attention_path', [])
            if gcs_steps and attention_scores:
                valid_steps = []
                valid_scores = []
                for i, score in enumerate(attention_scores):
                    if score is not None and i < len(gcs_steps):
                        valid_steps.append(gcs_steps[i])
                        valid_scores.append(score)
                if valid_scores:
                    normalized_scores = normalize_data(valid_scores) if mode == 'normalized' else np.array(valid_scores)
                    if len(normalized_scores) > 0:
                        ax3.plot(valid_steps, normalized_scores,
                               color=color_map.get(act_func, 'gray'),
                               label=act_func,
                               linewidth=linewidth,
                               alpha=0.9)

    ax3.set_xlabel('Training Epoch', fontsize=label_fontsize)
    ax3.set_ylabel('Normalized GCS' if mode == 'normalized' else 'GCS', fontsize=label_fontsize)
    ax3.legend(fontsize=legend_fontsize, framealpha=0.95, loc='lower right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if mode == 'normalized':
        ax3.set_ylim(-0.05, 1.05)
    ax3.set_xlim(0, None)
    add_grokking_markers_fn(ax3, show_legend=False)

    # Subplot 4 (right-bottom): MLP Flow GCS Score
    ax4.set_title('MLP Flow GCS' + (' (Normalized)' if mode == 'normalized' else ''), fontsize=title_fontsize)
    for act_func in activation_functions:
        if act_func in all_results and all_results[act_func] is not None:
            metrics = all_results[act_func]
            gcs_steps = metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', []))
            mlp_scores = metrics.get('path_gcs_scores_final_geo', metrics.get('path_gcs_scores', {})).get('mlp_path', [])
            if gcs_steps and mlp_scores:
                valid_steps = []
                valid_scores = []
                for i, score in enumerate(mlp_scores):
                    if score is not None and i < len(gcs_steps):
                        valid_steps.append(gcs_steps[i])
                        valid_scores.append(score)
                if valid_scores:
                    normalized_scores = normalize_data(valid_scores) if mode == 'normalized' else np.array(valid_scores)
                    if len(normalized_scores) > 0:
                        ax4.plot(valid_steps, normalized_scores,
                               color=color_map.get(act_func, 'gray'),
                               label=act_func,
                               linewidth=linewidth,
                               alpha=0.9)

    ax4.set_xlabel('Training Epoch', fontsize=label_fontsize)
    ax4.set_ylabel('Normalized GCS' if mode == 'normalized' else 'GCS', fontsize=label_fontsize)
    ax4.legend(fontsize=legend_fontsize, framealpha=0.95, loc='lower right')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    if mode == 'normalized':
        ax4.set_ylim(-0.05, 1.05)
    ax4.set_xlim(0, None)
    add_grokking_markers_fn(ax4, show_legend=False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_multitask_comparison(
    activation: str,
    task_metrics_dict: Optional[Dict[str, Union[Path, Any]]] = None,
    mode: str = 'raw',
    save_path: Optional[Union[str, Path]] = None,
    max_epochs_per_task: Optional[Dict[str, Optional[int]]] = None,
) -> plt.Figure:
    """
    Generate 2×2 multi-task comparison figure for one activation.

    Exactly matches generate_plots/generate_1layer_multitask_gcs_comparison.py.

    Layout:
    - Top-left: Modular Addition (3 paths)
    - Top-right: Modular Division (3 paths)
    - Bottom-left: Modular Multiplication (3 paths)
    - Bottom-right: Modular Subtraction (3 paths)

    Args:
        activation: Activation function name (e.g., 'ReLU')
        task_metrics_dict: Dict mapping task name to metrics path or loaded metrics.
            If None, auto-discovers from gcs_core/checkpoints/1layer/{activation}/.
        mode: 'raw' or 'normalized'
        save_path: Optional path to save figure
        max_epochs_per_task: Dict mapping task name to max epoch for truncation.
            If None, uses activation-specific defaults from MULTITASK_EPOCH_LIMITS.

    Returns:
        matplotlib Figure object
    """
    # Style settings - exact match from generate_1layer_multitask_gcs_comparison.py
    title_fontsize = 12
    label_fontsize = 11
    legend_fontsize = 9
    tick_fontsize = 10
    linewidth = 2

    tasks = ['add', 'div', 'mul', 'sub']
    task_titles = {
        'add': 'Modular Addition',
        'div': 'Modular Division',
        'mul': 'Modular Multiplication',
        'sub': 'Modular Subtraction',
    }
    task_positions = {
        'add': 0,  # top-left
        'div': 1,  # top-right
        'mul': 2,  # bottom-left
        'sub': 3,  # bottom-right
    }

    path_colors = {
        'overall': '#0173B2',    # Blue
        'attention': '#DE8F05',  # Orange
        'mlp': '#029E73',        # Green
    }

    # Grokking epochs per activation per task (from original script)
    grokking_epochs_by_activation = {
        'ReLU': {'add': 5800, 'sub': 17600, 'mul': 4200, 'div': 16300},
        'GeLU': {'add': 12700, 'sub': 24000, 'mul': 10500, 'div': 22000},
        'SiLU': {'add': 10400, 'sub': 34800, 'mul': 8500, 'div': 38700},
    }
    grokking_epochs = grokking_epochs_by_activation.get(activation, {})

    # Per-task epoch truncation (use activation-specific defaults if not provided)
    if max_epochs_per_task is None:
        max_epochs_per_task = MULTITASK_EPOCH_LIMITS.get(activation, {})

    ylabel = 'Normalized GCS' if mode == 'normalized' else 'GCS Score'

    # Auto-discover metrics if not provided
    if task_metrics_dict is None:
        task_metrics_dict = {}
        for task in tasks:
            # Use new checkpoints directory structure: 1layer/{activation}/{task}/
            path = CHECKPOINTS_DIR / '1layer' / activation / task / 'training_metrics_final_geo.pkl'
            if path.exists():
                task_metrics_dict[task] = path

    # Load all metrics
    all_metrics = {}
    for task, source in task_metrics_dict.items():
        if isinstance(source, dict):
            all_metrics[task] = source
        else:
            all_metrics[task] = load_metrics(source)

    # Create 2x2 figure - compact size with larger fonts
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes_flat = axes.flatten()

    for task in tasks:
        ax = axes_flat[task_positions[task]]

        if task not in all_metrics:
            ax.set_title(f'{task_titles[task]} (No Data)', fontsize=title_fontsize, weight='bold')
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=label_fontsize)
            continue

        metrics = all_metrics[task]

        # Extract GCS data
        gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', [])))
        path_gcs = metrics.get('path_gcs_scores_final_geo', metrics.get('path_gcs_scores', {}))

        gcs_overall = path_gcs.get('overall_path', [])
        gcs_attn = path_gcs.get('attention_path', [])
        gcs_mlp = path_gcs.get('mlp_path', [])

        # Get max epoch for this task
        max_epoch = max_epochs_per_task.get(task)

        # Plot Overall Path
        if len(gcs_overall) > 0:
            steps_plot = gcs_steps[:len(gcs_overall)]
            scores = np.array(gcs_overall)
            valid_mask = np.array([s is not None for s in scores])
            if np.any(valid_mask):
                steps_plot = steps_plot[valid_mask]
                scores = scores[valid_mask].astype(float)
                if mode == 'normalized':
                    scores = normalize_to_01(scores)
                steps_plot, scores = _apply_epoch_truncation(steps_plot, scores, max_epoch)
                ax.plot(steps_plot, scores, '-', color=path_colors['overall'],
                       linewidth=linewidth, label='Overall Flow', alpha=0.9)

        # Plot Attention Path
        if len(gcs_attn) > 0:
            steps_plot = gcs_steps[:len(gcs_attn)]
            scores = np.array(gcs_attn)
            valid_mask = np.array([s is not None for s in scores])
            if np.any(valid_mask):
                steps_plot = steps_plot[valid_mask]
                scores = scores[valid_mask].astype(float)
                if mode == 'normalized':
                    scores = normalize_to_01(scores)
                steps_plot, scores = _apply_epoch_truncation(steps_plot, scores, max_epoch)
                ax.plot(steps_plot, scores, '-', color=path_colors['attention'],
                       linewidth=linewidth, label='Attention Flow', alpha=0.9)

        # Plot MLP Path
        if len(gcs_mlp) > 0:
            steps_plot = gcs_steps[:len(gcs_mlp)]
            scores = np.array(gcs_mlp)
            valid_mask = np.array([s is not None for s in scores])
            if np.any(valid_mask):
                steps_plot = steps_plot[valid_mask]
                scores = scores[valid_mask].astype(float)
                if mode == 'normalized':
                    scores = normalize_to_01(scores)
                steps_plot, scores = _apply_epoch_truncation(steps_plot, scores, max_epoch)
                ax.plot(steps_plot, scores, '-', color=path_colors['mlp'],
                       linewidth=linewidth, label='MLP Flow', alpha=0.9)

        # Add grokking marker if available and within truncation range
        if task in grokking_epochs:
            grok_epoch = grokking_epochs[task]
            if max_epoch is None or grok_epoch <= max_epoch:
                ax.axvline(x=grok_epoch, color=GROK_COLOR, linestyle='--',
                          linewidth=2, alpha=0.7, label=f'Grokking (epoch {grok_epoch})')

        # Configure subplot - consistent style for paper
        ax.set_xlabel('Training Epoch', fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_title(task_titles[task], fontsize=title_fontsize)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.set_xlim(0, max_epoch)  # None = auto-scale

        if mode == 'normalized':
            ax.set_ylim(-0.05, 1.05)

    # Shared legend at bottom of figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.02), ncol=4,
                   fontsize=legend_fontsize, framealpha=0.95)

    # No overall title - will be added in paper
    plt.tight_layout(rect=[0, 0.04, 1, 1])  # Leave space for bottom legend

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Plot Functions - Multi-Layer Transformer
# =============================================================================

def plot_2layer_combined(
    metrics_path: Union[str, Path],
    mode: str = 'normalized',
    save_path: Optional[Union[str, Path]] = None,
    variant: Optional[str] = None,
    use_fixed_data: bool = False,
) -> plt.Figure:
    """
    Generate 2×3 combined figure for 2-layer Transformer.

    Layout:
    - Top row: Layer 1 (3 flows), Layer 2 (3 flows), ACC & Loss
    - Bottom row: Overall Flow (2 layers), Attention Flow (2 layers), MLP Flow (2 layers)

    Matches generate_2layer_gelu_stable_combined.py style.

    Args:
        metrics_path: Path to training_metrics_final_geo.pkl or training_metrics_FIXED.pkl
        mode: 'raw' or 'normalized'
        save_path: Optional path to save figure
        variant: Variant name (e.g., 'higher_wd_clip')
        use_fixed_data: If True, load training_metrics_FIXED.pkl instead

    Returns:
        matplotlib Figure object
    """
    metrics_path = Path(metrics_path)

    if use_fixed_data:
        actual_path = metrics_path.parent / 'training_metrics_FIXED.pkl'
    else:
        actual_path = metrics_path

    metrics = load_metrics(actual_path)

    style = UNIFIED_STYLE['paper_2x3_stable']

    # Extract data
    log_steps = metrics.get('log_steps', [])
    train_accs = metrics.get('train_accuracies', [])
    test_accs = metrics.get('test_accuracies', [])
    test_losses = metrics.get('test_losses', [])
    gcs_steps = np.array(metrics.get('gcs_steps_final_geo', []))

    path_gcs = metrics.get('path_gcs_scores_final_geo', {})

    num_layers = 2
    overall_by_layer = []
    attn_by_layer = []
    mlp_by_layer = []

    for layer_idx in range(num_layers):
        overall_by_layer.append(path_gcs.get(f'layer{layer_idx}_overall_path', []))
        attn_by_layer.append(path_gcs.get(f'layer{layer_idx}_attention_path', []))
        mlp_by_layer.append(path_gcs.get(f'layer{layer_idx}_mlp_path', []))

    # Color palettes
    path_colors = {
        'overall': '#0173B2',
        'attention': '#DE8F05',
        'mlp': '#029E73',
    }
    layer_colors = ['#0173B2', '#DE8F05']

    ylabel = 'Normalized GCS' if mode == 'normalized' else 'GCS'

    # Create figure
    fig = plt.figure(figsize=style['figsize'])
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1],
                         hspace=style['hspace'], wspace=style['wspace'])

    # === TOP ROW ===

    # Subplot 1: Layer 1 (3 flows)
    ax_layer1 = fig.add_subplot(gs[0, 0])
    layer_idx = 0
    if len(overall_by_layer[layer_idx]) > 0:
        steps_plot = gcs_steps[:len(overall_by_layer[layer_idx])]
        scores = np.array(overall_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(overall_by_layer[layer_idx]))
        ax_layer1.plot(steps_plot, scores, '-', color=path_colors['overall'],
                      linewidth=style['linewidth'], label='Overall Flow', alpha=0.9)
    if len(attn_by_layer[layer_idx]) > 0:
        steps_plot = gcs_steps[:len(attn_by_layer[layer_idx])]
        scores = np.array(attn_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(attn_by_layer[layer_idx]))
        ax_layer1.plot(steps_plot, scores, '-', color=path_colors['attention'],
                      linewidth=style['linewidth'], label='Attention Flow', alpha=0.9)
    if len(mlp_by_layer[layer_idx]) > 0:
        steps_plot = gcs_steps[:len(mlp_by_layer[layer_idx])]
        scores = np.array(mlp_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(mlp_by_layer[layer_idx]))
        ax_layer1.plot(steps_plot, scores, '-', color=path_colors['mlp'],
                      linewidth=style['linewidth'], label='MLP Flow', alpha=0.9)

    ax_layer1.set_xlabel('Training Epoch', fontsize=style['label_fontsize'])
    ax_layer1.set_ylabel(ylabel, fontsize=style['label_fontsize'])
    ax_layer1.set_title('Layer 1', fontsize=style['title_fontsize'])
    ax_layer1.legend(loc='lower left', fontsize=style['legend_fontsize'], framealpha=0.9)
    ax_layer1.grid(True, alpha=0.3, linestyle='--')
    ax_layer1.tick_params(labelsize=style['tick_fontsize'])

    if mode == 'raw':
        ax_layer1.relim()
        ax_layer1.autoscale_view()
        y_min, y_max = ax_layer1.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax_layer1.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_layer1.set_ylim(-0.05, 1.05)

    # Subplot 2: Layer 2 (3 flows)
    ax_layer2 = fig.add_subplot(gs[0, 1])
    layer_idx = 1
    if len(overall_by_layer[layer_idx]) > 0:
        steps_plot = gcs_steps[:len(overall_by_layer[layer_idx])]
        scores = np.array(overall_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(overall_by_layer[layer_idx]))
        ax_layer2.plot(steps_plot, scores, '-', color=path_colors['overall'],
                      linewidth=style['linewidth'], label='Overall Flow', alpha=0.9)
    if len(attn_by_layer[layer_idx]) > 0:
        steps_plot = gcs_steps[:len(attn_by_layer[layer_idx])]
        scores = np.array(attn_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(attn_by_layer[layer_idx]))
        ax_layer2.plot(steps_plot, scores, '-', color=path_colors['attention'],
                      linewidth=style['linewidth'], label='Attention Flow', alpha=0.9)
    if len(mlp_by_layer[layer_idx]) > 0:
        steps_plot = gcs_steps[:len(mlp_by_layer[layer_idx])]
        scores = np.array(mlp_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(mlp_by_layer[layer_idx]))
        ax_layer2.plot(steps_plot, scores, '-', color=path_colors['mlp'],
                      linewidth=style['linewidth'], label='MLP Flow', alpha=0.9)

    ax_layer2.set_xlabel('Training Epoch', fontsize=style['label_fontsize'])
    ax_layer2.set_ylabel(ylabel, fontsize=style['label_fontsize'])
    ax_layer2.set_title('Layer 2', fontsize=style['title_fontsize'])
    ax_layer2.legend(loc='upper left', fontsize=style['legend_fontsize'], framealpha=0.9)
    ax_layer2.grid(True, alpha=0.3, linestyle='--')
    ax_layer2.tick_params(labelsize=style['tick_fontsize'])

    if mode == 'raw':
        ax_layer2.relim()
        ax_layer2.autoscale_view()
        y_min, y_max = ax_layer2.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax_layer2.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_layer2.set_ylim(-0.05, 1.05)

    # Subplot 3: ACC & Loss
    ax_acc = fig.add_subplot(gs[0, 2])

    if len(train_accs) > 0:
        steps_plot = log_steps[:len(train_accs)]
        ax_acc.plot(steps_plot, train_accs, '-', color='#0173B2',
                   linewidth=style['linewidth'], label='Train Acc', alpha=0.9)

    if len(test_accs) > 0:
        steps_plot = log_steps[:len(test_accs)]
        ax_acc.plot(steps_plot, test_accs, '-', color='#029E73',
                   linewidth=style['linewidth'], label='Test Acc', alpha=0.9)

    ax_acc.set_xlabel('Training Epoch', fontsize=style['label_fontsize'])
    ax_acc.set_ylabel('Accuracy', fontsize=style['label_fontsize'], color='#0173B2')
    ax_acc.set_title('Accuracy & Loss', fontsize=style['title_fontsize'])
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.tick_params(axis='y', labelcolor='#0173B2', labelsize=style['tick_fontsize'])
    ax_acc.tick_params(axis='x', labelsize=style['tick_fontsize'])
    ax_acc.grid(True, alpha=0.3, linestyle='--')

    ax_loss = ax_acc.twinx()
    if len(test_losses) > 0:
        steps_plot = log_steps[:len(test_losses)]
        ax_loss.plot(steps_plot, test_losses, '--', color='#CC3311',
                    linewidth=style['linewidth'], label='Test Loss', alpha=0.9)

    ax_loss.set_ylabel('Loss (log scale)', fontsize=style['label_fontsize'], color='#CC3311')
    ax_loss.set_yscale('log')
    ax_loss.tick_params(axis='y', labelcolor='#CC3311', labelsize=style['tick_fontsize'])

    lines1, labels1 = ax_acc.get_legend_handles_labels()
    lines2, labels2 = ax_loss.get_legend_handles_labels()
    ax_acc.legend(lines1 + lines2, labels1 + labels2,
                 loc='center right', fontsize=style['legend_fontsize'], framealpha=0.9)

    # === BOTTOM ROW ===

    # Subplot 4: Overall Flow across layers
    ax_overall = fig.add_subplot(gs[1, 0])
    for layer_idx in range(num_layers):
        if len(overall_by_layer[layer_idx]) > 0:
            steps_plot = gcs_steps[:len(overall_by_layer[layer_idx])]
            scores = np.array(overall_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(overall_by_layer[layer_idx]))
            ax_overall.plot(steps_plot, scores, '-', color=layer_colors[layer_idx],
                           linewidth=style['linewidth'], label=f'Layer {layer_idx + 1}', alpha=0.9)

    ax_overall.set_xlabel('Training Epoch', fontsize=style['label_fontsize'])
    ax_overall.set_ylabel(ylabel, fontsize=style['label_fontsize'])
    ax_overall.set_title('Overall Flow', fontsize=style['title_fontsize'])
    ax_overall.legend(loc='lower left', fontsize=style['legend_fontsize'] + 1, framealpha=0.9)
    ax_overall.grid(True, alpha=0.3, linestyle='--')
    ax_overall.tick_params(labelsize=style['tick_fontsize'])

    if mode == 'raw':
        ax_overall.relim()
        ax_overall.autoscale_view()
        y_min, y_max = ax_overall.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax_overall.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_overall.set_ylim(-0.05, 1.05)

    # Subplot 5: Attention Flow across layers
    ax_attn = fig.add_subplot(gs[1, 1])
    for layer_idx in range(num_layers):
        if len(attn_by_layer[layer_idx]) > 0:
            steps_plot = gcs_steps[:len(attn_by_layer[layer_idx])]
            scores = np.array(attn_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(attn_by_layer[layer_idx]))
            ax_attn.plot(steps_plot, scores, '-', color=layer_colors[layer_idx],
                        linewidth=style['linewidth'], label=f'Layer {layer_idx + 1}', alpha=0.9)

    ax_attn.set_xlabel('Training Epoch', fontsize=style['label_fontsize'])
    ax_attn.set_ylabel(ylabel, fontsize=style['label_fontsize'])
    ax_attn.set_title('Attention Flow', fontsize=style['title_fontsize'])
    ax_attn.legend(loc='upper left', fontsize=style['legend_fontsize'] + 1, framealpha=0.9)
    ax_attn.grid(True, alpha=0.3, linestyle='--')
    ax_attn.tick_params(labelsize=style['tick_fontsize'])

    if mode == 'raw':
        ax_attn.relim()
        ax_attn.autoscale_view()
        y_min, y_max = ax_attn.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax_attn.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_attn.set_ylim(-0.05, 1.05)

    # Subplot 6: MLP Flow across layers
    ax_mlp = fig.add_subplot(gs[1, 2])
    for layer_idx in range(num_layers):
        if len(mlp_by_layer[layer_idx]) > 0:
            steps_plot = gcs_steps[:len(mlp_by_layer[layer_idx])]
            scores = np.array(mlp_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(mlp_by_layer[layer_idx]))
            ax_mlp.plot(steps_plot, scores, '-', color=layer_colors[layer_idx],
                       linewidth=style['linewidth'], label=f'Layer {layer_idx + 1}', alpha=0.9)

    ax_mlp.set_xlabel('Training Epoch', fontsize=style['label_fontsize'])
    ax_mlp.set_ylabel(ylabel, fontsize=style['label_fontsize'])
    ax_mlp.set_title('MLP Flow', fontsize=style['title_fontsize'])
    ax_mlp.legend(loc='upper left', fontsize=style['legend_fontsize'] + 1, framealpha=0.9)
    ax_mlp.grid(True, alpha=0.3, linestyle='--')
    ax_mlp.tick_params(labelsize=style['tick_fontsize'])

    if mode == 'raw':
        ax_mlp.relim()
        ax_mlp.autoscale_view()
        y_min, y_max = ax_mlp.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax_mlp.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_mlp.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_3layer_combined(
    metrics_path: Union[str, Path],
    mode: str = 'normalized',
    save_path: Optional[Union[str, Path]] = None,
    grokking_epoch: Optional[int] = None,
    max_epoch: int = 15000,
) -> plt.Figure:
    """
    Generate 2×3 combined figure for 3-layer Transformer.

    Layout:
    - Top row: Layer 1 (3 flows), Layer 2 (3 flows), Layer 3 (3 flows)
    - Bottom row: Overall Flow (3 layers), Attention Flow (3 layers), MLP Flow (3 layers)

    Matches generate_3layer_gelu_combined_2row.py style.

    Args:
        metrics_path: Path to training_metrics_final_geo.pkl
        mode: 'raw' or 'normalized'
        save_path: Optional path to save figure
        grokking_epoch: Grokking epoch (default 11000)
        max_epoch: Max epoch for truncation (default 15000)

    Returns:
        matplotlib Figure object
    """
    metrics_path = Path(metrics_path)
    metrics = load_metrics(metrics_path)

    if grokking_epoch is None:
        grokking_epoch = GROKKING_EPOCHS.get('3layer_gelu_add', 11000)

    style = UNIFIED_STYLE['paper_2x3']

    # Extract data
    gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', [])))
    path_gcs = metrics.get('path_gcs_scores_final_geo', {})

    num_layers = 3
    overall_by_layer = []
    attn_by_layer = []
    mlp_by_layer = []

    for layer_idx in range(num_layers):
        overall_by_layer.append(path_gcs.get(f'layer{layer_idx}_overall_path', []))
        attn_by_layer.append(path_gcs.get(f'layer{layer_idx}_attention_path', []))
        mlp_by_layer.append(path_gcs.get(f'layer{layer_idx}_mlp_path', []))

    # Color palettes
    path_colors = {
        'overall': '#0173B2',
        'attention': '#DE8F05',
        'mlp': '#029E73',
    }
    layer_colors = ['#0173B2', '#DE8F05', '#CC3311']

    ylabel = 'Normalized GCS' if mode == 'normalized' else 'GCS'

    # Create figure
    fig = plt.figure(figsize=style['figsize'])
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.4, wspace=0.35)

    # === TOP ROW: By-layer view ===
    for layer_idx in range(num_layers):
        ax = fig.add_subplot(gs[0, layer_idx])

        # Overall path
        if len(overall_by_layer[layer_idx]) > 0:
            steps_plot = gcs_steps[:len(overall_by_layer[layer_idx])]
            scores = np.array(overall_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(overall_by_layer[layer_idx]))
            mask = steps_plot <= max_epoch
            ax.plot(steps_plot[mask], scores[mask], '-', color=path_colors['overall'],
                   linewidth=style['linewidth'], label='Overall Flow', alpha=0.9)

        # Attention path
        if len(attn_by_layer[layer_idx]) > 0:
            steps_plot = gcs_steps[:len(attn_by_layer[layer_idx])]
            scores = np.array(attn_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(attn_by_layer[layer_idx]))
            mask = steps_plot <= max_epoch
            ax.plot(steps_plot[mask], scores[mask], '-', color=path_colors['attention'],
                   linewidth=style['linewidth'], label='Attention Flow', alpha=0.9)

        # MLP path
        if len(mlp_by_layer[layer_idx]) > 0:
            steps_plot = gcs_steps[:len(mlp_by_layer[layer_idx])]
            scores = np.array(mlp_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(mlp_by_layer[layer_idx]))
            mask = steps_plot <= max_epoch
            ax.plot(steps_plot[mask], scores[mask], '-', color=path_colors['mlp'],
                   linewidth=style['linewidth'], label='MLP Flow', alpha=0.9)

        # Add grokking marker
        if grokking_epoch <= max_epoch:
            ax.axvline(x=grokking_epoch, color=GROK_COLOR, linestyle='--',
                      linewidth=2, alpha=0.7, label=f'Grokking (epoch {grokking_epoch})')

        ax.set_xlabel('Training Epoch', fontsize=style['label_fontsize'])
        ax.set_ylabel(ylabel, fontsize=style['label_fontsize'])
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=style['title_fontsize'])
        ax.legend(loc='upper left', fontsize=style['legend_fontsize'], framealpha=0.9,
                 handlelength=1.2, labelspacing=0.2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=style['tick_fontsize'])

        if mode == 'raw':
            ax.relim()
            ax.autoscale_view()
            y_min, y_max = ax.get_ylim()
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            ax.set_ylim(-0.05, 1.05)

    # === BOTTOM ROW: By-flow view ===
    flow_data = [
        ('Overall Flow', overall_by_layer),
        ('Attention Flow', attn_by_layer),
        ('MLP Flow', mlp_by_layer),
    ]

    for col_idx, (flow_name, flow_by_layer) in enumerate(flow_data):
        ax = fig.add_subplot(gs[1, col_idx])

        for layer_idx in range(num_layers):
            if len(flow_by_layer[layer_idx]) > 0:
                steps_plot = gcs_steps[:len(flow_by_layer[layer_idx])]
                scores = np.array(flow_by_layer[layer_idx] if mode == 'raw' else normalize_to_01(flow_by_layer[layer_idx]))
                mask = steps_plot <= max_epoch
                ax.plot(steps_plot[mask], scores[mask], '-', color=layer_colors[layer_idx],
                       linewidth=style['linewidth'], label=f'Layer {layer_idx + 1}', alpha=0.9)

        # Add grokking marker
        if grokking_epoch <= max_epoch:
            ax.axvline(x=grokking_epoch, color=GROK_COLOR, linestyle='--',
                      linewidth=2, alpha=0.7, label=f'Grokking (epoch {grokking_epoch})')

        ax.set_xlabel('Training Epoch', fontsize=style['label_fontsize'])
        ax.set_ylabel(ylabel, fontsize=style['label_fontsize'])
        ax.set_title(flow_name, fontsize=style['title_fontsize'])
        ax.legend(loc='upper left', fontsize=style['legend_fontsize'], framealpha=0.9,
                 handlelength=1.2, labelspacing=0.2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=style['tick_fontsize'])

        if mode == 'raw':
            ax.relim()
            ax.autoscale_view()
            y_min, y_max = ax.get_ylim()
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            ax.set_ylim(-0.05, 1.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Batch Generation Functions
# =============================================================================

def generate_all_unified_figures(
    output_dir: Union[str, Path],
) -> Dict[str, List[Path]]:
    """
    Generate all unified figures matching the original scripts.

    This is the main entry point for batch generation. Uses data from
    gcs_core/checkpoints/ directory (no external dependencies).

    Args:
        output_dir: Directory to save generated figures

    Returns:
        Dict mapping figure type to list of generated paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = defaultdict(list)

    print("=" * 80)
    print("Unified Figure Generation")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Checkpoints directory: {CHECKPOINTS_DIR}")
    print()

    # 1. Generate 1-layer Transformer figures
    print("[1/6] Generating 1-layer Transformer figures...")

    for activation in ['ReLU', 'GeLU', 'SiLU']:
        for task in ['add', 'sub', 'mul', 'div']:
            metrics_path = CHECKPOINTS_DIR / '1layer' / activation / task / 'training_metrics_final_geo.pkl'
            if not metrics_path.exists():
                continue

            for mode in ['raw', 'normalized']:
                try:
                    filename = f"1layer_Transformer_{activation}_{task}_{mode}.png"
                    save_path = output_dir / filename
                    fig = plot_1layer_gcs_with_training(metrics_path, mode=mode, save_path=save_path)
                    plt.close(fig)
                    generated['1layer'].append(save_path)
                except Exception as e:
                    print(f"   ✗ Error generating {filename}: {e}")

    # 2. Generate SiLU with markers
    print("[2/6] Generating SiLU with markers figures...")

    silu_path = CHECKPOINTS_DIR / '1layer' / 'SiLU' / 'add' / 'training_metrics_final_geo.pkl'
    if silu_path.exists():
        for mode in ['raw', 'normalized']:
            try:
                filename = f"1layer_Transformer_SiLU_add_{mode}_with_markers.png"
                save_path = output_dir / filename
                fig = plot_1layer_with_markers(silu_path, mode=mode, save_path=save_path)
                plt.close(fig)
                generated['1layer_markers'].append(save_path)
            except Exception as e:
                print(f"   ✗ Error generating {filename}: {e}")

    # 3. Generate mod-add activation comparison (1layer_mod_add_gcs_comparison_*.png)
    print("[3/6] Generating mod-add activation comparison...")

    metrics_dict = {}
    for act in ['ReLU', 'GeLU', 'SiLU']:
        path = CHECKPOINTS_DIR / '1layer' / act / 'add' / 'training_metrics_final_geo.pkl'
        if path.exists():
            metrics_dict[act] = path

    if len(metrics_dict) > 0:
        for mode in ['raw', 'normalized']:
            try:
                filename = f"1layer_mod_add_gcs_comparison_{mode}.png"
                save_path = output_dir / filename
                fig = plot_activation_comparison(metrics_dict, task='add', mode=mode, save_path=save_path)
                plt.close(fig)
                generated['activation_comparison'].append(save_path)
            except Exception as e:
                print(f"   ✗ Error generating {filename}: {e}")

    # 4. Generate multitask comparison per activation (1layer_{act}_multitask_gcs_*.png)
    print("[4/6] Generating multitask comparison figures...")

    for activation in ['ReLU', 'GeLU', 'SiLU']:
        for mode in ['raw', 'normalized']:
            try:
                filename = f"1layer_{activation}_multitask_gcs_{mode}.png"
                save_path = output_dir / filename
                fig = plot_multitask_comparison(
                    activation, mode=mode, save_path=save_path
                )
                plt.close(fig)
                generated['multitask'].append(save_path)
            except Exception as e:
                print(f"   ✗ Error generating {filename}: {e}")

    # 5. Generate 2-layer combined figures
    print("[5/6] Generating 2-layer combined figures...")

    two_layer_path = CHECKPOINTS_DIR / '2layer' / 'GeLU' / 'add' / 'training_metrics_final_geo.pkl'
    if two_layer_path.exists():
        for mode in ['raw', 'normalized']:
            try:
                filename = f"2layer_Transformer_GeLU_mod_add_stable_combined_{mode}.png"
                save_path = output_dir / filename
                fig = plot_2layer_combined(
                    two_layer_path, mode=mode, save_path=save_path
                )
                plt.close(fig)
                generated['2layer'].append(save_path)
            except Exception as e:
                print(f"   ✗ Error generating {filename}: {e}")

    # 6. Generate 3-layer combined figures
    print("[6/6] Generating 3-layer combined figures...")

    three_layer_path = CHECKPOINTS_DIR / '3layer' / 'GeLU' / 'add' / 'training_metrics_final_geo.pkl'
    if three_layer_path.exists():
        for mode in ['raw', 'normalized']:
            try:
                filename = f"3layer_Transformer_GeLU_add_combined_{mode}.png"
                save_path = output_dir / filename
                fig = plot_3layer_combined(three_layer_path, mode=mode, save_path=save_path)
                plt.close(fig)
                generated['3layer'].append(save_path)
            except Exception as e:
                print(f"   ✗ Error generating {filename}: {e}")

    print()
    print("=" * 80)
    print("✅ Generation Complete!")
    print("=" * 80)

    total = sum(len(v) for v in generated.values())
    print(f"Total figures generated: {total}")
    for category, paths in generated.items():
        print(f"  {category}: {len(paths)} figures")

    return dict(generated)


# =============================================================================
# Module-level exports
# =============================================================================

def generate_mlp_figures(
    output_dir: Union[str, Path],
) -> Dict[str, List[Path]]:
    """
    Generate figures for MLP experiments.

    Looks for experiments in checkpoints/mlp_*/ directories.

    Args:
        output_dir: Directory to save generated figures

    Returns:
        Dict mapping figure type to list of generated paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = defaultdict(list)

    print("=" * 80)
    print("MLP Figure Generation")
    print("=" * 80)
    print(f"Output directory: {output_dir}")

    # Find MLP experiment directories
    mlp_base = CHECKPOINTS_DIR
    mlp_dirs = sorted(mlp_base.glob('mlp_*'))

    if not mlp_dirs:
        print("No MLP experiment directories found")
        return dict(generated)

    for mlp_dir in mlp_dirs:
        embed_type = mlp_dir.name.replace('mlp_', '')
        print(f"\nProcessing {mlp_dir.name}...")

        # Find experiment subdirectories
        exp_dirs = sorted(d for d in mlp_dir.iterdir()
                          if d.is_dir() and (d / 'training_metrics_final_geo.pkl').exists())

        for exp_dir in exp_dirs:
            metrics_path = exp_dir / 'training_metrics_final_geo.pkl'
            info = parse_experiment_info(metrics_path.parent)

            try:
                metrics = load_metrics(metrics_path)

                # Training curves
                from .plots import plot_training_curves
                filename = f"mlp_{embed_type}_{info.get('activation', 'unknown')}_{info.get('layers', '?')}layer_training.png"
                save_path = output_dir / filename
                fig = plot_training_curves(metrics, title=exp_dir.name, save_path=str(save_path))
                plt.close(fig)
                generated['mlp_training'].append(save_path)

            except Exception as e:
                print(f"   Error: {e}")

    total = sum(len(v) for v in generated.values())
    print(f"\nTotal MLP figures generated: {total}")

    return dict(generated)


__all__ = [
    # Style constants
    # Path constants
    'GCS_ROOT',
    'CHECKPOINTS_DIR',
    # Style constants
    'UNIFIED_STYLE',
    'GROKKING_EPOCHS',
    'TRAIN_ACC_1_EPOCHS',
    'GROK_COLOR',
    'GROK_MARKERS',
    'MULTITASK_EPOCH_LIMITS',
    # Helper functions
    'normalize_to_01',
    'parse_experiment_info',
    'load_metrics',
    # 1-layer plots
    'plot_1layer_gcs_with_training',
    'plot_1layer_with_markers',
    'plot_activation_comparison',
    'plot_multitask_comparison',
    # Multi-layer plots
    'plot_2layer_combined',
    'plot_3layer_combined',
    # Batch generation
    'generate_all_unified_figures',
    'generate_mlp_figures',
]
