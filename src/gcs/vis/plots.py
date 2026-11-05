"""
Visualization utilities for GCS analysis.

Provides plotting functions for:
- Training curves (loss/accuracy)
- GCS evolution (3-path decomposition)
- Ablation studies (N, d parameters)
- Cross-activation comparisons
- Multi-task GCS comparisons
- Attention pattern heatmaps
- Multi-layer GCS visualization
- PR vs GCS comparisons
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .config import (
    setup_style, get_path_color, get_activation_color, get_layer_color,
    get_ablation_color, get_model_color, get_figsize, normalize_scores,
    add_grokking_marker, style_axis, save_figure, PLOT_CONFIG, PATH_COLORS,
    ACTIVATION_COLORS, MODEL_COLORS, PATH_LABELS, LABELS
)


# =============================================================================
# Basic Training Curves
# =============================================================================

def plot_training_curves(metrics, title='Training Curves', save_path=None):
    """
    Plot training curves (loss and accuracy).

    Args:
        metrics: dict with 'log_steps', 'train_losses', 'test_losses',
                 'train_accuracies', 'test_accuracies'
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    setup_style()

    if hasattr(metrics, 'to_dict'):
        metrics = metrics.to_dict()

    fig, axes = plt.subplots(1, 2, figsize=get_figsize('1x2'))

    epochs = metrics.get('log_steps', [])

    # Loss plot
    ax = axes[0]
    ax.plot(epochs, metrics.get('train_losses', []),
            label='Train Loss', color=PLOT_CONFIG.get('train_color', '#0173B2'),
            linewidth=PLOT_CONFIG['linewidth'])
    ax.plot(epochs, metrics.get('test_losses', []),
            label='Test Loss', color=PLOT_CONFIG.get('test_color', '#CC3311'),
            linewidth=PLOT_CONFIG['linewidth'])
    ax.set_yscale('log')
    style_axis(ax, xlabel=LABELS['epoch'], ylabel=LABELS['loss'], title='Loss')

    # Accuracy plot
    ax = axes[1]
    ax.plot(epochs, metrics.get('train_accuracies', []),
            label='Train Acc', color=PLOT_CONFIG.get('train_color', '#0173B2'),
            linewidth=PLOT_CONFIG['linewidth'])
    ax.plot(epochs, metrics.get('test_accuracies', []),
            label='Test Acc', color=PLOT_CONFIG.get('test_color', '#029E73'),
            linewidth=PLOT_CONFIG['linewidth'])
    ax.set_ylim([0, 1.05])
    style_axis(ax, xlabel=LABELS['epoch'], ylabel=LABELS['accuracy'], title='Accuracy')

    plt.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# GCS Evolution Plots
# =============================================================================

def plot_gcs_3path(
    gcs_data: Dict,
    title: str = 'GCS 3-Path Evolution',
    normalized: bool = False,
    grokking_epoch: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Plot 3-path GCS evolution (Overall, Attention, MLP).

    Args:
        gcs_data: dict with 'gcs_steps' and path scores (overall_path, attention_path, mlp_path)
        title: Plot title
        normalized: Whether to normalize scores to [0,1]
        grokking_epoch: Epoch to mark as grokking point
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=get_figsize('single'))

    epochs = gcs_data.get('gcs_steps', gcs_data.get('epochs', []))

    # Handle nested structure
    if 'path_gcs_scores' in gcs_data:
        scores_dict = gcs_data['path_gcs_scores']
    else:
        scores_dict = gcs_data

    path_keys = [
        ('overall_path', 'overall'),
        ('attention_path', 'attention'),
        ('mlp_path', 'mlp')
    ]

    for key, path_name in path_keys:
        scores = scores_dict.get(key, [])
        if not scores:
            continue

        # Filter valid values
        valid_pairs = [(e, s) for e, s in zip(epochs, scores) if s is not None]
        if not valid_pairs:
            continue

        e, s = zip(*valid_pairs)
        if normalized:
            s = normalize_scores(s)

        ax.plot(e, s,
                label=PATH_LABELS[path_name],
                color=PATH_COLORS[path_name],
                linewidth=PLOT_CONFIG['linewidth'])

    if grokking_epoch:
        add_grokking_marker(ax, grokking_epoch, label=f'Grokking (epoch {grokking_epoch})')

    ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']
    style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel, title=title)

    if normalized:
        ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_gcs_with_accuracy(
    metrics: Dict,
    gcs_data: Dict,
    title: str = 'GCS vs Accuracy',
    normalized: bool = True,
    grokking_epoch: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Plot 3-path GCS with accuracy in 2-row layout.

    Like figures_unified/1layer_Transformer_SiLU_add_normalized_with_markers.png

    Args:
        metrics: Training metrics with 'log_steps', 'train_accuracies', 'test_accuracies'
        gcs_data: GCS data
        title: Plot title
        normalized: Whether to normalize GCS scores
        grokking_epoch: Epoch to mark as grokking
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: GCS 3-path
    ax = axes[0]
    epochs_gcs = gcs_data.get('gcs_steps', gcs_data.get('epochs', []))

    if 'path_gcs_scores' in gcs_data:
        scores_dict = gcs_data['path_gcs_scores']
    else:
        scores_dict = gcs_data

    for key, path_name in [('overall_path', 'overall'), ('attention_path', 'attention'), ('mlp_path', 'mlp')]:
        scores = scores_dict.get(key, [])
        if not scores:
            continue

        valid_pairs = [(e, s) for e, s in zip(epochs_gcs, scores) if s is not None]
        if not valid_pairs:
            continue

        e, s = zip(*valid_pairs)
        if normalized:
            s = normalize_scores(s)

        ax.plot(e, s, label=PATH_LABELS[path_name], color=PATH_COLORS[path_name],
                linewidth=PLOT_CONFIG['linewidth'])

    if grokking_epoch:
        add_grokking_marker(ax, grokking_epoch, label=f'Grokking (epoch {grokking_epoch})')

    ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']
    style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel)
    if normalized:
        ax.set_ylim([-0.05, 1.05])

    # Bottom: Accuracy
    ax = axes[1]
    epochs_train = metrics.get('log_steps', [])
    ax.plot(epochs_train, metrics.get('train_accuracies', []),
            label='Train Acc', color='#0173B2', linewidth=PLOT_CONFIG['linewidth'])
    ax.plot(epochs_train, metrics.get('test_accuracies', []),
            label='Test Acc', color='#029E73', linewidth=PLOT_CONFIG['linewidth'])

    if grokking_epoch:
        add_grokking_marker(ax, grokking_epoch, label=f'Grokking (epoch {grokking_epoch})')

    ax.set_ylim([0, 1.05])
    style_axis(ax, xlabel=LABELS['epoch'], ylabel=LABELS['accuracy'])

    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Ablation Study Plots
# =============================================================================

def plot_ablation_3path(
    results: Dict,
    param_name: str = 'N',
    normalized: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot 3-path GCS comparison for ablation study (1x3 layout).

    Like n_ablation_3path_flow.png or d_ablation_3path_flow.png

    Args:
        results: dict mapping parameter values to GCS data
                 e.g., {100: {'epochs': [...], 'overall_gcs': [...], ...}, ...}
        param_name: Name of ablated parameter ('N', 'd', etc.)
        normalized: Whether to normalize scores to [0,1]
        title: Optional title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=get_figsize('1x3'))

    path_keys = ['overall_gcs', 'attention_gcs', 'mlp_gcs']
    path_labels = ['Overall Path', 'Attention Path', 'MLP Path']

    sorted_params = sorted(results.keys())

    for ax, path_key, path_label in zip(axes, path_keys, path_labels):
        for idx, param_val in enumerate(sorted_params):
            data = results[param_val]
            epochs = data.get('epochs', [])
            scores = data.get(path_key, [])

            if not epochs or not scores:
                continue

            # Filter None values
            valid_pairs = [(e, s) for e, s in zip(epochs, scores) if s is not None]
            if not valid_pairs:
                continue

            e, s = zip(*valid_pairs)
            if normalized:
                s = normalize_scores(s)

            color = get_ablation_color(idx)
            ax.plot(e, s, label=f'{param_name}={param_val}',
                    color=color, linewidth=PLOT_CONFIG['linewidth'], alpha=0.85)

        ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']
        style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel, title=path_label,
                   legend_loc='lower right')

        if normalized:
            ax.set_ylim([0, 1.05])

    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Cross-Activation Comparison Plots
# =============================================================================

def plot_activation_comparison_2x2(
    all_metrics: Dict[str, Dict],
    activations: List[str] = None,
    normalized: bool = True,
    grokking_epochs: Optional[Dict[str, int]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot 2x2 comparison across activation functions.

    Layout:
    - Top-left: Test Accuracy
    - Top-right: Overall Flow GCS
    - Bottom-left: Attention Flow GCS
    - Bottom-right: MLP Flow GCS

    Like 1layer_mod_add_gcs_comparison.png

    Args:
        all_metrics: dict mapping activation names to metrics dicts
        activations: list of activation names to include (default: all)
        normalized: Whether to normalize GCS scores
        grokking_epochs: dict mapping activation names to grokking epochs
        title: Optional title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.9), gridspec_kw={'hspace': 0.34, 'wspace': 0.20})

    axis_label_fontsize = 14
    axis_tick_fontsize = 12
    axis_legend_fontsize = 11
    axis_title_fontsize = 13

    def apply_axis_typography(ax):
        """Apply larger, paper-friendly typography for this 2x2 figure."""
        ax.xaxis.label.set_size(axis_label_fontsize)
        ax.yaxis.label.set_size(axis_label_fontsize)
        ax.title.set_fontsize(axis_title_fontsize)
        ax.tick_params(labelsize=axis_tick_fontsize)
        ax.tick_params(axis='x', pad=6)
        ax.tick_params(axis='y', pad=6)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(axis_legend_fontsize)

    def add_left_epoch_padding(ax, pad_ratio=0.035):
        """Create a small visual gap between y-axis and first epoch tick."""
        left, right = ax.get_xlim()
        span = right - left
        if span > 0:
            ax.set_xlim(left=left - pad_ratio * span, right=right)

    def reduce_corner_overlap(ax):
        """Avoid lower-left x/y label collisions and keep positive epoch ticks."""
        xticks = ax.get_xticks()
        filtered = [tick for tick in xticks if tick > 0 and not np.isclose(tick, 0.0)]
        if filtered:
            ax.set_xticks(filtered)

    if activations is None:
        activations = list(all_metrics.keys())

    grokking_epochs = grokking_epochs or {}

    def infer_fastest_rise_epoch(metrics_dict):
        """Infer grokking epoch as the steepest positive rise in test accuracy."""
        steps = metrics_dict.get('log_steps', [])
        test_acc = metrics_dict.get('test_accuracies', [])
        if len(steps) < 2 or len(test_acc) < 2:
            return None

        best_epoch = None
        best_slope = float('-inf')
        for idx in range(1, min(len(steps), len(test_acc))):
            step_delta = steps[idx] - steps[idx - 1]
            if step_delta <= 0:
                continue
            acc_delta = test_acc[idx] - test_acc[idx - 1]
            slope = acc_delta / step_delta
            if slope > best_slope:
                best_slope = slope
                best_epoch = steps[idx]

        return best_epoch

    for act in activations:
        if act in all_metrics and act not in grokking_epochs:
            inferred = infer_fastest_rise_epoch(all_metrics[act])
            if inferred is not None:
                grokking_epochs[act] = inferred

    # Define grokking markers
    grok_markers = {'ReLU': 'o', 'GeLU': 's', 'SiLU': '^', 'SeLU': 'd'}

    def add_grok_markers(ax):
        """Add grokking point markers."""
        for act in activations:
            if act in grokking_epochs:
                epoch = grokking_epochs[act]
                color = get_activation_color(act)
                ax.axvline(x=epoch, color=color, linestyle='--',
                          linewidth=1.5, alpha=0.7)
                ylim = ax.get_ylim()
                y_pos = ylim[0] + 0.92 * (ylim[1] - ylim[0])
                ax.plot(epoch, y_pos, marker=grok_markers.get(act, 'o'),
                        color=color, markersize=8, zorder=10)

    # Top-left: Test Accuracy
    ax = axes[0, 0]
    max_test_epoch = None
    for act in activations:
        if act not in all_metrics:
            continue
        m = all_metrics[act]
        steps = m.get('log_steps', [])
        test_acc = m.get('test_accuracies', [])
        if steps and test_acc:
            ax.plot(steps, test_acc, label=act, color=get_activation_color(act),
                    linewidth=PLOT_CONFIG['linewidth'], alpha=0.9)
            series_max = max(steps)
            if max_test_epoch is None or series_max > max_test_epoch:
                max_test_epoch = series_max

    ax.set_ylim([-0.05, 1.05])
    if max_test_epoch is not None:
        ax.set_xlim(left=0, right=max_test_epoch)
    else:
        ax.set_xlim(left=0, right=1)
    add_left_epoch_padding(ax)
    style_axis(ax, xlabel=None, ylabel=LABELS['test_accuracy'],
               title='Test Accuracy', legend=False)
    add_grok_markers(ax)
    apply_axis_typography(ax)
    ax.tick_params(labelbottom=False)
    reduce_corner_overlap(ax)

    grok_handles = []
    grok_labels = []
    for act in activations:
        if act in grokking_epochs:
            grok_handles.append(Line2D(
                [0], [0],
                color=get_activation_color(act),
                linestyle='--',
                linewidth=1.5,
                marker=grok_markers.get(act, 'o'),
                markersize=7,
            ))
            grok_labels.append(f'{act} grok')
    if grok_handles:
        ax.legend(
            grok_handles,
            grok_labels,
            loc='lower right',
            framealpha=0.9,
            fontsize=axis_legend_fontsize,
        )

    # GCS plots
    path_configs = [
        (axes[0, 1], 'overall_path', 'Overall Flow GCS', False),
        (axes[1, 0], 'attention_path', 'Attention Flow GCS', True),
        (axes[1, 1], 'mlp_path', 'MLP Flow GCS', True),
    ]

    for ax, path_key, ax_title, show_xlabel in path_configs:
        max_path_epoch = None
        for act in activations:
            if act not in all_metrics:
                continue
            m = all_metrics[act]

            # Try final_geo first, then regular
            gcs_steps = m.get('gcs_steps_final_geo', m.get('gcs_steps', []))
            path_scores = m.get('path_gcs_scores_final_geo', m.get('path_gcs_scores', {}))
            scores = path_scores.get(path_key, [])

            if not gcs_steps or not scores:
                continue

            valid_pairs = [(e, s) for e, s in zip(gcs_steps, scores) if s is not None]
            if not valid_pairs:
                continue

            e, s = zip(*valid_pairs)
            if normalized:
                s = normalize_scores(s)

            ax.plot(e, s, label=act, color=get_activation_color(act),
                    linewidth=PLOT_CONFIG['linewidth'], alpha=0.9)
            series_max = max(e)
            if max_path_epoch is None or series_max > max_path_epoch:
                max_path_epoch = series_max

        ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']
        if normalized:
            ax_title = ax_title.replace('GCS', 'GCS (Normalized)')

        if max_path_epoch is not None:
            ax.set_xlim(left=0, right=max_path_epoch)
        else:
            ax.set_xlim(left=0, right=1)
        add_left_epoch_padding(ax)
        if normalized:
            ax.set_ylim([-0.05, 1.05])
        style_axis(ax, xlabel=LABELS['epoch'] if show_xlabel else None, ylabel=ylabel,
                   title=ax_title, legend=False)
        add_grok_markers(ax)
        apply_axis_typography(ax)
        if not show_xlabel:
            ax.tick_params(labelbottom=False)
        reduce_corner_overlap(ax)

    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')

    activation_handles = []
    activation_labels = []
    for act in activations:
        if act in all_metrics:
            activation_handles.append(Line2D(
                [0], [0],
                color=get_activation_color(act),
                linestyle='-',
                linewidth=PLOT_CONFIG['linewidth'],
            ))
            activation_labels.append(act)

    if activation_handles:
        fig.legend(
            activation_handles,
            activation_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.015),
            ncol=min(3, len(activation_handles)),
            framealpha=0.9,
            fontsize=axis_legend_fontsize,
        )

    fig.subplots_adjust(top=0.93, bottom=0.145, left=0.08, right=0.98)

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Multi-Task GCS Plots
# =============================================================================

def plot_multitask_gcs_2x2(
    task_metrics: Dict[str, Dict],
    tasks: List[str] = None,
    normalized: bool = False,
    grokking_epochs: Optional[Dict[str, int]] = None,
    x_limits_by_task: Optional[Dict[str, int]] = None,
    save_path: Optional[str] = None
):
    """
    Plot GCS for multiple tasks in 2x2 layout.

    Like 1layer_ReLU_multitask_gcs_raw.png

    Args:
        task_metrics: dict mapping task names to metrics
        tasks: list of task names to plot (default: all)
        normalized: Whether to normalize GCS scores
        grokking_epochs: dict mapping task names to grokking epochs
        x_limits_by_task: optional dict mapping task name -> max x epoch
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.9), gridspec_kw={'hspace': 0.34, 'wspace': 0.20})
    axes = axes.flatten()

    axis_label_fontsize = 14
    axis_tick_fontsize = 12
    axis_legend_fontsize = 11
    axis_title_fontsize = 13

    def apply_axis_typography(ax):
        ax.xaxis.label.set_size(axis_label_fontsize)
        ax.yaxis.label.set_size(axis_label_fontsize)
        ax.title.set_fontsize(axis_title_fontsize)
        ax.tick_params(labelsize=axis_tick_fontsize)
        ax.tick_params(axis='x', pad=6)
        ax.tick_params(axis='y', pad=6)

    def add_left_epoch_padding(ax, pad_ratio=0.035):
        """Create a small visual gap between y-axis and first epoch tick."""
        left, right = ax.get_xlim()
        span = right - left
        if span > 0:
            ax.set_xlim(left=left - pad_ratio * span, right=right)

    def reduce_corner_overlap(ax):
        """Avoid lower-left x/y label collisions and keep positive epoch ticks."""
        xticks = ax.get_xticks()
        filtered = [tick for tick in xticks if tick > 0 and not np.isclose(tick, 0.0)]
        if filtered:
            ax.set_xticks(filtered)

    def infer_fastest_rise_epoch(metrics_dict):
        """Infer grokking epoch as the steepest positive rise in test accuracy."""
        steps = metrics_dict.get('log_steps', [])
        test_acc = metrics_dict.get('test_accuracies', [])
        if len(steps) < 2 or len(test_acc) < 2:
            return None

        best_epoch = None
        best_slope = float('-inf')
        for idx in range(1, min(len(steps), len(test_acc))):
            step_delta = steps[idx] - steps[idx - 1]
            if step_delta <= 0:
                continue
            acc_delta = test_acc[idx] - test_acc[idx - 1]
            slope = acc_delta / step_delta
            if slope > best_slope:
                best_slope = slope
                best_epoch = steps[idx]

        return best_epoch

    if tasks is None:
        tasks = list(task_metrics.keys())[:4]

    grokking_epochs = grokking_epochs or {}
    x_limits_by_task = x_limits_by_task or {}

    for task in tasks[:4]:
        if task in task_metrics and task not in grokking_epochs:
            inferred = infer_fastest_rise_epoch(task_metrics[task])
            if inferred is not None:
                grokking_epochs[task] = inferred

    for idx, task in enumerate(tasks[:4]):
        ax = axes[idx]
        m = task_metrics.get(task, {})

        gcs_steps = m.get('gcs_steps_final_geo', m.get('gcs_steps', []))
        path_scores = m.get('path_gcs_scores_final_geo', m.get('path_gcs_scores', {}))

        for path_key, path_name in [('overall_path', 'overall'), ('attention_path', 'attention'), ('mlp_path', 'mlp')]:
            scores = path_scores.get(path_key, [])
            if not scores:
                continue

            valid_pairs = [(e, s) for e, s in zip(gcs_steps, scores) if s is not None]
            if not valid_pairs:
                continue

            e, s = zip(*valid_pairs)
            if normalized:
                s = normalize_scores(s)

            ax.plot(e, s, label=PATH_LABELS[path_name], color=PATH_COLORS[path_name],
                    linewidth=PLOT_CONFIG['linewidth'])

        if task in grokking_epochs:
            add_grokking_marker(ax, grokking_epochs[task])

        # Format task name for title
        task_title = task.replace('mod-', 'Modular ').replace('_', ' ').title()
        ylabel = LABELS['gcs_normalized'] if normalized else 'GCS Score'
        # Keep top-row tick labels for per-task epoch ranges, but only show
        # x-axis title text on the bottom row to reduce visual clutter.
        show_xlabel = idx >= 2
        style_axis(
            ax,
            xlabel=LABELS['epoch'] if show_xlabel else None,
            ylabel=ylabel,
            title=task_title,
            legend=False,
        )

        ax.set_xlim(left=0)
        if task in x_limits_by_task:
            ax.set_xlim(right=x_limits_by_task[task])
        add_left_epoch_padding(ax)

        if task in grokking_epochs:
            ylim = ax.get_ylim()
            y_pos = ylim[0] + 0.92 * (ylim[1] - ylim[0])
            ax.plot(grokking_epochs[task], y_pos, marker='o', color='#E74C3C', markersize=8, zorder=10)

        apply_axis_typography(ax)
        reduce_corner_overlap(ax)

    grok_handle = Line2D([0], [0], color='#E74C3C', linestyle='--', linewidth=1.5,
                         marker='o', markersize=7)
    axes[0].legend([grok_handle], ['Grokking marker'], loc='lower right',
                   framealpha=0.9, fontsize=axis_legend_fontsize)

    path_handles = [
        Line2D([0], [0], color=PATH_COLORS['overall'], linestyle='-', linewidth=PLOT_CONFIG['linewidth']),
        Line2D([0], [0], color=PATH_COLORS['attention'], linestyle='-', linewidth=PLOT_CONFIG['linewidth']),
        Line2D([0], [0], color=PATH_COLORS['mlp'], linestyle='-', linewidth=PLOT_CONFIG['linewidth']),
    ]
    path_labels = [PATH_LABELS['overall'], PATH_LABELS['attention'], PATH_LABELS['mlp']]
    fig.legend(path_handles, path_labels, loc='lower center', bbox_to_anchor=(0.5, 0.002),
               ncol=3, framealpha=0.9, fontsize=axis_legend_fontsize)

    fig.subplots_adjust(top=0.93, bottom=0.145, left=0.08, right=0.98)

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# PR vs GCS Comparison Plots
# =============================================================================

def plot_pr_vs_gcs_2x2(
    pr_data: Dict[str, Dict],
    gcs_data: Dict[str, Dict],
    activations: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot PR vs GCS path decomposition comparison in 2x2 layout.

    Like combined_comparison_2x2.png

    Args:
        pr_data: dict mapping activation names to PR DataFrames/dicts
                 with 'epoch', 'PR_attention', 'PR_mlp', 'PR_overall'
        gcs_data: dict mapping activation names to GCS DataFrames/dicts
                 with 'epoch', 'GCS_attention', 'GCS_mlp', 'GCS_overall'
        activations: list of activation names (default: both keys)
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(6, 5))

    if activations is None:
        activations = list(pr_data.keys())[:2]

    path_colors = {'Attention': '#0173B2', 'MLP': '#029E73', 'Overall': '#DE8F05'}

    for row_idx, act in enumerate(activations[:2]):
        pr = pr_data.get(act, {})
        gcs = gcs_data.get(act, {})

        # Convert DataFrame to dict if needed
        if hasattr(pr, 'to_dict'):
            pr = pr.to_dict('list')
        if hasattr(gcs, 'to_dict'):
            gcs = gcs.to_dict('list')

        # PR plot
        ax = axes[row_idx, 0]
        epochs_pr = pr.get('epoch', [])

        for path_name, pr_key in [('Attention', 'PR_attention'), ('MLP', 'PR_mlp'), ('Overall', 'PR_overall')]:
            scores = pr.get(pr_key, [])
            if epochs_pr and scores:
                ax.plot(epochs_pr, scores, color=path_colors[path_name],
                        linewidth=2, label=path_name, alpha=0.9)

        ylabel = LABELS['pr'] if row_idx == 1 else LABELS['pr']
        xlabel = LABELS['epoch'] if row_idx == 1 else None
        style_axis(ax, xlabel=xlabel, ylabel=ylabel,
                   title=f'{act}: PR Path Decomposition')

        # GCS plot
        ax = axes[row_idx, 1]
        epochs_gcs = gcs.get('epoch', [])

        for path_name, gcs_key in [('Attention', 'GCS_attention'), ('MLP', 'GCS_mlp'), ('Overall', 'GCS_overall')]:
            scores = gcs.get(gcs_key, [])
            if epochs_gcs and scores:
                ax.plot(epochs_gcs, scores, color=path_colors[path_name],
                        linewidth=2, label=path_name, alpha=0.9)

        ylabel = 'GCS Score'
        xlabel = LABELS['epoch'] if row_idx == 1 else None
        style_axis(ax, xlabel=xlabel, ylabel=ylabel,
                   title=f'{act}: GCS Path Decomposition')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Attention Pattern Heatmaps
# =============================================================================

def plot_attention_evolution(
    attention_data: Dict,
    phases: List[str] = None,
    num_heads: int = 4,
    gcs_scores: Optional[List[float]] = None,
    save_path: Optional[str] = None
):
    """
    Plot attention pattern evolution across training phases.

    Like relu_add_5phases.png (4 heads x 5 phases)

    Args:
        attention_data: dict mapping phase names to attention tensors
                       shape: [num_heads, seq_len, seq_len]
        phases: list of phase names (default: keys of attention_data)
        num_heads: number of attention heads to plot
        gcs_scores: optional GCS scores for each phase
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()

    if phases is None:
        phases = list(attention_data.keys())

    num_phases = len(phases)
    fig, axes = plt.subplots(num_heads, num_phases, figsize=(3 * num_phases, 3 * num_heads))

    if num_heads == 1:
        axes = axes.reshape(1, -1)
    if num_phases == 1:
        axes = axes.reshape(-1, 1)

    for col_idx, phase in enumerate(phases):
        attn = attention_data[phase]
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().numpy()

        # Handle different input shapes
        if attn.ndim == 4:  # [batch, heads, seq, seq]
            attn = attn[0]  # Take first batch

        for head_idx in range(min(num_heads, attn.shape[0])):
            ax = axes[head_idx, col_idx]
            pattern = attn[head_idx]

            im = ax.imshow(pattern, cmap='viridis', aspect='auto')

            # Column title with GCS if available
            if head_idx == 0:
                if gcs_scores and col_idx < len(gcs_scores):
                    title = f'{phase}\nGCS = {gcs_scores[col_idx]:.1f}'
                else:
                    title = phase
                ax.set_title(title, fontsize=PLOT_CONFIG['tick_fontsize'])

            # Row label
            if col_idx == 0:
                ax.set_ylabel(f'Head {head_idx}', fontsize=PLOT_CONFIG['tick_fontsize'])

            ax.set_xlabel('Operand a', fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f'Head {head_idx}\nOperand b', fontsize=8)

            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_attention_patterns(model, input_data, save_path=None):
    """
    Visualize attention patterns from a model.

    Args:
        model: Transformer model
        input_data: Input tensor [batch, seq_len]
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()

    cache = {}
    model.cache_all(cache)
    with torch.no_grad():
        model(input_data)
    model.remove_all_hooks()

    # Get attention patterns from first layer
    attn_key = 'blocks.0.attn.hook_attn'
    if attn_key not in cache:
        print("No attention patterns found in cache")
        return None

    attn = cache[attn_key]  # [batch, heads, seq, seq]
    num_heads = attn.shape[1]

    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    for head_idx, ax in enumerate(axes):
        attn_pattern = attn[0, head_idx].cpu().numpy()
        im = ax.imshow(attn_pattern, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Attention Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Multi-Layer GCS Plots
# =============================================================================

def plot_multilayer_bylayer(
    metrics: Dict,
    num_layers: int,
    normalized: bool = False,
    grokking_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Plot multi-layer GCS by layer (top row: each layer's 3 paths).

    Like top row of 3layer_Transformer_GeLU_add_combined_2row_raw.png

    Args:
        metrics: dict with path_gcs_scores containing layer-specific keys
                 e.g., 'layer0_overall_path', 'layer1_attention_path', etc.
        num_layers: number of layers
        normalized: Whether to normalize scores
        grokking_epoch: Epoch to mark as grokking
        max_epoch: Maximum epoch to display
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))

    if num_layers == 1:
        axes = [axes]

    gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', [])))
    path_scores = metrics.get('path_gcs_scores_final_geo', metrics.get('path_gcs_scores', {}))

    for layer_idx, ax in enumerate(axes):
        for path_name in ['overall', 'attention', 'mlp']:
            key = f'layer{layer_idx}_{path_name}_path'
            scores = path_scores.get(key, [])

            if not scores:
                continue

            steps_plot = gcs_steps[:len(scores)]
            scores_arr = np.array(scores)

            if max_epoch:
                mask = steps_plot <= max_epoch
                steps_plot = steps_plot[mask]
                scores_arr = scores_arr[mask]

            if normalized:
                scores_arr = normalize_scores(scores_arr)

            ax.plot(steps_plot, scores_arr,
                    label=PATH_LABELS[path_name],
                    color=PATH_COLORS[path_name],
                    linewidth=PLOT_CONFIG['linewidth'], alpha=0.9)

        if grokking_epoch and (not max_epoch or grokking_epoch <= max_epoch):
            add_grokking_marker(ax, grokking_epoch, label=f'Grokking (epoch {grokking_epoch})')

        ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']
        style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel,
                   title=f'Layer {layer_idx + 1}', legend_loc='upper left')

        if normalized:
            ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_multilayer_bypath(
    metrics: Dict,
    num_layers: int,
    normalized: bool = False,
    grokking_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Plot multi-layer GCS by path (each subplot shows all layers for one path).

    Like bottom row of 3layer_Transformer_GeLU_add_combined_2row_raw.png

    Args:
        metrics: dict with layer-specific path scores
        num_layers: number of layers
        normalized: Whether to normalize scores
        grokking_epoch: Epoch to mark as grokking
        max_epoch: Maximum epoch to display
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=get_figsize('1x3'))

    gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', [])))
    path_scores = metrics.get('path_gcs_scores_final_geo', metrics.get('path_gcs_scores', {}))

    path_names = ['overall', 'attention', 'mlp']

    for ax, path_name in zip(axes, path_names):
        for layer_idx in range(num_layers):
            key = f'layer{layer_idx}_{path_name}_path'
            scores = path_scores.get(key, [])

            if not scores:
                continue

            steps_plot = gcs_steps[:len(scores)]
            scores_arr = np.array(scores)

            if max_epoch:
                mask = steps_plot <= max_epoch
                steps_plot = steps_plot[mask]
                scores_arr = scores_arr[mask]

            if normalized:
                scores_arr = normalize_scores(scores_arr)

            ax.plot(steps_plot, scores_arr,
                    label=f'Layer {layer_idx + 1}',
                    color=get_layer_color(layer_idx),
                    linewidth=PLOT_CONFIG['linewidth'], alpha=0.9)

        if grokking_epoch and (not max_epoch or grokking_epoch <= max_epoch):
            add_grokking_marker(ax, grokking_epoch, label=f'Grokking (epoch {grokking_epoch})')

        ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']
        style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel,
                   title=PATH_LABELS[path_name], legend_loc='upper left')

        if normalized:
            ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_multilayer_combined_2row(
    metrics: Dict,
    num_layers: int,
    normalized: bool = False,
    grokking_epoch: Optional[int] = None,
    max_epoch: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    stack_legends_bottom: bool = False
):
    """
    Plot multi-layer GCS in combined 2-row layout.

    Top row: By-layer view (each subplot shows 3 paths for one layer)
    Bottom row: By-path view (each subplot shows all layers for one path)

    Like 3layer_Transformer_GeLU_add_combined_2row_raw.png

    Args:
        metrics: dict with layer-specific path scores
        num_layers: number of layers
        normalized: Whether to normalize scores
        grokking_epoch: Epoch to mark as grokking
        max_epoch: Maximum epoch to display
        title: Optional figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    num_cols = max(num_layers, 3)
    fig = plt.figure(figsize=(5 * num_cols, 7.2))
    gs = fig.add_gridspec(2, num_cols, height_ratios=[1, 1], hspace=0.62, wspace=0.35)

    gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', [])))
    path_scores = metrics.get('path_gcs_scores_final_geo', metrics.get('path_gcs_scores', {}))

    ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']

    top_legend_handles = []
    top_legend_labels = []
    bottom_legend_handles = []
    bottom_legend_labels = []

    def _append_unique(handles, labels, new_handles, new_labels):
        seen = set(labels)
        for handle, label in zip(new_handles, new_labels):
            if not label or label.startswith('_'):
                continue
            if label in seen:
                continue
            handles.append(handle)
            labels.append(label)
            seen.add(label)

    # Top row: By-layer view
    for layer_idx in range(num_layers):
        ax = fig.add_subplot(gs[0, layer_idx])

        for path_name in ['overall', 'attention', 'mlp']:
            key = f'layer{layer_idx}_{path_name}_path'
            scores = path_scores.get(key, [])

            if not scores:
                continue

            steps_plot = gcs_steps[:len(scores)]
            scores_arr = np.array(scores)

            if max_epoch:
                mask = steps_plot <= max_epoch
                steps_plot = steps_plot[mask]
                scores_arr = scores_arr[mask]

            if normalized:
                scores_arr = normalize_scores(scores_arr)

            ax.plot(steps_plot, scores_arr,
                    label=PATH_LABELS[path_name],
                    color=PATH_COLORS[path_name],
                    linewidth=PLOT_CONFIG['linewidth'], alpha=0.9)

        if grokking_epoch and (not max_epoch or grokking_epoch <= max_epoch):
            add_grokking_marker(ax, grokking_epoch, label=f'Grokking (epoch {grokking_epoch})')

        layer_handles, layer_labels = ax.get_legend_handles_labels()
        _append_unique(top_legend_handles, top_legend_labels, layer_handles, layer_labels)

        style_axis(
            ax,
            xlabel=LABELS['epoch'],
            ylabel=ylabel,
            title=f'Layer {layer_idx + 1}',
            legend=False,
        )

        if normalized:
            ax.set_ylim([-0.05, 1.05])

    # Fill remaining top-row slot with Training Dynamics if available
    if num_layers < num_cols and 'log_steps' in metrics:
        ax_train = fig.add_subplot(gs[0, num_cols - 1])
        log_steps = metrics['log_steps']

        ax_train.plot(log_steps, metrics.get('train_accuracies', []),
                      label='Train Acc', color='#0173B2',
                      linewidth=PLOT_CONFIG['linewidth'])
        ax_train.plot(log_steps, metrics.get('test_accuracies', []),
                      label='Test Acc', color='#029E73',
                      linewidth=PLOT_CONFIG['linewidth'])
        ax_train.set_ylabel(LABELS['accuracy'], color='#0173B2',
                            fontsize=PLOT_CONFIG['label_fontsize'])
        ax_train.set_ylim([0, 1.05])

        ax2 = ax_train.twinx()
        ax2.plot(log_steps, metrics.get('test_losses', []),
                 label='Test Loss', color='#CC3311', linestyle='--',
                 linewidth=PLOT_CONFIG['linewidth'])
        ax2.set_ylabel('Loss (log scale)', color='#CC3311',
                        fontsize=PLOT_CONFIG['label_fontsize'])
        ax2.set_yscale('log')

        ax_train.set_xlabel(LABELS['epoch'], fontsize=PLOT_CONFIG['label_fontsize'])
        ax_train.set_title('Training Dynamics', fontsize=PLOT_CONFIG['label_fontsize'])
        ax_train.grid(True, alpha=PLOT_CONFIG['grid_alpha'])

        if num_layers >= 3:
            lines1, labels1 = ax_train.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            _append_unique(top_legend_handles, top_legend_labels, lines1 + lines2, labels1 + labels2)

        if grokking_epoch and (not max_epoch or grokking_epoch <= max_epoch):
            add_grokking_marker(ax_train, grokking_epoch)

    # Bottom row: By-path view
    path_names = ['overall', 'attention', 'mlp']

    for col_idx, path_name in enumerate(path_names):
        ax = fig.add_subplot(gs[1, col_idx])

        for layer_idx in range(num_layers):
            key = f'layer{layer_idx}_{path_name}_path'
            scores = path_scores.get(key, [])

            if not scores:
                continue

            steps_plot = gcs_steps[:len(scores)]
            scores_arr = np.array(scores)

            if max_epoch:
                mask = steps_plot <= max_epoch
                steps_plot = steps_plot[mask]
                scores_arr = scores_arr[mask]

            if normalized:
                scores_arr = normalize_scores(scores_arr)

            ax.plot(steps_plot, scores_arr,
                    label=f'Layer {layer_idx + 1}',
                    color=get_layer_color(layer_idx),
                    linewidth=PLOT_CONFIG['linewidth'], alpha=0.9)

        if grokking_epoch and (not max_epoch or grokking_epoch <= max_epoch):
            add_grokking_marker(ax, grokking_epoch, label=f'Grokking (epoch {grokking_epoch})')

        path_handles, path_labels = ax.get_legend_handles_labels()
        _append_unique(bottom_legend_handles, bottom_legend_labels, path_handles, path_labels)

        style_axis(
            ax,
            xlabel=LABELS['epoch'],
            ylabel=ylabel,
            title=PATH_LABELS[path_name],
            legend=False,
        )

        if normalized:
            ax.set_ylim([-0.05, 1.05])

    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')

    if stack_legends_bottom:
        top_legend_y = 0.08
        bottom_legend_y = 0.01

        if top_legend_handles:
            fig.legend(
                top_legend_handles,
                top_legend_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, top_legend_y),
                ncol=min(4, len(top_legend_handles)),
                fontsize=PLOT_CONFIG['legend_fontsize'],
                framealpha=PLOT_CONFIG['legend_alpha'],
            )

        if bottom_legend_handles:
            fig.legend(
                bottom_legend_handles,
                bottom_legend_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, bottom_legend_y),
                ncol=min(4, len(bottom_legend_handles)),
                fontsize=PLOT_CONFIG['legend_fontsize'],
                framealpha=PLOT_CONFIG['legend_alpha'],
            )

        fig.subplots_adjust(top=0.95, bottom=0.19)
    else:
        top_legend_y = 0.545
        bottom_legend_y = 0.001

        if top_legend_handles:
            fig.legend(
                top_legend_handles,
                top_legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, top_legend_y),
                ncol=min(5, len(top_legend_handles)),
                fontsize=PLOT_CONFIG['legend_fontsize'],
                framealpha=PLOT_CONFIG['legend_alpha'],
            )

        if bottom_legend_handles:
            fig.legend(
                bottom_legend_handles,
                bottom_legend_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, bottom_legend_y),
                ncol=min(5, len(bottom_legend_handles)),
                fontsize=PLOT_CONFIG['legend_fontsize'],
                framealpha=PLOT_CONFIG['legend_alpha'],
            )

        fig.subplots_adjust(top=0.95, bottom=0.11)

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_2layer_bylayer_with_accuracy(
    metrics: Dict,
    normalized: bool = False,
    grokking_epoch: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot 2-layer GCS with by-layer view and accuracy.

    Layout:
    - Top row: Layer 0 GCS | Layer 1 GCS
    - Bottom: Accuracy + Loss (dual y-axis)

    Like 2layer_Transformer_GeLU_stable_add_bylayer_raw.png

    Args:
        metrics: Training metrics with GCS data
        normalized: Whether to normalize GCS
        grokking_epoch: Epoch to mark as grokking
        title: Optional title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('gcs_steps', [])))
    path_scores = metrics.get('path_gcs_scores_final_geo', metrics.get('path_gcs_scores', {}))
    log_steps = metrics.get('log_steps', [])

    ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']

    # Top row: By-layer view
    for layer_idx in range(2):
        ax = fig.add_subplot(gs[0, layer_idx])

        for path_name in ['overall', 'attention', 'mlp']:
            key = f'layer{layer_idx}_{path_name}_path'
            # Also try without layer prefix for 2-layer
            if key not in path_scores:
                key = f'{path_name}_path'

            scores = path_scores.get(key, [])

            if not scores:
                continue

            steps_plot = gcs_steps[:len(scores)]
            scores_arr = np.array(scores)

            if normalized:
                scores_arr = normalize_scores(scores_arr)

            ax.plot(steps_plot, scores_arr,
                    label=PATH_LABELS[path_name],
                    color=PATH_COLORS[path_name],
                    linewidth=PLOT_CONFIG['linewidth'])

        style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel,
                   title=f'Layer {layer_idx}', legend_loc='best')

    # Bottom: Accuracy + Loss
    ax1 = fig.add_subplot(gs[1, :])

    # Accuracy
    ax1.plot(log_steps, metrics.get('train_accuracies', []),
             label='Train Acc', color='#0173B2', linewidth=PLOT_CONFIG['linewidth'])
    ax1.plot(log_steps, metrics.get('test_accuracies', []),
             label='Test Acc', color='#029E73', linewidth=PLOT_CONFIG['linewidth'])
    ax1.set_ylabel(LABELS['accuracy'], color='#0173B2', fontsize=PLOT_CONFIG['label_fontsize'])
    ax1.set_ylim([0, 1.05])

    # Loss on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(log_steps, metrics.get('test_losses', []),
             label='Test Loss', color='#CC3311', linestyle='--', linewidth=PLOT_CONFIG['linewidth'])
    ax2.set_ylabel('Loss (log scale)', color='#CC3311', fontsize=PLOT_CONFIG['label_fontsize'])
    ax2.set_yscale('log')

    ax1.set_xlabel(LABELS['epoch'], fontsize=PLOT_CONFIG['label_fontsize'])
    ax1.grid(True, alpha=PLOT_CONFIG['grid_alpha'])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right',
               fontsize=PLOT_CONFIG['legend_fontsize'])

    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Convenience Plot from Checkpoint
# =============================================================================

def plot_from_metrics_file(
    metrics_path: str,
    plot_type: str = '3path',
    activation: str = None,
    num_layers: int = 1,
    **kwargs
):
    """
    Convenience function to generate plots from a metrics pickle file.

    Args:
        metrics_path: Path to training_metrics.pkl or training_metrics_final_geo.pkl
        plot_type: Type of plot ('3path', 'with_accuracy', 'multilayer', etc.)
        activation: Activation function name (for labeling)
        num_layers: Number of layers (for multi-layer plots)
        **kwargs: Additional arguments passed to specific plot function

    Returns:
        matplotlib Figure
    """
    import pickle

    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)

    if plot_type == '3path':
        return plot_gcs_3path(metrics, **kwargs)
    elif plot_type == 'with_accuracy':
        return plot_gcs_with_accuracy(metrics, metrics, **kwargs)
    elif plot_type == 'multilayer_bylayer':
        return plot_multilayer_bylayer(metrics, num_layers, **kwargs)
    elif plot_type == 'multilayer_bypath':
        return plot_multilayer_bypath(metrics, num_layers, **kwargs)
    elif plot_type == 'multilayer_combined':
        return plot_multilayer_combined_2row(metrics, num_layers, **kwargs)
    elif plot_type == '2layer_bylayer_acc':
        return plot_2layer_bylayer_with_accuracy(metrics, **kwargs)
    elif plot_type == 'training':
        return plot_training_curves(metrics, **kwargs)
    elif plot_type == 'mlp_perlayer':
        return plot_mlp_gcs_perlayer(metrics, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


# =============================================================================
# MLP GCS Plots
# =============================================================================

def plot_mlp_gcs_perlayer(
    gcs_data: Dict,
    num_layers: int = 2,
    normalized: bool = False,
    grokking_epoch: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot per-layer GCS evolution for an MLP experiment.

    Layout: 1 x (num_layers + 1) subplots: overall + per-layer

    Args:
        gcs_data: dict with 'gcs_steps_final_geo' and 'path_gcs_scores_final_geo'
                  containing 'overall', 'layer_0', 'layer_1', etc.
        num_layers: Number of MLP layers
        normalized: Whether to normalize scores
        grokking_epoch: Epoch to mark as grokking
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    num_cols = num_layers + 1
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 4))

    if num_cols == 1:
        axes = [axes]

    gcs_steps = np.array(gcs_data.get('gcs_steps_final_geo', gcs_data.get('epochs', [])))
    path_scores = gcs_data.get('path_gcs_scores_final_geo', gcs_data)

    # First subplot: overall
    ax = axes[0]
    scores = path_scores.get('overall', [])
    if scores:
        steps_plot = gcs_steps[:len(scores)]
        scores_arr = np.array(scores)
        valid = np.array([s is not None for s in scores_arr])
        if np.any(valid):
            steps_valid = steps_plot[valid]
            scores_valid = scores_arr[valid].astype(float)
            if normalized:
                scores_valid = normalize_scores(scores_valid)
            ax.plot(steps_valid, scores_valid, color='#0173B2',
                    linewidth=PLOT_CONFIG['linewidth'], label='Overall')

    if grokking_epoch:
        add_grokking_marker(ax, grokking_epoch)

    ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']
    style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel, title='Overall')
    if normalized:
        ax.set_ylim([-0.05, 1.05])

    # Per-layer subplots
    for layer_idx in range(num_layers):
        ax = axes[layer_idx + 1]
        key = f'layer_{layer_idx}'
        scores = path_scores.get(key, [])
        if scores:
            steps_plot = gcs_steps[:len(scores)]
            scores_arr = np.array(scores)
            valid = np.array([s is not None for s in scores_arr])
            if np.any(valid):
                steps_valid = steps_plot[valid]
                scores_valid = scores_arr[valid].astype(float)
                if normalized:
                    scores_valid = normalize_scores(scores_valid)
                color = get_layer_color(layer_idx)
                ax.plot(steps_valid, scores_valid, color=color,
                        linewidth=PLOT_CONFIG['linewidth'],
                        label=f'Layer {layer_idx}')

        if grokking_epoch:
            add_grokking_marker(ax, grokking_epoch)

        style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel,
                   title=f'Layer {layer_idx}')
        if normalized:
            ax.set_ylim([-0.05, 1.05])

    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_embedding_comparison(
    data_by_type: Dict[str, Dict],
    normalized: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Compare 3 MLP input types at same depth/width.

    Layout: 1x3 (one subplot per embedding type showing overall GCS)

    Args:
        data_by_type: dict mapping embedding type name to metrics dict
                      e.g., {'OneHot': {...}, 'EmbedConcat': {...}, 'EmbedSum': {...}}
        normalized: Whether to normalize scores
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=get_figsize('1x3'))

    type_colors = {
        'OneHot': MODEL_COLORS.get('onehot_mlp', '#D95F02'),
        'EmbedConcat': MODEL_COLORS.get('embed_concat_mlp', '#7570B3'),
        'EmbedSum': MODEL_COLORS.get('embed_sum_mlp', '#E7298A'),
    }

    for idx, (type_name, metrics) in enumerate(data_by_type.items()):
        ax = axes[idx] if len(data_by_type) > 1 else axes
        gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('epochs', [])))
        path_scores = metrics.get('path_gcs_scores_final_geo', metrics)
        scores = path_scores.get('overall', [])

        if scores:
            steps_plot = gcs_steps[:len(scores)]
            scores_arr = np.array(scores)
            valid = np.array([s is not None for s in scores_arr])
            if np.any(valid):
                steps_valid = steps_plot[valid]
                scores_valid = scores_arr[valid].astype(float)
                if normalized:
                    scores_valid = normalize_scores(scores_valid)
                color = type_colors.get(type_name, '#333333')
                ax.plot(steps_valid, scores_valid, color=color,
                        linewidth=PLOT_CONFIG['linewidth'], label='Overall GCS')

        ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']
        style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel, title=type_name)
        if normalized:
            ax.set_ylim([-0.05, 1.05])

    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_depth_comparison(
    data_by_depth: Dict[int, Dict],
    normalized: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Compare same architecture at different depths (overlay GCS curves).

    Args:
        data_by_depth: dict mapping num_layers to metrics dict
                       e.g., {1: {...}, 2: {...}, 3: {...}}
        normalized: Whether to normalize scores
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()
    fig, ax = plt.subplots(figsize=get_figsize('single'))

    for depth in sorted(data_by_depth.keys()):
        metrics = data_by_depth[depth]
        gcs_steps = np.array(metrics.get('gcs_steps_final_geo', metrics.get('epochs', [])))
        path_scores = metrics.get('path_gcs_scores_final_geo', metrics)
        scores = path_scores.get('overall', [])

        if not scores:
            continue

        steps_plot = gcs_steps[:len(scores)]
        scores_arr = np.array(scores)
        valid = np.array([s is not None for s in scores_arr])
        if not np.any(valid):
            continue

        steps_valid = steps_plot[valid]
        scores_valid = scores_arr[valid].astype(float)
        if normalized:
            scores_valid = normalize_scores(scores_valid)

        color = get_layer_color(depth - 1)
        ax.plot(steps_valid, scores_valid, color=color,
                linewidth=PLOT_CONFIG['linewidth'],
                label=f'{depth} layer{"s" if depth > 1 else ""}')

    ylabel = LABELS['gcs_normalized'] if normalized else LABELS['gcs']
    style_axis(ax, xlabel=LABELS['epoch'], ylabel=ylabel,
               title=title or 'Depth Comparison')
    if normalized:
        ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_architecture_grid(
    all_data: Dict[str, Dict[int, Dict]],
    normalized: bool = True,
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Comprehensive grid: rows=depths, columns=architectures.

    Each cell shows overall GCS + accuracy.

    Args:
        all_data: nested dict {arch_name: {depth: metrics_dict}}
        normalized: Whether to normalize scores
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()

    arch_names = list(all_data.keys())
    all_depths = sorted(set(d for arch in all_data.values() for d in arch.keys()))

    num_rows = len(all_depths)
    num_cols = len(arch_names)

    fig, axes = plt.subplots(num_rows, num_cols,
                              figsize=(4 * num_cols, 3 * num_rows),
                              squeeze=False)

    for col_idx, arch_name in enumerate(arch_names):
        for row_idx, depth in enumerate(all_depths):
            ax = axes[row_idx, col_idx]

            if depth not in all_data[arch_name]:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(f'{arch_name} ({depth}L)')
                continue

            metrics = all_data[arch_name][depth]
            gcs_steps = np.array(metrics.get('gcs_steps_final_geo',
                                              metrics.get('epochs', [])))
            path_scores = metrics.get('path_gcs_scores_final_geo', metrics)
            scores = path_scores.get('overall', [])

            if scores:
                steps_plot = gcs_steps[:len(scores)]
                scores_arr = np.array(scores)
                valid = np.array([s is not None for s in scores_arr])
                if np.any(valid):
                    steps_valid = steps_plot[valid]
                    scores_valid = scores_arr[valid].astype(float)
                    if normalized:
                        scores_valid = normalize_scores(scores_valid)
                    ax.plot(steps_valid, scores_valid, color='#0173B2',
                            linewidth=PLOT_CONFIG['linewidth'], label='GCS')

            # Overlay accuracy if available
            log_steps = metrics.get('log_steps', [])
            test_accs = metrics.get('test_accuracies', [])
            if log_steps and test_accs:
                ax2 = ax.twinx()
                ax2.plot(log_steps, test_accs, color='#029E73',
                         linewidth=1.5, alpha=0.7, label='Test Acc')
                ax2.set_ylim([0, 1.05])
                ax2.tick_params(labelsize=8)

            ylabel = 'Norm. GCS' if normalized else 'GCS'
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f'{arch_name} ({depth}L)', fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, linestyle='--')
            if normalized:
                ax.set_ylim([-0.05, 1.05])

    if title:
        fig.suptitle(title, fontsize=PLOT_CONFIG['title_fontsize'], fontweight='bold')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig
