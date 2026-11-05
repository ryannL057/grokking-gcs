"""
Unified plotting configuration for GCS visualization.

Provides standardized color schemes, font sizes, and styling for
consistent visualization across all plot types.
"""

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Color Schemes
# =============================================================================

# Primary path colors (for 3-path decomposition)
PATH_COLORS = {
    'overall': '#0173B2',      # Blue
    'attention': '#DE8F05',    # Orange
    'mlp': '#029E73',          # Green
}

# Activation function colors
ACTIVATION_COLORS = {
    'ReLU': '#0173B2',    # Blue
    'GeLU': '#DE8F05',    # Orange
    'SiLU': '#029E73',    # Green
    'SeLU': '#CC3311',    # Red
}

# Model/architecture colors (for cross-architecture comparison)
MODEL_COLORS = {
    'onehot_mlp': '#D95F02',
    'embed_concat_mlp': '#7570B3',
    'embed_sum_mlp': '#E7298A',
    'transformer_clock': '#1B9E77',
    'transformer_pizza': '#66A61E',
}

# Layer colors (for multi-layer plots)
LAYER_COLORS = ['#0173B2', '#DE8F05', '#CC3311', '#9467BD', '#8C564B']

# Training curve colors
TRAINING_COLORS = {
    'train': '#0173B2',   # Blue
    'test': '#029E73',    # Green (teal)
    'loss': '#CC3311',    # Red
}

# N/D ablation colors (for ablation studies)
ABLATION_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


# =============================================================================
# Plot Style Configuration
# =============================================================================

PLOT_CONFIG = {
    # Font sizes
    'title_fontsize': 14,
    'label_fontsize': 12,
    'tick_fontsize': 10,
    'legend_fontsize': 10,

    # Line properties
    'linewidth': 2.5,
    'linewidth_thin': 1.5,
    'marker_size': 8,

    # Grid
    'grid_alpha': 0.3,
    'grid_style': '--',

    # Figure sizes
    'figsize': {
        'single': (10, 6),
        '1x2': (14, 5),
        '1x3': (15, 5),
        '2x2': (10, 8),
        '2x3': (12, 8),
        '3x2': (10, 12),
        '4x5': (15, 12),
    },

    # Margins
    'legend_alpha': 0.9,
}


# =============================================================================
# Standard Labels
# =============================================================================

LABELS = {
    'epoch': 'Training Epoch',
    'gcs': 'GCS',
    'gcs_normalized': 'Normalized GCS',
    'accuracy': 'Accuracy',
    'test_accuracy': 'Test Accuracy',
    'train_accuracy': 'Train Accuracy',
    'loss': 'Loss',
    'test_loss': 'Test Loss',
    'train_loss': 'Train Loss',
    'pr': 'Participation Ratio',
}

PATH_LABELS = {
    'overall': 'Overall Flow',
    'attention': 'Attention Flow',
    'mlp': 'MLP Flow',
}


# =============================================================================
# Helper Functions
# =============================================================================

def setup_style():
    """Apply consistent matplotlib style settings."""
    plt.rcParams.update({
        'font.size': PLOT_CONFIG['tick_fontsize'],
        'axes.labelsize': PLOT_CONFIG['label_fontsize'],
        'axes.titlesize': PLOT_CONFIG['title_fontsize'],
        'legend.fontsize': PLOT_CONFIG['legend_fontsize'],
        'axes.grid': True,
        'grid.alpha': PLOT_CONFIG['grid_alpha'],
        'grid.linestyle': PLOT_CONFIG['grid_style'],
    })


def get_path_color(path_name):
    """Get color for a path type."""
    key = path_name.replace('_path', '').replace('_flow', '')
    return PATH_COLORS.get(key, '#333333')


def get_activation_color(activation):
    """Get color for an activation function."""
    return ACTIVATION_COLORS.get(activation, '#333333')


def get_layer_color(layer_idx):
    """Get color for a layer index."""
    return LAYER_COLORS[layer_idx % len(LAYER_COLORS)]


def get_model_color(model_type):
    """Get color for a model/architecture type."""
    return MODEL_COLORS.get(model_type, '#333333')


def get_ablation_color(idx):
    """Get color for ablation study line."""
    return ABLATION_COLORS[idx % len(ABLATION_COLORS)]


def get_figsize(layout):
    """Get figure size for a layout type."""
    return PLOT_CONFIG['figsize'].get(layout, (10, 6))


def normalize_scores(scores):
    """
    Normalize scores to [0, 1] range using min-max normalization.

    Args:
        scores: array-like of scores

    Returns:
        Normalized scores as numpy array
    """
    scores = np.array([s for s in scores if s is not None])
    if len(scores) == 0:
        return scores

    min_val = np.min(scores)
    max_val = np.max(scores)

    if max_val - min_val < 1e-10:
        return np.zeros_like(scores)

    return (scores - min_val) / (max_val - min_val)


def add_grokking_marker(ax, epoch, color='#E74C3C', label=None):
    """
    Add a vertical line marker for grokking epoch.

    Args:
        ax: matplotlib axis
        epoch: grokking epoch
        color: line color
        label: optional label for legend
    """
    ax.axvline(
        x=epoch,
        color=color,
        linestyle='--',
        linewidth=PLOT_CONFIG['linewidth_thin'],
        alpha=0.7,
        label=label
    )


def style_axis(
    ax,
    xlabel=None,
    ylabel=None,
    title=None,
    legend=True,
    legend_loc='best',
    legend_bbox_to_anchor=None,
    legend_ncol=1,
):
    """
    Apply consistent styling to an axis.

    Args:
        ax: matplotlib axis
        xlabel: x-axis label
        ylabel: y-axis label
        title: subplot title
        legend: whether to show legend
        legend_loc: legend location
        legend_bbox_to_anchor: optional legend anchor point for placing outside axis
        legend_ncol: number of columns in legend
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['label_fontsize'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['label_fontsize'])
    if title:
        ax.set_title(title, fontsize=PLOT_CONFIG['title_fontsize'])

    ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'], linestyle=PLOT_CONFIG['grid_style'])
    ax.tick_params(labelsize=PLOT_CONFIG['tick_fontsize'])

    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=PLOT_CONFIG['legend_fontsize'],
                  framealpha=PLOT_CONFIG['legend_alpha'],
                  loc=legend_loc,
                  bbox_to_anchor=legend_bbox_to_anchor,
                  ncol=legend_ncol)


def save_figure(fig, save_path, dpi=300):
    """
    Save figure with consistent settings.

    Args:
        fig: matplotlib figure
        save_path: path to save figure
        dpi: resolution
    """
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {save_path}")
