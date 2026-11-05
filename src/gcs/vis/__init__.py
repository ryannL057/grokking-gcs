"""
Visualization module for GCS analysis.

Provides comprehensive plotting functions for:
- Training curves (loss/accuracy)
- GCS evolution (3-path decomposition)
- Ablation studies (N, d parameters)
- Cross-activation comparisons
- Multi-task GCS comparisons
- Attention pattern heatmaps
- Multi-layer GCS visualization
- PR vs GCS comparisons
"""

from .config import (
    # Style configuration
    setup_style,
    PLOT_CONFIG,
    PATH_COLORS,
    ACTIVATION_COLORS,
    MODEL_COLORS,
    LAYER_COLORS,
    PATH_LABELS,
    LABELS,
    # Helper functions
    get_path_color,
    get_activation_color,
    get_model_color,
    get_layer_color,
    get_ablation_color,
    get_figsize,
    normalize_scores,
    add_grokking_marker,
    style_axis,
    save_figure,
)

from .plots import (
    # Basic training curves
    plot_training_curves,

    # GCS evolution plots
    plot_gcs_3path,
    plot_gcs_with_accuracy,

    # Ablation study plots
    plot_ablation_3path,

    # Cross-activation comparison
    plot_activation_comparison_2x2,

    # Multi-task plots
    plot_multitask_gcs_2x2,

    # PR vs GCS comparison
    plot_pr_vs_gcs_2x2,

    # Attention heatmaps
    plot_attention_evolution,
    plot_attention_patterns,

    # Multi-layer plots
    plot_multilayer_bylayer,
    plot_multilayer_bypath,
    plot_multilayer_combined_2row,
    plot_2layer_bylayer_with_accuracy,

    # MLP plots
    plot_mlp_gcs_perlayer,
    plot_embedding_comparison,
    plot_depth_comparison,
    plot_architecture_grid,

    # Convenience function
    plot_from_metrics_file,
)

_ARCHIVE_EXPORTS = (
    'UNIFIED_STYLE', 'GROKKING_EPOCHS', 'TRAIN_ACC_1_EPOCHS', 'GROK_COLOR',
    'GROK_MARKERS', 'normalize_to_01', 'parse_experiment_info', 'load_metrics',
    'plot_1layer_gcs_with_training', 'plot_1layer_with_markers',
    'plot_activation_comparison', 'plot_multitask_comparison',
    'plot_2layer_combined', 'plot_3layer_combined',
    'generate_all_unified_figures', 'generate_mlp_figures',
)


def __getattr__(name):
    # Only historical cohort plots need an explicit archive root.
    if name in _ARCHIVE_EXPORTS:
        from . import unified_figures
        return getattr(unified_figures, name)
    raise AttributeError(name)

__all__ = [
    # Config
    'setup_style',
    'PLOT_CONFIG',
    'PATH_COLORS',
    'ACTIVATION_COLORS',
    'MODEL_COLORS',
    'LAYER_COLORS',
    'PATH_LABELS',
    'LABELS',
    'get_path_color',
    'get_activation_color',
    'get_model_color',
    'get_layer_color',
    'get_ablation_color',
    'get_figsize',
    'normalize_scores',
    'add_grokking_marker',
    'style_axis',
    'save_figure',
    # Basic plots
    'plot_training_curves',
    # GCS plots
    'plot_gcs_3path',
    'plot_gcs_with_accuracy',
    # Ablation plots
    'plot_ablation_3path',
    # Activation comparison
    'plot_activation_comparison_2x2',
    # Multi-task plots
    'plot_multitask_gcs_2x2',
    # PR vs GCS
    'plot_pr_vs_gcs_2x2',
    # Attention plots
    'plot_attention_evolution',
    'plot_attention_patterns',
    # Multi-layer plots
    'plot_multilayer_bylayer',
    'plot_multilayer_bypath',
    'plot_multilayer_combined_2row',
    'plot_2layer_bylayer_with_accuracy',
    # MLP plots
    'plot_mlp_gcs_perlayer',
    'plot_embedding_comparison',
    'plot_depth_comparison',
    'plot_architecture_grid',
    # Convenience
    'plot_from_metrics_file',
    # Unified figures (for figures_unified/ - pixel-consistent with scripts/)
    'UNIFIED_STYLE',
    'GROKKING_EPOCHS',
    'TRAIN_ACC_1_EPOCHS',
    'GROK_COLOR',
    'GROK_MARKERS',
    'normalize_to_01',
    'parse_experiment_info',
    'load_metrics',
    'plot_1layer_gcs_with_training',
    'plot_1layer_with_markers',
    'plot_activation_comparison',
    'plot_multitask_comparison',
    'plot_2layer_combined',
    'plot_3layer_combined',
    'generate_all_unified_figures',
    'generate_mlp_figures',
]
