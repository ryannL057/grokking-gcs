"""
Analysis module: geometry construction and GCS computation.

Supports multi-layer transformers and MLP architectures with per-layer geometry construction.

Key functions:
- build_complete_geometric_structure: Build geometry for a specific layer
- analyze_3path_gmn_tsa: Analyze 3 paths (overall, attention, MLP) for target layer (Transformer)
- analyze_mlp_gcs: Analyze GCS for MLP models with per-layer decomposition
- compute_path_gmn_tsa: Compute GCS for a specific path (Transformer)
- compute_mlp_path_gcs: Compute GCS for an MLP path

Multi-layer Transformer usage:
    for layer_idx in range(num_layers):
        geometry = build_complete_geometric_structure(
            model, train_data, layer_idx=layer_idx
        )
        results = analyze_3path_gmn_tsa(model, geometry, target_layer=layer_idx)

MLP usage:
    geometry = build_complete_geometric_structure(
        model, train_data, layer_idx=0, model_type='onehot_mlp'
    )
    results = analyze_mlp_gcs(model, geometry, embedding_type='one_hot')
"""
from .geometry import (
    build_complete_geometric_structure,
    sample_data_points,
    sample_general_mlp_data_points,
    build_knn_graph,
    compute_tangent_space,
    validate_geometric_structure
)
from .gmn_tsa import (
    compute_path_gmn_tsa,
    compute_mlp_path_gcs,
    compute_alignment_score_cosine,
    compute_alignment_score_procrustes,
    compute_procrustes_alignment,
    compute_principal_angles,
    compute_eigenvalue_spectrum
)
from .paths import (
    analyze_3path_gmn_tsa,
    create_path_forward_function,
    create_mlp_path_forward,
    analyze_mlp_gcs,
    get_path_definitions,
    compute_path_statistics
)
from .batched_gcs import (
    build_coherence_matrix,
    compute_batched_transformed_bases,
    compute_path_gcs_batched,
)
from .reference_geometry import (
    build_reference_geometry_from_prompts,
    count_knn_components,
    extract_block_outputs,
    rebuild_geometry_variant,
)
