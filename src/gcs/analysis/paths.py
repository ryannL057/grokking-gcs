"""
Path analysis utilities for GMN-TSA.

Defines 3-path decomposition: overall, attention, and MLP paths.
"""

import torch
import numpy as np

from gcs.core.constants import PATH_IDENTITY_THRESHOLD


def create_path_forward_function(model, start_hook, end_hook, target_layer=0):
    """
    Create path-specific forward function.

    All paths take x_embedded as input and compute JVP from embedding space.
    The start_hook/end_hook parameters control which block output is returned:
    - Overall path (resid_pre, resid_post): embedding -> resid_post
    - Attention path (resid_pre, attn_out): embedding -> attn_out
    - MLP path (resid_mid, mlp_out): embedding -> mlp_out (through attention then MLP)

    Args:
        model: Transformer model
        start_hook: Controls internal computation flow within the block
        end_hook: Which hook point's output to return
        target_layer: Target layer index

    Returns:
        path_forward: Function mapping x_embedded -> hook output at last position
    """
    hook_order = ['hook_resid_pre', 'hook_attn_out', 'hook_resid_mid', 'hook_mlp_out', 'hook_resid_post']

    try:
        start_idx = hook_order.index(start_hook)
        end_idx = hook_order.index(end_hook)
    except ValueError as e:
        raise ValueError(f"Invalid hook: {e}")

    if start_idx >= end_idx:
        raise ValueError(f"Start hook must come before end hook")

    def path_forward(x_embedded):
        """Execute path-specific forward pass."""
        if x_embedded.dim() == 2:
            x = x_embedded.unsqueeze(0)  # [1, 3, d_model]
        elif x_embedded.dim() == 3:
            x = x_embedded  # [batch, 3, d_model]
        else:
            raise ValueError(f"Expected x_embedded dim 2 or 3, got {x_embedded.dim()}")

        x = model.pos_embed(x)

        # Process layers before target
        for layer_idx in range(target_layer):
            x = model.blocks[layer_idx](x)

        if target_layer < len(model.blocks):
            block = model.blocks[target_layer]

            if start_hook == 'hook_resid_pre':
                x_start = block.hook_resid_pre(x)

                if end_hook == 'hook_attn_out':
                    attn_out = block.hook_attn_out(block.attn(x_start))
                    return attn_out[:, -1, :].squeeze(0)

                attn_out = block.hook_attn_out(block.attn(x_start)) # (1, 3, 128)
                x_mid = block.hook_resid_mid(x_start + attn_out)

                if end_hook == 'hook_resid_mid':
                    return x_mid[:, -1, :].squeeze(0)

                mlp_out = block.hook_mlp_out(block.mlp(x_mid))

                if end_hook == 'hook_mlp_out':
                    return mlp_out[:, -1, :].squeeze(0)

                x_final = block.hook_resid_post(x_mid + mlp_out) # (1, 3, 128)

                if end_hook == 'hook_resid_post':
                    return x_final[:, -1, :].squeeze(0) # (1, 128)

            elif start_hook == 'hook_resid_mid':
                # First execute to resid_mid
                x_start = block.hook_resid_pre(x)
                attn_out = block.hook_attn_out(block.attn(x_start))
                x_mid = block.hook_resid_mid(x_start + attn_out)

                mlp_out = block.hook_mlp_out(block.mlp(x_mid))

                if end_hook == 'hook_mlp_out':
                    return mlp_out[:, -1, :].squeeze(0)

                x_final = block.hook_resid_post(x_mid + mlp_out)
                if end_hook == 'hook_resid_post':
                    return x_final[:, -1, :].squeeze(0)

        # Continue through remaining layers
        for layer_idx in range(target_layer + 1, len(model.blocks)):
            x = model.blocks[layer_idx](x)

        x = model.unembed(x)
        return x[:, -1, :].squeeze(0)

    return path_forward


def analyze_3path_gmn_tsa(model, complete_geometry, target_layer=0, device='cuda:0',
                          alignment_method='cosine', token_probe='final'):
    """
    Analyze GCS for 3 key paths: overall, attention, and MLP.

    Args:
        model: Transformer model
        complete_geometry: Complete geometric structure
        target_layer: Layer to analyze
        device: Computing device
        token_probe: embedding positions used for the JVP injection

    Returns:
        dict: GCS scores for each path
    """
    from .geometry import validate_geometric_structure
    from .gmn_tsa import compute_path_gmn_tsa

    print(f"Analyzing 3 paths for layer {target_layer}")
    validate_geometric_structure(complete_geometry)

    path_configs = [
        {
            'name': 'overall_path',
            'start': 'hook_resid_pre',
            'end': 'hook_resid_post',
            'description': 'Overall (embedding -> resid_post)'
        },
        {
            'name': 'attention_path',
            'start': 'hook_resid_pre',
            'end': 'hook_attn_out',
            'description': 'Attention (embedding -> attn_out)'
        },
        {
            'name': 'mlp_path',
            'start': 'hook_resid_mid',
            'end': 'hook_mlp_out',
            'description': 'MLP (embedding -> mlp_out)'
        }
    ]

    path_results = {}

    for config in path_configs:
        print(f"  Computing {config['description']}...")
        try:
            gcs_score = compute_path_gmn_tsa(
                model, complete_geometry,
                config['start'], config['end'],
                target_layer, device,
                alignment_method=alignment_method,
                token_probe=token_probe,
            )
            path_results[config['name']] = gcs_score
            print(f"    {config['name']}: {gcs_score:.4f}")
        except Exception as e:
            print(f"    {config['name']}: FAILED - {e}")
            path_results[config['name']] = None

    return path_results


def get_path_definitions():
    """Get standardized path definitions.

    All paths compute JVP from embedding space. The start/end hooks
    control which segment of the block output is returned, not the
    JVP input (which is always the embedding).
    """
    return {
        'overall_path': {
            'start': 'hook_resid_pre',
            'end': 'hook_resid_post',
            'description': 'embedding -> resid_post (full block)'
        },
        'attention_path': {
            'start': 'hook_resid_pre',
            'end': 'hook_attn_out',
            'description': 'embedding -> attn_out (attention output only)'
        },
        'mlp_path': {
            'start': 'hook_resid_mid',
            'end': 'hook_mlp_out',
            'description': 'embedding -> mlp_out (through attention then MLP)'
        }
    }


def create_mlp_path_forward(model, embedding_type, start_layer=0, end_layer=None):
    """
    Create path-specific forward function for MLP models.

    Args:
        model: MLP model (OneHotMLP, EmbedConcatMLP, or EmbedSumMLP)
        embedding_type: 'one_hot', 'embed_concat', or 'embed_sum'
        start_layer: Starting layer index (0 = first hidden layer)
        end_layer: Ending layer index (None = through unembed)

    Returns:
        path_forward: Function mapping embedded input to output
    """
    num_layers = model.num_layers

    include_unembed = (end_layer is None)  # Only overall path includes unembed
    if end_layer is None:
        end_layer = num_layers

    def path_forward(x_embedded):
        """Execute MLP path forward pass from embedded/projected input."""
        x = x_embedded
        for layer_idx in range(start_layer, min(end_layer, num_layers)):
            x = model.layers[layer_idx](x)

        if include_unembed:
            x = model.unembed(x)

        return x

    return path_forward


def analyze_mlp_gcs(model, geometry, embedding_type, device='cuda:0',
                    alignment_method='cosine', geometries_by_layer=None):
    """
    Analyze GCS for MLP model with per-layer decomposition.

    Analogous to analyze_3path_gmn_tsa for Transformer, but uses
    per-layer paths instead of attention/MLP paths.

    Args:
        model: MLP model (OneHotMLP, EmbedConcatMLP, or EmbedSumMLP)
        geometry: Geometric structure from build_complete_geometric_structure
        embedding_type: 'one_hot', 'embed_concat', or 'embed_sum'
        device: Computing device
        alignment_method: 'cosine' or 'procrustes'
        geometries_by_layer: Optional dict mapping layer_idx -> geometry.
            If provided, each layer_i score uses geometry built for layer_i,
            and overall uses geometry of the last layer. This enables
            per-layer final geometry protocol for multi-layer MLP.

    Returns:
        dict: GCS scores {'overall': score, 'layer_0': score, 'layer_1': score, ...}
    """
    from .gmn_tsa import compute_mlp_path_gcs

    print(f"Analyzing MLP GCS ({embedding_type})")

    num_layers = model.num_layers
    results = {}

    # Overall path: from embedding through all layers + unembed
    print(f"  Computing overall path...")
    try:
        if geometries_by_layer is not None:
            overall_geometry = geometries_by_layer.get(num_layers - 1, geometry)
        else:
            overall_geometry = geometry

        overall_forward = create_mlp_path_forward(model, embedding_type,
                                                   start_layer=0, end_layer=None)
        gcs_score = compute_mlp_path_gcs(
            model, overall_geometry, overall_forward, device, alignment_method
        )
        results['overall'] = gcs_score
        print(f"    overall: {gcs_score:.4f}")
    except Exception as e:
        print(f"    overall: FAILED - {e}")
        results['overall'] = None

    # Per-layer paths
    for layer_idx in range(num_layers):
        print(f"  Computing layer {layer_idx} path...")
        try:
            if geometries_by_layer is not None:
                layer_geometry = geometries_by_layer.get(layer_idx, geometry)
            else:
                layer_geometry = geometry

            layer_forward = create_mlp_path_forward(model, embedding_type,
                                                     start_layer=0,
                                                     end_layer=layer_idx + 1)
            gcs_score = compute_mlp_path_gcs(
                model, layer_geometry, layer_forward, device, alignment_method
            )
            results[f'layer_{layer_idx}'] = gcs_score
            print(f"    layer_{layer_idx}: {gcs_score:.4f}")
        except Exception as e:
            print(f"    layer_{layer_idx}: FAILED - {e}")
            results[f'layer_{layer_idx}'] = None

    return results


def compute_path_statistics(path_results):
    """Compute statistics for path analysis results."""
    valid_scores = [s for s in path_results.values() if s is not None]

    if not valid_scores:
        return {'count': 0}

    valid_scores = np.array(valid_scores)

    stats = {
        'count': len(valid_scores),
        'mean': np.mean(valid_scores),
        'std': np.std(valid_scores),
        'range': np.max(valid_scores) - np.min(valid_scores)
    }

    # Path differences
    if 'overall_path' in path_results and 'attention_path' in path_results:
        if path_results['overall_path'] and path_results['attention_path']:
            stats['overall_vs_attention'] = abs(
                path_results['overall_path'] - path_results['attention_path']
            )

    if 'attention_path' in path_results and 'mlp_path' in path_results:
        if path_results['attention_path'] and path_results['mlp_path']:
            stats['attention_vs_mlp'] = abs(
                path_results['attention_path'] - path_results['mlp_path']
            )

    return stats
