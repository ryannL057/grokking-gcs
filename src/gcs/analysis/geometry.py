"""
Geometric structure construction for GMN-TSA analysis.

Builds k-NN graphs and tangent spaces for manifold analysis.
"""

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from gcs.core.constants import (
    NUMERICAL_EPSILON,
    LOW_DIVERSITY_THRESHOLD,
    NEIGHBOR_DIVERSITY_RATIO,
    DEFAULT_VARIANCE_THRESHOLD
)


def sample_data_points(model, train_data, N=200, device='cuda:0',
                       layer_idx=-1, hook_point='hook_resid_post'):
    """
    Sample N points and extract network-processed representations.

    Args:
        model: Trained model
        train_data: Training dataset
        N: Number of points to sample
        device: Computing device
        layer_idx: Layer to extract from (-1 for last)
        hook_point: Hook point name

    Returns:
        (X_sample, X_prompts): Hidden states and original prompts
    """
    total_train_size = len(train_data) # (3830)
    sample_indices = torch.randperm(total_train_size)[:N] # (200)

    sampled_prompts = []
    for idx in sample_indices:
        prompt, _ = train_data[idx]
        sampled_prompts.append(prompt)

    X_prompts = torch.stack(sampled_prompts).to(device).to(torch.float64) # (200, 3)

    with torch.no_grad():
        model.eval()
        cache = {}
        model.cache_all(cache, incl_bwd=False)
        _ = model(X_prompts.to(torch.int64)) # (200, 3, 114)

        if layer_idx == -1:
            layer_idx = len(model.blocks) - 1 # 0

        hook_name = f'blocks.{layer_idx}.{hook_point}'

        if hook_name in cache:
            hidden_states = cache[hook_name]
            X_sample = hidden_states[:, -1, :]  # Last position
        else:
            raise ValueError(f"Hook {hook_name} not found")

        model.remove_all_hooks()

    return X_sample, X_prompts


def sample_mlp_data_points(model, train_data, N=200, device='cuda:0', layer_idx=0):
    """
    Sample data points for MLP models.

    Args:
        model: SimpleMLP model
        train_data: Training dataset
        N: Number of points
        device: Computing device
        layer_idx: Which layer to extract from

    Returns:
        (X_sample, X_prompts): Hidden states and prompts
    """
    total_train_size = len(train_data)
    sample_indices = torch.randperm(total_train_size)[:N]

    sampled_prompts = []
    for idx in sample_indices:
        prompt, _ = train_data[idx]
        sampled_prompts.append(prompt)

    X_prompts = torch.stack(sampled_prompts).to(device).to(torch.float64)

    with torch.no_grad():
        model.eval()
        cache = {}
        model.cache_all(cache, incl_bwd=False)
        _ = model(X_prompts.to(torch.int64))

        hook_name = f'layers.{layer_idx}.hook_post'

        if hook_name in cache:
            X_sample = cache[hook_name]
        else:
            raise ValueError(f"Hook {hook_name} not found")

        model.remove_all_hooks()

    return X_sample, X_prompts


def build_knn_graph(X_sample, k=15, normalize=False):
    """
    Build k-nearest neighbor graph.

    Args:
        X_sample: Sample data [N, d_model]
        k: Number of neighbors
        normalize: If True, normalize vectors to unit length before finding neighbors.
                   This makes Euclidean distance equivalent to cosine distance ranking,
                   which may be more appropriate if semantic similarity is angular.

    Returns:
        (knn_indices, knn_distances)
    """
    X_np = X_sample.cpu().numpy()  # (200, 128)

    if normalize:
        # L2 normalize: on unit sphere, Euclidean distance ranking = cosine distance ranking
        norms = np.linalg.norm(X_np, axis=1, keepdims=True)
        X_np = X_np / (norms + 1e-8)

    nn_finder = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', metric='euclidean')
    nn_finder.fit(X_np)
    distances, indices = nn_finder.kneighbors(X_np)  # (200, 16), (200, 16)

    # Exclude self
    return indices[:, 1:], distances[:, 1:]  # (200, 15)


def diagnose_knn_metric(X_sample, k=15):
    """
    Diagnose whether Euclidean or Cosine (normalized Euclidean) is more appropriate.

    This helps decide whether to use normalize=True in build_knn_graph.

    Args:
        X_sample: Sample data [N, d_model]
        k: Number of neighbors

    Returns:
        dict: Diagnostic information including:
            - norm_stats: Statistics about vector norms
            - neighbor_overlap: How many neighbors are shared between Euclidean and Cosine
            - recommendation: Suggested approach
    """
    X_np = X_sample.cpu().numpy()
    N = X_np.shape[0]

    # 1. Analyze norm distribution
    norms = np.linalg.norm(X_np, axis=1)
    norm_mean = norms.mean()
    norm_std = norms.std()
    norm_cv = norm_std / (norm_mean + 1e-8)  # Coefficient of variation

    # 2. Compare Euclidean vs Cosine neighbors
    # Euclidean neighbors
    nn_euclidean = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn_euclidean.fit(X_np)
    _, idx_euclidean = nn_euclidean.kneighbors(X_np)
    idx_euclidean = idx_euclidean[:, 1:]  # Exclude self

    # Cosine neighbors (via normalized Euclidean)
    X_normalized = X_np / (np.linalg.norm(X_np, axis=1, keepdims=True) + 1e-8)
    nn_cosine = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn_cosine.fit(X_normalized)
    _, idx_cosine = nn_cosine.kneighbors(X_normalized)
    idx_cosine = idx_cosine[:, 1:]  # Exclude self

    # 3. Compute overlap
    overlaps = []
    for i in range(N):
        set_e = set(idx_euclidean[i])
        set_c = set(idx_cosine[i])
        overlap = len(set_e & set_c) / k
        overlaps.append(overlap)
    mean_overlap = np.mean(overlaps)

    # 4. Recommendation
    if norm_cv < 0.1 and mean_overlap > 0.9:
        recommendation = "EUCLIDEAN_OK"
        reason = "Low norm variation and high neighbor overlap - Euclidean is fine"
    elif norm_cv > 0.3 or mean_overlap < 0.7:
        recommendation = "USE_NORMALIZE"
        reason = f"High norm variation (CV={norm_cv:.3f}) or low overlap ({mean_overlap:.3f}) - consider normalize=True"
    else:
        recommendation = "EITHER_OK"
        reason = "Moderate variation - both metrics should give similar results"

    return {
        'norm_stats': {
            'mean': float(norm_mean),
            'std': float(norm_std),
            'cv': float(norm_cv),
            'min': float(norms.min()),
            'max': float(norms.max()),
        },
        'neighbor_overlap': float(mean_overlap),
        'recommendation': recommendation,
        'reason': reason,
    }


def compute_tangent_space(i, X_sample, knn_graph, d):
    """
    Compute local tangent space basis vectors.

    Args:
        i: Point index
        X_sample: Sample data [N, d_model]
        knn_graph: k-NN indices [N, k]
        d: Intrinsic dimension

    Returns:
        tangent_basis: [d, d_model]
    """
    neighbor_indices = knn_graph[i, :] #(15, )
    neighborhood_points = X_sample[neighbor_indices] # (15, 128)
    centered_neighbors = neighborhood_points - X_sample[i] # (15, 128)

    centered_np = centered_neighbors.cpu().numpy()

    eps = NUMERICAL_EPSILON
    if np.allclose(centered_np, 0, atol=eps):
        centered_np += np.random.normal(0, eps, centered_np.shape)

    U, S, Vh = np.linalg.svd(centered_np, full_matrices=False) # Vh (15, 128); U:(15, 15), S:(15, )
    components = Vh[:d] # (8, 128)

    return torch.tensor(components, dtype=X_sample.dtype, device=X_sample.device)


def sample_general_mlp_data_points(model, train_data, N=200, device='cuda:0',
                                    layer_idx=0, embedding_type='one_hot'):
    """
    Sample data points for new MLP model types (OneHotMLP, EmbedConcatMLP, EmbedSumMLP).

    Args:
        model: MLP model (OneHotMLP, EmbedConcatMLP, or EmbedSumMLP)
        train_data: Training dataset
        N: Number of points
        device: Computing device
        layer_idx: Which layer to extract from
        embedding_type: 'one_hot', 'embed_concat', or 'embed_sum'

    Returns:
        (X_sample, X_prompts): Hidden states and prompts
    """
    total_train_size = len(train_data)
    sample_indices = torch.randperm(total_train_size)[:N]

    sampled_prompts = []
    for idx in sample_indices:
        prompt, _ = train_data[idx]
        sampled_prompts.append(prompt)

    X_prompts = torch.stack(sampled_prompts).to(device).to(torch.float64)

    with torch.no_grad():
        model.eval()
        cache = {}
        model.cache_all(cache, incl_bwd=False)
        _ = model(X_prompts.to(torch.int64))

        # For layer 0 of OneHotMLP, use hook_input (after input_proj)
        if embedding_type == 'one_hot' and layer_idx == 0:
            hook_name = 'hook_input'
        else:
            # Adjust layer index for one_hot (hook_input is "layer -1")
            actual_layer = layer_idx
            if embedding_type == 'one_hot' and layer_idx > 0:
                actual_layer = layer_idx - 1
            hook_name = f'layers.{actual_layer}.hook_post'

        if hook_name in cache:
            X_sample = cache[hook_name]
        else:
            available = [k for k in cache.keys()]
            raise ValueError(f"Hook {hook_name} not found. Available: {available}")

        model.remove_all_hooks()

    return X_sample, X_prompts


def build_complete_geometric_structure(model, train_data, N=200, k=15, d=8,
                                       device='cuda:0', layer_idx=-1,
                                       hook_point='hook_resid_post',
                                       model_type='transformer',
                                       embedding_type=None,
                                       normalize_for_knn=False):
    """
    Build complete geometric structure for GMN-TSA analysis.

    Args:
        model: Model to analyze
        train_data: Training dataset
        N: Number of sample points
        k: Number of nearest neighbors
        d: Intrinsic manifold dimension
        device: Computing device
        layer_idx: Layer index
        hook_point: Hook point name
        model_type: 'transformer', 'mlp', 'onehot_mlp', 'embed_concat_mlp', 'embed_sum_mlp'
        embedding_type: For MLP types, 'one_hot', 'embed_concat', or 'embed_sum'
        normalize_for_knn: If True, normalize representations before k-NN search.
                           This makes the neighborhood based on angular similarity (cosine)
                           rather than Euclidean distance, which may be more appropriate
                           when semantic similarity is directional.

    Returns:
        dict: Complete geometric structure
    """
    print(f"Building geometry: N={N}, k={k}, d={d}")

    # Sample data points
    if model_type in ('onehot_mlp', 'embed_concat_mlp', 'embed_sum_mlp'):
        et = embedding_type or model_type.replace('_mlp', '').replace('onehot', 'one_hot')
        X_sample, X_prompts = sample_general_mlp_data_points(
            model, train_data, N, device, layer_idx, et)
    elif model_type == 'mlp':
        X_sample, X_prompts = sample_mlp_data_points(model, train_data, N, device, layer_idx)
    else:
        X_sample, X_prompts = sample_data_points(model, train_data, N, device, layer_idx, hook_point)

    print(f"  Extracted representations: {X_sample.shape}")

    # Build k-NN graph
    knn_indices, knn_distances = build_knn_graph(X_sample, k, normalize=normalize_for_knn)
    print(f"  Built k-NN graph: {knn_indices.shape} (normalize={normalize_for_knn})")

    # Compute tangent spaces
    all_input_tangent_bases = []
    for i in range(N):
        tangent_basis = compute_tangent_space(i, X_sample, knn_indices, d)
        all_input_tangent_bases.append(tangent_basis)

    print(f"  Computed {N} tangent bases")

    # Precompute embeddings for JVP
    X_embeddings = []
    if model_type == 'transformer':
        with torch.no_grad():
            for i in range(N):
                prompt = X_prompts[i].to(torch.int64)
                embedded = model.embed(prompt.unsqueeze(0))
                X_embeddings.append(embedded.squeeze(0))

    elif model_type == 'onehot_mlp':
        with torch.no_grad():
            from gcs.core.mlp_models import _apply_activation
            for i in range(N):
                prompt = X_prompts[i].to(torch.int64).unsqueeze(0)
                onehot = model.encode_input(prompt)
                projected = _apply_activation(model.input_proj(onehot), model.act_type)
                X_embeddings.append(projected.squeeze(0))  # [hidden_dim]

    elif model_type == 'embed_concat_mlp':
        with torch.no_grad():
            for i in range(N):
                prompt = X_prompts[i].to(torch.int64).unsqueeze(0)
                embedded = model.embed_and_concat(prompt)
                X_embeddings.append(embedded.squeeze(0))  # [2*d_embed]

    elif model_type == 'embed_sum_mlp':
        with torch.no_grad():
            for i in range(N):
                prompt = X_prompts[i].to(torch.int64).unsqueeze(0)
                embedded = model.embed_and_sum(prompt)
                X_embeddings.append(embedded.squeeze(0))  # [d_embed]

    return {
        'X_sample': X_sample,
        'X_prompts': X_prompts,
        'knn_indices': knn_indices,
        'knn_distances': knn_distances,
        'N': N,
        'k': k,
        'd': d,
        'd_model': X_sample.shape[1],
        'X_embeddings': X_embeddings,
        'all_input_tangent_bases': all_input_tangent_bases,
        'layer_idx': layer_idx,
        'hook_point': hook_point,
        'model_type': model_type,
        'embedding_type': embedding_type
    }


def validate_geometric_structure(geometric_data):
    """Validate geometric structure completeness."""
    required_fields = ['d', 'X_embeddings', 'all_input_tangent_bases', 'N', 'k',
                       'X_sample', 'X_prompts', 'knn_indices', 'd_model']

    missing = [f for f in required_fields if f not in geometric_data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
