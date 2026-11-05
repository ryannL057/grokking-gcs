"""Helpers for controlled reference-geometry variants."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .geometry import build_knn_graph, compute_tangent_space


def count_knn_components(knn_indices: Sequence[Sequence[int]] | np.ndarray) -> int:
    """Count components after union-symmetrizing directed k-NN edges."""
    n = len(knn_indices)
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i, neighbors in enumerate(knn_indices):
        for raw_j in neighbors:
            j = int(raw_j)
            if j < 0 or j >= n:
                raise ValueError(f"Neighbor index {j} is outside [0, {n})")
            if i == j:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)

    unseen = set(range(n))
    components = 0
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            reached = adjacency[node] & unseen
            unseen.difference_update(reached)
            stack.extend(reached)
    return components


def _validate_dimensions(*, available_n: int, N: int, k: int, d: int, d_model: int) -> None:
    if N <= 1 or N > available_n:
        raise ValueError(f"Expected 1 < N <= {available_n}, got {N}")
    if k <= 0 or k >= N:
        raise ValueError(f"Expected 0 < k < N, got k={k}, N={N}")
    if d <= 0 or d > min(k, d_model):
        raise ValueError(
            f"Expected 0 < d <= min(k, d_model), got d={d}, "
            f"k={k}, d_model={d_model}"
        )


def _compute_tangent_bases(
    x_sample: torch.Tensor,
    knn_indices: np.ndarray,
    d: int,
) -> list[torch.Tensor]:
    return [
        compute_tangent_space(
            i=index,
            X_sample=x_sample,
            knn_graph=knn_indices,
            d=d,
        )
        for index in range(int(x_sample.shape[0]))
    ]


def rebuild_geometry_variant(
    fixed_geometry: dict[str, Any],
    *,
    N: int,
    k: int,
    d: int,
    device: str,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Rebuild k-NN/tangent structure while retaining an exact prompt prefix."""
    available_n = int(fixed_geometry["N"])
    source_sample = fixed_geometry["X_sample"]
    target_dtype = dtype or source_sample.dtype
    d_model = int(source_sample.shape[1])
    _validate_dimensions(
        available_n=available_n,
        N=int(N),
        k=int(k),
        d=int(d),
        d_model=d_model,
    )

    geometry = copy.deepcopy(fixed_geometry)
    x_sample = fixed_geometry["X_sample"][:N].to(
        device=device,
        dtype=target_dtype,
    )
    x_prompts = fixed_geometry["X_prompts"][:N].to(device=device)
    x_embeddings = [
        value.to(device=device, dtype=target_dtype)
        for value in fixed_geometry["X_embeddings"][:N]
    ]
    knn_indices, knn_distances = build_knn_graph(x_sample, k=k)
    tangent_bases = _compute_tangent_bases(x_sample, knn_indices, d)

    geometry.update(
        {
            "X_sample": x_sample,
            "X_prompts": x_prompts,
            "X_embeddings": x_embeddings,
            "knn_indices": knn_indices,
            "knn_distances": knn_distances,
            "all_input_tangent_bases": tangent_bases,
            "N": int(N),
            "k": int(k),
            "d": int(d),
            "d_model": d_model,
            "prompt_subset_rule": "canonical_saved_order_prefix",
            "knn_components": count_knn_components(knn_indices),
        }
    )
    return geometry


def extract_block_outputs(
    model: Any,
    prompts: torch.Tensor,
    *,
    layer_idx: int,
    device: str,
    hook_point: str = "hook_resid_post",
) -> torch.Tensor:
    """Extract final-token block outputs for an exact saved prompt cohort."""
    prompts_device = prompts.to(device=device, dtype=torch.int64)
    cache: dict[str, torch.Tensor] = {}
    model.eval()
    model.cache_all(cache, incl_bwd=False)
    try:
        with torch.no_grad():
            model(prompts_device)
        key = f"blocks.{int(layer_idx)}.{hook_point}"
        if key not in cache:
            raise KeyError(f"Missing hook {key}; available={sorted(cache)}")
        return cache[key][:, -1, :].detach()
    finally:
        model.remove_all_hooks()


def build_reference_geometry_from_prompts(
    model: Any,
    prompts: torch.Tensor,
    *,
    N: int,
    k: int,
    d: int,
    layer_idx: int,
    device: str,
    hook_point: str = "hook_resid_post",
) -> dict[str, Any]:
    """Build a geometry at a chosen layer without resampling prompts."""
    if prompts.ndim != 2:
        raise ValueError(f"Expected prompts [N, seq_len], got {tuple(prompts.shape)}")
    dtype = next(model.parameters()).dtype
    selected_prompts = prompts[:N].to(device=device)
    x_sample = extract_block_outputs(
        model,
        selected_prompts,
        layer_idx=layer_idx,
        device=device,
        hook_point=hook_point,
    ).to(dtype=dtype)
    d_model = int(x_sample.shape[1])
    _validate_dimensions(
        available_n=int(prompts.shape[0]),
        N=int(N),
        k=int(k),
        d=int(d),
        d_model=d_model,
    )

    knn_indices, knn_distances = build_knn_graph(x_sample, k=k)
    tangent_bases = _compute_tangent_bases(x_sample, knn_indices, d)
    with torch.no_grad():
        embedded = model.embed(selected_prompts.to(dtype=torch.int64)).detach()
    x_embeddings = [value for value in embedded]

    return {
        "X_sample": x_sample,
        "X_prompts": selected_prompts,
        "knn_indices": knn_indices,
        "knn_distances": knn_distances,
        "N": int(N),
        "k": int(k),
        "d": int(d),
        "d_model": d_model,
        "X_embeddings": x_embeddings,
        "all_input_tangent_bases": tangent_bases,
        "layer_idx": int(layer_idx),
        "hook_point": hook_point,
        "model_type": "transformer",
        "embedding_type": None,
        "prompt_subset_rule": "caller_saved_order_prefix",
        "knn_components": count_knn_components(knn_indices),
    }
