"""Reference-free cross-input path-Jacobian graph baseline.

The submitted GCS construction compares sample-local tangent maps on a graph
fixed by the final checkpoint.  This module implements the deliberately simpler
Reviewer-2 baseline: probe every sample with the same random directions, build
a checkpoint-local graph from pairwise Jacobian-sketch similarity, and reuse
the existing spectral aggregation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from .gmn_tsa import summarize_coherence_matrix


def make_fixed_random_directions(
    *,
    count: int,
    dimension: int,
    seed: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    """Create deterministic unit Gaussian directions.

    Generation always occurs on CPU so a seed identifies the same directions
    regardless of the CUDA device used for the experiment.
    """
    if count <= 0:
        raise ValueError(f"Expected count > 0, got {count}")
    if dimension <= 0:
        raise ValueError(f"Expected dimension > 0, got {dimension}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    directions = torch.randn(
        count,
        dimension,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    directions = F.normalize(directions, p=2, dim=1)
    return directions.to(device=device, dtype=dtype)


def compute_batched_jacobian_sketch(
    path_forward: Callable[[torch.Tensor], torch.Tensor],
    embeddings: torch.Tensor,
    directions: torch.Tensor,
    *,
    probe_position: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute concatenated normalized JVPs for a batch of samples.

    Args:
        path_forward: Maps ``[N, seq_len, d_model]`` embeddings to one path
            output per sample.
        embeddings: Checkpoint-local embedded inputs.
        directions: Shared ``[d, d_model]`` probe directions.
        probe_position: Token position receiving each direction.

    Returns:
        A ``[N, d * d_out]`` Jacobian sketch. Each ``d_out`` block is
        independently L2-normalized before concatenation.
    """
    if embeddings.ndim != 3:
        raise ValueError(
            "Expected embeddings with shape [N, seq_len, d_model], "
            f"got {tuple(embeddings.shape)}"
        )
    if directions.ndim != 2 or directions.shape[1] != embeddings.shape[2]:
        raise ValueError(
            "Direction shape must be [d, d_model]; got "
            f"{tuple(directions.shape)} for d_model={embeddings.shape[2]}"
        )

    seq_len = int(embeddings.shape[1])
    position = int(probe_position)
    if position < 0:
        position += seq_len
    if position < 0 or position >= seq_len:
        raise ValueError(
            f"probe_position={probe_position} is invalid for seq_len={seq_len}"
        )

    normalized_jvps: list[torch.Tensor] = []
    for direction in directions:
        tangent = torch.zeros_like(embeddings)
        tangent[:, position, :] = direction.unsqueeze(0)
        _, transformed = torch.autograd.functional.jvp(
            path_forward,
            embeddings,
            tangent,
            create_graph=False,
            strict=False,
        )
        if transformed.ndim == 1:
            transformed = transformed.unsqueeze(0)
        if transformed.ndim != 2 or transformed.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "Expected one flat path output per sample; got transformed "
                f"shape {tuple(transformed.shape)}"
            )
        normalized_jvps.append(F.normalize(transformed, p=2, dim=1, eps=eps))

    return torch.cat(normalized_jvps, dim=1)


def build_topk_similarity_graph(
    sketches: torch.Tensor,
    *,
    k: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the union-symmetrized top-k absolute-cosine graph."""
    if sketches.ndim != 2:
        raise ValueError(f"Expected sketches [N, features], got {tuple(sketches.shape)}")
    n = int(sketches.shape[0])
    if n < 2:
        raise ValueError("At least two samples are required")
    if k <= 0 or k >= n:
        raise ValueError(f"Expected 0 < k < N, got k={k}, N={n}")

    with torch.no_grad():
        normalized = F.normalize(sketches, p=2, dim=1, eps=eps)
        similarity = torch.abs(normalized @ normalized.T).clamp_(0.0, 1.0)

        candidates = similarity.clone()
        candidates.fill_diagonal_(float("-inf"))
        neighbors = torch.topk(
            candidates,
            k=k,
            dim=1,
            largest=True,
            sorted=True,
        ).indices

        directed = torch.zeros((n, n), dtype=torch.bool, device=sketches.device)
        directed.scatter_(1, neighbors, True)
        union = directed | directed.T

        graph = torch.where(union, similarity, torch.zeros_like(similarity))
        graph.fill_diagonal_(1.0)

    return graph, neighbors, similarity


def summarize_direct_jacobian_graph(
    sketches: torch.Tensor,
    *,
    k: int,
    include_matrices: bool = False,
) -> dict[str, Any]:
    """Apply the submitted spectral summary to a direct-Jacobian graph."""
    graph, neighbors, similarity = build_topk_similarity_graph(sketches, k=k)
    graph_cpu = graph.detach().to(device="cpu", dtype=torch.float64)
    neighbors_cpu = neighbors.detach().to(device="cpu")
    summary = dict(
        summarize_coherence_matrix(
            graph_cpu,
            knn_indices=neighbors_cpu,
            N=int(graph_cpu.shape[0]),
        )
    )
    summary["direct_jacobian_score"] = float(summary["gcs_score"])
    summary["graph_k"] = int(k)
    if include_matrices:
        summary["coherence_matrix"] = graph_cpu.numpy()
        summary["similarity_matrix"] = (
            similarity.detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        summary["knn_indices"] = neighbors_cpu.numpy()
    return summary
