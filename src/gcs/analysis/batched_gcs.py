"""Batch-equivalent GCS computation for sample-independent model paths."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .gmn_tsa import (
    _iter_union_edges,
    _token_probe_positions,
    compute_alignment_score_procrustes,
    compute_alignment_score_rank_permuted,
    summarize_coherence_matrix,
)
from .paths import create_path_forward_function


def compute_batched_transformed_bases(
    path_forward: Callable[[torch.Tensor], torch.Tensor],
    embeddings: torch.Tensor,
    tangent_bases: torch.Tensor,
    *,
    token_probe: str = "final",
) -> torch.Tensor:
    """Transform sample-local tangent bases using one batched JVP per rank.

    Model paths in this project do not mix examples across the batch dimension.
    A JVP over all samples is therefore exactly the stack of the corresponding
    per-sample JVPs, while requiring ``d`` calls rather than ``N * d`` calls.
    """
    if embeddings.ndim != 3:
        raise ValueError(
            "Expected embeddings [N, seq_len, d_model], got "
            f"{tuple(embeddings.shape)}"
        )
    if tangent_bases.ndim != 3:
        raise ValueError(
            "Expected tangent_bases [N, d, d_model], got "
            f"{tuple(tangent_bases.shape)}"
        )
    if embeddings.shape[0] != tangent_bases.shape[0]:
        raise ValueError("Embeddings and tangent bases must use the same N")
    if embeddings.shape[2] != tangent_bases.shape[2]:
        raise ValueError("Embedding and tangent-basis dimensions must match")

    probe_positions = _token_probe_positions(token_probe, int(embeddings.shape[1]))
    transformed_by_rank: list[torch.Tensor] = []
    for rank in range(int(tangent_bases.shape[1])):
        transformed_by_position: list[torch.Tensor] = []
        for position in probe_positions:
            tangent = torch.zeros_like(embeddings)
            tangent[:, position, :] = tangent_bases[:, rank, :]
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
                    "Expected one flat path output per sample, got "
                    f"{tuple(transformed.shape)}"
                )
            transformed_by_position.append(transformed)

        if len(transformed_by_position) == 1:
            transformed_rank = transformed_by_position[0]
        else:
            transformed_rank = torch.stack(
                transformed_by_position,
                dim=0,
            ).mean(dim=0)
        transformed_by_rank.append(transformed_rank)

    return torch.stack(transformed_by_rank, dim=1)


def build_coherence_matrix(
    transformed_bases: torch.Tensor,
    knn_indices: Sequence[Sequence[int]] | np.ndarray,
    *,
    alignment_method: str = "cosine",
    input_tangent_bases: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the submitted union-kNN coherence matrix."""
    if transformed_bases.ndim != 3:
        raise ValueError(
            "Expected transformed_bases [N, d, d_out], got "
            f"{tuple(transformed_bases.shape)}"
        )
    n = int(transformed_bases.shape[0])
    if len(knn_indices) != n:
        raise ValueError("knn_indices and transformed_bases must use the same N")

    matrix = torch.eye(
        n,
        dtype=torch.float64,
        device=transformed_bases.device,
    )
    edges = _iter_union_edges(knn_indices)
    if not edges:
        return matrix

    if alignment_method == "cosine":
        edge_i = torch.as_tensor(
            [edge[0] for edge in edges],
            dtype=torch.long,
            device=transformed_bases.device,
        )
        edge_j = torch.as_tensor(
            [edge[1] for edge in edges],
            dtype=torch.long,
            device=transformed_bases.device,
        )
        left = F.normalize(transformed_bases[edge_i], p=2, dim=2)
        right = F.normalize(transformed_bases[edge_j], p=2, dim=2)
        alignments = torch.abs((left * right).sum(dim=2)).mean(dim=1)
        matrix[edge_i, edge_j] = alignments.to(dtype=matrix.dtype)
        matrix[edge_j, edge_i] = alignments.to(dtype=matrix.dtype)
        return matrix

    if alignment_method == "procrustes" and input_tangent_bases is None:
        raise ValueError("Procrustes alignment requires input_tangent_bases")
    if alignment_method not in {"procrustes", "rank_permuted"}:
        raise ValueError(f"Unknown alignment_method={alignment_method!r}")

    for i, j in edges:
        if alignment_method == "procrustes":
            alignment = compute_alignment_score_procrustes(
                input_tangent_bases[i],
                input_tangent_bases[j],
                transformed_bases[i],
                transformed_bases[j],
            )
        else:
            alignment = compute_alignment_score_rank_permuted(
                transformed_bases[i],
                transformed_bases[j],
            )
        matrix[i, j] = alignment.item()
        matrix[j, i] = alignment.item()
    return matrix


def compute_path_gcs_batched(
    model: Any,
    geometric_data: dict[str, Any],
    start_hook: str,
    end_hook: str,
    *,
    target_layer: int = 0,
    device: str = "cuda:0",
    alignment_method: str = "cosine",
    token_probe: str = "final",
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Compute GCS with batch-equivalent JVPs on a fixed geometry."""
    model.to(device)
    model.eval()
    dtype = next(model.parameters()).dtype

    embeddings = torch.stack(
        [
            value.to(device=device, dtype=dtype)
            for value in geometric_data["X_embeddings"]
        ],
        dim=0,
    )
    tangent_bases = torch.stack(
        [
            value.to(device=device, dtype=dtype)
            for value in geometric_data["all_input_tangent_bases"]
        ],
        dim=0,
    )
    expected_n = int(geometric_data["N"])
    expected_d = int(geometric_data["d"])
    if embeddings.shape[0] != expected_n or tangent_bases.shape[:2] != (
        expected_n,
        expected_d,
    ):
        raise ValueError(
            "Geometry metadata does not match saved embeddings/tangent bases"
        )

    path_forward = create_path_forward_function(
        model,
        start_hook,
        end_hook,
        target_layer=target_layer,
    )
    transformed = compute_batched_transformed_bases(
        path_forward,
        embeddings,
        tangent_bases,
        token_probe=token_probe,
    )
    matrix = build_coherence_matrix(
        transformed,
        geometric_data["knn_indices"],
        alignment_method=alignment_method,
        input_tangent_bases=tangent_bases,
    )
    summary = dict(
        summarize_coherence_matrix(
            matrix,
            knn_indices=geometric_data["knn_indices"],
            N=expected_n,
        )
    )
    if return_details:
        summary["coherence_matrix"] = matrix.detach().cpu().numpy()
        summary["alignment_method"] = alignment_method
        summary["batched_jvp"] = True
        return summary
    return float(summary["gcs_score"])
