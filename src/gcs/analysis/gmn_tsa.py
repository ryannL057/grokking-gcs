"""
GMN-TSA (Geometric Manifold Network Tangent Space Analysis) core algorithms.

Computes GCS (Geometric Coherence Score = N - GMN-TSA).
"""

import torch
import torch.nn.functional as F
import numpy as np

from gcs.core.constants import EIGENVALUE_THRESHOLD, SYMMETRY_TOLERANCE


def _iter_union_edges(knn_indices):
    """
    Yield unique unordered edges from the union of directed k-NN relations.

    This matches the paper definition: include edge (i, j) if j is in i's
    neighborhood or i is in j's neighborhood, while computing each unordered
    pair only once.
    """
    edges = set()
    for i in range(len(knn_indices)):
        for j in knn_indices[i]:
            j = int(j)
            if i == j:
                continue
            edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def summarize_coherence_matrix(connectivity_matrix, *, knn_indices=None, N=None):
    """
    Summarize a coherence/connectivity matrix with both spectral and edge-level
    aggregations.

    Returns a dictionary containing the paper's entropy-based aggregation
    (`gmn_tsa_score`, `gcs_score`) alongside simpler baselines such as mean
    union-edge coherence and spectral participation rank.
    """
    if torch.is_tensor(connectivity_matrix):
        matrix_np = connectivity_matrix.detach().cpu().numpy()
    else:
        matrix_np = np.asarray(connectivity_matrix)

    size = int(N if N is not None else matrix_np.shape[0])
    eigenvals = np.linalg.eigvals(matrix_np)
    eigenvals = np.abs(eigenvals)
    eigenvals_sum = float(np.sum(eigenvals))

    if eigenvals_sum > 0:
        p = eigenvals / eigenvals_sum
    else:
        p = np.ones(len(eigenvals), dtype=np.float64) / max(len(eigenvals), 1)

    p_nonzero = p[p > EIGENVALUE_THRESHOLD]
    spectral_entropy = -np.sum(p_nonzero * np.log2(p_nonzero)) if p_nonzero.size else 0.0
    gmn_tsa_score = float(2 ** spectral_entropy)
    gcs_score = float(size - gmn_tsa_score)

    participation_denom = float(np.sum(eigenvals ** 2))
    if participation_denom > 0:
        participation_rank = float((eigenvals_sum ** 2) / participation_denom)
    else:
        participation_rank = 1.0

    if knn_indices is not None:
        edges = _iter_union_edges(knn_indices)
        edge_values = np.array([matrix_np[i, j] for i, j in edges], dtype=np.float64)
    else:
        tri_i, tri_j = np.triu_indices(size, k=1)
        edge_values = matrix_np[tri_i, tri_j].astype(np.float64, copy=False)

    mean_edge_coherence = float(edge_values.mean()) if edge_values.size else 1.0
    median_edge_coherence = float(np.median(edge_values)) if edge_values.size else 1.0

    return {
        'gcs_score': gcs_score,
        'gmn_tsa_score': gmn_tsa_score,
        'spectral_entropy': float(spectral_entropy),
        'participation_rank': participation_rank,
        'rank': int(np.sum(eigenvals > EIGENVALUE_THRESHOLD)),
        'mean_edge_coherence': mean_edge_coherence,
        'median_edge_coherence': median_edge_coherence,
        'max_eigenvalue': float(np.max(eigenvals)) if eigenvals.size else 0.0,
        'min_eigenvalue': float(np.min(eigenvals)) if eigenvals.size else 0.0,
        'mean_eigenvalue': float(np.mean(eigenvals)) if eigenvals.size else 0.0,
        'std_eigenvalue': float(np.std(eigenvals)) if eigenvals.size else 0.0,
        'edge_count': int(edge_values.size),
    }


def _orthonormalize_basis(basis):
    """
    Orthonormalize basis vectors using QR decomposition.

    Args:
        basis: [d, d_out] matrix where rows are basis vectors

    Returns:
        Q: [d, d_out] orthonormal basis (rows are orthonormal)
    """
    # QR decomposition: basis.T = Q @ R, so basis = R.T @ Q.T
    # We want orthonormal rows, so we work with basis.T
    Q, R = torch.linalg.qr(basis.T, mode='reduced')  # Q: [d_out, d], R: [d, d]
    return Q.T  # [d, d_out] with orthonormal rows


def compute_procrustes_alignment(basis_i, basis_j):
    """
    Compute optimal rotation matrix R to align basis_j to basis_i.

    Solves: R = argmin_R ||basis_i - R @ basis_j||_F
    Solution: M = basis_j @ basis_i.T, U, S, Vh = svd(M), R = U @ Vh

    Args:
        basis_i: [d, d_out] input tangent basis at point i
        basis_j: [d, d_out] input tangent basis at point j

    Returns:
        R: [d, d] optimal rotation matrix in tangent space
    """
    # M = basis_j @ basis_i.T, shape: [d, d]
    M = torch.mm(basis_j, basis_i.T)
    U, S, Vh = torch.linalg.svd(M)
    R = torch.mm(U, Vh)  # [d, d]
    return R


def compute_alignment_score_cosine(transformed_i, transformed_j):
    """
    Compute alignment score using row-wise cosine similarity (original method).

    Directly compares transformed tangent basis vectors row-by-row.

    Args:
        transformed_i: [d, d_out] transformed tangent basis at point i
        transformed_j: [d, d_out] transformed tangent basis at point j

    Returns:
        score: Mean absolute cosine similarity (scalar in [0, 1])
    """
    similarities = F.cosine_similarity(transformed_i, transformed_j, dim=1).abs()
    return similarities.mean()


def _fixed_rank_derangement(d, device):
    """Return a deterministic rank permutation with no fixed points when possible."""
    if d <= 1:
        return torch.arange(d, device=device)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(1729 + int(d))
    for _ in range(128):
        perm_cpu = torch.randperm(d, generator=generator)
        if bool(torch.all(perm_cpu != torch.arange(d))):
            return perm_cpu.to(device=device)

    return torch.roll(torch.arange(d, device=device), shifts=1)


def compute_alignment_score_rank_permuted(transformed_i, transformed_j):
    """
    Negative-control alignment that breaks SVD rank correspondence.

    The transformed vectors are otherwise unchanged; only the rank index used
    to match the neighbor basis is permuted. This tests whether a trajectory is
    mainly an artifact of rank-wise matching.
    """
    perm = _fixed_rank_derangement(transformed_j.shape[0], transformed_j.device)
    similarities = F.cosine_similarity(transformed_i, transformed_j[perm], dim=1).abs()
    return similarities.mean()


def compute_alignment_score_procrustes(basis_i, basis_j, transformed_i, transformed_j):
    """
    Compute Procrustes-aligned GCS score.

    This method first finds the optimal rotation R that aligns the input
    tangent bases, then measures the similarity of the transformed bases
    after applying this alignment. This preserves sensitivity to
    transformation differences while being robust to input basis ordering.

    Args:
        basis_i: [d, d_out] input tangent basis at point i
        basis_j: [d, d_out] input tangent basis at point j
        transformed_i: [d, d_out] transformed tangent basis at point i
        transformed_j: [d, d_out] transformed tangent basis at point j

    Returns:
        score: Alignment score after Procrustes alignment (scalar in [0, 1])
    """
    # Step 1: Compute optimal alignment from input bases
    R = compute_procrustes_alignment(basis_i, basis_j)  # [d, d]

    # Step 2: Align neighbor's transformed basis
    # transformed_j_aligned = R.T @ transformed_j (rotate in d-space)
    transformed_j_aligned = torch.mm(R.T, transformed_j)  # [d, d_out]

    # Step 3: Compute Frobenius cosine similarity
    numerator = (transformed_i * transformed_j_aligned).sum()
    norm_i = torch.norm(transformed_i)
    norm_j = torch.norm(transformed_j_aligned)

    score = numerator / (norm_i * norm_j + 1e-8)
    return score.clamp(0.0, 1.0)


def compute_principal_angles(basis1, basis2):
    """
    Compute principal angles between two subspaces (for analysis/debugging).

    Args:
        basis1: First tangent space basis [d, d_out]
        basis2: Second tangent space basis [d, d_out]

    Returns:
        angles: Principal angles in radians [d]
        cosines: Cosines of principal angles [d]
    """
    # Orthonormalize both bases
    Q1 = _orthonormalize_basis(basis1)
    Q2 = _orthonormalize_basis(basis2)

    M = torch.mm(Q1, Q2.T)
    singular_values = torch.linalg.svdvals(M)
    cosines = torch.clamp(singular_values, -1.0, 1.0)
    angles = torch.acos(cosines)
    return angles, cosines


def _token_probe_positions(token_probe: str, seq_len: int) -> list[int]:
    if seq_len <= 0:
        raise ValueError(f"Expected positive seq_len, got {seq_len}")

    if token_probe == 'final':
        return [seq_len - 1]
    if token_probe == 'operands':
        if seq_len < 2:
            raise ValueError(f"Operand probe requires seq_len >= 2, got {seq_len}")
        return [0, 1]
    if token_probe == 'all':
        return list(range(seq_len))

    raise ValueError(f"Unknown token_probe={token_probe!r}")


def compute_path_gmn_tsa(model, geometric_data, start_hook, end_hook,
                         target_layer=0, device='cuda:0', alignment_method='cosine',
                         token_probe='final', return_details=False):
    """
    Compute GCS (Geometric Coherence Score = N - GMN-TSA) for a path.

    Args:
        model: Transformer model
        geometric_data: Geometric structure data
        start_hook: Starting hook point
        end_hook: Ending hook point
        target_layer: Target layer index
        device: Computing device
        alignment_method: 'cosine' (original row-wise), 'procrustes'
            (Procrustes-aligned), or 'rank_permuted' (negative control)
        token_probe: which embedding positions receive the JVP tangent.
            'final' = last token only (paper default),
            'operands' = average of per-operand JVPs,
            'all' = average of per-token JVPs

    Returns:
        gcs_score by default, or a detail dictionary when return_details=True
    """
    from .paths import create_path_forward_function

    model.to(device)
    model.eval()

    N = geometric_data['N']
    d = geometric_data['d']
    X_embeddings = geometric_data['X_embeddings'] # 200个（3， 128）的张量组成的 list
    all_input_tangent_bases = geometric_data['all_input_tangent_bases']  # 200个(8, 128)的切向量组成的list
    knn_indices = geometric_data['knn_indices'] # (200, 15)

    path_forward = create_path_forward_function(model, start_hook, end_hook, target_layer)

    probe_positions = _token_probe_positions(token_probe, X_embeddings[0].shape[0])

    # Compute transformed tangent spaces via JVP
    all_transformed_bases = []
    for i in range(N):
        basis = all_input_tangent_bases[i] # (8, 128)
        transformed_basis = []

        for v in basis:
            transformed_vs = []
            for pos in probe_positions:
                direction = torch.zeros_like(X_embeddings[i]) # (3, 128)
                direction[pos, :] = v

                _, transformed_v = torch.autograd.functional.jvp(
                    path_forward,
                    X_embeddings[i], # (3, 128)
                    direction # (3, 128)
                )
                if transformed_v.dim() > 1:
                    transformed_v = transformed_v.squeeze()
                transformed_vs.append(transformed_v)

            if len(transformed_vs) == 1:
                transformed_v = transformed_vs[0]
            else:
                transformed_v = torch.stack(transformed_vs, dim=0).mean(dim=0)
            transformed_basis.append(transformed_v)

        all_transformed_bases.append(torch.stack(transformed_basis))

    # Build geodesic connectivity matrix G
    G = torch.eye(N, dtype=torch.float64, device=device)

    for i, j in _iter_union_edges(knn_indices):
        if alignment_method == 'procrustes':
            alignment = compute_alignment_score_procrustes(
                all_input_tangent_bases[i],
                all_input_tangent_bases[j],
                all_transformed_bases[i],
                all_transformed_bases[j]
            )
        elif alignment_method == 'rank_permuted':
            alignment = compute_alignment_score_rank_permuted(
                all_transformed_bases[i],
                all_transformed_bases[j]
            )
        else:  # 'cosine' (default, matches original behavior)
            alignment = compute_alignment_score_cosine(
                all_transformed_bases[i],
                all_transformed_bases[j]
            )
        G[i, j] = alignment.item()
        G[j, i] = alignment.item()

    summary = summarize_coherence_matrix(G, knn_indices=knn_indices, N=N)
    if return_details:
        summary = dict(summary)
        summary['coherence_matrix'] = G.detach().cpu().numpy()
        summary['alignment_method'] = alignment_method
        return summary
    return summary if return_details else summary['gcs_score']


def compute_mlp_path_gcs(model, geometric_data, path_forward_fn, device='cuda:0',
                          alignment_method='cosine', return_details=False):
    """
    Compute GCS for an MLP path.

    Like compute_path_gmn_tsa but for 1D flat embeddings (no last-position trick).

    Args:
        model: MLP model
        geometric_data: Geometric structure data
        path_forward_fn: Callable that maps embedded input to output
        device: Computing device
        alignment_method: 'cosine', 'procrustes', or 'rank_permuted'

    Returns:
        gcs_score by default, or a detail dictionary when return_details=True
    """
    model.to(device)
    model.eval()

    N = geometric_data['N']
    d = geometric_data['d']
    X_embeddings = geometric_data['X_embeddings']
    all_input_tangent_bases = geometric_data['all_input_tangent_bases']
    knn_indices = geometric_data['knn_indices']

    # Compute transformed tangent spaces via JVP
    all_transformed_bases = []
    for i in range(N):
        basis = all_input_tangent_bases[i]  # [d, embedding_dim]
        transformed_basis = []

        for v in basis:
            # For MLP: tangent direction is applied directly (1D, not 2D)
            direction = v  # [embedding_dim]

            # Handle dimension mismatch: tangent basis may have different dim
            # than embedding if geometry was built at a different layer
            emb = X_embeddings[i]  # [embedding_dim]
            if direction.shape[0] != emb.shape[0]:
                # Pad or truncate direction to match embedding dim
                if direction.shape[0] > emb.shape[0]:
                    direction = direction[:emb.shape[0]]
                else:
                    padded = torch.zeros_like(emb)
                    padded[:direction.shape[0]] = direction
                    direction = padded

            _, transformed_v = torch.autograd.functional.jvp(
                path_forward_fn,
                emb,
                direction
            )

            if transformed_v.dim() > 1:
                transformed_v = transformed_v.squeeze()
            transformed_basis.append(transformed_v)

        all_transformed_bases.append(torch.stack(transformed_basis))

    # Build geodesic connectivity matrix G
    G = torch.eye(N, dtype=torch.float64, device=device)

    for i, j in _iter_union_edges(knn_indices):
        if alignment_method == 'procrustes':
            alignment = compute_alignment_score_procrustes(
                all_input_tangent_bases[i],
                all_input_tangent_bases[j],
                all_transformed_bases[i],
                all_transformed_bases[j]
            )
        elif alignment_method == 'rank_permuted':
            alignment = compute_alignment_score_rank_permuted(
                all_transformed_bases[i],
                all_transformed_bases[j]
            )
        else:
            alignment = compute_alignment_score_cosine(
                all_transformed_bases[i],
                all_transformed_bases[j]
            )
        G[i, j] = alignment.item()
        G[j, i] = alignment.item()

    summary = summarize_coherence_matrix(G, knn_indices=knn_indices, N=N)
    if return_details:
        summary = dict(summary)
        summary['coherence_matrix'] = G.detach().cpu().numpy()
        summary['alignment_method'] = alignment_method
        return summary
    return summary if return_details else summary['gcs_score']


def compute_eigenvalue_spectrum(connectivity_matrix):
    """
    Analyze eigenvalue spectrum of connectivity matrix.

    Returns:
        dict: Spectrum analysis results
    """
    if torch.is_tensor(connectivity_matrix):
        matrix_np = connectivity_matrix.cpu().numpy()
    else:
        matrix_np = connectivity_matrix

    eigenvals = np.linalg.eigvals(matrix_np)
    eigenvals = np.abs(eigenvals)

    results = summarize_coherence_matrix(matrix_np, N=matrix_np.shape[0])
    results['eigenvalues'] = np.sort(eigenvals)[::-1]
    return results


def compute_gcs_statistics(gcs_scores):
    """Compute statistics for a series of GCS scores."""
    valid_scores = [s for s in gcs_scores if s is not None]

    if not valid_scores:
        return {'count': 0, 'mean': None, 'std': None}

    valid_scores = np.array(valid_scores)

    return {
        'count': len(valid_scores),
        'mean': np.mean(valid_scores),
        'std': np.std(valid_scores),
    }
