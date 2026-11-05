"""Independent checks for changes that would alter the paper's definition."""
import importlib

import numpy as np
import pytest
import torch


def method_module():
    try:
        return importlib.import_module('gcs.analysis.gmn_tsa')
    except ModuleNotFoundError:
        pytest.fail('The standalone GCS method package is not available')


def test_asymmetric_neighbor_relation_is_included_once():
    m = method_module()
    assert m._iter_union_edges([[1], [], [1, 2]]) == [(0, 1), (1, 2)]


@pytest.mark.parametrize('matrix,want', [(np.eye(3), 0.0), (np.ones((3, 3)), 2.0)])
def test_spectral_effective_rank_extremes(matrix, want):
    result = method_module().summarize_coherence_matrix(matrix)
    assert result['gcs_score'] == pytest.approx(want, abs=1e-12)


def test_rankwise_absolute_cosine_and_zero_response():
    m = method_module()
    a = torch.tensor([[1., 0.], [0., 2.]], dtype=torch.float64)
    assert m.compute_alignment_score_cosine(a, -a).item() == pytest.approx(1.)
    assert m.compute_alignment_score_cosine(a, torch.zeros_like(a)).item() == 0.
    # Permuting ranks is not an invariant of this statistic.
    assert m.compute_alignment_score_cosine(a, a.flip(0)).item() == 0.
