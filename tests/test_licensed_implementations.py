"""Contracts at the licensed backend boundary, independent of old source code."""
import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def licensed_model(**overrides):
    try:
        module = importlib.import_module('gcs.core.lens_model')
    except ModuleNotFoundError:
        pytest.fail('The licensed Transformer backend has not been integrated')
    config = dict(num_layers=1, d_vocab=8, d_model=8, d_mlp=16,
                  d_head=4, num_heads=2, n_ctx=3, act_type='GeLU', attention_rate=.35)
    return module.Transformer(**dict(config, **overrides)).double()


def fourier():
    try:
        return importlib.import_module('gcs.analysis.fourier')
    except ModuleNotFoundError:
        pytest.fail('The MIT-demo Fourier implementation has not been integrated')


def test_checkpoint_roundtrip_rejects_missing_weights_and_preserves_dtype():
    model = licensed_model(use_ln=True)
    state = model.state_dict()
    copy = licensed_model(use_ln=True)
    copy.load_state_dict(state)
    tokens = torch.tensor([[1, 2, 7], [3, 4, 7]])
    torch.testing.assert_close(model(tokens), copy(tokens), rtol=0, atol=0)
    assert state['embed.W_E'].shape == (8, 8)
    assert state['blocks.0.attn.W_Q'].shape == (2, 4, 8)
    assert state['blocks.0.mlp.W_in'].shape == (16, 8)
    del state['blocks.0.attn.W_Q']
    with pytest.raises(RuntimeError, match='missing|Missing'):
        copy.load_state_dict(state)
    incompatible = copy.load_state_dict(state, strict=False)
    assert incompatible.missing_keys == ['blocks.0.attn.W_Q']
    assert model.float()(tokens).dtype == torch.float32


@pytest.mark.parametrize('strict', [True, False])
def test_checkpoint_rejects_reshaped_projection_before_converting_axes(strict):
    model = licensed_model()
    state = model.state_dict()
    state['blocks.0.attn.W_O'] = state['blocks.0.attn.W_O'].flatten()
    with pytest.raises(RuntimeError, match='shape|size'):
        model.load_state_dict(state, strict=strict)


def test_onehot_geometry_uses_the_relocated_activation_helper():
    from gcs.core.models import OneHotMLP
    from gcs.analysis.geometry import build_complete_geometric_structure
    torch.manual_seed(42)
    model = OneHotMLP(8, hidden_dim=8, num_layers=1).double()
    prompts = torch.tensor([[a, b, 7] for a in range(7) for b in range(7)])
    data = torch.utils.data.TensorDataset(prompts, prompts[:, :2].sum(1) % 7)
    geometry = build_complete_geometric_structure(model, data, N=6, k=3, d=2,
        layer_idx=0, model_type='onehot_mlp', device='cpu')
    assert geometry['N'] == 6
    assert torch.isfinite(torch.stack(geometry['X_embeddings'])).all()


def test_native_and_legacy_hooks_share_patch_and_cleanup_lifecycle():
    model = licensed_model()
    tokens = torch.tensor([[1, 2, 7], [3, 4, 7]])
    baseline = model(tokens)
    point = model.blocks[0].attn.hook_z
    seen = {}
    def legacy_patch(value, name):
        seen[name] = tuple(value.shape)
        return value * .5
    point.add_hook(legacy_patch)
    patched = model(tokens)
    point.remove_hooks('fwd')
    torch.testing.assert_close(model(tokens), baseline, rtol=0, atol=0)
    with model.native.hooks(fwd_hooks=[('blocks.0.attn.hook_z', lambda value, hook: value * .5)]):
        torch.testing.assert_close(model(tokens), patched, rtol=0, atol=0)
    assert seen == {'blocks.0.attn.hook_z': (2, 2, 3, 4)}
    assert not torch.equal(patched, baseline)


def test_activation_patch_runner_uses_compatible_hook_names():
    from gcs.legacy.activation_patch_commutative import run_model_with_optional_patch
    host, donor = licensed_model(), licensed_model()
    tokens = torch.tensor([[1, 2, 7]])
    baseline = host(tokens).detach()
    observed = run_model_with_optional_patch(host, donor, tokens, hook_name='blocks.0.hook_attn_out')
    assert not torch.equal(observed, baseline)
    torch.testing.assert_close(host(tokens), baseline, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['transformer', 'onehot', 'concat', 'sum', 'simple'])
def test_forward_and_backward_caches_are_removed(kind):
    from gcs.core.models import OneHotMLP, EmbedConcatMLP, EmbedSumMLP, SimpleMLP
    constructors = dict(onehot=lambda: OneHotMLP(8, hidden_dim=8, num_layers=1),
                        concat=lambda: EmbedConcatMLP(8, d_embed=4, num_layers=1),
                        sum=lambda: EmbedSumMLP(8, d_embed=8, num_layers=1),
                        simple=lambda: SimpleMLP(8, hidden_dim=4, num_layers=1))
    model = licensed_model() if kind == 'transformer' else constructors[kind]().double()
    cache = {}
    model.cache_all(cache, incl_bwd=True)
    tokens = torch.tensor([[1, 2, 7]])
    model(tokens).sum().backward()
    key = 'blocks.0.mlp.hook_post' if kind == 'transformer' else 'layers.0.hook_post'
    assert key in cache and key + '_grad' in cache
    assert torch.isfinite(cache[key + '_grad']).all()
    model.remove_all_hooks()
    cache.clear()
    model(tokens).sum().backward()
    assert cache == {}


def test_fourier_projection_keeps_only_the_selected_addition_signal():
    m = fourier()
    grid = torch.cartesian_prod(torch.arange(7), torch.arange(7)).double()
    scale = torch.arange(7, dtype=torch.float64)[None, :]
    signal = torch.cos(2 * torch.pi * grid.sum(1) / 7)[:, None] * scale
    nuisance = torch.cos(2 * torch.pi * (grid[:, 0] - grid[:, 1]) / 7)[:, None] * scale
    torch.testing.assert_close(m.projected_logits(signal + nuisance, 7, [1]), signal, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(m.projected_logits(signal, 7, [])) == 0
    assert m.key_frequencies(torch.zeros(49, 3, dtype=torch.float64), 7).tolist() == [-1]
    with pytest.raises(ValueError):
        m.real_fourier_basis(6)


def test_fourier_losses_keep_bias_and_use_the_requested_split():
    m = fourier()
    grid = torch.cartesian_prod(torch.arange(7), torch.arange(7)).double()
    classes = torch.arange(7, dtype=torch.float64)[None, :]
    signal = torch.cos(2 * torch.pi * grid.sum(1) / 7)[:, None] * classes
    nuisance = torch.sin(2 * torch.pi * (grid[:, 0] - grid[:, 1]) / 7)[:, None] * classes
    logits = signal + nuisance + classes
    labels = grid.sum(1).long() % 7
    train = torch.arange(49) % 3 == 0
    test = ~train
    expected_restricted = F.cross_entropy((signal + classes)[test].float(), labels[test], reduction='none').mean()
    expected_excluded = F.cross_entropy((nuisance + classes)[train].float(), labels[train], reduction='none').mean()
    torch.testing.assert_close(m.restricted_loss(logits, 7, [1], train, test, labels, mode='test'), expected_restricted)
    torch.testing.assert_close(m.excluded_loss(logits, 7, [1], train, test, labels), expected_excluded)


def test_inactive_neuron_sentinel_keeps_the_recorded_basis_index_convention():
    m = fourier()
    # Historical discovery returns -1 for an inactive neuron. The recorded
    # projection uses basis rows -3/-2, not a negative Fourier frequency.
    coordinate = torch.arange(7, dtype=torch.float64)
    a = torch.sin(4 * torch.pi * coordinate / 7) / (7 / 2) ** .5
    b = torch.cos(6 * torch.pi * coordinate / 7) / (7 / 2) ** .5
    direction = (torch.outer(a, a) - torch.outer(b, b)).flatten() / 2 ** .5
    signal = direction[:, None] * torch.arange(7, dtype=torch.float64)[None, :]
    torch.testing.assert_close(m.projected_logits(signal, 7, [-1]), signal, rtol=1e-12, atol=1e-12)


def test_activation_patch_config_preserves_zero_attention_rate():
    from test_runner import tiny_config
    from gcs.legacy.activation_patch_commutative import build_model_from_cfg
    model = build_model_from_cfg(dict(tiny_config(), attention_rate=0.), 'cpu')
    cache = {}
    model.cache_all(cache)
    model(torch.tensor([[1, 2, 7]]))
    model.remove_all_hooks()
    torch.testing.assert_close(cache['blocks.0.attn.hook_attn'][0, 0],
        torch.tensor([[1., 0., 0.], [.5, .5, 0.], [1/3, 1/3, 1/3]], dtype=cache['blocks.0.attn.hook_attn'].dtype))


def test_frequency_capture_does_not_leave_hooks_or_remove_existing_patches():
    m = fourier()
    model = licensed_model()
    prompts = torch.tensor([[a, b, 7] for a in range(7) for b in range(7)])
    point = model.blocks[0].mlp.hook_post
    point.add_hook(lambda value, name: value * .5)
    before = model(prompts).detach()
    frequencies = m.calculate_key_freqs(SimpleNamespace(p=7, d_mlp=16), model, prompts, 'cpu')
    assert len(frequencies) > 0
    torch.testing.assert_close(model(prompts), before, rtol=0, atol=0)
    assert len(model.native.blocks[0].mlp.hook_post.fwd_hooks) == 1
    model.remove_all_hooks()


def test_centroid_kernel_matches_a_linear_function_and_zero_floor():
    from gcs.analysis import grokalign
    assert hasattr(grokalign, 'centroid_alignment'), 'The new centroid kernel has not been integrated'
    weights = torch.tensor([3., 4.], dtype=torch.float64)
    def forward(value):
        return (value * weights).sum(-1, keepdim=True)
    embedded = torch.tensor([[[3., 4.]], [[-3., -4.]], [[0., 0.]]], dtype=torch.float64)
    result = grokalign.centroid_alignment(forward, embedded)
    torch.testing.assert_close(result, torch.tensor([1., -1., 0.], dtype=torch.float64), rtol=0, atol=0)


def test_centroid_rejects_empty_sampling_instead_of_reporting_nan():
    from gcs.analysis.grokalign import compute_grokalign_score
    with pytest.raises(ValueError, match='sample|empty'):
        compute_grokalign_score(licensed_model(), [], device='cpu', max_samples=10)


def test_fourier_runner_covers_all_checkpoints(tmp_path):
    from test_runner import tiny_config
    from gcs.runner import train, load_model
    from gcs.legacy.compare_gcs_nanda_measures import compute_nanda_trajectory, build_full_dataset
    train(tiny_config(), tmp_path / 'run')
    checkpoints = sorted((tmp_path / 'run/checkpoints').glob('*.pth'))
    model, config = load_model(checkpoints[-1])
    prompts, labels, train_mask, test_mask = build_full_dataset(config)
    result = compute_nanda_trajectory(model, checkpoints, config, prompts, labels, train_mask, test_mask, 'cpu')
    assert result['failed_epochs'] == []
    assert result['epochs'] == [0, 1, 2]
    assert len(result['restricted_loss']) == len(result['excluded_loss_2d_full']) == 3
    assert all(torch.isfinite(torch.tensor(result[key])).all() for key in ['restricted_loss', 'excluded_loss_2d_full', 'coeff_mean'])
