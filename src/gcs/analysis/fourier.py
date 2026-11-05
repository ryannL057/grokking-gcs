"""Fourier progress measures based on the MIT TransformerLens Grokking Demo.

Projection and loss design follows demo cells 102, 120, 131, 137 at
a634e5757cd2ae43add9beb492be7a2dfb443a30. The demo's MIT notice is retained in
gcs/licenses/TransformerLens-LICENSE. The kernels retain the paper's
frequency-selection and float32 per-example loss conventions.
"""
import math

import torch
import torch.nn.functional as F


def real_fourier_basis(p, *, dtype=None, device='cpu'):
    if p < 3 or p % 2 != 1:
        raise ValueError('The recorded baseline uses an odd modulus')
    dtype = dtype or torch.get_default_dtype()
    phase = torch.outer(torch.arange(1, (p+1)//2, dtype=dtype, device=device),
                        torch.arange(p, dtype=dtype, device=device)) * (2*math.pi/p)
    pairs = torch.stack((phase.cos(), phase.sin()), dim=1).flatten(0, 1)
    pairs = F.normalize(pairs, dim=1)
    return torch.cat((torch.full((1, p), 1/math.sqrt(p), dtype=dtype, device=device), pairs))


def addition_directions(p, frequencies, *, dtype, device):
    basis = real_fourier_basis(p, dtype=dtype, device=device)
    frequencies = torch.as_tensor(list(frequencies), dtype=torch.long, device=device)
    # Discovery uses -1 for inactive neurons. Preserve the archived negative
    # basis indices as well as the valid cos(a+b)/sin(a+b) projections.
    cosine, sine = basis[2 * frequencies - 1], basis[2 * frequencies]
    cc = torch.einsum('fx,fy->fxy', cosine, cosine)
    ss = torch.einsum('fx,fy->fxy', sine, sine)
    cs = torch.einsum('fx,fy->fxy', cosine, sine)
    sc = torch.einsum('fx,fy->fxy', sine, cosine)
    return torch.stack((cc - ss, cs + sc), dim=1).reshape(2 * len(frequencies), p * p) / math.sqrt(2)


def projected_logits(logits, p, frequencies):
    directions = addition_directions(p, frequencies, dtype=logits.dtype, device=logits.device)
    return directions.T @ (directions @ logits)


def evaluate_logits(logits, p, train, test, labels, *, mode='all', original=None):
    # Based on MIT demo cell 102. Keep the paper baseline's float32 loss
    # convention, despite the historical helper's "high_precision" name.
    if logits.shape[1] == p*p:
        logits = logits.T
    logits = logits.reshape(p*p, -1)[:, :p]
    if original is not None:
        logits = logits + (original[:, :p] - logits).mean(dim=0)
    selected = {'all': slice(None), 'train': train, 'test': test}[mode]
    return F.cross_entropy(logits[selected].float(), labels[selected], reduction='none').mean()


def restricted_loss(logits, p, frequencies, train, test, labels, mode='all'):
    return evaluate_logits(projected_logits(logits, p, frequencies), p, train, test,
                           labels, mode=mode, original=logits)


def excluded_loss(logits, p, frequencies, train, test, labels):
    return evaluate_logits(logits-projected_logits(logits, p, frequencies), p,
                           train, test, labels, mode='train')


def transform_2d(values, p, basis):
    cube = values.reshape(p, p, -1)
    transformed = torch.einsum('ax,xyc,by->abc', basis, cube, basis)
    return transformed.reshape_as(values)


def key_frequencies(activations, p):
    centered = activations - activations.mean(dim=0)
    basis = real_fourier_basis(p, dtype=activations.dtype, device=activations.device)
    coefficients = transform_2d(centered, p, basis).reshape(p, p, -1)
    energies = coefficients.square()
    total = energies.sum(dim=(0, 1))
    candidates = list(range(1, p//2))
    if not candidates:
        return torch.tensor([-1], device=activations.device)
    energy = torch.stack([energies[[0, 2*k-1, 2*k]][:, [0, 2*k-1, 2*k]].sum(dim=(0, 1))
                          for k in candidates])
    winners = energy.argmax(dim=0) + 1
    return torch.unique(torch.where(total > 0, winners, -1)).sort().values


def mean_cosine_coefficient(logits, p):
    # Streaming form of the MIT demo's normalized cos(a+b-c) projections.
    coordinate = torch.arange(p, dtype=logits.dtype, device=logits.device)
    phase = coordinate[:, None, None] + coordinate[None, :, None] - coordinate[None, None, :]
    values = []
    for frequency in range(1, (p+1)//2):
        direction = (phase * (frequency*2*math.pi/p)).cos().reshape(p*p, p)
        values.append(torch.sum(F.normalize(direction.flatten(), dim=0).reshape_as(direction)*logits))
    return torch.stack(values).mean()



DEMO_SOURCE = 'https://github.com/TransformerLensOrg/TransformerLens/blob/a634e5757cd2ae43add9beb492be7a2dfb443a30/demos/Grokking_Demo.ipynb'
MEASURE_SOURCES = {
    'restricted_loss': DEMO_SOURCE + '#get_restricted_loss',
    'excluded_loss_2d_full': DEMO_SOURCE + '#get_excluded_loss',
    'implementation': 'gcs.analysis.fourier; logits projection and recorded precision conventions',
}


def calculate_key_freqs(config, model, all_data, device):
    """Capture first-block activations without retaining or clearing user hooks."""
    captured = {}
    def save(value, hook):
        captured['post'] = value[:, -1].detach()
    with model.native.hooks(fwd_hooks=[('blocks.0.mlp.hook_post', save)]):
        with torch.no_grad():
            model(all_data.to(device))
    return key_frequencies(captured['post'], config.p).cpu().numpy()


def sum_sq_weights(model):
    return sum(float(parameter.detach().square().sum()) for parameter in model.parameters())
