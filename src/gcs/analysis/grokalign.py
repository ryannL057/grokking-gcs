"""Centroid alignment, following the mathematical method of Walker et al.

Method reference: https://github.com/ThomasWalker1/GrokAlign
This implementation uses PyTorch differentiation and does not import or vendor
the upstream GrokAlign source. The paper's sampling and norm floor are retained.
"""
from pathlib import Path
import re

import torch


def centroid_alignment(forward_from_embedding, embedded):
    """cos(x, grad_x sum(last-position logits)), with a product-norm floor."""
    def total_output(sample):
        return forward_from_embedding(sample.unsqueeze(0))[0, -1].sum()

    derivatives = torch.vmap(torch.func.grad(total_output))(embedded)
    left, right = embedded.flatten(1), derivatives.flatten(1)
    scale = torch.linalg.vector_norm(left, dim=1) * torch.linalg.vector_norm(right, dim=1)
    return (left * right).sum(dim=1) / scale.clamp_min(1e-8)


def compute_centroid_alignment(model, x_embedded, device='cuda:0'):
    """Return one alignment per embedded prompt, without parameter gradients."""
    def forward(embedded):
        residual = model.pos_embed(embedded)
        for block in model.blocks:
            residual = block(residual)
        return model.unembed(residual)

    return centroid_alignment(forward, x_embedded.detach().to(device)).detach()


def compute_grokalign_score(model, dataset, device='cuda:0', max_samples=500):
    """Mean alignment over the paper's random-permutation prompt sample."""
    count = min(len(dataset), max_samples)
    if count <= 0:
        raise ValueError('Cannot measure an empty sample; max_samples must be positive')
    model.eval().to(device)
    indices = torch.randperm(len(dataset))[:count]
    prompts = torch.stack([dataset[int(index)][0] for index in indices]).to(device=device, dtype=torch.long)
    with torch.no_grad():
        embedded = model.embed(prompts)
    values = [compute_centroid_alignment(model, batch, device) for batch in embedded.split(256)]
    return torch.cat(values).mean().item()


def _extract_epoch(filename):
    match = re.search(r'epoch_(\d+)', str(filename))
    return int(match.group(1)) if match else 0


def compute_grokalign_trajectory(model, dataset, checkpoint_dir, checkpoint_freq=200,
                                device='cuda:0', max_samples=500):
    """Measure every selected checkpoint; fail on an incomplete trajectory."""
    if checkpoint_freq <= 0:
        raise ValueError('checkpoint_freq must be positive')
    paths = sorted(Path(checkpoint_dir).glob('*.pth'), key=lambda path: _extract_epoch(path.name))
    selected = [(path, _extract_epoch(path.name)) for path in paths
                if _extract_epoch(path.name) % checkpoint_freq == 0]
    if not selected:
        raise FileNotFoundError(f'No selected checkpoints in {checkpoint_dir}')
    torch.manual_seed(42)
    result = {'epochs': [], 'grokalign_scores': []}
    for path, epoch in selected:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
        score = compute_grokalign_score(model, dataset, device=device, max_samples=max_samples)
        result['epochs'].append(epoch)
        result['grokalign_scores'].append(score)
    return result
