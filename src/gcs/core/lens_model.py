"""TransformerLens backend with the paper's checkpoint and path interfaces.

Backend: TransformerLens 2.16.1 (MIT), commit
 a634e5757cd2ae43add9beb492be7a2dfb443a30.
The authors originally used its Grokking Demo notebook. License and source
attribution are retained in gcs/licenses and THIRD_PARTY.md.
"""
import math
from collections import OrderedDict

import torch
from torch import nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from .hooks import HookView, HookCacheMixin


class AttentionView:
    def __init__(self, attention, owner):
        self.native = attention
        self.owner = owner

    def __call__(self, value):
        return self.native(value, value, value)

    @property
    def W_O(self):
        return self.native.W_O.flatten(0, 1).T

    def __getattr__(self, name):
        if name in {'W_Q', 'W_K', 'W_V'}:
            return getattr(self.native, name).transpose(-1, -2)
        result = getattr(self.native, {'hook_attn': 'hook_pattern',
                                        'hook_attn_pre': 'hook_attn_scores'}.get(name, name))
        return HookView(result) if name.startswith('hook_') else result


class MLPView:
    def __init__(self, mlp, owner):
        self.native, self.owner = mlp, owner

    def __call__(self, value):
        return self.native(value)

    def __getattr__(self, name):
        result = getattr(self.native, name)
        if name in {'W_in', 'W_out'}:
            return result.T
        return HookView(result) if name.startswith('hook_') else result


class BlockView:
    def __init__(self, block, owner):
        self.native = block
        self.owner = owner
        self.attn = AttentionView(block.attn, owner)
        self.mlp = MLPView(block.mlp, owner)

    def __call__(self, value):
        return self.native(value)

    def __getattr__(self, name):
        result = getattr(self.native, name)
        return HookView(result) if name.startswith('hook_') else result


class EmbeddingView:
    def __init__(self, embedding):
        self.native = embedding

    def __call__(self, tokens):
        return self.native(tokens)

    @property
    def W_E(self):
        return self.native.W_E.T


class Transformer(HookCacheMixin, nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads,
                 n_ctx, act_type, use_cache=False, use_ln=False, attention_rate=1.0):
        super().__init__()
        if not 0 <= attention_rate <= 1:
            raise ValueError('attention_rate must be between zero and one')
        rng = torch.random.get_rng_state()
        cfg = HookedTransformerConfig(
            n_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp,
            d_head=d_head, n_heads=num_heads, n_ctx=n_ctx,
            act_fn={'ReLU': 'relu', 'GeLU': 'gelu', 'SiLU': 'silu', 'SeLU': 'relu'}[act_type],
            normalization_type='LN' if use_ln else None, device='cpu',
            dtype=torch.get_default_dtype(), default_prepend_bos=False,
        )
        self.native = HookedTransformer(cfg)
        if act_type == 'SeLU':
            for block in self.native.blocks:
                block.mlp.act_fn = torch.nn.functional.selu
        # The paper architecture has no normalization after the last residual.
        self.native.ln_final = nn.Identity()
        self.num_layers, self.use_ln = num_layers, use_ln
        self.attention_rate, self.use_cache = attention_rate, use_cache
        self.cache = {}
        # Extra native biases must never become additional trainable parameters.
        for name, parameter in self.native.named_parameters():
            if name.endswith(('attn.b_Q', 'attn.b_K', 'attn.b_V', 'attn.b_O', 'unembed.b_U')):
                parameter.data.zero_()
                parameter.requires_grad_(False)
        # Recreate the documented initialization in legacy checkpoint axes.
        # Native default initialization would change the training experiment.
        torch.random.set_rng_state(rng)
        initial = self.state_dict()
        order = ['embed.W_E', 'pos_embed.W_pos']
        for index in range(num_layers):
            order += [f'blocks.{index}.attn.{key}' for key in ['W_K', 'W_Q', 'W_V', 'W_O']]
            order += [f'blocks.{index}.mlp.{key}' for key in ['W_in', 'W_out']]
        order += ['unembed.W_U']
        for key in order:
            scale = d_vocab if key == 'unembed.W_U' else d_model
            initial[key] = torch.randn(initial[key].shape, dtype=cfg.dtype) / math.sqrt(scale)
        self.load_state_dict(initial)
        if attention_rate < 1:
            def interpolate(pattern, hook):
                length = pattern.shape[-1]
                uniform = torch.ones_like(pattern).tril()
                uniform = uniform / torch.arange(1, length+1, device=pattern.device,
                                                  dtype=pattern.dtype)[None, None, :, None]
                return attention_rate * pattern + (1-attention_rate) * uniform
            for block in self.native.blocks:
                block.attn.hook_pattern.add_hook(interpolate, is_permanent=True)

    @property
    def embed(self):
        return EmbeddingView(self.native.embed)

    @property
    def blocks(self):
        return [BlockView(block, self) for block in self.native.blocks]

    @property
    def unembed(self):
        return self.native.unembed

    def pos_embed(self, value):
        return value + self.native.pos_embed.W_pos[:value.shape[-2]]

    def forward(self, tokens):
        return self.native(tokens)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        native = self.native.state_dict(keep_vars=keep_vars)
        result = OrderedDict() if destination is None else destination
        for name, value in native.items():
            if name.endswith(('attn.b_Q', 'attn.b_K', 'attn.b_V', 'attn.b_O',
                              'unembed.b_U', 'attn.IGNORE')):
                continue
            if name == 'embed.W_E' or name.endswith(('.mlp.W_in', '.mlp.W_out')):
                value = value.T
            elif name.endswith(('.attn.W_Q', '.attn.W_K', '.attn.W_V')):
                value = value.transpose(-1, -2)
            elif name.endswith('.attn.W_O'):
                value = value.flatten(0, 1).T
            elif name.endswith('.attn.mask'):
                value = value.to(self.native.W_E.dtype)
            elif '.ln' in name:
                name = name.removesuffix('.w') + '.weight' if name.endswith('.w') else name.removesuffix('.b') + '.bias'
            result[prefix + name] = value
        return result

    def load_state_dict(self, source, strict=True, assign=False):
        schema = self.state_dict()
        expected = set(schema)
        missing = sorted(expected - set(source))
        unexpected = sorted(set(source) - expected)
        if strict and (missing or unexpected):
            raise RuntimeError(f'Checkpoint keys differ: missing={missing}, unexpected={unexpected}')
        native = self.native.state_dict()
        for name, value in source.items():
            if name not in expected:
                continue
            if not torch.is_tensor(value) or value.shape != schema[name].shape:
                shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                raise RuntimeError(f'Checkpoint shape mismatch for {name}: '
                                   f'expected {tuple(schema[name].shape)}, got {shape}')
            if name == 'embed.W_E' or name.endswith(('.mlp.W_in', '.mlp.W_out')):
                value = value.T
            elif name.endswith(('.attn.W_Q', '.attn.W_K', '.attn.W_V')):
                value = value.transpose(-1, -2)
            elif name.endswith('.attn.W_O'):
                value = value.T.reshape_as(native[name])
            elif name.endswith('.attn.mask'):
                value = value.bool()
            elif '.ln' in name:
                name = name.removesuffix('.weight') + '.w' if name.endswith('.weight') else name.removesuffix('.bias') + '.b'
            native[name] = value
        self.native.load_state_dict(native, strict=True, assign=assign)
        return torch.nn.modules.module._IncompatibleKeys(missing, unexpected)

    def hook_points(self):
        return [HookView(point) for point in self.native.hook_dict.values()]

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.native.cfg.device = str(self.native.W_E.device)
        self.native.cfg.dtype = self.native.W_E.dtype
        return result
