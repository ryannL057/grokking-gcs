"""The authors' MLP variants, with MIT TransformerLens hook infrastructure."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .hooks import HookPoint, HookCacheMixin


class Embed(nn.Module):
    """Embedding with the historical checkpoint matrix layout."""
    def __init__(self, d_vocab, d_model):
        super().__init__()
        weights = torch.randn((d_model, d_vocab)) / math.sqrt(d_model)
        self.W_E = nn.Parameter(weights)

    def forward(self, tokens):
        return F.embedding(tokens, self.W_E.transpose(0, 1))


class MLPLayer(nn.Module):
    """Single MLP layer with hook point and optional residual/LayerNorm."""

    def __init__(self, hidden_dim, act_type='ReLU', use_residual=False, use_layernorm=False):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act_type = act_type
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.hook_post = HookPoint()

        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_dim)

        assert act_type in ['ReLU', 'GeLU', 'SiLU', 'SeLU']

    def forward(self, x):
        residual = x
        if self.use_layernorm:
            x = self.ln(x)
        x = self.linear(x)
        x = _apply_activation(x, self.act_type)
        if self.use_residual:
            x = x + residual
        return self.hook_post(x)


def _apply_activation(x, act_type):
    """Apply activation function by name."""
    if act_type == 'ReLU':
        return F.relu(x)
    elif act_type == 'GeLU':
        return F.gelu(x)
    elif act_type == 'SiLU':
        return F.silu(x)
    elif act_type == 'SeLU':
        return F.selu(x)
    else:
        raise ValueError(f"Unknown activation: {act_type}")


class OneHotMLP(HookCacheMixin, nn.Module):
    """
    MLP with one-hot encoded input for modular arithmetic grokking experiments.

    Input: [one_hot(a, p); one_hot(b, p)] -> dim 2*p
    Architecture: Linear(2*p, hidden_dim) -> [MLPLayer(hidden_dim) x num_layers] -> Linear(hidden_dim, d_vocab)
    """

    def __init__(self, d_vocab, hidden_dim=128, num_layers=2, act_type='ReLU', use_cache=False,
                 use_residual=False, use_layernorm=False):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.d_vocab = d_vocab
        self.p = d_vocab - 1  # d_vocab = p + 1 (includes equals token)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.act_type = act_type
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        # Input projection: 2*p -> hidden_dim (with activation)
        self.input_proj = nn.Linear(2 * self.p, hidden_dim)
        self.hook_input = HookPoint()

        # Hidden layers
        self.layers = nn.ModuleList([
            MLPLayer(hidden_dim, act_type, use_residual=use_residual, use_layernorm=use_layernorm)
            for _ in range(num_layers)
        ])

        # Output projection
        self.unembed = nn.Linear(hidden_dim, d_vocab)

        self._name_hooks()

    def encode_input(self, x):
        """Convert integer tokens to one-hot concatenation.

        Args:
            x: [batch, 3] integer tokens (a, b, =)

        Returns:
            [batch, 2*p] one-hot encoded (a and b only)
        """
        a = x[:, 0].long()
        b = x[:, 1].long()
        a_onehot = F.one_hot(a, self.p).to(x.dtype if x.is_floating_point() else torch.float64)
        b_onehot = F.one_hot(b, self.p).to(x.dtype if x.is_floating_point() else torch.float64)
        return torch.cat([a_onehot, b_onehot], dim=-1)  # [batch, 2*p]

    def forward(self, x):
        """Full forward pass from integer tokens."""
        onehot = self.encode_input(x)  # [batch, 2*p]
        x_proj = _apply_activation(self.input_proj(onehot), self.act_type)
        x_proj = self.hook_input(x_proj)  # [batch, hidden_dim]

        for layer in self.layers:
            x_proj = layer(x_proj)

        return self.unembed(x_proj)  # [batch, d_vocab]

    def forward_from_projected(self, x_proj):
        """Forward pass from projected space (after input_proj + activation).

        Used for JVP computation to avoid non-differentiable one-hot encoding.

        Args:
            x_proj: [hidden_dim] or [batch, hidden_dim] projected input

        Returns:
            [d_vocab] or [batch, d_vocab] logits
        """
        x = x_proj
        for layer in self.layers:
            x = layer(x)
        return self.unembed(x)


class EmbedConcatMLP(HookCacheMixin, nn.Module):
    """
    MLP with learned embedding and concatenation for modular arithmetic.

    Input: [Embed(a); Embed(b)] -> dim 2*d_embed
    Architecture: Embed(d_vocab, d_embed) -> concat -> [MLPLayer(2*d_embed) x num_layers] -> Linear(2*d_embed, d_vocab)
    """

    def __init__(self, d_vocab, d_embed=128, num_layers=2, act_type='ReLU', use_cache=False,
                 use_residual=False, use_layernorm=False):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.hidden_dim = 2 * d_embed  # Concatenated dimension
        self.num_layers = num_layers
        self.act_type = act_type
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        # Embedding layer
        self.embedding = nn.Embedding(d_vocab, d_embed)
        nn.init.normal_(self.embedding.weight, std=1.0 / np.sqrt(d_embed))

        # Hidden layers operate on 2*d_embed
        self.layers = nn.ModuleList([
            MLPLayer(self.hidden_dim, act_type, use_residual=use_residual, use_layernorm=use_layernorm)
            for _ in range(num_layers)
        ])

        # Output projection
        self.unembed = nn.Linear(self.hidden_dim, d_vocab)

        self._name_hooks()

    def embed_and_concat(self, x):
        """Embed tokens and concatenate.

        Args:
            x: [batch, 3] integer tokens (a, b, =)

        Returns:
            [batch, 2*d_embed] concatenated embeddings
        """
        a = x[:, 0].long()
        b = x[:, 1].long()
        a_emb = self.embedding(a)  # [batch, d_embed]
        b_emb = self.embedding(b)  # [batch, d_embed]
        return torch.cat([a_emb, b_emb], dim=-1)  # [batch, 2*d_embed]

    def forward(self, x):
        """Full forward pass from integer tokens."""
        x_flat = self.embed_and_concat(x)  # [batch, 2*d_embed]

        for layer in self.layers:
            x_flat = layer(x_flat)

        return self.unembed(x_flat)  # [batch, d_vocab]

    def forward_from_embedded(self, x_flat):
        """Forward pass from concatenated embedding space.

        Used for JVP computation.

        Args:
            x_flat: [2*d_embed] or [batch, 2*d_embed] concatenated embeddings

        Returns:
            [d_vocab] or [batch, d_vocab] logits
        """
        x = x_flat
        for layer in self.layers:
            x = layer(x)
        return self.unembed(x)


class EmbedSumMLP(HookCacheMixin, nn.Module):
    """
    MLP with learned embedding and summation for modular arithmetic.

    Input: Embed(a) + Embed(b) -> dim d_embed
    Architecture: Embed(d_vocab, d_embed) -> sum -> [MLPLayer(d_embed) x num_layers] -> Linear(d_embed, d_vocab)
    """

    def __init__(self, d_vocab, d_embed=128, num_layers=2, act_type='ReLU', use_cache=False,
                 use_residual=False, use_layernorm=False):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.hidden_dim = d_embed
        self.num_layers = num_layers
        self.act_type = act_type
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        # Embedding layer
        self.embedding = nn.Embedding(d_vocab, d_embed)
        nn.init.normal_(self.embedding.weight, std=1.0 / np.sqrt(d_embed))

        # Hidden layers operate on d_embed
        self.layers = nn.ModuleList([
            MLPLayer(d_embed, act_type, use_residual=use_residual, use_layernorm=use_layernorm)
            for _ in range(num_layers)
        ])

        # Output projection
        self.unembed = nn.Linear(d_embed, d_vocab)

        self._name_hooks()

    def embed_and_sum(self, x):
        """Embed tokens and sum.

        Args:
            x: [batch, 3] integer tokens (a, b, =)

        Returns:
            [batch, d_embed] summed embeddings
        """
        a = x[:, 0].long()
        b = x[:, 1].long()
        a_emb = self.embedding(a)  # [batch, d_embed]
        b_emb = self.embedding(b)  # [batch, d_embed]
        return a_emb + b_emb  # [batch, d_embed]

    def forward(self, x):
        """Full forward pass from integer tokens."""
        x_sum = self.embed_and_sum(x)  # [batch, d_embed]

        for layer in self.layers:
            x_sum = layer(x_sum)

        return self.unembed(x_sum)  # [batch, d_vocab]

    def forward_from_embedded(self, x_sum):
        """Forward pass from summed embedding space.

        Used for JVP computation.

        Args:
            x_sum: [d_embed] or [batch, d_embed] summed embeddings

        Returns:
            [d_vocab] or [batch, d_vocab] logits
        """
        x = x_sum
        for layer in self.layers:
            x = layer(x)
        return self.unembed(x)


class SimpleMLP(HookCacheMixin, nn.Module):
    """
    Simple MLP for modular arithmetic grokking experiments.

    Architecture: Embed -> MLP layers -> Unembed
    """

    def __init__(self, d_vocab, hidden_dim=200, num_layers=2, act_type='ReLU', use_cache=False):
        super().__init__()

        self.cache = {}
        self.use_cache = use_cache
        self.d_vocab = d_vocab
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = Embed(d_vocab, hidden_dim)
        self.layers = nn.ModuleList([
            MLPLayer(hidden_dim * 3, act_type) for _ in range(num_layers)
        ])
        self.unembed = nn.Linear(hidden_dim * 3, d_vocab)

        self._name_hooks()

    def forward(self, x):
        x = self.embed(x)  # [batch, 3, hidden_dim]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # [batch, 3*hidden_dim]

        for layer in self.layers:
            x = layer(x)

        return self.unembed(x)  # [batch, d_vocab]
