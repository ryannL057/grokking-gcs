"""
Data handling utilities for grokking experiments.

Supports modular arithmetic tasks: add, sub, mul, div.
Supports finite-group composition tasks via abstract element IDs, including
cyclic, dihedral, and symmetric groups.
Supports MNIST tokenized sequence classification for Transformer analysis.
"""

import torch
import numpy as np
from itertools import permutations
from math import factorial
import torch.nn.functional as F


def create_data(
    p=113,
    fraction=0.3,
    seed=777,
    device='cuda:0',
    task='add',
    reserve_fraction=0.0,
    include_reserved_in_train=False,
):
    """
    Create modular arithmetic dataset.

    Args:
        p: Prime modulus
        fraction: Training fraction
        seed: Random seed
        device: Device for tensors
        task: 'add', 'sub', 'mul', or 'div'
        reserve_fraction: Additional held-out fraction removed from the
            evaluation split. This is useful for matched-control designs
            where multiple conditions share the same base train set and
            evaluation set, but only some conditions are allowed to train on
            the reserved pool.
        include_reserved_in_train: If True, append the reserved pool to the
            base training set. With the same seed and reserve_fraction, this
            yields a matched split across baseline / swap / random-extra
            conditions.

    Returns:
        (train_dataset, test_dataset)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    x, y = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    x = x.flatten()
    y = y.flatten()

    # Filter out b=0 for division
    if task == 'div':
        mask = y != 0
        x = x[mask]
        y = y[mask]

    # Create prompts: [a, b, equals_token]
    equals_token = p
    equals = torch.ones(x.shape, dtype=torch.int64) * equals_token
    prompts = torch.stack([x, y, equals], dim=1).to(device)

    # Generate answers
    if task == 'add':
        answers = ((x + y) % p).to(device)
    elif task == 'sub':
        answers = ((x - y) % p).to(device)
    elif task == 'mul':
        answers = ((x * y) % p).to(device)
    elif task == 'div':
        answers = _compute_modular_division(x, y, p).to(device)
    else:
        raise ValueError(f"Unsupported task: {task}")

    data = torch.utils.data.TensorDataset(prompts, answers)
    n_total = len(data)
    train_size = int(fraction * n_total)
    reserve_size = int(reserve_fraction * n_total)

    if train_size <= 0:
        raise ValueError(f"Training split is empty: fraction={fraction}, n_total={n_total}")
    if reserve_fraction < 0:
        raise ValueError(f"reserve_fraction must be >= 0, got {reserve_fraction}")
    if train_size + reserve_size >= n_total:
        raise ValueError(
            f"Training + reserved splits consume the full dataset: "
            f"train={train_size}, reserve={reserve_size}, total={n_total}"
        )

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)

    train_idx = perm[:train_size]
    reserve_idx = perm[train_size: train_size + reserve_size]
    test_idx = perm[train_size + reserve_size:]

    if include_reserved_in_train:
        effective_train_idx = torch.cat([train_idx, reserve_idx], dim=0)
    else:
        effective_train_idx = train_idx

    train_dataset = torch.utils.data.Subset(data, effective_train_idx.tolist())
    test_dataset = torch.utils.data.Subset(data, test_idx.tolist())
    return train_dataset, test_dataset


def _mnist_images_to_tokens(images, image_size=14, num_bins=10):
    """Convert raw MNIST images [N, 28, 28] to token sequences [N, image_size^2]."""
    x = images.to(torch.float32).unsqueeze(1) / 255.0  # [N, 1, 28, 28]

    if image_size != 28:
        x = F.interpolate(x, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # Quantize to discrete token ids in [0, num_bins-1]
    x = torch.clamp((x * num_bins).long(), min=0, max=num_bins - 1)
    return x.squeeze(1).reshape(x.shape[0], image_size * image_size)


def create_mnist_data(
    seed=777,
    image_size=14,
    num_bins=10,
    train_fraction=1.0,
    data_root='data',
    download=True,
):
    """
    Create tokenized MNIST dataset for Transformer sequence modeling.

    Input format: [pixel_token_0, ..., pixel_token_(H*W-1), cls_token].
    Token ids:
      - 0..num_bins-1: quantized pixel bins
      - num_bins: CLS token
    Labels: 0..9 digit class.
    """
    if not (0 < train_fraction <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    from torchvision import datasets

    try:
        train_raw = datasets.MNIST(root=data_root, train=True, download=download)
        test_raw = datasets.MNIST(root=data_root, train=False, download=download)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load/download MNIST at data_root='{data_root}'. "
            f"Original error: {e}"
        ) from e

    train_tokens = _mnist_images_to_tokens(train_raw.data, image_size=image_size, num_bins=num_bins)
    test_tokens = _mnist_images_to_tokens(test_raw.data, image_size=image_size, num_bins=num_bins)

    cls_token = num_bins
    train_cls = torch.full((train_tokens.shape[0], 1), cls_token, dtype=torch.int64)
    test_cls = torch.full((test_tokens.shape[0], 1), cls_token, dtype=torch.int64)

    train_prompts = torch.cat([train_tokens, train_cls], dim=1)
    test_prompts = torch.cat([test_tokens, test_cls], dim=1)

    train_labels = train_raw.targets.to(torch.int64)
    test_labels = test_raw.targets.to(torch.int64)

    if train_fraction < 1.0:
        keep_n = int(len(train_prompts) * train_fraction)
        generator = torch.Generator().manual_seed(seed)
        keep_idx = torch.randperm(len(train_prompts), generator=generator)[:keep_n]
        train_prompts = train_prompts[keep_idx]
        train_labels = train_labels[keep_idx]

    train_data = torch.utils.data.TensorDataset(train_prompts, train_labels)
    test_data = torch.utils.data.TensorDataset(test_prompts, test_labels)
    return train_data, test_data


def _compute_modular_division(a, b, p):
    """Compute (a / b) % p using Fermat's little theorem."""
    result = torch.zeros_like(a)
    a_np, b_np = a.numpy(), b.numpy()

    for i in range(len(a)):
        if b_np[i] == 0:
            result[i] = 0
        else:
            b_inv = pow(int(b_np[i]), p - 2, p)
            result[i] = (a_np[i] * b_inv) % p

    return result


def setup_training_config(p=113, fraction=0.3, seed=777, total_epochs=15000,
                          num_layers=1, act_type='ReLU', task='add',
                          attention_rate=1.0, d_model=128, weight_decay=1.0,
                          reserve_fraction=0.0, include_reserved_in_train=False,
                          use_ln=False):
    """
    Setup standard training configuration.

    Args:
        attention_rate: Controls Clock vs Pizza algorithm (1.0=Clock, 0.0=Pizza)
        d_model: Model dimension (width)

    Returns:
        dict: Configuration dictionary
    """
    return {
        # Data
        'p': p,
        'fraction': fraction,
        'reserve_fraction': reserve_fraction,
        'include_reserved_in_train': include_reserved_in_train,
        'seed': seed,
        'd_vocab': p + 1,
        'task': task,

        # Model
        'num_layers': num_layers,
        'd_model': d_model,
        'd_mlp': d_model * 4,  # Standard 4x expansion
        'd_head': d_model // 4,  # 4 heads
        'num_heads': 4,
        'n_ctx': 3,
        'act_type': act_type,
        'use_cache': False,
        'use_ln': use_ln,
        'attention_rate': attention_rate,

        # Training
        'total_epochs': total_epochs,
        'lr': 1e-3,
        'weight_decay': weight_decay,
        'optimizer': 'AdamW',
        'betas': (0.9, 0.98),

        # Analysis
        'analysis_N': 200,
        'analysis_k': 15,
        'analysis_d': 8,
        'record_frequency': 200,

        # Device
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }


def setup_mnist_training_config(
    seed=777,
    total_steps=20000,
    batch_size=256,
    eval_batch_size=512,
    num_layers=1,
    act_type='SiLU',
    attention_rate=1.0,
    d_model=128,
    weight_decay=1.0,
    lr=1e-3,
    image_size=14,
    num_bins=10,
    train_fraction=1.0,
    download_mnist=True,
):
    """Setup training configuration for tokenized MNIST classification."""
    n_ctx = image_size * image_size + 1  # +1 for CLS token
    return {
        # Data
        'dataset': 'mnist',
        'seed': seed,
        'task': 'mnist',
        'num_classes': 10,
        'image_size': image_size,
        'num_bins': num_bins,
        'cls_token_id': num_bins,
        'n_ctx': n_ctx,
        'd_vocab': num_bins + 1,
        'train_fraction': train_fraction,
        'download_mnist': download_mnist,

        # Model
        'num_layers': num_layers,
        'd_model': d_model,
        'd_mlp': d_model * 4,
        'd_head': d_model // 4,
        'num_heads': 4,
        'act_type': act_type,
        'use_cache': False,
        'use_ln': False,
        'attention_rate': attention_rate,

        # Training
        'total_steps': total_steps,
        'total_epochs': total_steps,  # compatibility field for downstream tooling
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'optimizer': 'AdamW',
        'betas': (0.9, 0.98),

        # Analysis
        'analysis_N': 200,
        'analysis_k': 15,
        'analysis_d': 8,
        'record_frequency': 200,

        # Device
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }


def get_model_config_from_data(config):
    """Extract model configuration from full config."""
    return {
        'num_layers': config['num_layers'],
        'd_vocab': config['d_vocab'],
        'd_model': config['d_model'],
        'd_mlp': config['d_mlp'],
        'd_head': config['d_head'],
        'num_heads': config['num_heads'],
        'n_ctx': config['n_ctx'],
        'act_type': config['act_type'],
        'use_cache': config['use_cache'],
        'use_ln': config['use_ln'],
        'attention_rate': config.get('attention_rate', 1.0)
    }


def setup_mlp_training_config(p=113, fraction=0.3, seed=777, total_epochs=15000,
                               num_layers=2, act_type='ReLU', task='add',
                               hidden_dim=128, weight_decay=1.0, lr=1e-3,
                               embedding_type='one_hot',
                               use_residual=False, use_layernorm=False):
    """
    Setup training configuration for MLP experiments.

    Args:
        p: Prime modulus
        fraction: Training fraction
        seed: Random seed
        total_epochs: Total training epochs
        num_layers: Number of hidden MLP layers
        act_type: Activation function
        task: Arithmetic task
        hidden_dim: Hidden dimension (also d_embed for embed types)
        weight_decay: Weight decay for optimizer
        lr: Learning rate for optimizer
        embedding_type: 'one_hot', 'embed_concat', or 'embed_sum'
        use_residual: Add residual connections to MLP layers
        use_layernorm: Add LayerNorm before each MLP layer

    Returns:
        dict: Configuration dictionary
    """
    return {
        # Data
        'p': p,
        'fraction': fraction,
        'seed': seed,
        'd_vocab': p + 1,
        'task': task,

        # Model
        'model_type': 'mlp',
        'embedding_type': embedding_type,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'd_embed': hidden_dim,
        'act_type': act_type,
        'use_cache': False,
        'use_residual': use_residual,
        'use_layernorm': use_layernorm,

        # Training
        'total_epochs': total_epochs,
        'lr': lr,
        'weight_decay': weight_decay,
        'optimizer': 'AdamW',
        'betas': (0.9, 0.98),

        # Analysis
        'analysis_N': 200,
        'analysis_k': 15,
        'analysis_d': 8,
        'record_frequency': 200,

        # Device
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }


def parse_group_spec(group_spec):
    """Parse a finite-group spec string into (family, parameter).

    Supported forms:
      - ``cyclic:120`` or ``c120``
      - ``dihedral:60`` or ``d60``   (order is 2n)
      - ``symmetric:5`` or ``s5``    (order is n!)
    """
    spec = str(group_spec).strip().lower()
    if not spec:
        raise ValueError("group_spec must be non-empty")

    if spec.startswith('cyclic:'):
        family, param = 'cyclic', int(spec.split(':', 1)[1])
    elif spec.startswith('dihedral:'):
        family, param = 'dihedral', int(spec.split(':', 1)[1])
    elif spec.startswith('symmetric:'):
        family, param = 'symmetric', int(spec.split(':', 1)[1])
    elif spec.startswith('c') and spec[1:].isdigit():
        family, param = 'cyclic', int(spec[1:])
    elif spec.startswith('d') and spec[1:].isdigit():
        family, param = 'dihedral', int(spec[1:])
    elif spec.startswith('s') and spec[1:].isdigit():
        family, param = 'symmetric', int(spec[1:])
    else:
        raise ValueError(
            f"Unsupported group spec '{group_spec}'. Use cyclic:n, dihedral:n, or symmetric:n."
        )

    if family == 'cyclic' and param < 1:
        raise ValueError(f"cyclic groups require n >= 1, got {param}")
    if family == 'dihedral' and param < 2:
        raise ValueError(f"dihedral groups require n >= 2, got {param}")
    if family == 'symmetric' and param < 1:
        raise ValueError(f"symmetric groups require n >= 1, got {param}")

    return family, param


def group_spec_to_slug(group_spec):
    """Convert a group spec to a filesystem-safe slug."""
    family, param = parse_group_spec(group_spec)
    return f'{family}{param}'


def group_order(group_spec):
    """Return the order of the finite group described by ``group_spec``."""
    family, param = parse_group_spec(group_spec)
    if family == 'cyclic':
        return param
    if family == 'dihedral':
        return 2 * param
    if family == 'symmetric':
        return factorial(param)
    raise ValueError(f'Unhandled group family: {family}')


def _build_cyclic_table(n):
    """Build the Cayley table for the cyclic group C_n."""
    elems = list(range(n))
    table = np.fromfunction(lambda i, j: (i + j) % n, (n, n), dtype=int)
    return elems, table.astype(np.int64)


def _build_dihedral_table(n):
    """Build the Cayley table for the dihedral group D_n of order 2n.

    Elements are encoded as ``(rotation, flip)`` pairs, mapped to ids via
    ``rotation + n * flip``.
    """
    elems = [(r, f) for f in (0, 1) for r in range(n)]
    size = 2 * n
    table = np.zeros((size, size), dtype=np.int64)

    def encode(rotation, flip):
        return rotation + n * flip

    for i, (r1, f1) in enumerate(elems):
        for j, (r2, f2) in enumerate(elems):
            sign = -1 if f1 else 1
            rotation = (r1 + sign * r2) % n
            flip = f1 ^ f2
            table[i, j] = encode(rotation, flip)

    return elems, table


def _build_symmetric_tables(degree):
    """Build the Cayley table for the symmetric group S_degree."""
    perms = sorted(permutations(range(degree)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    n = len(perms)
    compose_table = np.zeros((n, n), dtype=np.int64)
    for i, sigma in enumerate(perms):
        for j, tau in enumerate(perms):
            # composition: (sigma ∘ tau)(x) = sigma(tau(x))
            composed = tuple(sigma[tau[k]] for k in range(degree))
            compose_table[i, j] = perm_to_idx[composed]
    return perms, compose_table


def build_group_table(group_spec):
    """Build the Cayley table for a supported finite group."""
    family, param = parse_group_spec(group_spec)
    if family == 'cyclic':
        return _build_cyclic_table(param)
    if family == 'dihedral':
        return _build_dihedral_table(param)
    if family == 'symmetric':
        return _build_symmetric_tables(param)
    raise ValueError(f'Unhandled group family: {family}')


def _relabel_group_table(compose_table, seed):
    """Apply a deterministic random relabeling of finite-group elements."""
    n = compose_table.shape[0]
    generator = np.random.default_rng(seed)
    label_perm = generator.permutation(n).astype(np.int64)  # old_id -> new_id
    inverse_perm = np.argsort(label_perm)
    relabeled = label_perm[compose_table[inverse_perm][:, inverse_perm]]
    return relabeled, label_perm


def create_group_data(group_spec='symmetric:5', fraction=0.3, seed=777,
                      device='cuda:0', shuffle_labels=False):
    """Create finite-group composition data from a generic Cayley table.

    The task uses the same prompt format as modular arithmetic and S5:
    ``[g1_id, g2_id, equals_token]`` with the answer ``g1 ∘ g2``.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    _, compose_table = build_group_table(group_spec)
    if shuffle_labels:
        compose_table, _ = _relabel_group_table(compose_table, seed=seed)

    n = compose_table.shape[0]
    x, y = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    x = x.flatten()
    y = y.flatten()

    equals_token = n
    equals = torch.full_like(x, equals_token)
    prompts = torch.stack([x, y, equals], dim=1).to(device)
    answers = torch.from_numpy(compose_table[x.numpy(), y.numpy()]).to(device)

    data = torch.utils.data.TensorDataset(prompts, answers)
    n_total = len(data)
    train_size = int(fraction * n_total)
    test_size = n_total - train_size
    if train_size <= 0 or test_size <= 0:
        raise ValueError(
            f"Invalid train/test split for {group_spec}: fraction={fraction}, total={n_total}"
        )

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    train_dataset = torch.utils.data.Subset(data, train_idx.tolist())
    test_dataset = torch.utils.data.Subset(data, test_idx.tolist())
    return train_dataset, test_dataset


def create_s5_data(fraction=0.3, seed=777, device='cuda:0'):
    """Create S5 permutation group composition dataset.

    Task: given σ₁, σ₂ ∈ S5, predict σ₁ ∘ σ₂.
    |S5| = 120, total pairs = 14,400.

    Input format: [σ₁_id, σ₂_id, equals_token] (n_ctx=3, same as mod arith).
    d_vocab = 121 (120 permutations + 1 equals token).

    Args:
        fraction: Training fraction
        seed: Random seed
        device: Device for tensors

    Returns:
        (train_dataset, test_dataset)
    """
    return create_group_data(
        group_spec='symmetric:5',
        fraction=fraction,
        seed=seed,
        device=device,
        shuffle_labels=False,
    )


def setup_group_training_config(group_spec='symmetric:5', fraction=0.3, seed=777,
                                total_epochs=20000, num_layers=1,
                                act_type='SiLU', attention_rate=1.0,
                                d_model=128, weight_decay=1.0,
                                shuffle_labels=False):
    """Setup training configuration for generic finite-group composition."""
    family, param = parse_group_spec(group_spec)
    order = group_order(group_spec)
    return {
        # Data
        'p': order,  # group order (analogous to p for mod arith)
        'fraction': fraction,
        'seed': seed,
        'd_vocab': order + 1,
        'task': 'group_comp',
        'group_spec': f'{family}:{param}',
        'group_family': family,
        'group_param': param,
        'group_order': order,
        'group_slug': group_spec_to_slug(group_spec),
        'shuffle_group_labels': shuffle_labels,

        # Model
        'num_layers': num_layers,
        'd_model': d_model,
        'd_mlp': d_model * 4,
        'd_head': d_model // 4,
        'num_heads': 4,
        'n_ctx': 3,
        'act_type': act_type,
        'use_cache': False,
        'use_ln': False,
        'attention_rate': attention_rate,

        # Training
        'total_epochs': total_epochs,
        'lr': 1e-3,
        'weight_decay': weight_decay,
        'optimizer': 'AdamW',
        'betas': (0.9, 0.98),

        # Analysis
        'analysis_N': 200,
        'analysis_k': 15,
        'analysis_d': 8,
        'record_frequency': 200,

        # Device
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }


def setup_s5_training_config(fraction=0.3, seed=777, total_epochs=20000,
                              num_layers=1, act_type='SiLU',
                              attention_rate=1.0, d_model=128,
                              weight_decay=1.0):
    """Setup training configuration for S5 composition task."""
    config = setup_group_training_config(
        group_spec='symmetric:5',
        fraction=fraction,
        seed=seed,
        total_epochs=total_epochs,
        num_layers=num_layers,
        act_type=act_type,
        attention_rate=attention_rate,
        d_model=d_model,
        weight_decay=weight_decay,
        shuffle_labels=False,
    )
    config['task'] = 's5'
    return config


def _validate_sparse_parity_args(n_bits, k, relevant_positions, task_type,
                                 n_samples, fraction):
    """Validate sparse parity dataset/config arguments."""
    relevant_positions = tuple(int(pos) for pos in relevant_positions)

    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(relevant_positions) != k:
        raise ValueError(
            f"len(relevant_positions) must equal k, got {len(relevant_positions)} vs {k}"
        )
    if len(set(relevant_positions)) != k:
        raise ValueError(f"relevant_positions must be distinct, got {relevant_positions}")
    if any(pos < 0 or pos >= n_bits for pos in relevant_positions):
        raise ValueError(
            f"relevant_positions must lie in [0, {n_bits - 1}], got {relevant_positions}"
        )
    if task_type not in ('xor', 'and_xor'):
        raise ValueError(f"task_type must be 'xor' or 'and_xor', got {task_type}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    return relevant_positions


def create_sparse_parity_data(n_bits=20, k=3, relevant_positions=(3, 11, 17),
                               task_type='xor', n_samples=4000,
                               fraction=0.5, seed=777, device='cuda:0'):
    """Create sparse parity (XOR) or AND-XOR dataset.

    Input: [x_0, x_1, ..., x_{n-1}, equals_token] with x_i in {0,1}, eq=2
    Label: XOR or AND-XOR of bits at relevant_positions (0 or 1)

    XOR parity (symmetric): y = x_{i1} ^ x_{i2} ^ x_{i3}
    AND-XOR (asymmetric):   y = (x_{i1} & x_{i2}) ^ x_{i3}
      - swapping i1<->i3 changes the result

    Args:
        n_bits: Total number of input bits
        k: Number of relevant bits (must match len(relevant_positions))
        relevant_positions: Tuple of bit indices that determine the label
        task_type: 'xor' (symmetric) or 'and_xor' (asymmetric)
        n_samples: Total number of samples to generate
        fraction: Training fraction
        seed: Random seed
        device: Device for tensors

    Returns:
        (train_dataset, test_dataset)
    """
    relevant_positions = _validate_sparse_parity_args(
        n_bits=n_bits,
        k=k,
        relevant_positions=relevant_positions,
        task_type=task_type,
        n_samples=n_samples,
        fraction=fraction,
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Generate random binary inputs
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_samples, n_bits)).astype(np.int64)

    # Compute labels from relevant positions
    r = [relevant_positions[i] for i in range(k)]
    if task_type == 'xor':
        # y = x[r0] ^ x[r1] ^ x[r2]
        labels = X[:, r[0]] ^ X[:, r[1]] ^ X[:, r[2]]
    elif task_type == 'and_xor':
        # y = (x[r0] & x[r1]) ^ x[r2]
        labels = (X[:, r[0]] & X[:, r[1]]) ^ X[:, r[2]]

    # Build prompts: [x_0, ..., x_{n-1}, equals_token]
    equals_token = 2  # d_vocab = 3: {0, 1, 2}
    eq_col = np.full((n_samples, 1), equals_token, dtype=np.int64)
    prompts_np = np.concatenate([X, eq_col], axis=1)  # [n_samples, n_bits+1]

    prompts = torch.from_numpy(prompts_np).to(device)
    answers = torch.from_numpy(labels.astype(np.int64)).to(device)

    data = torch.utils.data.TensorDataset(prompts, answers)
    train_size = int(fraction * len(data))
    test_size = len(data) - train_size

    return torch.utils.data.random_split(data, [train_size, test_size])


def setup_sparse_parity_config(n_bits=20, k=3, relevant_positions=(3, 11, 17),
                                task_type='xor', n_samples=4000,
                                fraction=0.5, seed=777, total_epochs=30000,
                                num_layers=1, act_type='SiLU',
                                attention_rate=1.0, d_model=128,
                                weight_decay=1.0):
    """Setup training configuration for sparse parity task."""
    relevant_positions = _validate_sparse_parity_args(
        n_bits=n_bits,
        k=k,
        relevant_positions=relevant_positions,
        task_type=task_type,
        n_samples=n_samples,
        fraction=fraction,
    )
    n_ctx = n_bits + 1  # bits + equals token
    return {
        # Data
        'n_bits': n_bits,
        'k': k,
        'relevant_positions': relevant_positions,
        'task_type': task_type,
        'n_samples': n_samples,
        'fraction': fraction,
        'seed': seed,
        'd_vocab': 3,  # {0, 1, equals_token=2}
        'task': f'parity_{task_type}',

        # Model
        'num_layers': num_layers,
        'd_model': d_model,
        'd_mlp': d_model * 4,
        'd_head': d_model // 4,
        'num_heads': 4,
        'n_ctx': n_ctx,
        'act_type': act_type,
        'use_cache': False,
        'use_ln': False,
        'attention_rate': attention_rate,

        # Training
        'total_epochs': total_epochs,
        'lr': 1e-3,
        'weight_decay': weight_decay,
        'optimizer': 'AdamW',
        'betas': (0.9, 0.98),

        # Analysis
        'analysis_N': 200,
        'analysis_k': 15,
        'analysis_d': 8,
        'record_frequency': 200,

        # Device
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }


def create_batch_iterator(dataset, batch_size=None, shuffle=False):
    """Create a data loader for the dataset."""
    if batch_size is None:
        batch_size = len(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
