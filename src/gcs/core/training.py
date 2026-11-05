"""
Training utilities for grokking experiments.

Contains loss functions, accuracy metrics, and training loops.
"""

import torch
import torch.nn.functional as F
import numpy as np


def full_loss(model, data, device):
    """Compute cross-entropy loss on complete dataset."""
    loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
    x, labels = next(iter(loader)) # (3830, 3), (3830)
    x = x.to(device)
    labels = labels.to(device)

    logits = model(x) # (3830, 3, 114)
    # Handle both Transformer (3D output) and MLP (2D output)
    if logits.dim() == 3:
        logits = logits[:, -1]  # Take last position for Transformer (3830, 114)
    return F.cross_entropy(logits, labels) # (3830, 114) , (3830)


def full_loss_with_swap(model, data, device):
    """Compute cross-entropy loss on original + operand-swapped inputs.

    For each (a, b, =) in data, also feeds (b, a, =) with the SAME label.
    This teaches the model that f(a,b) = f(b,a), which is true for
    commutative operations (add, mul) and informative for non-commutative ones.

    Returns average loss over original + swapped examples.
    """
    loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
    x, labels = next(iter(loader))
    x = x.to(device)
    labels = labels.to(device)

    # Original forward
    logits = model(x)
    if logits.dim() == 3:
        logits = logits[:, -1]
    loss_orig = F.cross_entropy(logits, labels)

    # Swapped forward: [a, b, =] -> [b, a, =]
    x_swap = x.clone()
    x_swap[:, 0], x_swap[:, 1] = x[:, 1].clone(), x[:, 0].clone()
    logits_swap = model(x_swap)
    if logits_swap.dim() == 3:
        logits_swap = logits_swap[:, -1]
    loss_swap = F.cross_entropy(logits_swap, labels)

    return (loss_orig + loss_swap) / 2


def full_accuracy(model, data, device):
    """Compute accuracy on complete dataset."""
    loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
    x, labels = next(iter(loader))
    x = x.to(device)
    labels = labels.to(device)

    logits = model(x)
    if logits.dim() == 3:
        logits = logits[:, -1]
    predictions = torch.argmax(logits, dim=1)
    return torch.sum(predictions == labels).item() / len(labels)


def batched_loss(model, data, device, batch_size=512):
    """Compute mean cross-entropy loss over a dataset in mini-batches."""
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)

            logits = model(x)
            if logits.dim() == 3:
                logits = logits[:, -1]

            loss = F.cross_entropy(logits, labels, reduction='sum')
            total_loss += loss.item()
            total_count += labels.shape[0]

    return total_loss / max(total_count, 1)


def batched_accuracy(model, data, device, batch_size=512):
    """Compute accuracy over a dataset in mini-batches."""
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)

            logits = model(x)
            if logits.dim() == 3:
                logits = logits[:, -1]

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]

    return correct / max(total, 1)


def compute_parameter_norm(model):
    """Compute L2 norm of all model parameters."""
    return np.sqrt(sum(param.pow(2).sum().item() for param in model.parameters()))


def compute_parameter_norm_components(model):
    """Compute total / attention / MLP parameter L2 norms by parameter name."""
    attn_tags = ('W_Q', 'W_K', 'W_V', 'W_O')
    mlp_tags = ('W_in', 'W_out', 'b_in', 'b_out')

    total_sq = 0.0
    attn_sq = 0.0
    mlp_sq = 0.0

    for name, param in model.named_parameters():
        sq = param.pow(2).sum().item()
        total_sq += sq
        if any(tag in name for tag in attn_tags):
            attn_sq += sq
        if any(tag in name for tag in mlp_tags):
            mlp_sq += sq

    return np.sqrt(total_sq), np.sqrt(attn_sq), np.sqrt(mlp_sq)


def create_optimizer(model, lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98),
                     optimizer_type='AdamW', wd_attn=None, wd_mlp=None):
    """Create optimizer for the model.

    Args:
        model: Model to optimize.
        lr: Learning rate.
        weight_decay: Default weight decay for all parameters.
        betas: Adam betas.
        optimizer_type: 'AdamW', 'Adam', or 'SGD'.
        wd_attn: Per-group weight decay for attention params (None = use weight_decay).
        wd_mlp: Per-group weight decay for MLP params (None = use weight_decay).
    """
    if wd_attn is not None or wd_mlp is not None:
        attn_params = []
        mlp_params = []
        other_params = []
        attn_tags = ('W_Q', 'W_K', 'W_V', 'W_O')
        mlp_tags = ('W_in', 'W_out', 'b_in', 'b_out')
        for name, param in model.named_parameters():
            if any(tag in name for tag in attn_tags):
                attn_params.append(param)
            elif any(tag in name for tag in mlp_tags):
                mlp_params.append(param)
            else:
                other_params.append(param)
        param_groups = [
            {'params': attn_params, 'weight_decay': wd_attn if wd_attn is not None else weight_decay},
            {'params': mlp_params, 'weight_decay': wd_mlp if wd_mlp is not None else weight_decay},
            {'params': other_params, 'weight_decay': weight_decay},
        ]
        if optimizer_type == 'AdamW':
            return torch.optim.AdamW(param_groups, lr=lr, betas=betas)
        elif optimizer_type == 'Adam':
            return torch.optim.Adam(param_groups, lr=lr, betas=betas)
        elif optimizer_type == 'SGD':
            return torch.optim.SGD(param_groups, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    else:
        if optimizer_type == 'AdamW':
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif optimizer_type == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif optimizer_type == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")


class TrainingMetrics:
    """Track training metrics."""

    def __init__(self):
        self.log_steps = []
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.norms = []
        self.norms_attn = []
        self.norms_mlp = []

    def record(self, step, train_loss, test_loss, train_acc, test_acc,
               param_norm, param_norm_attn=None, param_norm_mlp=None):
        self.log_steps.append(step)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        self.norms.append(param_norm)
        self.norms_attn.append(param_norm_attn)
        self.norms_mlp.append(param_norm_mlp)

    def to_dict(self):
        return {
            'log_steps': self.log_steps,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'norms': self.norms,
            'norms_attn': self.norms_attn,
            'norms_mlp': self.norms_mlp,
        }


def basic_training_loop(model, train_data, test_data, total_epochs=15000,
                        lr=1e-3, weight_decay=1.0, record_frequency=200,
                        device='cuda:0', verbose=True, checkpoint_callback=None,
                        start_epoch=0, optimizer_state_dict=None,
                        max_grad_norm=None, wd_attn=None, wd_mlp=None,
                        augment_swap=False):
    """
    Basic training loop with optional checkpoint callbacks.

    Args:
        model: Model to train
        train_data: Training dataset
        test_data: Test dataset
        total_epochs: Total training epochs
        lr: Learning rate
        weight_decay: Weight decay
        record_frequency: Metric recording frequency
        device: Computing device
        verbose: Print progress
        checkpoint_callback: Optional callback(model, epoch, metrics) for saving
        start_epoch: Epoch to start from (for resume training)
        optimizer_state_dict: Optimizer state dict to restore (for resume training)
        max_grad_norm: Maximum gradient norm for clipping (None = no clipping)
        wd_attn: Per-group weight decay for attention params (None = use weight_decay)
        wd_mlp: Per-group weight decay for MLP params (None = use weight_decay)
        augment_swap: If True, train on both (a,b) and (b,a) with same label

    Returns:
        TrainingMetrics: Recorded training metrics
    """
    from tqdm.auto import tqdm

    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay,
                                 wd_attn=wd_attn, wd_mlp=wd_mlp)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    metrics = TrainingMetrics()

    if verbose:
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch} to {total_epochs}, recording every {record_frequency}")
        else:
            print(f"Training for {total_epochs} epochs, recording every {record_frequency}")

    loss_fn = full_loss_with_swap if augment_swap else full_loss

    for epoch in tqdm(range(start_epoch, total_epochs), desc="Training", disable=not verbose):
        model.train()
        train_loss = loss_fn(model, train_data, device)

        if epoch % record_frequency == 0:
            with torch.no_grad():
                model.eval()
                test_loss = full_loss(model, test_data, device)
                train_acc = full_accuracy(model, train_data, device)
                test_acc = full_accuracy(model, test_data, device)
                param_norm, param_norm_attn, param_norm_mlp = compute_parameter_norm_components(model)

                metrics.record(epoch, train_loss.item(), test_loss.item(),
                              train_acc, test_acc, param_norm,
                              param_norm_attn, param_norm_mlp)

                if verbose and epoch % (record_frequency * 4) == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss.item():.4f}, "
                          f"Test Acc: {test_acc:.4f}, Norm: {param_norm:.3f}")

                # Call checkpoint callback if provided
                if checkpoint_callback is not None:
                    checkpoint_callback(model, epoch, metrics)

        train_loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    if verbose:
        final_test_acc = full_accuracy(model, test_data, device)
        print(f"Training completed! Final test accuracy: {final_test_acc:.4f}")

    return metrics


def minibatch_training_loop(model, train_data, test_data, total_steps=20000,
                            batch_size=256, eval_batch_size=512,
                            lr=1e-3, weight_decay=1.0, record_frequency=200,
                            device='cuda:0', verbose=True, checkpoint_callback=None,
                            start_step=0, optimizer_state_dict=None):
    """
    Mini-batch training loop for larger datasets (e.g., MNIST).

    Args:
        model: Model to train
        train_data: Training dataset
        test_data: Test dataset
        total_steps: Total optimization steps
        batch_size: Training mini-batch size
        eval_batch_size: Batch size for full-dataset evaluation
        lr: Learning rate
        weight_decay: Weight decay
        record_frequency: Metric recording frequency (in steps)
        device: Computing device
        verbose: Print progress
        checkpoint_callback: Optional callback(model, step, metrics)
        start_step: Starting step for resumed training
        optimizer_state_dict: Optimizer state dict to restore

    Returns:
        TrainingMetrics: Recorded training metrics
    """
    from tqdm.auto import tqdm

    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    metrics = TrainingMetrics()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    train_iter = iter(train_loader)

    if verbose:
        if start_step > 0:
            print(f"Resuming training from step {start_step} to {total_steps}, recording every {record_frequency}")
        else:
            print(f"Training for {total_steps} steps, batch_size={batch_size}, recording every {record_frequency}")

    for step in tqdm(range(start_step, total_steps), desc="Training", disable=not verbose):
        model.train()

        try:
            x, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, labels = next(train_iter)

        x = x.to(device)
        labels = labels.to(device)

        logits = model(x)
        if logits.dim() == 3:
            logits = logits[:, -1]
        train_loss = F.cross_entropy(logits, labels)

        if step % record_frequency == 0:
            with torch.no_grad():
                model.eval()
                train_loss_eval = batched_loss(model, train_data, device=device, batch_size=eval_batch_size)
                test_loss_eval = batched_loss(model, test_data, device=device, batch_size=eval_batch_size)
                train_acc = batched_accuracy(model, train_data, device=device, batch_size=eval_batch_size)
                test_acc = batched_accuracy(model, test_data, device=device, batch_size=eval_batch_size)
                param_norm, param_norm_attn, param_norm_mlp = compute_parameter_norm_components(model)

                metrics.record(
                    step,
                    float(train_loss_eval),
                    float(test_loss_eval),
                    float(train_acc),
                    float(test_acc),
                    float(param_norm),
                    float(param_norm_attn),
                    float(param_norm_mlp),
                )

                if verbose and step % (record_frequency * 4) == 0:
                    print(
                        f"Step {step}: Train Loss: {train_loss_eval:.4f}, "
                        f"Test Acc: {test_acc:.4f}, Norm: {param_norm:.3f}"
                    )

        if checkpoint_callback is not None:
            checkpoint_callback(model, step, metrics)

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if verbose:
        final_test_acc = batched_accuracy(model, test_data, device=device, batch_size=eval_batch_size)
        print(f"Training completed! Final test accuracy: {final_test_acc:.4f}")

    return metrics


def setup_training_environment(seed=777, device='cuda:0'):
    """Setup training environment with proper seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.float64)

    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    return device
