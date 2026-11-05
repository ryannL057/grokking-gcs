
from gcs.runtime import data_root as _gcs_data_root
#!/usr/bin/env python3
"""
Train 1-3 layer Transformer models on modular arithmetic tasks.

Usage:
    python train_transformer.py --layers 1 --activation ReLU --task add
    python train_transformer.py --layers 2 --activation GeLU --epochs 30000
    python train_transformer.py --layers 3 --activation SiLU --task mul
    python train_transformer.py --task add --p 127 --fraction 0.3

Clock vs Pizza algorithm control:
    python train_transformer.py --attention-rate 1.0  # Clock (full attention)
    python train_transformer.py --attention-rate 0.0  # Pizza (constant attention)
    python train_transformer.py --attention-rate 0.5  # Interpolated

Width ablation:
    python train_transformer.py --d-model 64
    python train_transformer.py --d-model 256
"""

import argparse
import hashlib
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime

import torch

# Add gcs_core directory to path
pass  # Package imports do not search the experiment-data directory.

from gcs.core.models import Transformer
from gcs.core.data import create_data, setup_training_config, get_model_config_from_data
from gcs.core.training import basic_training_loop, setup_training_environment
from gcs.analysis import build_complete_geometric_structure, analyze_3path_gmn_tsa
from gcs.vis import plot_training_curves, plot_gcs_3path

# Parameter-name tags for --freeze (same tags as the reset tests).
_ATTN_TAGS = ('.attn.W_Q', '.attn.W_K', '.attn.W_V', '.attn.W_O')
FREEZE_GROUPS = {
    'mlp': ('.mlp.W_in', '.mlp.b_in', '.mlp.W_out', '.mlp.b_out'),
    'attn': _ATTN_TAGS,
    'attn_pos': _ATTN_TAGS + ('pos_embed.W_pos',),
    'attn_pos_embed': _ATTN_TAGS + ('pos_embed.W_pos', 'embed.W_E'),
}


def train_transformer(
    num_layers: int = 1,
    act_type: str = 'ReLU',
    task: str = 'add',
    p: int = 113,
    fraction: float = 0.3,
    total_epochs: int = 15000,
    save_checkpoints: bool = True,
    checkpoint_freq: int = 200,
    compute_gcs: bool = True,
    gcs_freq: int = 200,
    gcs_last_layer_only: bool = False,
    output_dir: str = None,
    device: str = 'cuda:0',
    attention_rate: float = 1.0,
    d_model: int = 128,
    weight_decay: float = 1.0,
    seed: int = 777,
    grad_clip: float = None,
    wd_attn: float = None,
    wd_mlp: float = None,
    use_ln: bool = False,
    augment_swap: bool = False,
    reserve_fraction: float = 0.0,
    include_reserved_in_train: bool = False,
    init_checkpoint: str = None,
    start_epoch: int = 0,
    freeze: str = 'none',
    record_frequency: int = None,
    config_overrides: dict = None,
    checkpoint_epochs: list = None,
    gcs_epochs: list = None,
    canonical_layer: int = None,
):
    """
    Train a Transformer model and optionally compute GCS.

    Args:
        num_layers: Number of transformer layers (1-3)
        act_type: Activation function ('ReLU', 'GeLU', 'SiLU', 'SeLU')
        task: Arithmetic task ('add', 'sub', 'mul', 'div')
        p: Prime modulus for modular arithmetic dataset
        fraction: Train split fraction
        total_epochs: Total training epochs
        save_checkpoints: Whether to save model checkpoints
        checkpoint_freq: Checkpoint saving frequency
        compute_gcs: Whether to compute GCS after training
        gcs_freq: Frequency for GCS computation during reanalysis
        output_dir: Output directory (auto-generated if None)
        device: Computing device
        attention_rate: Controls Clock vs Pizza (1.0=Clock, 0.0=Pizza)
        d_model: Model dimension (width)
        weight_decay: Weight decay for AdamW optimizer
        seed: Random seed for reproducibility
        grad_clip: Maximum gradient norm for clipping (None = no clipping)
        wd_attn: Weight decay for attention params (None = use weight_decay)
        wd_mlp: Weight decay for MLP params (None = use weight_decay)
        use_ln: Use pre-LayerNorm transformer blocks
        augment_swap: Train on both (a,b) and (b,a) with same label
        reserve_fraction: Held-out reserve fraction removed from evaluation.
            Used for matched-control experiments.
        include_reserved_in_train: If True, append the reserved split to the
            base training set.
        init_checkpoint: Branch from this checkpoint: all parameters are
            loaded from it before training (optimizer moments start fresh,
            since saved checkpoints hold no optimizer state).
        start_epoch: Epoch index of init_checkpoint; training runs from here
            to total_epochs, so epoch numbering matches the source run.
        freeze: Parameter group held fixed during training ('none', 'mlp',
            'attn' = W_Q/W_K/W_V/W_O, 'attn_pos' = attn + W_pos,
            'attn_pos_embed' = attn_pos + W_E). Frozen parameters receive
            neither gradient updates nor weight decay.
        record_frequency: Metric/checkpoint recording frequency (None = the
            config default).
    """
    algo_type = "Clock" if attention_rate > 0.5 else ("Pizza" if attention_rate < 0.5 else "Mixed")
    print(f"{'='*70}")
    print(f"Training {num_layers}-layer {act_type} Transformer on {task}")
    clip_str = f", grad_clip={grad_clip}" if grad_clip is not None else ""
    wd_targeted_str = ""
    if wd_attn is not None or wd_mlp is not None:
        wd_targeted_str = f", wd_attn={wd_attn}, wd_mlp={wd_mlp}"
    swap_str = ", augment_swap=True" if augment_swap else ""
    ln_str = ", use_ln=True" if use_ln else ""
    reserve_str = (
        f", reserve_fraction={reserve_fraction}, include_reserved_in_train={include_reserved_in_train}"
        if reserve_fraction > 0 or include_reserved_in_train else ""
    )
    print(f"  attention_rate={attention_rate} ({algo_type}), d_model={d_model}, weight_decay={weight_decay}, seed={seed}{clip_str}{wd_targeted_str}{ln_str}{swap_str}{reserve_str}")
    print(f"{'='*70}")

    # Setup environment
    device = setup_training_environment(seed=seed, device=device)

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        p_str = f"_p{p}"
        alpha_str = f"_a{attention_rate:.1f}" if attention_rate != 1.0 else ""
        width_str = f"_d{d_model}" if d_model != 128 else ""
        seed_str = f"_s{seed}" if seed != 777 else ""
        frac_str = "" if abs(fraction - 0.3) < 1e-12 else f"_f{fraction:.2f}".replace(".", "p")
        reserve_slug = ""
        if reserve_fraction > 0:
            reserve_slug = f"_r{reserve_fraction:.2f}".replace(".", "p")
        reserve_mode_slug = "_randextra" if include_reserved_in_train else ""
        output_dir = (
            f"experiments/{act_type.lower()}_{num_layers}layer_{task}"
            f"{p_str}{alpha_str}{width_str}{frac_str}{reserve_slug}{reserve_mode_slug}{seed_str}_{timestamp}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path = output_path / 'checkpoints'
    checkpoints_path.mkdir(exist_ok=True)

    print(f"Output: {output_path}")

    # Setup configuration
    config = setup_training_config(
        p=p,
        fraction=fraction,
        num_layers=num_layers,
        act_type=act_type,
        task=task,
        total_epochs=total_epochs,
        attention_rate=attention_rate,
        d_model=d_model,
        weight_decay=weight_decay,
        seed=seed,
        use_ln=use_ln,
        reserve_fraction=reserve_fraction,
        include_reserved_in_train=include_reserved_in_train,
    )
    config['device'] = device
    config['wd_attn'] = wd_attn
    config['wd_mlp'] = wd_mlp
    config['augment_swap'] = augment_swap
    if record_frequency is not None:
        config['record_frequency'] = record_frequency
    if config_overrides is not None:
        config.update(config_overrides)
        config['device'] = device
    if init_checkpoint is not None or freeze != 'none':
        config['init_checkpoint'] = init_checkpoint
        config['init_checkpoint_sha256'] = (
            hashlib.sha256(Path(init_checkpoint).read_bytes()).hexdigest()
            if init_checkpoint is not None else None
        )
        config['start_epoch'] = start_epoch
        config['freeze'] = freeze

    # Save config
    with open(output_path / 'config.pkl', 'wb') as f:
        pickle.dump(config, f)

    # Create data
    print("\nCreating dataset...")
    train_data, test_data = create_data(
        p=config['p'],
        fraction=config['fraction'],
        seed=config['seed'],
        device=device,
        task=task,
        reserve_fraction=config.get('reserve_fraction', 0.0),
        include_reserved_in_train=config.get('include_reserved_in_train', False),
    )
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # Create model
    print("\nCreating model...")
    model_config = get_model_config_from_data(config)
    model = Transformer(**model_config).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if init_checkpoint is not None:
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        print(f"  Initialized from {init_checkpoint} (start epoch {start_epoch})")
    if freeze != 'none':
        frozen_names = [name for name, _ in model.named_parameters()
                        if any(tag in name for tag in FREEZE_GROUPS[freeze])]
        for name, param in model.named_parameters():
            if name in frozen_names:
                param.requires_grad_(False)
        print(f"  Frozen ({freeze}): {', '.join(frozen_names)}")

    # Checkpoint callback
    saved_checkpoints = []

    def checkpoint_callback(model, epoch, metrics):
        if save_checkpoints and (epoch in checkpoint_epochs if checkpoint_epochs is not None else epoch % checkpoint_freq == 0):
            ckpt_path = checkpoints_path / f'{act_type.lower()}_epoch_{epoch:05d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config
            }, ckpt_path)
            saved_checkpoints.append(epoch)

    # Train
    print(f"\nTraining for {total_epochs} epochs...")
    metrics = basic_training_loop(
        model=model,
        train_data=train_data,
        test_data=test_data,
        total_epochs=total_epochs,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        record_frequency=config['record_frequency'],
        device=device,
        verbose=True,
        checkpoint_callback=checkpoint_callback,
        max_grad_norm=grad_clip,
        wd_attn=wd_attn,
        wd_mlp=wd_mlp,
        augment_swap=augment_swap,
        start_epoch=start_epoch,
    )

    # Save final checkpoint
    final_ckpt_path = checkpoints_path / f'{act_type.lower()}_epoch_{total_epochs-1:05d}.pth'
    torch.save({
        'epoch': total_epochs - 1,
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_ckpt_path)

    # Save training metrics
    metrics_dict = metrics.to_dict()
    metrics_dict['config'] = config
    with open(output_path / 'training_metrics.pkl', 'wb') as f:
        pickle.dump(metrics_dict, f)

    # Generate training curves plot
    plot_training_curves(
        metrics,
        title=f'{num_layers}L {act_type} on {task}',
        save_path=output_path / 'training_curves.png'
    )

    # Compute GCS if requested
    if compute_gcs:
        print("\nComputing GCS with final geometry...")

        # Build final geometry/geometries
        if num_layers > 1:
            last_layer = num_layers - 1
            layers_to_compute = [last_layer] if gcs_last_layer_only else list(range(num_layers))
            if gcs_last_layer_only:
                print(f"  Building final geometry for last layer (layer {last_layer}) only...")
            else:
                print(f"  Building per-layer final geometries for {num_layers} layers...")
            final_geometries = {}
            for layer_idx in layers_to_compute:
                final_geometries[layer_idx] = build_complete_geometric_structure(
                    model=model,
                    train_data=train_data,
                    N=config['analysis_N'],
                    k=config['analysis_k'],
                    d=config['analysis_d'],
                    device=device,
                    layer_idx=layer_idx,
                    hook_point='hook_resid_post'
                )

                with open(output_path / f'final_geometry_layer{layer_idx}.pkl', 'wb') as f:
                    pickle.dump(final_geometries[layer_idx], f)

            # Backward compatibility: keep canonical geometry filename
            compat_layer = canonical_layer if canonical_layer is not None else (last_layer if gcs_last_layer_only else 0)
            if compat_layer not in layers_to_compute:
                raise ValueError('Canonical layer must be one of the measured layers')
            with open(output_path / 'final_geometry.pkl', 'wb') as f:
                pickle.dump(final_geometries[compat_layer], f)
        else:
            final_geometry = build_complete_geometric_structure(
                model=model,
                train_data=train_data,
                N=config['analysis_N'],
                k=config['analysis_k'],
                d=config['analysis_d'],
                device=device
            )
            final_geometries = {0: final_geometry}

            with open(output_path / 'final_geometry.pkl', 'wb') as f:
                pickle.dump(final_geometry, f)

        # Compute GCS for checkpoints
        path_scores = {
            'overall_path': [],
            'attention_path': [],
            'mlp_path': [],
        }
        if num_layers > 1:
            for layer_idx in layers_to_compute:
                path_scores[f'layer{layer_idx}_overall_path'] = []
                path_scores[f'layer{layer_idx}_attention_path'] = []
                path_scores[f'layer{layer_idx}_mlp_path'] = []

        gcs_results = {
            'epochs': [],
            'path_gcs_scores': path_scores,
            'geometry_type': 'final_geometry_per_layer' if num_layers > 1 else 'final_geometry',
            'analysis_params': {
                'N': config['analysis_N'],
                'k': config['analysis_k'],
                'd': config['analysis_d'],
                'num_layers': num_layers,
            }
        }

        # Backward compatibility with older flat schema consumers
        gcs_results['overall_path'] = gcs_results['path_gcs_scores']['overall_path']
        gcs_results['attention_path'] = gcs_results['path_gcs_scores']['attention_path']
        gcs_results['mlp_path'] = gcs_results['path_gcs_scores']['mlp_path']

        checkpoint_files = sorted(checkpoints_path.glob('*.pth'))
        for ckpt_file in checkpoint_files:
            epoch = None
            try:
                ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
                epoch = ckpt['epoch']

                if not (epoch in gcs_epochs if gcs_epochs is not None else epoch % gcs_freq == 0):
                    continue

                model.load_state_dict(ckpt['model_state_dict'])

                if num_layers > 1:
                    layer_path_results = {}
                    for layer_idx in layers_to_compute:
                        path_results = analyze_3path_gmn_tsa(
                            model=model,
                            complete_geometry=final_geometries[layer_idx],
                            target_layer=layer_idx,
                            device=device
                        )
                        layer_path_results[layer_idx] = path_results

                    gcs_results['epochs'].append(epoch)

                    for layer_idx in layers_to_compute:
                        path_results = layer_path_results[layer_idx]

                        gcs_results['path_gcs_scores'][f'layer{layer_idx}_overall_path'].append(
                            path_results.get('overall_path')
                        )
                        gcs_results['path_gcs_scores'][f'layer{layer_idx}_attention_path'].append(
                            path_results.get('attention_path')
                        )
                        gcs_results['path_gcs_scores'][f'layer{layer_idx}_mlp_path'].append(
                            path_results.get('mlp_path')
                        )

                        # Canonical 3-path: last layer if last-layer-only, else layer 0
                        if layer_idx == compat_layer:
                            gcs_results['path_gcs_scores']['overall_path'].append(path_results.get('overall_path'))
                            gcs_results['path_gcs_scores']['attention_path'].append(path_results.get('attention_path'))
                            gcs_results['path_gcs_scores']['mlp_path'].append(path_results.get('mlp_path'))
                else:
                    path_results = analyze_3path_gmn_tsa(
                        model=model,
                        complete_geometry=final_geometries[0],
                        target_layer=0,
                        device=device
                    )

                    gcs_results['epochs'].append(epoch)

                    gcs_results['path_gcs_scores']['overall_path'].append(path_results.get('overall_path'))
                    gcs_results['path_gcs_scores']['attention_path'].append(path_results.get('attention_path'))
                    gcs_results['path_gcs_scores']['mlp_path'].append(path_results.get('mlp_path'))

            except Exception as e:
                print(f"  GCS failed for {ckpt_file.name}: {e}")
                if epoch is not None and (epoch in gcs_epochs if gcs_epochs is not None else epoch % gcs_freq == 0):
                    gcs_results['epochs'].append(epoch)
                    for key in gcs_results['path_gcs_scores']:
                        gcs_results['path_gcs_scores'][key].append(None)

        # Save GCS results
        with open(output_path / 'gcs_results.pkl', 'wb') as f:
            pickle.dump(gcs_results, f)

        # Plot GCS evolution
        plot_gcs_3path(
            gcs_results,
            title=f'{num_layers}L {act_type} GCS Evolution',
            save_path=output_path / 'gcs_evolution.png'
        )

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")
    print(f"Output saved to: {output_path}")
    print(f"Checkpoints: {len(saved_checkpoints) + 1}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Train Transformer on modular arithmetic')
    parser.add_argument('--layers', type=int, default=1, choices=[1, 2, 3],
                        help='Number of transformer layers')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'GeLU', 'SiLU', 'SeLU'],
                        help='Activation function')
    parser.add_argument('--task', type=str, default='add',
                        choices=['add', 'sub', 'mul', 'div'],
                        help='Arithmetic task')
    parser.add_argument('--p', type=int, default=113,
                        help='Prime modulus for modular arithmetic dataset')
    parser.add_argument('--fraction', type=float, default=0.3,
                        help='Training data fraction (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=15000,
                        help='Total training epochs')
    parser.add_argument('--checkpoint-freq', type=int, default=200,
                        help='Checkpoint saving frequency')
    parser.add_argument('--no-checkpoints', action='store_true',
                        help='Disable checkpoint saving')
    parser.add_argument('--gcs-last-layer-only', action='store_true',
                        help='For multi-layer models: only compute GCS for the last layer (faster)')
    parser.add_argument('--no-gcs', action='store_true',
                        help='Disable GCS computation')
    parser.add_argument('--gcs-freq', type=int, default=200,
                        help='GCS computation frequency')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Computing device')
    # Clock vs Pizza control
    parser.add_argument('--attention-rate', type=float, default=1.0,
                        help='Attention rate α: 1.0=Clock (full attention), 0.0=Pizza (constant attention)')
    # Width ablation
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model dimension (width). MLP will be 4x this value.')
    # Weight decay control
    parser.add_argument('--weight-decay', type=float, default=1.0,
                        help='Weight decay for AdamW optimizer')
    # Gradient clipping
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='Maximum gradient norm for clipping (default: None = no clipping)')
    # Targeted weight decay
    parser.add_argument('--wd-attn', type=float, default=None,
                        help='Weight decay for attention params (default: same as --weight-decay)')
    parser.add_argument('--wd-mlp', type=float, default=None,
                        help='Weight decay for MLP params (default: same as --weight-decay)')
    # Architecture control
    parser.add_argument('--use-ln', '--use-layernorm', action='store_true', dest='use_ln',
                        help='Use pre-LayerNorm transformer blocks')
    # Operand-swap augmentation
    parser.add_argument('--augment-swap', action='store_true',
                        help='Train on both (a,b) and (b,a) with same label')
    parser.add_argument('--reserve-fraction', type=float, default=0.0,
                        help='Held-out reserve fraction removed from evaluation for matched-control experiments')
    parser.add_argument('--include-reserved-in-train', action='store_true',
                        help='Append the reserved split to the base training set (for random-extra controls)')
    # Random seed
    parser.add_argument('--seed', type=int, default=777,
                        help='Random seed for reproducibility (default: 777)')
    # Branching from a saved checkpoint
    parser.add_argument('--init-checkpoint', type=str, default=None,
                        help='Load all parameters from this checkpoint before training (fresh optimizer state)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Epoch index of --init-checkpoint; training continues to --epochs')
    parser.add_argument('--freeze', type=str, default='none',
                        choices=['none'] + sorted(FREEZE_GROUPS),
                        help='Parameter group held fixed (no update, no weight decay)')
    parser.add_argument('--record-freq', type=int, default=None,
                        help='Metric/checkpoint recording frequency (default: config value)')

    args = parser.parse_args()
    if args.start_epoch and args.init_checkpoint is None:
        parser.error('--start-epoch requires --init-checkpoint')

    train_transformer(
        num_layers=args.layers,
        act_type=args.activation,
        task=args.task,
        p=args.p,
        fraction=args.fraction,
        total_epochs=args.epochs,
        save_checkpoints=not args.no_checkpoints,
        checkpoint_freq=args.checkpoint_freq,
        compute_gcs=not args.no_gcs,
        gcs_freq=args.gcs_freq,
        gcs_last_layer_only=args.gcs_last_layer_only,
        output_dir=args.output,
        device=args.device,
        attention_rate=args.attention_rate,
        d_model=args.d_model,
        weight_decay=args.weight_decay,
        seed=args.seed,
        grad_clip=args.grad_clip,
        wd_attn=args.wd_attn,
        wd_mlp=args.wd_mlp,
        use_ln=args.use_ln,
        augment_swap=args.augment_swap,
        reserve_fraction=args.reserve_fraction,
        include_reserved_in_train=args.include_reserved_in_train,
        init_checkpoint=args.init_checkpoint,
        start_epoch=args.start_epoch,
        freeze=args.freeze,
        record_frequency=args.record_freq,
    )


if __name__ == '__main__':
    main()
