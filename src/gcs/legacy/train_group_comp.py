
from gcs.runtime import data_root as _gcs_data_root
#!/usr/bin/env python3
"""
Train a Transformer on generic finite-group composition and optionally compute GCS.

Examples:
    python scripts/train_group_comp.py --group cyclic:120
    python scripts/train_group_comp.py --group dihedral:60 --shuffle-labels
    python scripts/train_group_comp.py --group symmetric:5 --activation ReLU
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import torch

pass  # Package imports do not search the experiment-data directory.

from gcs.analysis import build_complete_geometric_structure, analyze_3path_gmn_tsa
from gcs.core.data import (
    create_group_data,
    get_model_config_from_data,
    group_order,
    group_spec_to_slug,
    setup_group_training_config,
)
from gcs.core.models import Transformer
from gcs.core.training import basic_training_loop, setup_training_environment
from gcs.vis import plot_gcs_3path, plot_training_curves


def train_group_comp(
    group_spec='cyclic:120',
    shuffle_labels=False,
    num_layers=1,
    act_type='SiLU',
    total_epochs=20000,
    save_checkpoints=True,
    checkpoint_freq=200,
    compute_gcs=True,
    gcs_freq=200,
    output_dir=None,
    device='cuda:0',
    attention_rate=1.0,
    d_model=128,
    weight_decay=1.0,
    seed=777,
    fraction=0.3,
):
    order = group_order(group_spec)
    slug = group_spec_to_slug(group_spec)
    print(f"{'=' * 78}")
    print(f"Training {num_layers}-layer {act_type} Transformer on {group_spec} composition")
    print(f"  order={order}, d_model={d_model}, weight_decay={weight_decay}, seed={seed}")
    print(f"  shuffle_labels={shuffle_labels}")
    print(f"{'=' * 78}")

    device = setup_training_environment(seed=seed, device=device)

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        seed_str = f"_s{seed}" if seed != 777 else ""
        shuffle_str = "_shuf" if shuffle_labels else ""
        output_dir = (
            f"experiments/group_{slug}{shuffle_str}_{act_type.lower()}_"
            f"{num_layers}layer{seed_str}_{timestamp}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoints_path = output_path / 'checkpoints'
    checkpoints_path.mkdir(exist_ok=True)

    print(f"Output: {output_path}")

    config = setup_group_training_config(
        group_spec=group_spec,
        fraction=fraction,
        seed=seed,
        total_epochs=total_epochs,
        num_layers=num_layers,
        act_type=act_type,
        attention_rate=attention_rate,
        d_model=d_model,
        weight_decay=weight_decay,
        shuffle_labels=shuffle_labels,
    )
    config['device'] = device

    with open(output_path / 'config.pkl', 'wb') as f:
        pickle.dump(config, f)

    print("\nCreating finite-group dataset...")
    train_data, test_data = create_group_data(
        group_spec=group_spec,
        fraction=config['fraction'],
        seed=config['seed'],
        device=device,
        shuffle_labels=shuffle_labels,
    )
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"  |G|={order}, total pairs={order * order}, d_vocab={config['d_vocab']}")

    print("\nCreating model...")
    model_config = get_model_config_from_data(config)
    model = Transformer(**model_config).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    saved_checkpoints = []

    def checkpoint_callback(model, epoch, metrics):
        if save_checkpoints and epoch % checkpoint_freq == 0:
            ckpt_path = checkpoints_path / f'{act_type.lower()}_epoch_{epoch:05d}.pth'
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'config': config,
                },
                ckpt_path,
            )
            saved_checkpoints.append(epoch)

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
    )

    final_ckpt_path = checkpoints_path / f'{act_type.lower()}_epoch_{total_epochs - 1:05d}.pth'
    torch.save(
        {
            'epoch': total_epochs - 1,
            'model_state_dict': model.state_dict(),
            'config': config,
        },
        final_ckpt_path,
    )

    metrics_dict = metrics.to_dict()
    metrics_dict['config'] = config
    with open(output_path / 'training_metrics.pkl', 'wb') as f:
        pickle.dump(metrics_dict, f)

    plot_training_curves(
        metrics,
        title=f'{group_spec} {num_layers}L {act_type}',
        save_path=output_path / 'training_curves.png',
    )

    if compute_gcs:
        print("\nComputing GCS with final geometry...")
        final_geometry = build_complete_geometric_structure(
            model=model,
            train_data=train_data,
            N=config['analysis_N'],
            k=config['analysis_k'],
            d=config['analysis_d'],
            device=device,
        )

        with open(output_path / 'final_geometry.pkl', 'wb') as f:
            pickle.dump(final_geometry, f)

        gcs_results = {
            'epochs': [],
            'path_gcs_scores': {
                'overall_path': [],
                'attention_path': [],
                'mlp_path': [],
            },
            'geometry_type': 'final_geometry',
            'analysis_params': {
                'N': config['analysis_N'],
                'k': config['analysis_k'],
                'd': config['analysis_d'],
            },
        }

        checkpoint_files = sorted(checkpoints_path.glob('*.pth'))
        for ckpt_file in checkpoint_files:
            epoch = None
            try:
                ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
                epoch = ckpt['epoch']
                if epoch % gcs_freq != 0:
                    continue

                model.load_state_dict(ckpt['model_state_dict'])
                path_results = analyze_3path_gmn_tsa(
                    model=model,
                    complete_geometry=final_geometry,
                    target_layer=0,
                    device=device,
                )

                gcs_results['epochs'].append(epoch)
                for key in ('overall_path', 'attention_path', 'mlp_path'):
                    gcs_results['path_gcs_scores'][key].append(path_results.get(key))
            except Exception as exc:
                print(f"  GCS failed for {ckpt_file.name}: {exc}")
                if epoch is not None and epoch % gcs_freq == 0:
                    gcs_results['epochs'].append(epoch)
                    for key in ('overall_path', 'attention_path', 'mlp_path'):
                        gcs_results['path_gcs_scores'][key].append(None)

        with open(output_path / 'gcs_results.pkl', 'wb') as f:
            pickle.dump(gcs_results, f)

        plot_gcs_3path(
            gcs_results,
            title=f'{group_spec} {num_layers}L {act_type} GCS Evolution',
            save_path=output_path / 'gcs_evolution.png',
        )

        metrics_dict['gcs_steps_final_geo'] = gcs_results['epochs']
        metrics_dict['path_gcs_scores_final_geo'] = gcs_results['path_gcs_scores']
        with open(output_path / 'training_metrics_final_geo.pkl', 'wb') as f:
            pickle.dump(metrics_dict, f)

    print(f"\n{'=' * 78}")
    print("Training complete!")
    print(f"{'=' * 78}")
    print(f"Output saved to: {output_path}")
    print(f"Checkpoints: {len(saved_checkpoints) + 1}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Train Transformer on finite-group composition')
    parser.add_argument('--group', type=str, default='cyclic:120')
    parser.add_argument('--shuffle-labels', action='store_true')
    parser.add_argument('--layers', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--activation', type=str, default='SiLU', choices=['ReLU', 'GeLU', 'SiLU'])
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--checkpoint-freq', type=int, default=200)
    parser.add_argument('--no-checkpoints', action='store_true')
    parser.add_argument('--no-gcs', action='store_true')
    parser.add_argument('--gcs-freq', type=int, default=200)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--weight-decay', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--fraction', type=float, default=0.3)

    args = parser.parse_args()
    train_group_comp(
        group_spec=args.group,
        shuffle_labels=args.shuffle_labels,
        num_layers=args.layers,
        act_type=args.activation,
        total_epochs=args.epochs,
        save_checkpoints=not args.no_checkpoints,
        checkpoint_freq=args.checkpoint_freq,
        compute_gcs=not args.no_gcs,
        gcs_freq=args.gcs_freq,
        output_dir=args.output,
        device=args.device,
        d_model=args.d_model,
        weight_decay=args.weight_decay,
        seed=args.seed,
        fraction=args.fraction,
    )


if __name__ == '__main__':
    main()
