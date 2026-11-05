"""Integrated primary protocol: train, then measure without reseeding geometry."""
from pathlib import Path
import json
import os
import pickle
import tempfile


def plan_stages(config: dict, schedule: dict) -> list[int]:
    """Validate the finite schedule before allocating a model or output directory."""
    stages=schedule.get('training_stages',[config['total_epochs']])
    if not stages or stages!=sorted(set(stages)) or stages[0]<=0 or stages[-1]!=config['total_epochs']:
        raise ValueError('Training stages must increase and end at total_epochs')
    checkpoints=schedule['checkpoint_epochs'];measurements=schedule['gcs_epochs']
    if not set(measurements)<=set(checkpoints):
        raise ValueError('Every analysis epoch must have a checkpoint')
    endpoints={end-1 for end in stages}
    if not endpoints<=set(checkpoints):raise ValueError('Every training-stage endpoint must be retained')
    if any(e<0 or e>=stages[-1] or (e not in endpoints and e%config['record_frequency']) for e in checkpoints):
        raise ValueError('Checkpoints must be metric records or post-update stage endpoints')
    return list(stages)


def train_primary(config: dict, schedule: dict, output: Path, *, device='cpu') -> dict:
    # Import matplotlib only after selecting a portable noninteractive backend.
    os.environ['MPLBACKEND']='Agg'
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'gcs-training-mpl'))
    from .runner import ensure_device
    from .legacy.train_transformer import train_transformer
    ensure_device(device)
    output=Path(output)
    checkpoint_epochs=schedule['checkpoint_epochs']
    gcs_epochs=schedule['gcs_epochs']
    final_epoch=config['total_epochs']-1
    stages=plan_stages(config,schedule)
    output.mkdir(parents=True,exist_ok=False)
    resolved=dict(config,device=device)
    (output/'config.json').write_text(json.dumps(resolved,indent=2)+'\n')
    (output/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
    first=dict(resolved,total_epochs=stages[0],checkpoint_freq=config['record_frequency'])
    train_transformer(num_layers=config['num_layers'],act_type=config['act_type'],task=config['task'],
        p=config['p'],fraction=config['fraction'],total_epochs=stages[0],
        output_dir=str(output),device=device,seed=config['seed'],d_model=config['d_model'],
        attention_rate=config.get('attention_rate',1.),weight_decay=config['weight_decay'],
        use_ln=config.get('use_ln',False),record_frequency=config['record_frequency'],
        config_overrides=first,checkpoint_epochs=[e for e in checkpoint_epochs if e<stages[0]],
        gcs_epochs=gcs_epochs,compute_gcs=len(stages)==1,canonical_layer=config['num_layers']-1)
    if len(stages)>1:
        from .legacy.resume_training import main as resume
        for end in stages[1:]:
            resume([str(output.resolve()),'--epochs',str(end),'--device',device,
                    '--gcs-freq',str(config['record_frequency']),
                    '--gcs-epochs',','.join(str(e)for e in gcs_epochs if e<end)])
    with (output/'gcs_results.pkl').open('rb') as stream: scores=pickle.load(stream)
    import numpy as np
    if scores['epochs']!=gcs_epochs or any(not np.isfinite(v).all() for v in scores['path_gcs_scores'].values()):
        raise RuntimeError('Integrated training finished with incomplete GCS; inspect gcs_results.pkl')
    result={'measurement_layer':config['num_layers']-1,'analysis_epochs':scores['epochs'],
            'training_stages':stages,'optimizer_restarts':len(stages)-1,
            'geometry_rng':('Continued training RNG, preserving original per-layer construction order.' if len(stages)==1
                            else 'Original resume reseeding/model reconstruction; final reference built only for last block.'),
            'final_checkpoint':str((output/'checkpoints'/f"{config['act_type'].lower()}_epoch_{final_epoch:05d}.pth").resolve())}
    (output/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
