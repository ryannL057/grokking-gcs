"""Explicit local entrypoints around the original model and training loop."""
from __future__ import annotations

import copy
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .core.models import Transformer
from .core.data import create_data
from .core.training import basic_training_loop, setup_training_environment, full_accuracy
from .analysis.geometry import build_complete_geometric_structure
from .analysis.gmn_tsa import compute_path_gmn_tsa
from .analysis.paths import get_path_definitions
from .artifacts import sha256


def model_config(config: dict) -> dict:
    keys = set(inspect.signature(Transformer).parameters)
    return {key:value for key,value in config.items() if key in keys}


def dataset(config: dict):
    return create_data(p=config['p'],fraction=config['fraction'],seed=config['seed'],task=config['task'],
                       device=config.get('device','cpu'))


def ensure_device(device: str) -> None:
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable. Choose --device cpu explicitly.')


def train(config: dict, output: Path, *, device: str = 'cpu') -> dict:
    ensure_device(device)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    config = dict(config,device=device)
    setup_training_environment(config['seed'],device)
    train_data,test_data = dataset(config)
    model = Transformer(**model_config(config)).to(device)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with (output/'config.pkl').open('wb') as stream:pickle.dump(config,stream)
    (output/'checkpoints').mkdir()
    def save(path,epoch,updates):
        torch.save({'model_state_dict':model.state_dict(),'config':config,'epoch':epoch,
                    'completed_updates':updates},path)
    save(output/'initial.pth',0,0)
    def checkpoint(model,epoch,metrics):
        save(output/'checkpoints'/f"{config['act_type'].lower()}_epoch_{epoch:05d}.pth",epoch,epoch)
    metrics = basic_training_loop(model,train_data,test_data,total_epochs=config['total_epochs'],
            lr=config['lr'],weight_decay=config['weight_decay'],record_frequency=config['record_frequency'],
            device=device,verbose=False,checkpoint_callback=checkpoint)
    epoch = config['total_epochs']-1
    final = output/'checkpoints'/f"{config['act_type'].lower()}_epoch_{epoch:05d}.pth"
    save(final,epoch,config['total_epochs'])
    (output/'metrics.json').write_text(json.dumps(metrics.to_dict(),indent=2)+'\n')
    with (output/'training_metrics.pkl').open('wb') as stream:pickle.dump({**metrics.to_dict(),'config':config},stream)
    result={'final_checkpoint':str(final.resolve()),'epochs':config['total_epochs'],
            'final_test_accuracy':full_accuracy(model,test_data,device),
            'checkpoint_epoch_note':'Regular records precede that iteration update; final record follows all updates.'}
    (output/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def continue_fixed(host_path: Path,donor_path: Path|None,output: Path,*,component: str='mlp',
                   updates: int,config: dict|None=None,device: str='cpu',record_frequency: int=100,
                   stop_after_success: int|None=None) -> dict:
    """Fresh-AdamW continuation with an exactly fixed transplanted component.

    Updates are additional updates, independent of the historical epoch label.
    The caller supplies the matched budget F+1-t_minus (or three times it).
    """
    from .core.training import create_optimizer,full_loss
    if updates<=0 or record_frequency<=0:raise ValueError('updates and record_frequency must be positive')
    if component not in ['mlp','attention','none']:raise ValueError('Unknown fixed component')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    host,config=load_model(host_path,config=config,device=device)
    config=dict(config,device=device)
    if donor_path is not None:
        if component=='none':raise ValueError('Unfrozen branch does not take a donor')
        donor,_=load_model(donor_path,config=config,device=device)
        host=reset_component(host,donor,component)
    before={k:p.detach().clone() for k,p in host.named_parameters()}
    frozen=[]
    for name,param in host.named_parameters():
        should_freeze=('.mlp.' in name if component=='mlp' else '.attn.W_' in name if component=='attention' else False)
        if should_freeze:param.requires_grad_(False);frozen.append(name)
    optimizer=create_optimizer(host,lr=config['lr'],weight_decay=config['weight_decay'])
    train_data,test_data=dataset(config);curve=[];first=None;completed=0
    for update in range(updates+1):
        if update%record_frequency==0 or update==updates:
            host.eval();accuracy=full_accuracy(host,test_data,device)
            curve.append({'updates':update,'test_accuracy':accuracy})
            if accuracy>=.99 and first is None:first=update
        if update==updates or (first is not None and stop_after_success is not None and update>=first+stop_after_success):
            completed=update;break
        host.train();optimizer.zero_grad();loss=full_loss(host,train_data,device);loss.backward();optimizer.step()
    state=dict(host.named_parameters())
    frozen_exact=all(torch.equal(before[k],state[k]) for k in frozen)
    if not frozen_exact:raise RuntimeError('Frozen parameters changed')
    torch.save({'model_state_dict':host.state_dict(),'config':config,'completed_updates':completed},output/'endpoint.pth')
    result={'component':component,'budget_updates':updates,'completed_updates':completed,'first_observed_99_updates':first,
            'reached_99':first is not None,'frozen_exact':frozen_exact,'frozen_parameters':frozen,
            'trainable_parameters_changed':any(not torch.equal(before[k],p) for k,p in state.items() if k not in frozen),
            'curve':curve,'host_sha256':sha256(host_path),'donor_sha256':sha256(donor_path) if donor_path else None,
            'optimizer':'fresh AdamW; beta=(0.9,0.98), eps=1e-8','record_frequency':record_frequency}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    return result


def load_model(checkpoint: Path, *, config: dict | None = None, device: str = 'cpu'):
    ensure_device(device)
    payload = torch.load(checkpoint,map_location=device,weights_only=False)
    config = config or payload.get('config') or payload.get('model_config')
    if config is None:
        raise ValueError('Checkpoint has no configuration; pass an explicit config.')
    config=dict(config,device=device)
    torch.set_default_dtype(torch.float64)
    model = Transformer(**model_config(config)).to(device=device,dtype=torch.float64)
    model.load_state_dict(payload.get('model_state_dict',payload),strict=True)
    return model.eval(),config


def save_geometry(geometry: dict, path: Path) -> None:
    arrays = {}
    for key,value in geometry.items():
        if torch.is_tensor(value): value=value.detach().cpu().numpy()
        elif isinstance(value,list) and value and torch.is_tensor(value[0]):
            value=np.stack([x.detach().cpu().numpy() for x in value])
        if value is not None: arrays[key]=np.asarray(value)
    np.savez_compressed(path,**arrays)


def load_geometry(path: Path,device: str) -> dict:
    tensors={'X_sample','X_prompts','X_embeddings','all_input_tangent_bases'}
    with np.load(path,allow_pickle=False) as archive:
        result={k:(torch.from_numpy(archive[k]).to(device) if k in tensors else
                   archive[k].item() if archive[k].ndim==0 else archive[k]) for k in archive.files}
    return result


def analyze(checkpoint: Path,output: Path,*,config: dict|None=None,reference: Path|None=None,
            geometry_path: Path|None=None,device: str='cpu',token_probe: str='final',
            alignment: str='cosine') -> dict:
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    model,config=load_model(checkpoint,config=config,device=device)
    if geometry_path:
        geometry=load_geometry(geometry_path,device)
    else:
        refmodel,_=load_model(reference or checkpoint,config=config,device=device)
        train_data,_=dataset(config)
        torch.manual_seed(config['seed'])
        geometry=build_complete_geometric_structure(refmodel,train_data,
            N=config['analysis_N'],k=config['analysis_k'],d=config['analysis_d'],
            device=device,layer_idx=config['num_layers']-1)
    save_geometry(geometry,output/'geometry.npz')
    scores={name:float(compute_path_gmn_tsa(model,geometry,path['start'],path['end'],
              target_layer=config['num_layers']-1,device=device,alignment_method=alignment,token_probe=token_probe))
            for name,path in get_path_definitions().items()}
    result={'gcs':scores,'N':int(geometry['N']),'k':int(geometry['k']),'d':int(geometry['d']),
            'checkpoint_sha256':sha256(Path(checkpoint)),'geometry_sha256':sha256(output/'geometry.npz'),
            'token_probe':token_probe,'alignment':alignment,'reference':str(reference or geometry_path or checkpoint)}
    (output/'scores.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    return result


def reset_component(host,donor,component: str):
    if component not in ['attention','mlp']:
        raise ValueError('component must be attention or mlp')
    changed=copy.deepcopy(host)
    tags=('.attn.W_Q','.attn.W_K','.attn.W_V','.attn.W_O') if component=='attention' else ('.mlp.W_in','.mlp.W_out','.mlp.b_in','.mlp.b_out')
    state=changed.state_dict()
    donor_state=donor.state_dict()
    for key in state:
        if any(tag in key for tag in tags):state[key]=donor_state[key].clone()
    changed.load_state_dict(state,strict=True)
    return changed


def reset_experiment(host_path: Path,donor_path: Path,output: Path,*,config: dict|None=None,device: str='cpu') -> dict:
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    host,config=load_model(host_path,config=config,device=device)
    donor,_=load_model(donor_path,config=config,device=device)
    _,test_data=dataset(config)
    base=full_accuracy(host,test_data,device)
    result={'base_test_accuracy':base,'host_sha256':sha256(host_path),'donor_sha256':sha256(donor_path)}
    for component in ['attention','mlp']:
        changed=reset_component(host,donor,component)
        acc=full_accuracy(changed,test_data,device)
        result[f'{component}_reset_accuracy']=acc
        result[f'{component}_accuracy_drop']=base-acc
    (output/'reset.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
