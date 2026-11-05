import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


def runner():
    try:
        return importlib.import_module('gcs.runner')
    except ModuleNotFoundError:
        pytest.fail('Standalone train/analyze runner is not implemented')


def tiny_config():
    return dict(p=7,fraction=.3,seed=42,task='add',num_layers=1,d_vocab=8,d_model=8,
                d_mlp=16,d_head=4,num_heads=2,n_ctx=3,act_type='GeLU',use_ln=False,
                total_epochs=3,lr=.001,weight_decay=1.,record_frequency=1,
                analysis_N=8,analysis_k=3,analysis_d=2)


def test_train_save_load_analyze_and_no_overwrite(tmp_path):
    m = runner()
    torch.set_num_threads(1)
    result = m.train(tiny_config(), tmp_path/'run', device='cpu')
    initial, cfg = m.load_model(tmp_path/'run/initial.pth', device='cpu')
    final, _ = m.load_model(Path(result['final_checkpoint']), device='cpu')
    assert next(final.parameters()).dtype == torch.float64
    assert not torch.equal(initial.embed.W_E, final.embed.W_E)
    scores = m.analyze(Path(result['final_checkpoint']), tmp_path/'analysis', device='cpu')
    assert set(scores['gcs']) == {'overall_path','attention_path','mlp_path'}
    assert all(np.isfinite(x) for x in scores['gcs'].values())
    with pytest.raises(FileExistsError):
        m.train(tiny_config(), tmp_path/'run', device='cpu')


def test_reset_changes_only_selected_component(tmp_path):
    m = runner()
    torch.set_num_threads(1)
    result = m.train(tiny_config(), tmp_path/'run', device='cpu')
    host, _ = m.load_model(Path(result['final_checkpoint']),device='cpu')
    donor, _ = m.load_model(tmp_path/'run/initial.pth',device='cpu')
    changed = m.reset_component(host, donor, 'attention')
    host_state, donor_state = host.state_dict(), donor.state_dict()
    for key, value in changed.state_dict().items():
        expected = donor_state[key] if '.attn.W_' in key else host_state[key]
        torch.testing.assert_close(value, expected, rtol=0,atol=0)


def test_frozen_mlp_receives_neither_gradient_nor_weight_decay(tmp_path):
    m = runner()
    torch.set_num_threads(1)
    result = m.train(tiny_config(), tmp_path/'run', device='cpu')
    outcome = m.continue_fixed(Path(result['final_checkpoint']),tmp_path/'run/initial.pth',
        tmp_path/'branch',component='mlp',updates=3,record_frequency=1,device='cpu')
    assert outcome['frozen_exact']
    assert outcome['completed_updates'] == 3
    assert outcome['trainable_parameters_changed']


def test_paper_protocol_emits_requested_epochs_and_geometry(tmp_path,monkeypatch):
    import pickle
    monkeypatch.setenv('MPLBACKEND','Agg')
    monkeypatch.setenv('MPLCONFIGDIR',str(tmp_path/'mpl'))
    torch.set_num_threads(1)
    config=tiny_config()
    schedule={'checkpoint_epochs':[0,1,2], 'gcs_epochs':[0,2]}
    try:
        from gcs.primary import train_primary
    except ModuleNotFoundError:
        pytest.fail('Primary training does not yet preserve the integrated geometry protocol')
    result=train_primary(config,schedule,tmp_path/'paper',device='cpu')
    with (tmp_path/'paper/gcs_results.pkl').open('rb') as f: scores=pickle.load(f)
    assert scores['epochs']==[0,2]
    assert all(np.isfinite(scores['path_gcs_scores'][k]).all() for k in ['overall_path','attention_path','mlp_path'])
    assert (tmp_path/'paper/final_geometry.pkl').is_file()
    assert result['measurement_layer']==0
    with pytest.raises(FileExistsError):
        train_primary(config,schedule,tmp_path/'paper',device='cpu')


def test_primary_multilayer_canonical_outputs_measure_last_block(tmp_path,monkeypatch):
    import pickle
    from gcs.primary import train_primary
    monkeypatch.setenv('MPLBACKEND','Agg');torch.set_num_threads(1)
    config=dict(tiny_config(),num_layers=2)
    train_primary(config,{'checkpoint_epochs':[0,1,2],'gcs_epochs':[0,2]},tmp_path/'run')
    scores=pickle.loads((tmp_path/'run/gcs_results.pkl').read_bytes())['path_gcs_scores']
    np.testing.assert_array_equal(scores['attention_path'],scores['layer1_attention_path'])
    canonical=pickle.loads((tmp_path/'run/final_geometry.pkl').read_bytes())
    last=pickle.loads((tmp_path/'run/final_geometry_layer1.pkl').read_bytes())
    np.testing.assert_array_equal(canonical['X_sample'],last['X_sample'])


def test_staged_primary_retains_post_update_boundary_and_restarts_optimizer(tmp_path,monkeypatch):
    from gcs.primary import train_primary
    monkeypatch.setenv('MPLBACKEND','Agg');torch.set_num_threads(1)
    config=dict(tiny_config(),total_epochs=6,record_frequency=2)
    schedule={'checkpoint_epochs':[0,2,3,4,5],'gcs_epochs':[0,2,4],'training_stages':[4,6]}
    result=train_primary(config,schedule,tmp_path/'staged')
    before,_=runner().load_model(tmp_path/'staged/checkpoints/gelu_epoch_00003.pth')
    after,_=runner().load_model(tmp_path/'staged/checkpoints/gelu_epoch_00004.pth')
    for k,v in before.state_dict().items():torch.testing.assert_close(v,after.state_dict()[k],atol=0,rtol=0)
    assert result['training_stages']==[4,6]
    assert result['optimizer_restarts']==1
    continuous=runner().train(config,tmp_path/'continuous')
    a,_=runner().load_model(Path(result['final_checkpoint']))
    b,_=runner().load_model(Path(continuous['final_checkpoint']))
    assert any(not torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_all_primary_manifests_have_executable_stage_schedules():
    import json
    from gcs.primary import plan_stages
    root=Path(__file__).resolve().parents[1]
    rows=json.loads((root/'manifests/primary_runs.json').read_text())
    plans=[plan_stages(json.loads((root/r['config']).read_text()),r)for r in rows]
    assert len(plans)==104
    assert sum(len(p)>1 for p in plans)==7
