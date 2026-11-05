import json
import os
import csv
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
import torch

from test_runner import tiny_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def nkd_loader(tmp_path, monkeypatch):
    monkeypatch.setenv('GCS_DATA_ROOT', str(tmp_path / 'reader-data'))
    from gcs.legacy.run_gcs_nkd_selector_sensitivity import load_frozen_cohort
    return load_frozen_cohort


def _write_nkd_manifest(path, rows, *, source_root=None):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['act', 'task', 'seed', 'exp_dir', 'status'])
        writer.writeheader()
        writer.writerows(rows)
    if source_root is not None:
        path.with_suffix('.paths.json').write_text(json.dumps({
            'source_root': str(source_root),
            'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }))


def test_nkd_archived_cohort_relocates_with_protocol_inputs(tmp_path, nkd_loader):
    root = tmp_path / 'reader-data'
    shutil.copytree(ROOT / 'configs/archive_protocols', root)
    manifest = root / 'rebuttal_2026/direct_jacobian_graph_audit_20260723/run_manifest.csv'
    cohort = nkd_loader(manifest, root=root)
    assert len(cohort) == 23
    assert cohort[0] == ('ReLU', 'add', 777,
        str(root / 'experiments/relu_1layer_add_20260213_004649'))
    assert all(Path(row[3]).is_relative_to(root) for row in cohort)


def test_nkd_relative_manifest_paths_use_data_root(tmp_path, monkeypatch, nkd_loader):
    root = tmp_path / 'reader-data'
    manifest = tmp_path / 'runs.csv'
    _write_nkd_manifest(manifest, [dict(act='ReLU', task='add', seed=123,
        exp_dir='experiments/add', status='completed')])
    monkeypatch.chdir(tmp_path)
    cohort = nkd_loader(manifest, root=root,
        registered_runs=[('ReLU', 'add', 123, 'experiments/add')], expected_count=1)
    assert cohort == [('ReLU', 'add', 123, str(root / 'experiments/add'))]


def test_nkd_path_mapping_rejects_a_changed_manifest(tmp_path, nkd_loader):
    manifest = tmp_path / 'runs.csv'
    _write_nkd_manifest(manifest, [dict(act='ReLU', task='add', seed=123,
        exp_dir=str(tmp_path / 'experiments/add'), status='completed')], source_root=tmp_path)
    with manifest.open('a') as handle:
        handle.write('\n')
    with pytest.raises(RuntimeError, match='checksum'):
        nkd_loader(manifest, root=tmp_path,
            registered_runs=[('ReLU', 'add', 123, 'experiments/add')], expected_count=1)


@pytest.mark.parametrize('problem, message', [
    ('duplicate', 'Duplicate'), ('unknown', 'Unexpected'),
    ('unfinished', 'not complete'), ('foreign_path', 'outside'),
    ('wrong_path', 'path mismatch'), ('row_count', 'row count'),
])
def test_nkd_relocation_preserves_cohort_rejections(tmp_path, problem, message, nkd_loader):
    archive = tmp_path / 'archive'
    root = tmp_path / 'reader-data'
    rows = [dict(act='ReLU', task='add', seed=123,
                 exp_dir=str(archive / 'experiments/add'), status='completed'),
            dict(act='GeLU', task='mul', seed=777,
                 exp_dir=str(archive / 'experiments/mul'), status='skipped_complete')]
    if problem == 'duplicate': rows[1] = dict(rows[0])
    elif problem == 'unknown': rows[1]['task'] = 'sub'
    elif problem == 'unfinished': rows[1]['status'] = 'failed'
    elif problem == 'foreign_path': rows[1]['exp_dir'] = str(tmp_path / 'foreign/experiments/mul')
    elif problem == 'wrong_path': rows[1]['exp_dir'] = str(archive / 'experiments/wrong')
    elif problem == 'row_count': rows.pop()
    manifest = tmp_path / 'runs.csv'
    _write_nkd_manifest(manifest, rows, source_root=archive)
    with pytest.raises(RuntimeError, match=message):
        nkd_loader(manifest, root=root,
            registered_runs=[('ReLU', 'add', 123, 'experiments/add'),
                             ('GeLU', 'mul', 777, 'experiments/mul')], expected_count=2)


def test_nkd_foreign_absolute_paths_require_explicit_origin(tmp_path, nkd_loader):
    manifest = tmp_path / 'runs.csv'
    _write_nkd_manifest(manifest, [dict(act='ReLU', task='add', seed=123,
        exp_dir=str(tmp_path / 'foreign/experiments/add'), status='completed')])
    with pytest.raises(RuntimeError, match='path mismatch'):
        nkd_loader(manifest, root=tmp_path / 'reader-data',
            registered_runs=[('ReLU', 'add', 123, 'experiments/add')], expected_count=1)


def test_legacy_loader_keeps_float64_in_fresh_process():
    code='''import torch,sys
from gcs.legacy.reanalyze_gcs import load_checkpoint
state=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['model_state_dict']
model,_=load_checkpoint(sys.argv[1],device='cpu')
assert model.embed.W_E.dtype==torch.float64
torch.testing.assert_close(model.embed.W_E,state['embed.W_E'],rtol=0,atol=0)
'''
    r=subprocess.run([sys.executable,'-c',code,str(ROOT/'examples/checkpoints/t_minus.pth')],
        env=dict(os.environ,MPLBACKEND='Agg'),capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_n_sweep_uses_final_reference_for_every_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setenv('MPLBACKEND','Agg');torch.set_num_threads(1)
    from gcs.runner import train,analyze
    from gcs.legacy.ablation_n import run_n_ablation
    train(dict(tiny_config(),task='sub'),tmp_path/'run')
    result=run_n_ablation(tmp_path/'run',[8],k=3,d=2,checkpoint_freq=1,device='cpu',seed=42)
    assert result[8]['epochs']==[0,1,2]
    assert result[8]['reference_epoch']==2
    geom=tmp_path/'run/n_ablation_study/geometry_N8.npz'
    replay=analyze(tmp_path/'run/checkpoints/gelu_epoch_00000.pth',tmp_path/'replay',geometry_path=geom)
    for field,path in [('overall_gcs','overall_path'),('attention_gcs','attention_path'),('mlp_gcs','mlp_path')]:
        assert abs(result[8][field][0]-replay['gcs'][path])<1e-10


def test_grokalign_producer_is_consumable_by_selector_evaluator(tmp_path,monkeypatch):
    monkeypatch.setenv('MPLBACKEND','Agg');monkeypatch.setenv('GCS_DATA_ROOT',str(tmp_path));torch.set_num_threads(1)
    from gcs.runner import train
    from gcs.legacy.systematic_grokalign_comparison import produce_selector_cache
    from gcs.legacy.evaluate_grokalign_selector_benchmark import load_grokalign_cache,centroid_centers
    train(tiny_config(),tmp_path/'run')
    dest=tmp_path/'grokalign.pkl'
    produce_selector_cache([('GeLU','add',42,str(tmp_path/'run'))],dest,device='cpu',checkpoint_freq=1,max_samples=4)
    loaded=load_grokalign_cache(dest)[('GeLU','add',42)]['grokalign']
    assert loaded['epochs']==[0,1,2]
    assert np.isfinite(loaded['grokalign_scores']).all()
    centers=centroid_centers(np.array(loaded['epochs']),np.array(loaded['grokalign_scores']),3)
    assert set(centers)=={'centroid_alignment_peak','centroid_alignment_abs','centroid_alignment_rise','centroid_alignment_drop'}
