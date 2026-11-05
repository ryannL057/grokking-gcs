import importlib
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    try:return importlib.import_module('gcs.reproduce.appendix')
    except ModuleNotFoundError:pytest.fail('Appendix narrative reducers are not implemented')


def test_truncation_and_non_grok_cohorts_do_not_merge():
    result=module().reproduce('app_endpoints',ROOT/'results/reference')
    assert result['truncated']['n']==24
    assert result['truncated']['combined_signature']==2
    assert result['non_grok']['n']==7


def test_recovery_preserves_short_budgets():
    result=module().reproduce('app_interventions',ROOT/'results/reference')
    assert result['recovery']['reset_attn']['median_updates']==20
    assert result['recovery']['reset_mlp']['median_updates']==1


def test_order_uses_signed_upturn_lag():
    result=module().reproduce('app_order',ROOT/'results/reference')
    assert [r['attention_first'] for r in result['by_depth']]==[33,24,11]


def test_reference_counts_distinguish_each_path_from_joint():
    result=module().reproduce('app_reference_variants',ROOT/'results/reference')
    assert sorted((r['n'],r['mlp_recovery']) for r in result['task_references'].values()) == [(34,19),(35,30),(68,37)]
    assert result['shuffled']['attention']==10
    assert result['shuffled']['mlp']==10
    assert result['shuffled']['joint']==8
    assert result['earlier']['block_upturn_before_grok']=={'original':8,'alternative':4}


def test_prompt_source_counts_are_recomputed_from_recovered_trajectories():
    result=module().reproduce('app_reference_variants',ROOT/'results/reference')
    assert 'prompt_sources' in result, 'Recovered prompt-source evidence is not included'
    pilot=result['prompt_sources']
    assert pilot['n_outputs']==36
    assert {k:(v['n'],v['attention'],v['mlp'],v['joint'])
            for k,v in pilot['by_split'].items()} == {
                'train':(12,12,12,12), 'test':(12,11,12,11), 'full_grid':(12,10,12,10)}
    assert len(pilot['per_run'])==36


@pytest.mark.parametrize('damage', ['missing_run', 'duplicate_run', 'truncated_trajectory'])
def test_prompt_source_rejects_incomplete_or_duplicated_cohorts(tmp_path, damage):
    import json
    import shutil
    assert hasattr(module(), 'prompt_sources'), 'Prompt-source reducer is missing'
    source=ROOT/'results/reference/evidence/prompt_sources'
    dest=tmp_path/'evidence/prompt_sources'
    shutil.copytree(source,dest)
    path=dest/'trajectories.json'
    rows=json.loads(path.read_text())
    if damage=='missing_run':rows.pop()
    elif damage=='duplicate_run':rows[-1]=rows[0]
    else:rows[0]['path_gcs_scores']['attention_path'].pop()
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):module().prompt_sources(tmp_path)


def test_prompt_source_counts_change_when_the_test_split_trajectory_changes(tmp_path):
    import json
    import shutil
    assert hasattr(module(), 'prompt_sources'), 'Prompt-source reducer is missing'
    dest=tmp_path/'evidence/prompt_sources'
    shutil.copytree(ROOT/'results/reference/evidence/prompt_sources',dest)
    path=dest/'trajectories.json'
    rows=json.loads(path.read_text())
    for row in rows:
        if row['prompt_source']=='test':
            row['path_gcs_scores']['attention_path']=[1.0]*len(row['epochs'])
    path.write_text(json.dumps(rows))
    result=module().prompt_sources(tmp_path)['by_split']
    assert result['test']=={'n':12,'attention':0,'mlp':12,'joint':0}
    assert result['train']['attention']==12
    assert result['full_grid']['attention']==10


def test_threshold_sweep_and_reported_selector_uncertainty_are_recomputed():
    ref=ROOT/'results/reference'
    robust=module().reproduce('app_robustness',ref)
    assert robust['threshold_sweep']['joint_range']==[88,97]
    selectors=module().reproduce('app_selectors',ref)
    ci=selectors['six_seed_attention_ci']
    assert ci['test_accuracy_rise']['median_ci']==pytest.approx([.830,.950],abs=.0005)
    assert ci['attention_equivariance_drop']['median_ci']==pytest.approx([.934,.975],abs=.0005)
    offsets=selectors['six_seed_proximity_windows']
    assert {r['n']for r in offsets}=={71}
    assert [min(r['median_abs_start_gap']for r in offsets),max(r['median_abs_start_gap']for r in offsets)]==[3200,5600]
    assert all(r['median_jaccard']==0 for r in offsets)
    assert max(r['median_center_shift']for r in selectors['geometry_sensitivity'])==2400
    assert selectors['grok_free']['null_rates']['gcs']['P1']==pytest.approx(.111,abs=.001)
    measurement=module().reproduce('app_measurement',ref)
    assert measurement['procrustes_shape']=={'n':8,'cosine_pre':6,'procrustes_pre':0,'procrustes_fall':8,'procrustes_post_recovery':6}


def test_algebra_endpoints_include_the_non_grokking_planned_run():
    result=module().reproduce('app_interventions',ROOT/'results/reference')['task_algebra']
    assert result['endpoints']['commutative']['n']==18
    assert result['endpoints']['noncommutative']['n']==18
    assert result['endpoints']['commutative']['maximum']==pytest.approx(.026,abs=.0005)
    assert result['p97']['reset_accuracies']['reset_attn']==pytest.approx(.0439,abs=.00005)
    assert result['p97']['site_swap']['mlp_out']==pytest.approx(.0817,abs=.00005)
