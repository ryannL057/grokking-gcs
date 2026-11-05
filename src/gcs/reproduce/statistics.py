"""Paper summaries computed from immutable run-level numerical records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def primary(ref: Path) -> dict:
    from .null_rules import attention_pass,mlp_pass,detect_grok_epoch
    keys=['depth','activation','task','seed']
    raw=pd.read_csv(ref/'source_data/round12/schedule_raw_trajectories.csv')
    anchors=pd.read_csv(ref/'source_data/round12/schedule_protocol_per_run.csv').set_index(keys)
    metrics=pd.read_csv(ref/'primary/training_metrics.csv.gz')
    metrics={k:g.sort_values('epoch') for k,g in metrics.groupby(keys)}
    rows=[]
    for key,group in raw.groupby(keys):
        group=group.sort_values('epoch');g=metrics[key]
        anchor=detect_grok_epoch({'log_steps':g.epoch.to_numpy(),'test_accuracies':g.test_accuracy.to_numpy()})
        if anchor != anchors.loc[key,'grok_epoch']: raise ValueError(f'Grok epoch mismatch: {key}')
        e=group.epoch.to_numpy();a=group.attention.to_numpy();m=group.mlp.to_numpy()
        rows.append(dict(zip(keys,key)) | {'grok_epoch':anchor,
            'attention_rise_dip':attention_pass(a,e,anchor),'mlp_rise_dip':attention_pass(m,e,anchor),
            'attention_fall_recovery':mlp_pass(a,e,anchor),'mlp_fall_recovery':mlp_pass(m,e,anchor)})
    data=pd.DataFrame(rows)
    saved=pd.read_csv(ref/'source_data/round13/crossed_shape_rules_per_run.csv')
    pd.testing.assert_frame_equal(data.drop(columns='grok_epoch').sort_values(keys).reset_index(drop=True),
        saved[data.drop(columns='grok_epoch').columns].sort_values(keys).reset_index(drop=True))
    def summarize(frame):
        cols = ['attention_rise_dip', 'mlp_rise_dip', 'attention_fall_recovery', 'mlp_fall_recovery']
        return {'n': len(frame), **{k: int(frame[k].sum()) for k in cols},
                'joint': int((frame.attention_rise_dip & frame.mlp_fall_recovery).sum())}
    return {**summarize(data), 'by_depth': {str(k): summarize(g) for k, g in data.groupby('depth')},
            'by_task': {str(k): summarize(g) for k, g in data.groupby('task')},'per_run':rows}


def branch_outcome(record: dict, budget: int) -> dict:
    start = record['branch_epoch']
    hits = [int(e-start) for e, a in zip(record['curve']['epochs'], record['curve']['test_acc'])
            if 0 <= e-start <= budget and a >= .99]
    return {'success': bool(hits), 'first_success_updates': min(hits) if hits else None}


def freeze(ref: Path) -> dict:
    base = ref / 'source_data/two_stage_freeze_20261103'
    cohort = pd.read_csv(base / 'per_run.csv')
    arms = ['VV','VL','LV','LL','0L','0V','0V_long','V_unfrozen','L_attn_frozen']
    rows = []
    for run in cohort.itertuples():
        for arm in arms:
            record = json.loads((base / 'raw' / run.run / f'{arm}.json').read_text())
            budget = int(run.budget) * (3 if arm == '0V_long' else 1)
            if arm in ['V_unfrozen', 'L_attn_frozen']:
                outcome = branch_outcome(record, budget)
                if not record['frozen_params_unchanged']:
                    raise ValueError(f'Frozen parameters changed: {run.run}/{arm}')
            else:
                if int(record['budget_updates']) != budget:
                    raise ValueError(f'Budget mismatch: {run.run}/{arm}')
                if not all(record.get('frozen_exact', {}).values()):
                    raise ValueError(f'Frozen parameters changed: {run.run}/{arm}')
                first = record['first_observed_99_updates']
                success = bool(record['reached_99']) and first is not None and first <= budget
                outcome = {'success': success, 'first_success_updates': first if success else None}
            rows.append({'run': run.run, 'arm': arm, 'budget': budget, **outcome})
    return {'records':len(rows), 'runs':len(cohort),
            'successes':{arm:sum(r['success'] for r in rows if r['arm']==arm) for arm in arms},
            'budget_range':[int(cohort.budget.min()),int(cohort.budget.max())], 'per_run':rows}


def resets(ref: Path) -> dict:
    data = pd.read_csv(ref / 'primary/surgery_stats.csv')
    attn = data.base_test_acc - data.reset_attn_test_acc
    mlp = data.base_test_acc - data.reset_mlp_test_acc
    return {'n':len(data), 'attention_hurts_more':int((attn > mlp).sum()),
            'mean_attention_drop':float(attn.mean()), 'mean_mlp_drop':float(mlp.mean()),
            'per_run':data.assign(attention_drop=attn, mlp_drop=mlp).to_dict('records')}


def references(ref: Path) -> dict:
    own = pd.read_csv(ref / 'source_data/round13/reference_joint_per_run.csv')
    other = pd.read_csv(ref / 'source_data/round13/cross_seed_joint_per_pair.csv')
    result = {stage:{'n':len(own), **{key:int(own[f'{stage}_{key}'].sum()) for key in ['attention','mlp','joint']}}
              for stage in ['final','pre','post']}
    result['intersection'] = {'n':len(own), **{key:int(own[[f'{stage}_{key}' for stage in ['final','pre','post']]].all(axis=1).sum())
                                              for key in ['attention','mlp','joint']}}
    result['other_seed'] = {'n':len(other),'attention':int(other.attn_peak_dip.sum()),'mlp':int(other.mlp_ushape.sum()),'joint':int(other.joint.sum())}
    return result


def nulls(ref: Path) -> dict:
    observed = primary(ref)
    result = {}
    for name in ['N1_random_anchor','N2_iaaft','N3_permutation']:
        frame = pd.read_csv(ref / f'source_data/round25/{name}_replicate_counts.csv')
        result[name] = {'replicates':len(frame), **{col:{'mean':float(frame[col].mean()),
                         'q05':float(frame[col].quantile(.05)), 'q95':float(frame[col].quantile(.95))}
                         for col in ['attention','mlp','joint']}}
    return {'observed':observed, 'nulls':result}


def workflow(ref: Path, *, extended: bool=False) -> dict:
    path='six_seed.csv.gz' if extended else 'joined.csv.gz'
    rows=pd.read_csv(ref/f'evidence/workflow/{path}')
    rows=rows[(rows.split=='heldout_seed123_777') & (rows.window_len==11)]
    rows=rows[rows.target.isin(['test_accuracy_rise','attention_equivariance_drop'])]
    summary=[]
    for (selector,target),group in rows.groupby(['selector','target']):
        n=len(group[['act','task','seed']].drop_duplicates())
        summary.append({'selector':selector,'target':target,'n_runs':n,'n_rows':len(group),
                        'median_percentile':float(group.percentile.median()),
                        'top20_rate':float((group.percentile>=.8).mean())})
    if not extended:
        direct=pd.read_csv(ref/'evidence/workflow/direct_j.csv')
        for (rule,target),group in direct.groupby(['rule','target']):
            summary.append({'selector':'direct_j_'+rule,'target':target,'n_runs':len(group),
                'median_percentile':float(group.percentile.median()),'top20_rate':float((group.percentile>=.8).mean())})
    return {'seeds':sorted(int(x) for x in rows.seed.unique()),'summary':summary,
            'random_note':'Random rows are repeated draws within runs, not independent training runs.'}


def aggregation(ref: Path) -> dict:
    from scipy.stats import spearmanr
    from .null_rules import attention_pass
    trajectories=pd.read_csv(ref/'source_data/round13/aggregation_trajectories.csv')
    anchors=pd.read_csv(ref/'source_data/round12/schedule_protocol_per_run.csv')
    anchors=anchors[anchors.depth==1]
    anchor={f'{r.activation.lower()}_{r.task}_s{r.seed}':r.grok_epoch for r in anchors.itertuples()}
    rows=[]
    for (run,path),g in trajectories.groupby(['run_id','path']):
        g=g.sort_values('epoch');epochs=g.epoch.to_numpy();t=anchor[run]
        rows.append({'run_id':run,'path':path,'rho_mean':float(spearmanr(g.gcs,g.mean_edge).statistic),
                     'rho_pr':float(spearmanr(g.gcs,g.participation_rank).statistic),
                     'gcs_rise_dip':attention_pass(g.gcs.to_numpy(),epochs,t),
                     'mean_rise_dip':attention_pass(g.mean_edge.to_numpy(),epochs,t),
                     'uniform_rise_dip':attention_pass(g.uniform_gcs.to_numpy(),epochs,t)})
    frame=pd.DataFrame(rows);summary={}
    for path,g in frame.groupby('path'):
        summary[path]={'n':len(g),**{k:int(g[k].sum()) for k in ['gcs_rise_dip','mean_rise_dip','uniform_rise_dip']},
            **{k:{'median':float(g[k].median()),'q25':float(g[k].quantile(.25)),'q75':float(g[k].quantile(.75))}for k in ['rho_mean','rho_pr']}}
    return {'summary':summary,'per_run':rows}


def mechanistic(ref: Path) -> dict:
    from . import mechanistic as engine
    r=ref/'evidence'
    selectors,_=engine.build_selector_epochs(engine.read_csv(r/'mechanistic/donors.csv'),
                  engine.read_csv(r/'workflow/direct_j.csv'),engine.read_csv(r/'mechanistic/dynamic.csv'))
    selected,random_rows,keys=engine.build_selected_rows(engine.read_csv(r/'mechanistic/windows.csv'),selectors)
    summaries=engine.summarize_selectors(selected)
    expected=pd.read_csv(r/'mechanistic/expected.csv').set_index('selector')
    for row in summaries:
        for key in ['median_mechanistic_percentile','median_percentile_bootstrap_ci_low','median_percentile_bootstrap_ci_high']:
            np.testing.assert_allclose(row[key],expected.loc[row['selector'],key],atol=1e-12)
    return {'n':len(keys),'bootstrap_resamples':20000,'summary':summaries,
            'random':engine.summarize_random(random_rows),'selected_donors':selected}


def windows(ref: Path) -> dict:
    base=ref/'source_data/reviewed_window_example'
    result={}
    for name in ['gcs','cka','l2']:
        sites=pd.read_csv(base/f'{name}_case_study_sites.csv')
        heads=pd.read_csv(base/f'{name}_case_study_heads.csv')
        result[name]={'sites':sites.to_dict('records'),'heads':heads.to_dict('records')}
    return result


def fourier(ref: Path) -> dict:
    from scipy.stats import spearmanr,rankdata
    rows=[]
    for folder in sorted((ref/'evidence/fourier').iterdir()):
        summary=pd.read_csv(folder/'comparison_summary.csv').iloc[0]
        frame=pd.read_csv(folder/'comparison_timeseries.csv')
        frame=frame[frame.epoch<=summary.grok_epoch]
        progress=-frame['restricted_loss_raw' if 'restricted_loss_raw' in frame else 'trig_loss_raw'].to_numpy()
        attention=frame.attention_gcs_raw.to_numpy();epoch=frame.epoch.to_numpy()
        ranks=np.column_stack([np.ones(len(frame)),rankdata(epoch)])
        residuals=[rankdata(v)-ranks@np.linalg.lstsq(ranks,rankdata(v),rcond=None)[0] for v in [attention,progress]]
        rows.append({'activation':summary.activation,'seed':int(summary.seed),'n':len(frame),
            'rho_levels':float(spearmanr(attention,progress).statistic),
            'rho_epoch_progress':float(spearmanr(epoch,progress).statistic),
            'rho_differences':float(spearmanr(np.diff(attention),np.diff(progress)).statistic),
            'rank_partial_time':float(np.corrcoef(*residuals)[0,1])})
    return {'per_run':rows,'medians':{k:float(np.median([r[k] for r in rows])) for k in ['rho_levels','rho_epoch_progress','rho_differences','rank_partial_time']}}


def submodules(ref: Path) -> dict:
    base=ref/'evidence/interventions/raw'
    rows=[]
    for file in sorted((base/'submodule_cohort_20260419').glob('*_surgery.csv')):
        frame=pd.read_csv(file);frame=frame[frame.base_label=='final']
        original=float(frame[frame.condition=='base'].test_acc.iloc[0])
        for r in frame[frame.condition!='base'].itertuples():
            rows.append({'run':file.stem,'condition':r.condition,'parameters':int(r.reset_param_count),'drop':original-float(r.test_acc)})
    matched=[]
    for file in sorted((base/'matched_null_neuron_cohort_20260419').glob('*_surgery.csv')):
        frame=pd.read_csv(file);frame=frame[frame.base_label=='final']
        original=float(frame[frame.condition=='base'].test_acc.iloc[0])
        subset=frame[frame.match_family=='mlp_count_match']
        if len(subset)!=32:raise ValueError(f'Expected 32 count-matched draws in {file}')
        matched.append({'run':file.stem,'n':len(subset),'max_drop':float((original-subset.test_acc).max())})
    frame=pd.DataFrame(rows)
    return {'per_run':rows,'summary':{name:{'n':len(g),'mean_drop':float(g['drop'].mean()),'parameters':int(g.parameters.iloc[0])}
        for name,g in frame.groupby('condition')},'matched_per_run':matched,
        'matched_mean_of_run_maxima':float(np.mean([r['max_drop'] for r in matched]))}
