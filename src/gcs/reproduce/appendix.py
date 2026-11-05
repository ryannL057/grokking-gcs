"""Additional reported analyses, with separate cohort denominators."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import statistics


def reproduce(target: str,ref: Path) -> dict:
    if target=='gap_reference_sampling':
        from .recovered_controls import neighbor_enrichment
        return neighbor_enrichment(ref)
    if target=='gap_uniform_factorial':
        from .recovered_controls import uniform_factorial
        return uniform_factorial(ref)
    base=ref/'evidence/narrative'
    read=lambda name:pd.read_csv(base/(name+'.csv'))
    if target=='app_endpoints':
        data=read('near_solved');cut=data[data.exact_band==True];non=read('non_grok')
        return {'truncated':{'n':len(cut),**{k:int(cut[k].sum()) for k in ['late_attention','early_mlp','combined_signature']}},
                'non_grok':{'n':len(non),'locator_like':int(non.full_locator_like_pattern.sum()),
                            'final_accuracy':non[['family','act','task','seed','final_test_acc']].to_dict('records')}}
    if target=='app_scope':
        result={}
        for name in ['layernorm_one_layer','layernorm_two_layer']:
            data=read(name);eligible=data[data.trajectory_resolved==True]
            if 'successful' in data:eligible=eligible[eligible.successful==True]
            result[name]={'total':len(data),'eligible':len(eligible),
                'joint':int((eligible.mlp_u_shape_strict & eligible.attention_peak_dip_strict).sum()),
                'attention_drop_recover':int(eligible.attention_drop_recover.sum()),
                'mlp_drop_recover':int(eligible.mlp_drop_recover.sum()),
                'both_drop_recover':int((eligible.attention_drop_recover & eligible.mlp_drop_recover).sum())}
        data=read('s5_full_cohort');grok=data[data.grok_observed_in_archived_table==True]
        result['s5']={'total':len(data),'grok':len(grok),'joint':int((grok.attn_peak_dip & grok.mlp_ushape).sum())}
        data=read('groups');result['groups']={k:{'n':len(g),'attention':int(g.attn_peak_dip.sum()),'mlp':int(g.mlp_ushape.sum())}for k,g in data.groupby('group_spec')}
        return result
    if target=='app_measurement':
        data=read('path_purity');result={'detached_attention':{'n':len(data),'mlp_agreement':int(data.mlp_u_agree.sum()),
                    'median_mlp_rho':float(data.spearman_mlp.median()),'median_block_rho':float(data.spearman_overall.median())}}
        data=read('procrustes');result['procrustes']={k:{'n':len(g),'median_rho':float(g.spearman.median()),
                    'maxima_within_one':int((g.argmax_idx_delta.abs()<=1).sum())}for k,g in data.groupby('path')}
        result['procrustes_shape']=procrustes_shape(read('procrustes_trajectories'))
        data=read('token_probe');result['probe_positions']=[{'path':k[0],'probe':k[1],'n':len(g),'median_rho':float(g.spearman_rho.median())}for k,g in data.groupby(['path','probe_comparison'])]
        result['toy_controls']=read('toy_controls').to_dict('records')
        controls=read('method_control_runs')
        result['reduced_method_controls']={k:{'n':len(g),'median_rho':float(g.spearman_vs_main.median()),
             'maxima_within_one':int((g.argmax_idx_delta.abs()<=1).sum())}for k,g in controls.groupby('condition')}
        neighborhoods=np.load(base/'svd_neighborhoods.npz',allow_pickle=False)
        ratios=[];adjacent=[]
        for act in ['ReLU','GeLU','SiLU']:
            x=neighborhoods[act+'_states'];neighbors=neighborhoods[act+'_neighbors']
            for i in range(len(x)):
                values=np.linalg.svd(x[neighbors[i]]-x[i],compute_uv=False)
                ratios.append(float(values[0]/values[7]));adjacent.extend(values[:7]/values[1:8])
        result['svd']={'n':len(ratios),'mean_first_over_eighth':float(np.mean(ratios)),
                       'min_first_over_eighth':float(np.min(ratios)),'min_adjacent_ratio':float(np.min(adjacent))}
        result['unverified_claims']=['JVP norm 19,200 / minimum 0.135: original per-response input missing.']
        return result
    if target=='app_reference_variants':
        result={'prompt_sources':prompt_sources(ref)}
        task=pd.read_csv(ref/'evidence/task_reference/task_reference_ablation_taskref_full_20260426_combined.csv')
        result['task_references']={k:{'n':len(g),'attention':int(g.attn_peak_dip.sum()),
            'mlp_recovery':int(g.mlp_ushape.sum()),'joint':int((g.attn_peak_dip & g.mlp_ushape).sum())}for k,g in task.groupby('relation')}
        shuffled=read('shuffled_reference')
        result['shuffled']={'n':len(shuffled),'original_joint':int((shuffled.matched_attention_peak_then_dip & shuffled.matched_mlp_u_shape).sum()),
                           'attention':int(shuffled.shuffled_attention_peak_then_dip.sum()),
                           'mlp':int(shuffled.shuffled_mlp_u_shape.sum()),
                           'joint':int((shuffled.shuffled_attention_peak_then_dip & shuffled.shuffled_mlp_u_shape).sum()),
                           'median_attention_rho':float(shuffled.attention_path_spearman_matched_vs_shuffled.median()),
                           'median_mlp_rho':float(shuffled.mlp_path_spearman_matched_vs_shuffled.median())}
        for variant in ['earlier','h1','h2']:
            path=ref/'evidence/reference_variants'/variant
            metrics=pd.read_csv(path/'per_run_path_metrics.csv');verdicts=pd.read_csv(path/'strict_diagnostics.csv')
            result[variant]={'n':len(verdicts),'median_rho':{k:float(g.spearman_rho.median())for k,g in metrics.groupby('path')},
                'block_upturn_before_grok':{'original':int(verdicts.overall_canonical_positive_lead.sum()),
                                          'alternative':int(verdicts.overall_preceding_positive_lead.sum())},
                **{component:{'original':int(verdicts[component+'_canonical'].sum()),'alternative':int(verdicts[component+'_preceding'].sum()),
                              'agreement':int(verdicts[component+'_agreement'].sum())}for component in ['attention','mlp']}}
        return result
    if target=='app_order':
        raw=pd.read_csv(ref/'source_data/round12/schedule_raw_trajectories.csv')
        anchors=statistics.primary(ref)['per_run'];keys=['depth','activation','task','seed']
        lookup={tuple(r[k] for k in keys):r['grok_epoch'] for r in anchors}
        records=[]
        for key,g in raw.groupby(keys):
            g=g.sort_values('epoch');g=g[g.epoch<=lookup[key]];epochs=g.epoch.to_numpy()
            turns={}
            for path in ['attention','mlp']:
                values=g[path].to_numpy();changes=np.diff(values)
                candidates=[i for i in range(len(changes)-2) if (changes[i:i+3]>0).all()]
                turns[path+'_turn_epoch']=int(epochs[candidates[0] if candidates else np.argmin(values)])
            records.append(dict(zip(keys,key))|turns)
        data=pd.DataFrame(records)
        rows=[]
        for depth,g in data.groupby('depth'):
            lag=g.mlp_turn_epoch-g.attention_turn_epoch
            rows.append({'depth':int(depth),'n':len(g),'attention_first':int((lag>0).sum()),
                         'ties':int((lag==0).sum()),'median_signed_lag':float(lag.median())})
        return {'by_depth':rows,'pooled_median_signed_lag':float((data.mlp_turn_epoch-data.attention_turn_epoch).median()),
                'null_summary_archive':json.loads((ref/'source_data/round25/order_summary.json').read_text()),
                'null_status':'Archived seeded null summary; individual order-null draws were not saved.'}
    if target=='app_selectors':
        result={'six_seed':statistics.workflow(ref,extended=True)}
        result.update(selector_uncertainty(ref))
        data=read('method_neutral');result['grok_free']={'n':len(data),
            **{name:{criterion:int(data[name+'_'+criterion].sum()) for criterion in ['S1','S2','P1','S3']}
               for name in ['gcs','direct_j','cka','oriented_l2']}}
        null=read('method_neutral_null')
        result['grok_free']['null_rates']={k:{'P1':float(g.null_P1_rate.median()),'S3':float(g.null_S3_rate.median()),
             'aggregation':'median of per-run null rates','n':len(g)}for k,g in null.groupby('family')}
        result['geometry_sensitivity']=geometry_sensitivity(read('nkd_windows'),read('nkd_sensitivity'))
        data=read('reference_windows');data=data[(data.router=='fixed_attention') & data.reference_condition.isin(['own_final','own_pre_ref']) & data.family.isin(['gcs','cka','l2'])]
        result['reference_windows']=[{'reference':k[0],'family':k[1],'n':len(g),'median_percentile':float(g.equivariance_window_percentile.median())}for k,g in data.groupby(['reference_condition','family'])]
        data=read('answer_only');result['answer_graph']=[{'variant':k[0],'target':k[1],'n':len(g),'median_percentile':float(g.percentile.median())}for k,g in data.groupby(['variant','target'])]
        return result
    if target=='app_interventions':
        data=pd.read_csv(ref/'evidence/interventions/recovery.csv');result={'recovery':{}}
        result['task_algebra']=task_algebra(ref)
        for condition,g in data.groupby('condition'):
            first=g[g.test_acc>=.99].groupby(['act','task','seed']).budget_epochs.min()
            result['recovery'][condition]={'n':len(first),'median_updates':float(first.median()),'max_updates':int(first.max())}
        data=pd.read_csv(ref/'evidence/interventions/selected_donors.csv')
        result['selected_donors']=[{'donor':label,'n':len(g),'attention_hurts_more':int((g.delta_attn>g.delta_mlp).sum()),
            'mean_attention_drop':float(g.delta_attn.mean()),'mean_mlp_drop':float(g.delta_mlp.mean())}for label,g in data.groupby('donor_label')]
        data=pd.read_csv(ref/'evidence/interventions/rescue_share.csv')
        result['rescue_shares']={name:float(data[name].median()) for name in ['attn_rescue_share','resid_mid_rescue_share','mlp_rescue_share']}
        result['freeze']=statistics.freeze(ref)
        return result
    if target=='app_uniform_graphs':
        adjacency=np.load(ref/'source_data/round13/reference_adjacencies.npz',allow_pickle=False)
        rows=[];weights=np.linspace(0,1,1001)
        for name in adjacency.files:
            a=adjacency[name];eigenvalues=np.linalg.eigvalsh(a)
            values=np.abs(1+weights[:,None]*eigenvalues[None,:]);probs=values/values.sum(axis=1,keepdims=True)
            logs=np.zeros_like(probs);mask=probs>1e-12;logs[mask]=np.log2(probs[mask])
            score=len(a)-2**(-(probs*logs).sum(axis=1))
            drawdown=float(np.max(np.maximum.accumulate(score)-score)/np.ptp(score))
            rows.append({'run':name,'monotone':bool(np.all(np.diff(score)>=-1e-10)),'drawdown_fraction':drawdown})
        return {'n':len(rows),'monotone_count':sum(r['monotone'] for r in rows),'max_drawdown_fraction':max(r['drawdown_fraction'] for r in rows),'per_run':rows}
    if target=='app_robustness':return robustness(ref)
    raise ValueError(f'Unknown appendix analysis: {target}')


def prompt_sources(ref: Path) -> dict:
    """Recompute the recovered 12-run pilot from complete exported trajectories."""
    from .null_rules import attention_pass, mlp_pass, detect_grok_epoch

    base = ref / 'evidence/prompt_sources'
    trajectories = json.loads((base / 'trajectories.json').read_text())
    metrics = json.loads((base / 'training_metrics.json').read_text())
    expected = {(a, t, 777, s) for a in ['ReLU', 'GeLU', 'SiLU']
                for t in ['add', 'sub', 'mul', 'div']
                for s in ['train', 'test', 'full_grid']}
    keys = [(r['activation'], r['task'], r['seed'], r['prompt_source'])
            for r in trajectories]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Expected 12 distinct runs with all three prompt sources')
    records = []
    schedules = {}
    for record in trajectories:
        identity = (record['activation'], record['task'], record['seed'])
        epochs = np.asarray(record['epochs'], dtype=float)
        scores = {k: np.asarray(v, dtype=float)
                  for k, v in record['path_gcs_scores'].items()}
        if (epochs.ndim != 1 or len(epochs) < 3
                or record['epochs'] != record['selected_epochs']
                or not np.isfinite(epochs).all() or not (np.diff(epochs) > 0).all()
                or set(scores) != {'overall_path', 'attention_path', 'mlp_path'}
                or any(v.shape != epochs.shape or not np.isfinite(v).all()
                       for v in scores.values())):
            raise ValueError(f'Incomplete or invalid prompt-source trajectory: {identity}')
        if identity in schedules and schedules[identity] != record['epochs']:
            raise ValueError(f'Prompt sources use different checkpoint schedules: {identity}')
        schedules[identity] = record['epochs']
        # The historical directory string is only a dictionary key, never a local path.
        training = metrics[record['experiment_dir']]
        anchor = detect_grok_epoch(training)
        if anchor is None:
            raise ValueError(f'Missing grok anchor for {identity}')
        attention = attention_pass(scores['attention_path'], epochs, anchor)
        mlp = mlp_pass(scores['mlp_path'], epochs, anchor)
        if attention is None or mlp is None:
            raise ValueError(f'Insufficient checkpoint coverage for {identity}')
        records.append({'activation': identity[0], 'task': identity[1], 'seed': identity[2],
                        'prompt_source': record['prompt_source'], 'n_epochs': len(epochs),
                        't_grok': anchor, 'attention': attention, 'mlp': mlp,
                        'joint': attention and mlp})
    by_split = {}
    for split in ['train', 'test', 'full_grid']:
        rows = [r for r in records if r['prompt_source'] == split]
        by_split[split] = {'n': len(rows), **{k: sum(r[k] for r in rows)
                                            for k in ['attention', 'mlp', 'joint']}}
    return {'n_outputs': len(records), 'by_split': by_split, 'per_run': records}


def robustness(ref):
    data=pd.read_csv(ref/'evidence/narrative/robustness_per_run.csv')
    clusters=['task','act','seed'];rng=np.random.default_rng(20260426);draws=5000
    # Preserve the original RNG consumption for the five preceding continuous summaries.
    for col,exclude_zero in [('lead_margin',False),('mlp_u_margin',True),('attn_rise_margin',True),('slope_shift',False),('pathway_gap',False)]:
        sub=data.dropna(subset=[col]);sub=sub[sub[col]!=0] if exclude_zero else sub
        n=len(sub[clusters].drop_duplicates())
        for _ in range(2):rng.integers(0,n,size=(draws,n))
    result={}
    for col in ['mlp_u_shape_hit','attn_peak_dip_hit']:
        sub=data.dropna(subset=[col]);groups=sub[clusters].drop_duplicates()
        values=[sub[(sub[clusters]==row).all(axis=1)][col].to_numpy(float) for _,row in groups.iterrows()]
        chosen=rng.integers(0,len(groups),size=(draws,len(groups)))
        sums=np.asarray([v.sum() for v in values]);counts=np.asarray([len(v) for v in values])
        boot=sums[chosen].sum(axis=1)/counts[chosen].sum(axis=1)
        leave=[float(sub[sub[f]!=v][col].mean())for f in ['depth','task','act','seed'] for v in sub[f].unique()]
        result[col]={'n':len(sub),'mean':float(sub[col].mean()),'ci025':float(np.quantile(boot,.025)),
                     'ci975':float(np.quantile(boot,.975)),'min_leave_one_support':min(leave)}
    return {'resamples':draws,'cluster':'task × activation × seed; depths stay together','summary':result,
            'threshold_sweep':threshold_sweep(ref)}


def threshold_sweep(ref):
    from .null_rules import attention_pass,mlp_pass
    raw=pd.read_csv(ref/'source_data/round12/schedule_raw_trajectories.csv')
    keys=['depth','activation','task','seed']
    anchors={tuple(r[k]for k in keys):r['grok_epoch']for r in statistics.primary(ref)['per_run']}
    rows=[]
    for a in [.025,.05,.10]:
        for m in [.05,.10,.20]:
            counts={'attention':0,'mlp':0,'joint':0}
            for key,g in raw.groupby(keys):
                g=g.sort_values('epoch');epochs=g.epoch.to_numpy();anchor=anchors[key]
                attn=attention_pass(g.attention.to_numpy(),epochs,anchor,a)
                mlp=mlp_pass(g.mlp.to_numpy(),epochs,anchor,m)
                counts['attention']+=bool(attn);counts['mlp']+=bool(mlp);counts['joint']+=bool(attn and mlp)
            rows.append({'attention_threshold':a,'mlp_threshold':m,'n':len(anchors),**counts})
    expected=pd.read_csv(ref/'source_data/round12/threshold_grid_counts.csv').sort_values(['attention_threshold','mlp_threshold'])
    np.testing.assert_array_equal(pd.DataFrame(rows)[['attention','mlp','joint']],expected[['attention','mlp','joint']])
    return {'grid':rows,'joint_range':[min(r['joint']for r in rows),max(r['joint']for r in rows)]}


def procrustes_shape(data):
    from .null_rules import mlp_pass
    result=dict(n=data.run.nunique(),cosine_pre=0,procrustes_pre=0,procrustes_fall=0,procrustes_post_recovery=0)
    for _,g in data.groupby('run'):
        g=g.sort_values('epoch');epochs=g.epoch.to_numpy();v=g.procrustes.to_numpy();anchor=g.grok_epoch.iloc[0]
        result['cosine_pre']+=bool(mlp_pass(g.cosine.to_numpy(),epochs,anchor))
        result['procrustes_pre']+=bool(mlp_pass(v,epochs,anchor))
        trough=int(np.argmin(v));span=np.ptp(v)
        result['procrustes_fall']+=bool(v[0]-v[trough]>=.1*span)
        recovery=np.flatnonzero((np.arange(len(v))>trough)&(v-v[trough]>.1*span))
        result['procrustes_post_recovery']+=bool(len(recovery) and epochs[recovery[0]]>anchor)
    return result


def selector_uncertainty(ref):
    rows=pd.read_csv(ref/'evidence/workflow/six_seed.csv.gz')
    rows=rows[(rows.split=='heldout_seed123_777')&(rows.window_len==11)]
    rng=np.random.default_rng(20260502);ci={}
    # Attention-GCS is first in the original selector list; preserve acc/equiv draw order.
    for target in ['test_accuracy_rise','attention_equivariance_drop']:
        g=rows[(rows.selector=='gcs_attention')&(rows.target==target)]
        values=g.percentile.to_numpy();draws=rng.integers(0,len(values),size=(10000,len(values)))
        ci[target]={'n':len(values),'resamples':10000,'seed':20260502,'median':float(np.median(values)),
                    'median_ci':np.quantile(np.median(values[draws],axis=1),[.025,.975]).tolist()}
    accuracy=rows[rows.target=='test_accuracy_rise'];keys=['act','task','seed']
    gcs=accuracy[accuracy.selector=='gcs_attention'];offsets=[]
    for name in ['final_repr_l2_drop','final_repr_cka_rise','path_l2_attention_drop','path_cka_attention_rise']:
        paired=gcs.merge(accuracy[accuracy.selector==name],on=keys,suffixes=('_gcs','_other'),validate='one_to_one')
        a,b=paired.window_start_epoch_gcs,paired.window_end_epoch_gcs
        c,d=paired.window_start_epoch_other,paired.window_end_epoch_other
        jaccard=np.maximum(0,np.minimum(b,d)-np.maximum(a,c))/(np.maximum(b,d)-np.minimum(a,c))
        offsets.append({'comparison':name,'n':len(paired),'median_abs_start_gap':float(np.median(abs(a-c))),
                        'median_jaccard':float(np.median(jaccard))})
    return {'six_seed_attention_ci':ci,'six_seed_proximity_windows':offsets}


def geometry_sensitivity(data,expected):
    import hashlib
    result=[]
    for (variant,target),g in data.groupby(['variant','target']):
        seed=int.from_bytes(hashlib.sha256(f'{variant}:{target}'.encode()).digest()[:8],'little')^20260724
        values=g.percentile.to_numpy();rng=np.random.default_rng(seed)
        ci=np.quantile(np.median(rng.choice(values,size=(20000,len(values)),replace=True),axis=1),[.025,.975])
        saved=expected[(expected.variant==variant)&(expected.target==target)].iloc[0]
        np.testing.assert_allclose(ci,[saved.median_bootstrap_ci_low,saved.median_bootstrap_ci_high],atol=1e-12)
        result.append({'variant':variant,'target':target,'n':len(g),'median_percentile':float(np.median(values)),
                       'median_center_shift':float(g.absolute_center_shift_epochs.median()),
                       'median_ci':ci.tolist(),'resamples':20000,'bootstrap_seed':seed})
    return result


def task_algebra(ref):
    base=ref/'evidence/interventions'
    endpoints=pd.read_csv(base/'planned_endpoint36.csv');endpoints['relative']=endpoints.mean_diff/endpoints.mean_norm
    groups={}
    for commutative,g in endpoints.groupby(endpoints.task.isin(['add','mul'])):
        groups['commutative' if commutative else 'noncommutative']={'n':len(g),'minimum':float(g.relative.min()),'maximum':float(g.relative.max())}
    surgery=pd.read_csv(base/'p97_surgery.csv');sites=pd.read_csv(base/'p97_site_swap.csv');sites=sites[sites.seed==2026]
    values=sites.mean_diff/sites.mean_norm
    np.testing.assert_allclose(values,sites.relative_dequiv,atol=1e-12)
    acquisition=pd.read_csv(base/'sitewise_acquisition.csv')
    return {'endpoints':groups,'p97':{'reset_accuracies':surgery.set_index('condition').test_acc.to_dict(),
             'site_swap':dict(zip(sites.site,values))},
             'sitewise_earlier_than_attention':{k:{'n':len(g),'earlier':int((g.acquisition_minus_attn_out<0).sum())}for k,g in acquisition.groupby('site')}}
