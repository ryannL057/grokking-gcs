"""Tabular views of recomputed statistics; metadata tables remain labelled."""
from pathlib import Path
import json
import pandas as pd
from . import statistics as s


def table(number: int,root: Path):
    ref=root/'results/reference'
    if number in [1,9]:
        result=s.primary(ref)
        groups=result['by_depth' if number==1 else 'by_task']
        rows=[{'depth' if number==1 else 'task':k,**v} for k,v in groups.items()]
        if number==1:
            null=json.loads((ref/'source_data/round25/summary.json').read_text())
            for row in rows:row['random_anchor_joint_mean']=null['by_depth']['N1_random_anchor'][row['depth']]['joint']['null_mean']
        return rows,result
    if number==2:
        result=s.references(ref);return [{'reference':k,**v} for k,v in result.items()],result
    if number==3:
        result=s.freeze(ref)
        return [{'arm':a,'successes':result['successes'][a],'n':result['runs']} for a in ['VV','VL','LV','LL','0L','0V']],result
    if number in [5,7]:
        data=json.loads((ref/f'metadata/table{number:02d}.json').read_text())
        return data['rows'],data
    if number==6:
        data=pd.read_csv(ref/'source_data/round14/cohort_horizons.csv')
        rows=[{'depth':int(k),'n':len(g),'horizon_min':int(g.saved_horizon_epochs.min()),'horizon_max':int(g.saved_horizon_epochs.max()),
               'gcs_samples_min':int(g.gcs_checkpoint_count.min()),'gcs_samples_max':int(g.gcs_checkpoint_count.max())} for k,g in data.groupby('depth')]
        return rows,{'summary':rows,'per_run':data.to_dict('records')}
    if number==4:
        result=s.workflow(ref)
        selected={'gcs_attention','final_repr_cka_rise','path_l2_attention_drop','random',
                  'centroid_alignment_peak','centroid_alignment_abs','centroid_alignment_rise','centroid_alignment_drop',
                  'direct_j_post_valley_peak','direct_j_largest_absolute_change'}
        return [r for r in result['summary'] if r['selector'] in selected],result
    if number==8:
        result=s.aggregation(ref)
        return [{'path':path,'n':r['n'],'rho_mean_median':r['rho_mean']['median'],'rho_mean_q25':r['rho_mean']['q25'],
                 'rho_mean_q75':r['rho_mean']['q75'],'rho_pr_median':r['rho_pr']['median'],'rho_pr_q25':r['rho_pr']['q25'],
                 'rho_pr_q75':r['rho_pr']['q75'],'gcs_rise_dip':r['gcs_rise_dip'],'mean_rise_dip':r['mean_rise_dip'],
                 'uniform_rise_dip':r['uniform_rise_dip']} for path,r in result['summary'].items()],result
    if number==10:
        result=s.nulls(ref)
        return [{'null':name,'metric':metric,**r} for name,metrics in result['nulls'].items() for metric,r in metrics.items() if isinstance(r,dict)],result
    if number==11:
        result=s.mechanistic(ref);return result['summary'],result
    if number in [12,13]:
        result=s.windows(ref)
        if number==13:
            return [{k:r[k] for k in ['label','epoch','attn_out_dequiv','mlp_out_dequiv','logits_centered_dequiv']}
                    for r in result['gcs']['sites'] if r['label']!='valley'],result
        rows=[]
        for name,r in result.items():
            index={v['label']:v for v in r['sites']};a,b=index['window-start'],index['window-end']
            rows.append({'selector':name,'start_epoch':a['epoch'],'end_epoch':b['epoch'],
                         'start_accuracy':a['test_accuracy'],'end_accuracy':b['test_accuracy'],
                         'start_swap_distance':a['attn_out_dequiv'],'end_swap_distance':b['attn_out_dequiv']})
        return rows,result
    if number==14:
        result=s.fourier(ref);return result['per_run'],result
    if number==15:
        result=s.submodules(ref)
        return [{'condition':k,**v} for k,v in result['summary'].items()]+[{'condition':'matched_mlp_max_within_run',
            'n':17,'mean_drop':result['matched_mean_of_run_maxima'],'parameters':65535}],result
    raise ValueError(f'Unknown table: {number}')
