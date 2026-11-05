import os
from pathlib import Path
import pandas as pd
import numpy as np
from paper_style import INK
ROOT = Path(os.environ['GCS_RENDER_ROOT'])

def normalize_text(fig):
    for ax in fig.axes:
        for text in ax.texts:
            text.set_color(INK)
            text.set_fontweight('normal')
        for text in [ax.xaxis.label, ax.yaxis.label,
                     *ax.get_xticklabels(), *ax.get_yticklabels(),
                     ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]:
            text.set_color(INK)
            text.set_fontweight('normal')
        legend = ax.get_legend()
        if legend is not None:
            for text in [legend.get_title(), *legend.get_texts()]:
                text.set_color(INK)
                text.set_fontweight('normal')
    for text in fig.texts:
        text.set_color(INK)
        text.set_fontweight('normal')

def cohort_summary():
    """Recompute the plotted median/IQR from the 35 individual trajectories."""
    saved = pd.read_csv(ROOT / 'source_data/figure1_summary.csv')
    saved = saved[saved.relative_epoch <= 1.1].copy()
    runs = pd.read_csv(ROOT / 'source_data/figure1_run_trajectories.csv')
    result = saved[['relative_epoch']].copy()
    for path in ['attention','mlp']:
        curves = []
        for _, run in runs.groupby('run_id'):
            run = run.sort_values('epoch')
            raw = run[path+'_raw'].to_numpy()
            smooth = np.convolve(np.pad(raw,(1,1),mode='edge'),np.ones(3)/3,mode='valid')
            scaled = (smooth-smooth.min())/(smooth.max()-smooth.min())
            curves.append(np.interp(result.relative_epoch,run.relative_epoch,scaled))
        if len(curves) != 35: raise ValueError('Expected 35 primary one-layer runs')
        result[path+'_n'] = len(curves)
        for label,q in [('q25',.25),('median',.5),('q75',.75)]:
            result[path+'_'+label] = np.quantile(curves,q,axis=0)
            np.testing.assert_allclose(result[path+'_'+label],saved[path+'_'+label],atol=1e-10,rtol=0)
    return result
