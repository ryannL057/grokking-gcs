"""Figure 2 (attention and MLP coherence by depth) in the house style of Figures 1 and 3.

One row of three panels; attention solid vermilion, MLP dashed blue, IQR bands, dotted t_grok line labelled 'grok',
Times New Roman at the sizes of Figure 1, 7.0-inch canvas included at \\linewidth like Figures 1 and 3.
Data, medians, IQR and per-depth time limits are those of the saved depth curves (source_data/reviewed_depth_curves).
"""
import sys, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gcs-camera-ready-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_style import apply_style, ATTENTION, MLP, CONTROL, INK
SRC = Path(sys.argv[1])
apply_style()
plt.rcParams.update({'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.linewidth': .6,
                     'axes.spines.top': False, 'axes.spines.right': False})
d = np.load(SRC / 'source_data/reviewed_depth_curves/depth_curves.npz')
W, H = 7.0, 2.25
fig, axes = plt.subplots(1, 3, figsize=(W, H), sharey=True)
fig.subplots_adjust(left=.075, right=.995, bottom=.21, top=.88, wspace=.12)
for col, depth in enumerate([1, 2, 3]):
    ax = axes[col]
    for name, color, ls, label in [('attention', ATTENTION, '-', 'Attention'), ('mlp', MLP, '--', 'MLP')]:
        k = f'{name}_{depth}'
        x = d[k + '_x']
        stack=np.array([np.interp(x,d[f'{k}_run{i:02}_x'],d[f'{k}_run{i:02}_y'],left=np.nan,right=np.nan)
                        for i in range(int(d[k+'_n']))])
        q25,median,q75=np.nanquantile(stack,[.25,.5,.75],axis=0)
        for suffix,values in [('q25',q25),('median',median),('q75',q75)]:
            np.testing.assert_allclose(values,d[k+'_'+suffix],atol=1e-12)
        ax.fill_between(x, q25, q75, color=color, alpha=.15, linewidth=0, zorder=1)
        ax.plot(x, median, color=color, lw=1.7, ls=ls, zorder=3, label=label)
    ax.axvline(1., color=CONTROL, ls=':', lw=.8, zorder=0)
    ax.text(.97, .03, 'grok', ha='right', fontsize=8.0, color=INK)
    ax.set_xlim(0, float(d[f'attention_{depth}_cutoff']))
    ax.set_ylim(-.02, 1.04)
    ax.set_xticks([0, .5, 1.] if depth == 1 else [0, .5, 1., 1.5])
    ax.set_yticks([0, .5, 1.])
    ax.set_xticklabels(['0', '0.5', '1.0'] if depth == 1 else ['0', '0.5', '1.0', '1.5'])
    ax.set_yticklabels(['0', '0.5', '1.0'])
    ax.tick_params(length=2.5, width=.6)
    ax.set_xlabel(r'Epoch / $t_{\mathrm{grok}}$')
    n = int(d[f'attention_{depth}_n'])
    ax.set_title(f'{depth} layer{"s" if depth > 1 else ""} ($n={n}$)', fontsize=9.0, pad=4)
axes[0].set_ylabel('Scaled GCS')
axes[0].legend(loc='upper left', bbox_to_anchor=(.06, 1.0), frameon=False, fontsize=8.0, handlelength=1.7, labelspacing=.2, columnspacing=1.0, borderpad=0, ncol=2)
out = SRC / 'figures/depth_coherence.pdf'
fig.savefig(out, metadata={'Creator': None, 'Producer': None})
fig.savefig(out.with_suffix('.png'), dpi=220)
fig.savefig(out.with_suffix('.svg'))
print(out, {f'n{k}': int(d[f"attention_{k}_n"]) for k in (1, 2, 3)})
