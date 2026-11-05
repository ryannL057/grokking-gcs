"""Redraw the appendix figures in the house style of Figures 1-3 (2026-11-03).

Same data as the submitted or repaired displays; only typography, colour semantics and size change.
House style: Times New Roman (paper_style.apply_style), 7.0 in canvas included at \\linewidth, 9 pt labels and
8.5 pt ticks (about 7 pt in print), attention vermilion solid, MLP blue dashed, full block dark grey,
controls and alternatives in greys, IQR bands at alpha .15, dotted grey t_grok line, panel captions "(a) ..." below.

Usage: MPLBACKEND=Agg python3 plot_appendix_house.py <manuscript_source> [figure ...]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import csv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import apply_style, MLP, ATTENTION, REFERENCE, CONTROL, LIGHT, INK, GRID as GRID_C

apply_style()
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    'axes.linewidth': .65, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
})
W = 7.0
CAPTION_SIZE = 9
PATHS = [('overall', 'Full block', REFERENCE, '-'), ('attention', 'Attention', ATTENTION, '-'), ('mlp', 'MLP', MLP, '--')]


def kfmt(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x / 1000:g}k'))


def panel_captions(fig, axes_groups, labels, y):
    for axs, label in zip(axes_groups, labels):
        x0 = min(a.get_position().x0 for a in axs); x1 = max(a.get_position().x1 for a in axs)
        fig.text((x0 + x1) / 2, y, label, ha='center', va='baseline', fontsize=CAPTION_SIZE)


def grok_line(ax, x, label=True):
    ax.axvline(x, color=CONTROL, ls=':', lw=.8, zorder=0)


def save(fig, out):
    out = Path(out)
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(out.with_suffix('.' + suffix), dpi=300, metadata={'Creator': None, 'Producer': None} if suffix == 'pdf' else None)
    plt.close(fig)
    print(out)


# ---------------------------------------------------------------- Figure 4: probe positions
def probe_positions(ms):
    D = ms / 'source_data/reviewed_display_repairs'
    rows = list(csv.DictReader((D / 'probe_aggregate.csv').open()))
    index = {(r['path'], r['probe_comparison']): r for r in rows}
    art = np.load(D / 'probe_artists.npz')
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.15), gridspec_kw={'width_ratios': [1, 1, 1.35]})
    fig.subplots_adjust(left=.075, right=.985, bottom=.27, top=.86, wspace=.42)
    comps = [('overall_path', 'Block', REFERENCE), ('attention_path', 'Attn.', ATTENTION), ('mlp_path', 'MLP', MLP)]
    for ax, field, ylabel in zip(axes[:2], ['median_spearman_rho', 'mean_abs_diff_norm01'],
                                 [r'Median Spearman $\rho$', 'Mean absolute difference']):
        for i, (path, _, color) in enumerate(comps):
            for j, probe in enumerate(['operands', 'all']):
                v = float(index[(path, probe)][field])
                x = i + (j - .5) * .34
                if probe == 'operands':
                    ax.bar(x, v, width=.32, color=color, edgecolor=color, lw=.8)
                else:
                    ax.bar(x, v, width=.32, facecolor='white', edgecolor=color, hatch='//////', lw=.8)
        ax.set_xticks(range(3), [c[1] for c in comps])
        ax.set_xlim(-.6, 2.6)
        ax.set_ylabel(ylabel, labelpad=3)
    axes[0].set_ylim(-.5, 1.05); axes[0].set_yticks([-.5, 0, .5, 1])
    axes[0].axhline(0, color=CONTROL, lw=.5)
    axes[1].set_ylim(0, .62); axes[1].set_yticks([0, .2, .4, .6])
    plt.rcParams['hatch.linewidth'] = .6
    handles = [Patch(facecolor=CONTROL, edgecolor=CONTROL, label='Operand positions'),
               Patch(facecolor='white', edgecolor=CONTROL, hatch='//////', label='All positions')]
    p0, p1 = axes[0].get_position(), axes[1].get_position()
    fig.legend(handles=handles, frameon=False, ncol=2, loc='lower center', bbox_to_anchor=((p0.x0 + p1.x1) / 2, .87),
               handlelength=1.4, columnspacing=1.2, borderpad=0)
    ax = axes[2]
    for j, (label, color, ls, lw) in enumerate([('Prediction', ATTENTION, '-', 1.6), ('Operands', CONTROL, '--', 1.3)]):
        ax.plot(art[f'2:line{j}:x'], art[f'2:line{j}:y'], color=color, ls=ls, lw=lw, label=label)
    ax.set_ylabel('Raw attention GCS', labelpad=3); ax.set_xlabel('Epoch', labelpad=2)
    ax.set_xlim(0, 15000); ax.set_xticks([0, 5000, 10000, 15000]); kfmt(ax)
    ax.set_ylim(24.5, 31.2); ax.set_yticks([25, 27, 29, 31])
    p2 = ax.get_position()
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2, loc='lower right', bbox_to_anchor=(p2.x1, .87),
               handlelength=1.4, handletextpad=.4, columnspacing=.8, borderpad=0)
    panel_captions(fig, [[axes[0]], [axes[1]], [axes[2]]], ['(a) Correlation', '(b) Difference', '(c) Attention trajectory'], .03)
    save(fig, ms / 'figures/probe_positions_readable.pdf')


# ---------------------------------------------------------------- Figure 12: learned vs uniform attention
def attention_controls(ms):
    d = np.load(ms / 'source_data/reviewed_display_repairs/attention_controls.npz')
    values = np.concatenate([d[kind + '_' + path][np.isfinite(d[kind + '_' + path])]
                             for kind in ['learned', 'uniform'] for path, _, _, _ in PATHS])
    lower, upper = values.min(), values.max()
    assert upper > lower
    fig, axes = plt.subplots(2, 2, figsize=(W, 2.85), sharex='col')
    fig.subplots_adjust(left=.085, right=.985, bottom=.2, top=.97, wspace=.2, hspace=.2)
    for col, kind in enumerate(['learned', 'uniform']):
        for path, label, color, ls in PATHS:
            x, y = d[kind + '_epochs'], d[kind + '_' + path]
            ok = np.isfinite(x) & np.isfinite(y)
            axes[0, col].plot(x[ok], (y[ok] - lower) / (upper - lower), color=color, ls=ls, lw=1.5, label=label)
        for path, label, color, ls in [('train', 'Train', CONTROL, '-'), ('test', 'Test', INK, '--')]:
            axes[1, col].plot(d[kind + '_train_epochs'], d[kind + '_' + path], color=color, ls=ls, lw=1.4, label=label)
        for row in (0, 1):
            ax = axes[row, col]
            grok_line(ax, 10400)
            ax.set_xlim(0, 20000); ax.set_xticks([0, 5000, 10000, 15000, 20000]); kfmt(ax)
        axes[0, col].set_ylim(-.03, 1.06); axes[0, col].set_yticks([0, .5, 1])
        axes[1, col].set_ylim(-.03, 1.06); axes[1, col].set_yticks([0, .5, 1])
        axes[1, col].set_xlabel('Epoch', labelpad=2)
        axes[1, col].text(10400 + 300, .5, 'grok', fontsize=8, color=INK, va='center')
    axes[0, 0].set_ylabel('Normalized GCS', labelpad=3)
    axes[1, 0].set_ylabel('Accuracy', labelpad=3)
    axes[0, 0].legend(loc='lower left', frameon=False, ncol=3, handlelength=1.8, columnspacing=1.0, borderpad=.1)
    axes[1, 0].legend(loc='center right', frameon=False, ncol=1, handlelength=1.8, borderpad=.1)
    panel_captions(fig, [[axes[1, 0]], [axes[1, 1]]], ['(a) Learned attention', '(b) Uniform attention'], .025)
    save(fig, ms / 'figures/attention_controls_readable.pdf')

# ---------------------------------------------------------------- shared helpers for the cohort trajectories
GROK_X = 'Epoch / $t_{\\mathrm{grok}}$'


def grok_marker(ax, y=.03, right=False):
    ax.axvline(1., color=CONTROL, ls=':', lw=.8, zorder=0)
    ax.text(1.03 if right else .97, y, 'grok', ha='left' if right else 'right', va='bottom', fontsize=8, color=INK)


def band(ax, x, med, q25, q75, color, ls, lw=1.6, label=None, alpha=.15, z=3):
    ok = np.isfinite(med)
    ax.fill_between(x[ok], q25[ok], q75[ok], color=color, alpha=alpha, lw=0, zorder=z - 2)
    ax.plot(x[ok], med[ok], color=color, ls=ls, lw=lw, zorder=z, label=label)


def scaled_axes(ax, xmax, first):
    ax.set_xlim(0, xmax); ax.set_ylim(-.03, 1.05); ax.set_yticks([0, .5, 1], ['0', '0.5', '1.0'])
    ax.set_xticks([0, .5, 1, 1.5], ['0', '0.5', '1.0', '1.5'])
    ax.set_xlabel(GROK_X, labelpad=2)
    if first:
        ax.set_ylabel('Scaled GCS', labelpad=3)
    else:
        ax.tick_params(labelleft=False)


def half_rule(stack):
    """Median and IQR on the grid where at least half of the group's runs are observed."""
    n = np.sum(np.isfinite(stack), axis=0)
    keep = n >= stack.shape[0] / 2
    with np.errstate(all='ignore'):
        q = np.nanpercentile(stack, [50, 25, 75], axis=0)
    q[:, ~keep] = np.nan
    return q


# ---------------------------------------------------------------- Figure 5: three references from the same run
def reference_trajectories(ms):
    d = np.load(ms / 'source_data/reviewed_appendix_figures/reference_trajectories.npz', allow_pickle=True)
    x = d['grid']
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.185, top=.80, wspace=.13)
    for i, (ax, (path, title, color, _)) in enumerate(zip(axes, PATHS)):
        for ref, ls, c, lw, alpha, label in [('pre', '--', REFERENCE, 1.2, .0, 'Pre-grok reference'),
                                             ('post', ':', REFERENCE, 1.4, .0, 'Post-grok reference'),
                                             ('final', '-', color, 1.7, .15, 'Final reference')]:
            k = f'{path}_{ref}'
            band(ax, x, d[k + '_median'], d[k + '_q25'], d[k + '_q75'], c, ls, lw=lw, label=label, alpha=alpha)
        ax.set_title(f'{title} ($n={d[path + "_final_stack"].shape[0]}$)', fontsize=9, pad=4)
        grok_marker(ax)
        scaled_axes(ax, 1.5, i == 0)
    h, l = axes[0].get_legend_handles_labels()
    order = [2, 0, 1]
    handles = [Line2D([], [], color=INK, ls='-', lw=1.7), h[0], h[1]]
    fig.legend(handles, [l[o] for o in order], frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.53, 1.0),
               handlelength=2.0, columnspacing=1.4, borderpad=.1)
    save(fig, ms / 'figures/reviewed/reference_trajectories.pdf')


# ---------------------------------------------------------------- Figure 9: detached-attention control
def detached_attention(ms):
    d = np.load(ms / 'source_data/reviewed_appendix_figures/detached_attention.npz', allow_pickle=True)
    x = d['grid']
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.185, top=.80, wspace=.13)
    for i, (ax, (path, title, color, _)) in enumerate(zip(axes, [PATHS[2], PATHS[1], PATHS[0]])):
        k = f'{path}_original'
        band(ax, x, d[k + '_median'], d[k + '_q25'], d[k + '_q75'], color, '-', lw=1.7, label='Original')
        if f'{path}_control_median' in d.files:
            k = f'{path}_control'
            band(ax, x, d[k + '_median'], d[k + '_q25'], d[k + '_q75'], INK, ':', lw=1.4, label='Detached control', alpha=.08, z=4)
        ax.set_title(f'{title} ($n={d[path + "_original_stack"].shape[0]}$)', fontsize=9, pad=4)
        grok_marker(ax)
        scaled_axes(ax, 1.5, i == 0)
    handles = [Line2D([], [], color=INK, ls='-', lw=1.7), Line2D([], [], color=INK, ls=':', lw=1.4)]
    fig.legend(handles, ['Original', 'Detached-attention control'], frameon=False, ncol=2, loc='upper center',
               bbox_to_anchor=(.53, 1.0), handlelength=2.0, columnspacing=1.4, borderpad=.1)
    save(fig, ms / 'figures/reviewed/detached_attention.pdf')


# ---------------------------------------------------------------- Figures 10 and 11: per-run cohort curves
def cohort(ms, name, groups, color, ls, keyfn, title, grok_right=False):
    d = np.load(ms / f'source_data/reviewed_appendix_figures/{name}.npz', allow_pickle=True)
    grid = d['grid']
    fig, axes = plt.subplots(1, len(groups), figsize=(W, 1.95), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.22, top=.89, wspace=.13)
    runs = d['trace_run']
    for i, (ax, g) in enumerate(zip(axes, groups)):
        members = keyfn(d, g)
        stack = np.concatenate([d[m + '_stack'] for m in members[1]], axis=0)
        for r in members[0]:
            sel = runs == r
            ax.plot(d['trace_x'][sel], d['trace_y'][sel], color=LIGHT, lw=.5, zorder=1)
        med, q25, q75 = half_rule(stack)
        band(ax, grid, med, q25, q75, color, ls, lw=1.7)
        ax.set_title(f'{title(g)} ($n={len(members[0])}$)', fontsize=9, pad=4)
        grok_marker(ax, right=grok_right)
        scaled_axes(ax, 1.6, i == 0)
    save(fig, ms / f'figures/reviewed/{name}.pdf')


def cohort_mlp(ms):
    def key(d, act):
        runs = np.where(d['run_activation'] == act)[0]
        return runs, [f'{act}_{t}' for t in ['add', 'mul', 'sub', 'div'] if f'{act}_{t}_stack' in d.files]
    cohort(ms, 'cohort_mlp', ['ReLU', 'GeLU', 'SiLU'], MLP, '--', key, lambda g: g.replace('GeLU', 'GELU'), grok_right=True)


def cohort_attention(ms):
    def key(d, task):
        return np.where(d['run_task'] == task)[0], [task]
    names = {'add': 'Addition', 'mul': 'Multiplication', 'sub': 'Subtraction', 'div': 'Division'}
    cohort(ms, 'cohort_attention', ['add', 'mul', 'sub', 'div'], ATTENTION, '-', key, names.get)



# ---------------------------------------------------------------- Figure 13: attention maps at five checkpoints
def attention_maps(ms):
    from matplotlib.colors import LinearSegmentedColormap
    d = np.load(ms / 'source_data/reviewed_appendix_figures/attention_maps.npz', allow_pickle=True)
    maps, epochs = d['attention_maps'], d['phase_epochs']
    names = ['Initial', 'First valley', 'Peak', 'Second valley', 'Final']
    cmap = LinearSegmentedColormap.from_list('attn', ['#FFFFFF', ATTENTION, '#4A1800'])
    nh, nc = maps.shape[1], maps.shape[0]
    fig, axes = plt.subplots(nh, nc, figsize=(W, 4.9))
    fig.subplots_adjust(left=.1, right=.87, bottom=.08, top=.92, wspace=.1, hspace=.08)
    for c in range(nc):
        for h in range(nh):
            ax = axes[h, c]
            im = ax.imshow(maps[c, h].T, origin='lower', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([0, 56, 112]); ax.set_yticks([0, 56, 112])
            if h == nh - 1:
                for t, ha in zip(ax.get_xticklabels(), ['left', 'center', 'right']):
                    t.set_ha(ha)
            ax.tick_params(labelbottom=(h == nh - 1), labelleft=(c == 0), length=2)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(.5)
            if c == 0:
                ax.set_ylabel(f'Head {h}\nOperand $b$', labelpad=2)
            if h == nh - 1:
                ax.set_xlabel('Operand $a$', labelpad=2)
        axes[0, c].set_title(f'{names[c]}\nepoch {epochs[c]:,}', fontsize=9, pad=4)
    cax = fig.add_axes([.89, .08, .016, .84])
    cb = fig.colorbar(im, cax=cax, ticks=[0, .5, 1])
    cb.outline.set_linewidth(.5); cb.ax.tick_params(length=2)
    cb.set_label('Attention weight', labelpad=4)
    save(fig, ms / 'figures/reviewed/attention_maps.pdf')


# ---------------------------------------------------------------- Figure 14: parameter-count-matched MLP resets
def matched_mlp(ms):
    d = np.load(ms / 'source_data/reviewed_appendix_figures/matched_mlp.npz', allow_pickle=True)
    labels = [str(l) for l in d['run_labels']]
    n = len(labels)
    x = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.45), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.28, top=.83, wspace=.06)
    rng = np.random.default_rng(0)
    for ax, key, k in [(axes[0], 'count_subset_drop', 32), (axes[1], 'norm_closest_drop', 8)]:
        sub = d[key]
        jit = rng.uniform(-.2, .2, size=sub.shape)
        ax.scatter((x[:, None] + jit).ravel(), sub.ravel(), s=4, color=CONTROL, alpha=.55, lw=0, zorder=2,
                   label=f'255-unit MLP subsets')
        ax.scatter(x, d['mlp_drop'], s=16, marker='s', color=MLP, lw=0, zorder=3, label='Full MLP reset')
        ax.scatter(x, d['attention_drop'], s=18, marker='o', color=ATTENTION, lw=0, zorder=4, label='Full attention reset')
        ax.set_xticks(x, [l.split('/')[1] for l in labels], rotation=90, fontsize=7.5)
        ax.tick_params(axis='x', length=0, pad=2)
        ax.set_xlim(-.7, n - .3); ax.set_ylim(-.03, .85); ax.set_yticks([0, .2, .4, .6, .8])
        ax.axvline(7.5, color=GRID_C, lw=.6, zorder=0)
        for lo, hi, name in [(0, 7, 'Addition'), (8, 16, 'Multiplication')]:
            ax.text((lo + hi) / 2, -.235, name, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8.5)
    axes[0].set_ylabel('Test-accuracy drop', labelpad=3)
    axes[0].set_title('(a) All 32 subsets per run', fontsize=9, pad=4)
    axes[1].set_title('(b) Eight subsets closest in norm', fontsize=9, pad=4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc='upper center', bbox_to_anchor=(.53, 1.0), frameon=False, ncol=3, handletextpad=.3,
               columnspacing=1.6, borderpad=.1, markerscale=1.2)
    save(fig, ms / 'figures/reviewed/matched_mlp.pdf')


FIGS = {'probe_positions': probe_positions, 'attention_controls': attention_controls,
        'reference_trajectories': reference_trajectories, 'detached_attention': detached_attention,
        'cohort_mlp': cohort_mlp, 'cohort_attention': cohort_attention,
        'attention_maps': attention_maps, 'matched_mlp': matched_mlp}

if __name__ == '__main__':
    ms = Path(sys.argv[1]).resolve()
    for name in (sys.argv[2:] or FIGS):
        FIGS[name](ms)
