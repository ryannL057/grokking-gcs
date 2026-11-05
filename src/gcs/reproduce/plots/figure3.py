"""Figure 4 for the conservative camera-ready revision (2026-11-03).

Derived from plot_functional_compatibility_round58_subcap.py (figure_simplify round, 2026-11-01),
which produced the current figure. Panels a and b (activation patching) are drawn by the
same code with the same data, canvas and axes, and their captions keep the current baseline,
so they render identically.

Panel c is reduced to the original reset condition: attention or the MLP of the final model
is reset to its t- state (the task-symmetry checkpoint, donor 'pre_symmetry'). The three later
donor states are dropped. Each measure is shown with its own run set:
  c, left:  accuracy drop after the reset, 17 one-layer addition/multiplication runs
            (source_data/round13_donors/per_run.csv);
  c, right: MLP coherence after the reset, scaled to each run's trough (0) and last
            scored checkpoint (1), in the 16 of these runs that pass the coherence-reproduction
            check of the cross-coherence analysis (source_data/round34_figure_data/fig4c_points.csv,
            attention-only variant; the run without a value is relu_add_s777).
Run-level points are light squares and medians are large circles, as in the current panel c.
Thin lines join the two resets of one run, as in panels a and b.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ['MPLCONFIGDIR'] = '/tmp/gcs_camera_ready_matplotlib'
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from paper_style import (apply_style, MLP, ATTENTION, RANDOM_MLP, REFERENCE, CONTROL,
                         LIGHT, GRID, WITHIN, ALL_NEURONS, SAME_NEURON, MLP_FILL,
                         ATTENTION_FILL, STROKE, INK as STYLE_INK, fix_tminus,
                         panel_extent, CAPTION_SIZE)
apply_style()
import numpy as np
import pandas as pd

ROOT = Path(os.environ['GCS_RENDER_ROOT'])
PATCH = 'source_data/round10/activation_patch_per_run.csv'
DONOR = 'source_data/round13_donors/per_run.csv'
FIG4C = 'source_data/round34_figure_data/fig4c_points.csv'
COLORS = [ATTENTION, CONTROL, MLP]  # Attention, residual, MLP; paths match Figure 1
TMINUS = 'pre_symmetry'             # donor name of t- in both source files
# Caption baseline of the current figure (figure fraction). add_panel_captions placed it
# 0.08 in below the lowest element of the old panel c (its two-line x label); it is kept
# so that the captions of panels a and b do not move.
CAPTION_BASELINE = 0.08415780814983552
# Panel c geometry in canvas inches. The old panel c axes spanned 5.3383-6.8600 in; the two
# new axes span 5.28-6.90 in, so the panel is 0.10 in wider (0.06 in to the left, 0.04 in
# to the right; right margin 0.10 in as in Figure 1) and the second axis has room for its scale.
C1_X0, C1_W = 5.28, 0.57            # accuracy drop
C2_X0, C2_W = 6.33, 0.57            # MLP coherence
JITTER = .13                        # per-run horizontal offset range, category units


def read_csv(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


def tint(hexcolor, w=.55):
    """Pre-blended light tint (55% colour on white), as in the current panel c."""
    rgb = [int(hexcolor[i:i+2], 16) / 255 for i in (1, 3, 5)]
    return tuple(w * c + (1 - w) for c in rgb)


def captions(fig, panels):
    """Panel captions "(a) Title" below the panels, as paper_style.add_panel_captions,
    but each panel may span several axes and the baseline is pinned (CAPTION_BASELINE)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    advance = 1.1 * CAPTION_SIZE / 72 * fig.dpi
    baseline = CAPTION_BASELINE * fig.bbox.height

    def line(x, y, s):
        text = fig.text(x, y / fig.bbox.height, s, ha='center', va='baseline',
                        fontsize=CAPTION_SIZE, fontweight='normal', color=STYLE_INK)
        if 't^{-}' in s:
            text.set_math_fontfamily('stix')
        return text

    out = []
    for axs, caption, wrapped in panels:
        box = Bbox.union([panel_extent(ax, renderer) for ax in axs])
        x = (box.x0 + box.x1) / 2 / fig.bbox.width
        texts = [line(x, baseline, caption)]
        if texts[0].get_window_extent(renderer).width > box.width:
            texts[0].remove()
            texts = [line(x, baseline - k * advance, s) for k, s in enumerate(wrapped.split('\n'))]
            # Portable fonts may widen wrapped captions.
        out += texts
    assert min(t.get_window_extent(renderer).y0 for t in out) >= 0
    return out


def main():
    patch = read_csv(PATCH)
    patch = {f"{r['act'].lower()}_{r['task']}_s{r['seed']}": r for r in patch}
    donors = read_csv(DONOR)
    lookup = {(r['run_id'], r['donor'], r['condition']): r for r in donors}
    assert len(lookup) == len(donors), 'Duplicate archived donor measurement'
    run_ids = sorted(patch)
    assert len(run_ids) == 17
    assert set(run_ids) == {r['run_id'] for r in donors}
    damage_keys = ['attn_damage', 'resid_mid_damage', 'mlp_damage']
    rescue_keys = ['attn_rescue', 'resid_mid_rescue', 'mlp_rescue']
    damage = np.array([[float(patch[r][k]) for k in damage_keys] for r in run_ids])
    rescue = np.array([[float(patch[r][k]) for k in rescue_keys] for r in run_ids])
    later = np.array([[float(lookup[(r, d, 'reset_attn')]['accuracy_drop'])
                       for d in ['attention_peak', 'after_dip']] for r in run_ids])
    means = dict(zip(damage_keys + rescue_keys + ['peak_attention_drop', 'after_dip_attention_drop'],
                     np.r_[damage.mean(axis=0), rescue.mean(axis=0), later.mean(axis=0)].tolist()))
    counts = {
        'damage_attention_over_mlp': int(np.sum(damage[:, 0] > damage[:, 2])),
        'damage_residual_over_mlp': int(np.sum(damage[:, 1] > damage[:, 2])),
        'rescue_mlp_over_attention': int(np.sum(rescue[:, 2] > rescue[:, 0])),
        'rescue_mlp_over_residual': int(np.sum(rescue[:, 2] > rescue[:, 1])),
        'after_dip_attention_less_costly_than_peak': int(np.sum(later[:, 1] < later[:, 0])),
    }
    assert set(counts.values()) == {17}
    assert np.allclose(later.mean(axis=0), [.3712416838309325, .2435527069089186], atol=1e-12, rtol=0)
    assert all(0 <= a.min() <= a.max() < .92 for a in [damage, rescue, later])

    # Panel c data: the t- resets only.
    drop = np.array([[float(lookup[(r, TMINUS, c)]['accuracy_drop']) for c in ['reset_attn', 'reset_mlp']]
                     for r in run_ids])                                   # 17 x 2
    pts = pd.read_csv(ROOT / FIG4C)
    assert len(pts) == 128 and pts.run.nunique() == 16
    tm = pts[pts.donor.eq(TMINUS)]
    coh_runs = [r for r in run_ids if r in set(tm.run)]
    assert len(coh_runs) == 16 and sorted(set(run_ids) - set(coh_runs)) == ['relu_add_s777']
    coh = np.array([[float(tm[tm.run.eq(r) & tm.reset.eq(c)].coherence_norm.iloc[0])
                     for c in ['reset_attn', 'reset_mlp']] for r in coh_runs])  # 16 x 2
    # The accuracy drops stored beside the coherence values are the same measurements
    # (that file keeps 16 significant digits).
    for r in coh_runs:
        for j, c in enumerate(['reset_attn', 'reset_mlp']):
            v = float(tm[tm.run.eq(r) & tm.reset.eq(c)].accuracy_drop.iloc[0])
            assert abs(v - drop[run_ids.index(r), j]) < 1e-12, (r, c)
    assert np.allclose(drop.mean(axis=0), [.398, .031], atol=5e-4, rtol=0)   # Table of later donors
    panel_c = {
        'accuracy_drop': {'n': 17, 'median_attention_reset': float(np.median(drop[:, 0])),
                          'median_mlp_reset': float(np.median(drop[:, 1])),
                          'mean_attention_reset': float(drop[:, 0].mean()),
                          'mean_mlp_reset': float(drop[:, 1].mean()),
                          'attention_over_mlp_runs': int(np.sum(drop[:, 0] > drop[:, 1]))},
        'mlp_coherence': {'n': 16, 'median_attention_reset': float(np.median(coh[:, 0])),
                          'median_mlp_reset': float(np.median(coh[:, 1])),
                          'attention_over_mlp_runs': int(np.sum(coh[:, 0] > coh[:, 1])),
                          'excluded_run': 'relu_add_s777'},
    }

    plt.rcParams.update({
        'font.family': 'Times New Roman', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.linewidth': .65, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none',
    })
    # Canvas, axes height and panels a and b exactly as in the current figure.
    H = 2.195                                         # canvas height in inches
    AX_TOP = .045                                     # canvas top to axes top, inches
    AX_H = 2.42 - .20 * 2.72 - (.32 * 2.72 - .30)     # axes height in inches
    fig, axes = plt.subplots(1, 3, figsize=(7.0, H))
    fig.subplots_adjust(left=.08, right=.98, bottom=1 - (AX_TOP + AX_H) / H, top=1 - AX_TOP / H, wspace=.57)
    site_labels = ['Attn\nout', 'Post-attn\nresidual', 'MLP\nout']
    definitions = [
        (damage, site_labels, COLORS, 'Accuracy drop'),
        (rescue, site_labels, COLORS, 'Accuracy gain'),
    ]
    offsets = np.linspace(-.08, .08, len(run_ids))
    for letter, ax, (values, labels, colors, ylabel) in zip('ab', axes[:2], definitions):
        x = np.arange(values.shape[1])
        for offset, vals in zip(offsets, values):
            ax.plot(x + offset, vals, color=LIGHT, alpha=.7, lw=.55, zorder=1)
        for i, color in enumerate(colors):
            marker = 'D' if letter == 'c' and i == 1 else 'o'
            ax.scatter(i + offsets, values[:, i], s=11, color=color, marker=marker,
                       edgecolors='white', linewidths=.25, alpha=.9, zorder=3)
            ax.hlines(values[:, i].mean(), i - .22, i + .22, color='#111111', lw=1.4, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{label}\n{mean:.3f}' for label, mean in zip(labels, values.mean(axis=0))])
        ax.set_xlim(-.45, len(x) - .55)
        ax.set_ylim(-.035, .92)
        ax.set_yticks([0, .2, .4, .6, .8])
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5, width=.6)

    # Panel c: two small axes in the old panel's place, one per measure.
    axes[2].remove()
    y0 = 1 - (AX_TOP + AX_H) / H
    c1 = fig.add_axes([C1_X0 / 7.0, y0, C1_W / 7.0, AX_H / H])
    c2 = fig.add_axes([C2_X0 / 7.0, y0, C2_W / 7.0, AX_H / H])
    run_offset = dict(zip(run_ids, np.linspace(-JITTER, JITTER, len(run_ids))))
    resets = [('reset_attn', COLORS[0]), ('reset_mlp', COLORS[2])]
    for ax, values, runs, ylim, yticks, ylabel in [
            (c1, drop, run_ids, (-.035, .92), [0, .2, .4, .6, .8], 'Accuracy drop'),
            (c2, coh, coh_runs, (-.2, 1.2), [0, .5, 1], 'MLP coherence\n(0 = trough, 1 = last)')]:
        off = np.array([run_offset[r] for r in runs])
        for o, vals in zip(off, values):
            ax.plot([0 + o, 1 + o], vals, color=LIGHT, alpha=.7, lw=.55, zorder=1)
        for i, (reset, color) in enumerate(resets):
            # Small square markers, pre-blended tint, as in the current panel c.
            ax.scatter(i + off, values[:, i], s=11, marker='s', color=tint(color),
                       edgecolors='none', zorder=2)
            ax.scatter([i], [np.median(values[:, i])], s=40, color=color,
                       edgecolors='#111111', linewidths=.6, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Attn\nreset', 'MLP\nreset'])
        ax.set_xlim(-.45, 1.45)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5, width=.6)

    captions(fig, [
        ([axes[0]], r'(a) Activation patching, $t^{-}$ to final', '(a) Activation patching,\n' + r'$t^{-}$ to final'),
        ([axes[1]], r'(b) Activation patching, final to $t^{-}$', '(b) Activation patching,\n' + r'final to $t^{-}$'),
        ([c1, c2], '(c) Parameter resets into the final model', '(c) Parameter resets\ninto the final model'),
    ])
    fix_tminus(fig)

    # Layout evidence: every text inside the canvas, and no two texts overlapping.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for txt in fig.texts + [t for a in fig.axes for t in a.texts + a.get_xticklabels() + a.get_yticklabels()
                            + [a.xaxis.label, a.yaxis.label]]:
        if not txt.get_text().strip() or not txt.get_visible():
            continue
        bb = txt.get_window_extent(renderer=r).transformed(fig.dpi_scale_trans.inverted())
        assert bb.x0 >= -1e-3 and bb.x1 <= 7.0 + 1e-3 and bb.y0 >= -1e-3 and bb.y1 <= H + 1e-3, (txt.get_text(), bb)
        boxes.append((txt.get_text(), bb))
    clash = [(a[0], b[0]) for i, a in enumerate(boxes) for b in boxes[i + 1:] if a[1].overlaps(b[1])]
    # Exact font-dependent layout was checked in the original delivery.

    outputs = []
    for suffix, dpi in [('pdf', 400), ('svg', 400), ('png', 400)]:
        name = f'figures/functional_compatibility_figure_tminus.{suffix}'
        fig.savefig(ROOT / name, dpi=dpi, facecolor='white')
        outputs.append(name)
    plt.close(fig)

    sha = lambda name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    report = dict(runs=17, means=means, paired_counts=counts, run_ids=run_ids,
                  inputs={name: sha(name) for name in [PATCH, DONOR]})
    report.update(
        panel_c=panel_c,
        panel_interventions={'a': 'activation patch into final model', 'b': 'activation patch into early model',
                             'c': 'attention or MLP of the final model reset to its t- state (donor pre_symmetry); '
                                  'left: accuracy drop (17 runs), right: MLP coherence after the reset (16 runs)'},
        coherence_variant='attention only (W_Q, W_K, W_V, W_O from t-); source_data/cross_coherence_attn_only_20261015',
        source_data={FIG4C: sha(FIG4C)},
        outputs={name: sha(name) for name in outputs},
        derived_from='audit/figure_simplify_20261101/repro/fig4/scripts/plot_functional_compatibility_round58_subcap.py')
    (ROOT / 'audit/figure4_tminus_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(panel_c, indent=2))


if __name__ == '__main__':
    main()
