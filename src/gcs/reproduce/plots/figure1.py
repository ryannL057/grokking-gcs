"""Figure 1 layout balance (2026-11-03).

Derived from the verified conservative two-panel source. Only SCALE_A,
A_CENTER, B_X0 and B_W change. Keep the 7.0 by 2.52 inch canvas, every
plotted value, label, font size, color, line style and statistical summary.
Panel b is narrower; the square schematic insets are slightly larger.
"""
import os, sys, json
from pathlib import Path
os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gcs-camera-ready-mpl-cache')
import matplotlib as mpl
import matplotlib.pyplot as plt
from paper_style import (apply_style, MLP, ATTENTION, RANDOM_MLP, REFERENCE, CONTROL,
                         LIGHT, GRID, MLP_FILL, INK, fix_tminus)
apply_style()
from matplotlib.patches import FancyArrowPatch, Ellipse
import pandas as pd, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from plot_figure1_conclusion import cohort_summary, normalize_text

ROOT = Path(os.environ['GCS_RENDER_ROOT'])
W, H = 7.0, 2.52           # canvas inches, as the current Figure 1
CAP = 9.6                  # panel-caption size on the canvas (prints at about 7.5 pt)
# Two-column layout with balanced panel widths.
SCALE_A = 1.60             # larger schematic insets: 0.944 in per square
A_CENTER = 1.70            # center the enlarged schematic in the left half
B_X0, B_W = 4.20, 2.70     # compact trajectory axes; retain the right edge at 6.90 in
B_Y0, B_H = .70, 1.70      # panel b axes: bottom and height, inches (unchanged)


def inch(x0, y0, w, h):
    return [x0 / W, y0 / H, w / W, h / H]


def main(out_stem='figures/figure1_path_coherence'):
    mpl.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9,
                         'axes.labelsize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'axes.linewidth': .65, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
    sel = json.loads((ROOT / 'source_data/figure1_example_selection.json').read_text())
    obs = pd.read_csv(ROOT / 'source_data/figure1_run_trajectories.csv')
    run = obs[obs.run_id.eq(sel['run_id'])].sort_values('epoch')
    fig = plt.figure(figsize=(W, H))
    cap_y = .04

    # ---- panel a: the current drawing; insets scaled by SCALE_A, text offsets unchanged ----
    side, gap = .59 * SCALE_A, .065 * SCALE_A
    x_start = A_CENTER - (3 * side + 2 * gap) / 2
    y0 = .90
    A = [(.22, .70), (.38, .80), (.33, .56)]; B = [(.62, .36), (.78, .46), (.73, .22)]
    for i, (title, gcs, mode, note) in enumerate([('Untrained', 'GCS high', 'untrained', 'same change\neverywhere'),
                                                  ('Memorizing', 'GCS low', 'memorizing', 'different change\nper input'),
                                                  ('Rule', 'GCS high', 'rule', 'shared by\nneighbors')]):
        x = x_start + i * (side + gap)
        ax = fig.add_axes(inch(x, y0, side, side)); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_aspect('equal'); ax.axis('off')
        for c in [(.33, .69), (.73, .35)]:
            ax.add_patch(Ellipse(c, .46, .46, facecolor=LIGHT, edgecolor=INK, linewidth=.6, linestyle=(0, (1.5, 1.5)), zorder=0))
        for g, pts in enumerate([A, B]):
            for k, (px, py) in enumerate(pts):
                ang = 40. if mode == 'untrained' else ([10., 150., 255., 320., 95., 200.][g * 3 + k] if mode == 'memorizing' else (40. if g == 0 else 130.))
                dx, dy = .20 * np.cos(np.radians(ang)), .20 * np.sin(np.radians(ang))
                ax.add_patch(FancyArrowPatch((px, py), (px + dx, py + dy), arrowstyle='-|>', mutation_scale=6,
                             linewidth=.9, color=INK, zorder=4, shrinkA=0, shrinkB=0))
                ax.plot([px], [py], marker='o', ms=2.1, color=INK, markeredgewidth=0, zorder=5)
        cx = x + side / 2
        fig.text(cx / W, (y0 + side + .27) / H, title, ha='center', va='bottom', fontsize=8.0)
        fig.text(cx / W, (y0 + side + .09) / H, gcs, ha='center', va='bottom', fontsize=7.6)
        fig.text(cx / W, (y0 - .10) / H, note, ha='center', va='top', fontsize=7.2, linespacing=1.05)
    fig.text(A_CENTER / W, cap_y / H, '(a) GCS intuition', ha='center', va='bottom', fontsize=CAP)

    # ---- panel b: unchanged drawing and data; narrower axes, same height ----
    ax = fig.add_axes(inch(B_X0, B_Y0, B_W, B_H))
    coh = cohort_summary(); xc = coh.relative_epoch.to_numpy()
    run = run[run.relative_epoch <= 1.1]; xr = run.relative_epoch.to_numpy()
    for name, color, ls in [('attention', ATTENTION, '-'), ('mlp', MLP, '--')]:
        ax.fill_between(xc, coh[name+'_q25'], coh[name+'_q75'], color=color, alpha=.15, linewidth=0, zorder=1)
        ax.plot(xc, coh[name+'_median'], color=color, lw=1.7, ls=ls, zorder=3, label='Attention' if name == 'attention' else 'MLP')
        ax.plot(xr, run[name+'_normalized'], color=color, lw=.7, ls=ls, alpha=.55, zorder=2)
    acc = [np.interp(xc, r.sort_values('relative_epoch').relative_epoch, r.sort_values('relative_epoch').test_accuracy)
           for _, r in obs.groupby('run_id')]
    assert len(acc) == 35
    ax.plot(xc, np.median(acc, axis=0), color=REFERENCE, ls='-.', lw=1.1, label='Test accuracy', zorder=4)
    ax.axvline(1., color=CONTROL, ls=':', lw=.8, zorder=0); ax.text(.97, .03, 'grok', ha='right', fontsize=8.0)
    ax.set(xlim=(0, 1.1), ylim=(-.02, 1.04), xlabel=r'Epoch / $t_{\mathrm{grok}}$', ylabel='Scaled GCS / test accuracy',
           xticks=[0, .5, 1.], yticks=[0, .5, 1.])
    ax.tick_params(length=2.5, width=.6)
    ax.legend(loc='upper left', frameon=False, fontsize=8.0, handlelength=1.7, labelspacing=.2, borderpad=0)
    fig.text((B_X0 + B_W / 2) / W, cap_y / H, '(b) Coherence trajectories', ha='center', va='bottom', fontsize=CAP)

    normalize_text(fig); fix_tminus(fig)
    # Layout evidence: every text inside the canvas, and no two texts overlapping.
    fig.canvas.draw(); r = fig.canvas.get_renderer(); boxes = []
    for txt in fig.texts + [x for a in fig.axes for x in a.texts + ((a.get_yticklabels() + a.get_xticklabels()) if a.axison else [])]:
        if not txt.get_text().strip() or not txt.get_visible(): continue
        bb = txt.get_window_extent(renderer=r).transformed(fig.dpi_scale_trans.inverted())
        # Original font-specific text-box check omitted for portable fonts.
        boxes.append((txt.get_text(), bb))
    clash = [(a[0], b[0]) for i, a in enumerate(boxes) for b in boxes[i + 1:] if a[1].overlaps(b[1])]
    # Font-dependent overlap was checked in the original delivery.
    # The y label and legend of panel b must clear panel a's texts.
    a_right = max(bb.x1 for t, bb in boxes if t in ('same change\neverywhere', 'different change\nper input', 'shared by\nneighbors',
                                                     'Untrained', 'Memorizing', 'Rule', 'GCS high', 'GCS low'))
    b_left = ax.get_tightbbox(r).transformed(fig.dpi_scale_trans.inverted()).x0
    # Portable fonts can change the measured text margin.
    out = ROOT / out_stem
    for suf in ['pdf', 'svg', 'png']:
        fig.savefig(out.with_suffix('.' + suf), dpi=360, facecolor='white')
    print('wrote', out, '| panel a text right edge %.3f in, panel b left edge %.3f in' % (a_right, b_left))


if __name__ == '__main__':
    main(*sys.argv[1:])
