"""Shared typography and semantic colors for the camera-ready figures."""
import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

# One meaning per color in every figure (Okabe-Ito hues, color-vision safe).
# Series that are not components or arms differ by line style or marker instead.
INK = '#222222'          # text, axes and ticks
MLP = '#0072B2'          # blue: the MLP and its learned structure
ATTENTION = '#D55E00'    # vermilion: attention and the parameters upstream of the MLP
RANDOM_MLP = '#CC79A7'   # reddish purple: untrained or freshly drawn random MLPs
REFERENCE = '#3D4248'    # dark grey: unchanged restart and reference cohorts
CONTROL = '#899198'      # mid grey: nulls, controls and reference lines
LIGHT = '#C5CAD0'        # light grey: individual runs, baseline and control bars
GRID = '#E5E7EB'         # gridlines and separators
WITHIN = '#009E73'       # bluish green: permutation within frequency groups
ALL_NEURONS = '#E69F00'  # orange: permutation across all neurons
SAME_NEURON = '#56B4E9'  # sky blue: same-neuron permutation (appendix only)
# Schematic fills: light tints with a dark-grey outline; text stays INK.
MLP_FILL = '#D6EAF8'
ATTENTION_FILL = '#FBE2C5'
STROKE = '#374151'


def fix_tminus(fig):
    """Typeset t^- in STIX math, a Times-style face.

    Times New Roman sets a superscript minus high and short, so $t^{-}$ prints
    like an overlined t. Only labels containing t^- are changed.
    """
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        if 't^{-}' in text.get_text():
            text.set_math_fontfamily('stix')


# Panel captions: points on the 7.0-in canvas, printed at \linewidth = 5.5 in (about 7.5 pt).
CAPTION_SIZE = 9.6


def panel_extent(ax, renderer):
    """Display box of one panel: the axes, its tick labels, its full axis labels and
    the texts, lines and legends it draws outside the frame."""
    from matplotlib.transforms import Bbox
    boxes = [ax.get_tightbbox(renderer), ax.xaxis.get_tightbbox(renderer),
             ax.yaxis.get_tightbbox(renderer)]
    return Bbox.union([box for box in boxes if box is not None])


def add_panel_captions(fig, panels, gap=0.08):
    """Caption each panel below the panel row as "(a) Title".

    `panels` holds (axes, caption, wrapped) triples. Each caption is centered on its
    panel's extent. All captions share one baseline, placed so that plain caption
    text hangs `gap` inches below the lowest element of the panel row. `wrapped` is
    the caption broken at a natural word boundary; it is used only when the one-line
    caption is wider than its panel, and then each line is its own baseline-anchored
    text, 1.1 x the type size apart (multi-line text spaces its lines differently in
    the PDF and PNG renderers).
    """
    from matplotlib.font_manager import FontProperties
    fix_tminus(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [panel_extent(ax, renderer) for ax, _, _ in panels]
    plain = FontProperties(size=CAPTION_SIZE, weight='normal')
    _, lp_height, lp_descent = renderer.get_text_width_height_descent('lp', plain, ismath=False)
    baseline = min(box.y0 for box in boxes) - gap * fig.dpi - (lp_height - lp_descent)
    advance = 1.1 * CAPTION_SIZE / 72 * fig.dpi

    def line(x, y, s):
        text = fig.text(x, y / fig.bbox.height, s, ha='center', va='baseline',
                        fontsize=CAPTION_SIZE, fontweight='normal', color=INK)
        if 't^{-}' in s:
            text.set_math_fontfamily('stix')
        return text

    captions = []
    for (ax, caption, wrapped), box in zip(panels, boxes):
        x = (box.x0 + box.x1) / 2 / fig.bbox.width
        texts = [line(x, baseline, caption)]
        if texts[0].get_window_extent(renderer).width > box.width:
            if wrapped is None:
                raise ValueError(f'{caption!r} is wider than its panel; give a wrapped form')
            texts[0].remove()
            texts = [line(x, baseline - k * advance, s) for k, s in enumerate(wrapped.split('\n'))]
            if max(t.get_window_extent(renderer).width for t in texts) > box.width:
                raise ValueError(f'{wrapped!r} is still wider than its panel')
        captions += texts
    if min(t.get_window_extent(renderer).y0 for t in captions) < 0:
        raise ValueError('A caption leaves the canvas at the bottom; raise the panels')
    return captions


def apply_style():
    """Use the actual Times New Roman faces, without a family-name fallback."""
    font_dir = os.environ.get('GCS_FIGURE_FONT_DIR')
    family = 'DejaVu Serif'
    if font_dir:
        for name in ('times.ttf','timesi.ttf','timesbd.ttf','timesbi.ttf'):
            font_manager.fontManager.addfont(Path(font_dir)/name)
        family = 'Times New Roman'
    mpl.rcParams.update({
        'font.family': family,
        'mathtext.fontset': 'custom',
        'mathtext.rm': family + '',
        'mathtext.it': family + ':italic',
        'mathtext.bf': family + ':bold',
        'mathtext.sf': family + '',
        'mathtext.tt': family + '',
        'mathtext.cal': family + ':italic',
        'mathtext.fallback': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'text.color': INK,
        'axes.labelcolor': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'savefig.facecolor': 'white',
    })
