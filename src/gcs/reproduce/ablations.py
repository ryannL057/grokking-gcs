"""Replot the recovered dimension and sample-size trajectories without model code."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np


PARAMETERS = {'d': [2, 4, 6, 10, 12, 14], 'N': [100, 200, 400, 750, 1850]}
PATHS = ['overall_gcs', 'attention_gcs', 'mlp_gcs']


def load_trajectories(ref: Path, parameter: str) -> dict:
    """Validate each saved epoch grid; leave precision and unequal horizons intact."""
    source = ref / f'evidence/ablations/{parameter.lower()}_trajectories.json'
    data = json.loads(source.read_text())
    expected = PARAMETERS[parameter]
    if not isinstance(data, dict) or set(data) != {str(value) for value in expected}:
        raise ValueError(f'Expected {parameter} groups {expected} in {source.name}')
    result = {}
    for value in expected:
        row = data[str(value)]
        if not isinstance(row, dict) or set(row) != {'epochs', *PATHS}:
            raise ValueError(f'Invalid fields for {parameter}={value}')
        epochs = np.asarray(row['epochs'], dtype=float)
        if (epochs.ndim != 1 or len(epochs) < 2 or not np.isfinite(epochs).all()
                or np.any(epochs < 0) or np.any(epochs != np.floor(epochs))
                or np.any(np.diff(epochs) <= 0)):
            raise ValueError(f'Invalid epoch grid for {parameter}={value}')
        for path in PATHS:
            scores = np.asarray(row[path], dtype=float)
            if scores.shape != epochs.shape or not np.isfinite(scores).all():
                raise ValueError(f'Invalid {path} trajectory for {parameter}={value}')
        result[value] = row
    return result


def plot_trajectories(data: dict, parameter: str):
    """Scale each path/parameter curve independently, as in Figures 6 and 7."""
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'gcs-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
        for ax, path, title in zip(axes, PATHS, ['Overall Path', 'Attention Path', 'MLP Path']):
            for index, value in enumerate(sorted(data)):
                row = data[value]
                scores = np.asarray(row[path], dtype=float)
                span = np.ptp(scores)
                scaled = np.zeros_like(scores) if span < 1e-10 else (scores - scores.min()) / span
                ax.plot(row['epochs'], scaled, label=f'{parameter}={value}',
                        color=plt.get_cmap('tab10')(index), linewidth=2, alpha=.85)
            ax.set(xlabel='Epoch', ylabel='Normalized GCS', title=title, ylim=(0, 1.05))
            ax.legend(loc='lower right', frameon=False)
    return fig


def render(target: str, ref: Path, output: Path) -> list[str]:
    parameter = {'fig06': 'd', 'fig07': 'N'}[target]
    data = load_trajectories(ref, parameter)
    fig = plot_trajectories(data, parameter)
    import matplotlib.pyplot as plt
    paths = [f'{target}.{suffix}' for suffix in ['pdf', 'png', 'svg']]
    try:
        for name in paths:
            fig.savefig(output / name, dpi=180)
    finally:
        plt.close(fig)
    return paths
