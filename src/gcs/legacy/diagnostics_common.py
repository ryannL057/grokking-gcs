from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import pickle
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


RUNS_1L = [
    ('ReLU', 'add', 777, 'experiments/relu_1layer_add_20260213_004649'),
    ('ReLU', 'add', 42, 'experiments/relu_1layer_add_s42_20260213_004649'),
    ('ReLU', 'add', 123, 'experiments/relu_1layer_add_s123_20260213_004649'),
    ('GeLU', 'add', 42, 'experiments/gelu_1layer_add_p113_s42_20260301_041540'),
    ('GeLU', 'add', 123, 'experiments/gelu_1layer_add_p113_s123_20260301_044607'),
    ('GeLU', 'sub', 42, 'experiments/gelu_1layer_sub_p113_s42_20260301_101650'),
    ('GeLU', 'sub', 123, 'experiments/gelu_1layer_sub_p113_s123_20260301_112309'),
    ('GeLU', 'mul', 42, 'experiments/gelu_1layer_mul_p113_s42_20260301_063635'),
    ('GeLU', 'mul', 123, 'experiments/gelu_1layer_mul_p113_s123_20260301_071116'),
    ('GeLU', 'div', 42, 'experiments/gelu_1layer_div_p113_s42_20260301_140319'),
    ('GeLU', 'div', 123, 'experiments/gelu_1layer_div_p113_s123_20260301_150648'),
    ('ReLU', 'sub', 42, 'experiments/relu_1layer_sub_p113_s42_20260301_074151'),
    ('ReLU', 'sub', 123, 'experiments/relu_1layer_sub_p113_s123_20260301_082109'),
    ('ReLU', 'div', 42, 'experiments/relu_1layer_div_p113_s42_20260301_085906'),
    ('ReLU', 'div', 123, 'experiments/relu_1layer_div_p113_s123_20260301_094158'),
    ('ReLU', 'mul', 42, 'experiments/relu_1layer_mul_p113_s42_20260301_051814'),
    ('ReLU', 'mul', 123, 'experiments/relu_1layer_mul_p113_s123_20260301_223159'),
    ('SiLU', 'sub', 42, 'experiments/silu_1layer_sub_p113_s42_20260301_160848'),
    ('SiLU', 'sub', 123, 'experiments/silu_1layer_sub_p113_s123_20260301_171748'),
    ('SiLU', 'div', 42, 'experiments/silu_1layer_div_p113_s42_20260301_183736'),
    ('SiLU', 'div', 123, 'experiments/silu_1layer_div_p113_s123_20260301_200724'),
    ('SiLU', 'mul', 42, 'experiments/silu_1layer_mul_p113_s42_20260301_055000'),
    ('SiLU', 'mul', 123, 'experiments/silu_1layer_mul_p113_s123_20260301_231450'),
    ('SiLU', 'add', 42, 'experiments/silu_1layer_add_s42_20260213_000145'),
    ('GeLU', 'add', 777, 'checkpoints/1layer/GeLU/add'),
    ('SiLU', 'add', 777, 'checkpoints/1layer/SiLU/add'),
    ('ReLU', 'mul', 777, 'checkpoints/1layer/ReLU/mul'),
    ('GeLU', 'mul', 777, 'checkpoints/1layer/GeLU/mul'),
    ('SiLU', 'mul', 777, 'checkpoints/1layer/SiLU/mul'),
    ('ReLU', 'sub', 777, 'checkpoints/1layer/ReLU/sub'),
    ('GeLU', 'sub', 777, 'checkpoints/1layer/GeLU/sub'),
    ('SiLU', 'sub', 777, 'checkpoints/1layer/SiLU/sub'),
    ('ReLU', 'div', 777, 'checkpoints/1layer/ReLU/div'),
    ('GeLU', 'div', 777, 'checkpoints/1layer/GeLU/div'),
    ('SiLU', 'div', 777, 'checkpoints/1layer/SiLU/div'),
]

RUNS_2L = {
    'add': [
        ('ReLU', 'add', 777, 'checkpoints/2layer/ReLU/add'),
        ('GeLU', 'add', 777, 'checkpoints/2layer/GeLU/add'),
        ('SiLU', 'add', 777, 'checkpoints/2layer/SiLU/add'),
        ('ReLU', 'add', 42, 'experiments/relu_2layer_add_p113_s42_20260302_005228'),
        ('ReLU', 'add', 123, 'experiments/relu_2layer_add_p113_s123_20260302_024651'),
        ('GeLU', 'add', 42, 'experiments/gelu_2layer_add_p113_s42_20260302_043217'),
        ('GeLU', 'add', 123, 'experiments/gelu_2layer_add_p113_s123_20260302_061712'),
        ('SiLU', 'add', 42, 'experiments/silu_2layer_add_p113_s42_20260302_075848'),
        ('SiLU', 'add', 123, 'experiments/silu_2layer_add_p113_s123_20260302_094750'),
    ],
    'mul': [
        ('ReLU', 'mul', 777, 'experiments/relu_2layer_mul_p113_20260303_072852'),
        ('ReLU', 'mul', 42, 'experiments/relu_2layer_mul_p113_s42_20260303_092008'),
        ('ReLU', 'mul', 123, 'experiments/relu_2layer_mul_p113_s123_20260303_111515'),
        ('GeLU', 'mul', 777, 'experiments/gelu_2layer_mul_p113_20260303_130648'),
        ('GeLU', 'mul', 42, 'experiments/gelu_2layer_mul_p113_s42_20260303_145740'),
        ('GeLU', 'mul', 123, 'experiments/gelu_2layer_mul_p113_s123_20260303_165047'),
        ('SiLU', 'mul', 777, 'experiments/silu_2layer_mul_p113_20260303_184519'),
        ('SiLU', 'mul', 42, 'experiments/silu_2layer_mul_p113_s42_20260303_204704'),
        ('SiLU', 'mul', 123, 'experiments/silu_2layer_mul_p113_s123_20260303_224614'),
    ],
    'sub': [
        ('ReLU', 'sub', 777, 'experiments/relu_2layer_sub_p113_20260305_203610'),
        ('ReLU', 'sub', 42, 'experiments/relu_2layer_sub_p113_s42_20260305_225541'),
        ('ReLU', 'sub', 123, 'experiments/relu_2layer_sub_p113_s123_20260306_011330'),
        ('GeLU', 'sub', 777, 'experiments/gelu_2layer_sub_p113_20260306_100234'),
        ('GeLU', 'sub', 42, 'experiments/gelu_2layer_sub_p113_s42_20260306_122317'),
        ('GeLU', 'sub', 123, 'experiments/gelu_2layer_sub_p113_s123_20260306_143514'),
        ('SiLU', 'sub', 777, 'experiments/silu_2layer_sub_p113_20260305_203609'),
        ('SiLU', 'sub', 42, 'experiments/silu_2layer_sub_p113_s42_20260305_230013'),
        ('SiLU', 'sub', 123, 'experiments/silu_2layer_sub_p113_s123_20260306_011557'),
    ],
    'div': [
        ('ReLU', 'div', 777, 'experiments/relu_2layer_div_p113_20260306_032107'),
        ('ReLU', 'div', 42, 'experiments/relu_2layer_div_p113_s42_20260306_053715'),
        ('ReLU', 'div', 123, 'experiments/relu_2layer_div_p113_s123_20260306_074430'),
        ('GeLU', 'div', 777, 'experiments/gelu_2layer_div_p113_20260306_164651'),
        ('GeLU', 'div', 42, 'experiments/gelu_2layer_div_p113_s42_20260306_185736'),
        ('GeLU', 'div', 123, 'experiments/gelu_2layer_div_p113_s123_20260306_210853'),
        ('SiLU', 'div', 777, 'experiments/silu_2layer_div_p113_20260306_032539'),
        ('SiLU', 'div', 42, 'experiments/silu_2layer_div_p113_s42_20260306_054259'),
        ('SiLU', 'div', 123, 'experiments/silu_2layer_div_p113_s123_20260306_075256'),
    ],
}

RUNS_3L = {
    'add': [
        ('ReLU', 'add', 777, 'checkpoints/3layer/ReLU/add'),
        ('GeLU', 'add', 777, 'checkpoints/3layer/GeLU/add'),
        ('SiLU', 'add', 777, 'checkpoints/3layer/SiLU/add'),
        ('ReLU', 'add', 42, 'experiments/relu_3layer_add_p113_s42_20260302_113308'),
        ('ReLU', 'add', 123, 'experiments/relu_3layer_add_p113_s123_20260302_144618'),
        ('GeLU', 'add', 42, 'experiments/gelu_3layer_add_p113_s42_20260302_180241'),
        ('GeLU', 'add', 123, 'experiments/gelu_3layer_add_p113_s123_20260302_212452'),
        ('SiLU', 'add', 42, 'experiments/silu_3layer_add_p113_s42_20260303_004304'),
        ('SiLU', 'add', 123, 'experiments/silu_3layer_add_p113_s123_20260303_040717'),
    ],
    'mul': [
        ('ReLU', 'mul', 777, 'experiments/relu_3layer_mul_p113_20260304_004253'),
        ('ReLU', 'mul', 42, 'experiments/relu_3layer_mul_p113_s42_20260304_041323'),
        ('ReLU', 'mul', 123, 'experiments/relu_3layer_mul_p113_s123_20260304_074825'),
        ('GeLU', 'mul', 777, 'experiments/gelu_3layer_mul_p113_20260304_110440'),
        ('GeLU', 'mul', 42, 'experiments/gelu_3layer_mul_p113_s42_20260304_143215'),
        ('GeLU', 'mul', 123, 'experiments/gelu_3layer_mul_p113_s123_20260304_173538'),
        ('SiLU', 'mul', 777, 'experiments/silu_3layer_mul_p113_20260304_181741'),
        ('SiLU', 'mul', 42, 'experiments/silu_3layer_mul_p113_s42_20260304_185921'),
        ('SiLU', 'mul', 123, 'experiments/silu_3layer_mul_p113_s123_20260304_194115'),
    ],
    'sub': [
        ('ReLU', 'sub', 777, 'experiments/relu_3layer_sub_p113_20260306_101323'),
        ('ReLU', 'sub', 42, 'experiments/relu_3layer_sub_p113_s42_20260306_135729'),
        ('ReLU', 'sub', 123, 'experiments/relu_3layer_sub_p113_s123_20260306_174409'),
        ('GeLU', 'sub', 777, 'experiments/gelu_3layer_sub_p113_20260305_203610'),
        ('GeLU', 'sub', 42, 'experiments/gelu_3layer_sub_p113_s42_20260306_003021'),
        ('GeLU', 'sub', 123, 'experiments/gelu_3layer_sub_p113_s123_20260306_041759'),
        ('SiLU', 'sub', 777, 'experiments/silu_3layer_sub_p113_20260306_192023'),
        ('SiLU', 'sub', 42, 'experiments/silu_3layer_sub_p113_s42_20260306_230150'),
        ('SiLU', 'sub', 123, 'experiments/silu_3layer_sub_p113_s123_20260307_015314'),
    ],
    'div': [
        ('ReLU', 'div', 777, 'experiments/relu_3layer_div_p113_20260306_212834'),
        ('ReLU', 'div', 42, 'experiments/relu_3layer_div_p113_s42_20260307_003533'),
        ('ReLU', 'div', 123, 'experiments/relu_3layer_div_p113_s123_20260307_032423'),
        ('GeLU', 'div', 777, 'experiments/gelu_3layer_div_p113_20260306_080350'),
        ('GeLU', 'div', 42, 'experiments/gelu_3layer_div_p113_s42_20260306_114753'),
        ('GeLU', 'div', 123, 'experiments/gelu_3layer_div_p113_s123_20260306_153419'),
        ('SiLU', 'div', 777, 'experiments/silu_3layer_div_p113_20260307_044059'),
        ('SiLU', 'div', 42, 'experiments/silu_3layer_div_p113_s42_20260307_070607'),
        ('SiLU', 'div', 123, 'experiments/silu_3layer_div_p113_s123_20260307_091936'),
    ],
}


def repo_root() -> Path:
    return _gcs_data_root()


def resolve_path(base: str | Path) -> Path:
    path = Path(base)
    if path.is_absolute():
        return path
    return repo_root() / path


def smooth_series(values: Iterable[float], window: int = 3) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    if np.isnan(arr).any():
        arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
    if arr.size < window:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[: arr.size]


def normalize_01(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def load_pickle(path: str | Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_metrics(base: str | Path):
    return load_pickle(resolve_path(base) / 'training_metrics.pkl')


def load_gcs(base: str | Path):
    base_path = resolve_path(base)
    # Prefer corrected union files, fall back to originals
    for name in ('gcs_results_union.pkl', 'gcs_results.pkl'):
        p = base_path / name
        if p.exists():
            payload = load_pickle(p)
            return np.asarray(payload['epochs'], dtype=float), payload['path_gcs_scores']
    for name in ('training_metrics_final_geo_union.pkl', 'training_metrics_final_geo.pkl'):
        p = base_path / name
        if p.exists():
            payload = load_pickle(p)
            return np.asarray(payload['gcs_steps_final_geo'], dtype=float), payload['path_gcs_scores_final_geo']
    raise FileNotFoundError(f'No GCS data found in {base_path}')


def path_key(depth: int, path_type: str) -> str:
    if depth == 1:
        return f'{path_type}_path'
    return f'layer{depth - 1}_{path_type}_path'


def load_path_series(base: str | Path, depth: int, path_type: str) -> tuple[np.ndarray, np.ndarray]:
    epochs, scores = load_gcs(base)
    key = path_key(depth, path_type)
    return epochs, np.asarray(scores[key], dtype=float)


def load_all_series(base: str | Path, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epochs, scores = load_gcs(base)
    return (
        epochs,
        np.asarray(scores[path_key(depth, 'overall')], dtype=float),
        np.asarray(scores[path_key(depth, 'attention')], dtype=float),
        np.asarray(scores[path_key(depth, 'mlp')], dtype=float),
    )


def detect_grok_epoch(metrics: dict, threshold: float = 0.99, smooth_window: int = 3) -> Optional[int]:
    steps = np.asarray(metrics['log_steps'], dtype=int)
    accs = np.asarray(metrics['test_accuracies'], dtype=float)
    if steps.size < 4 or accs.max() < threshold:
        return None

    first_idx = next(i for i, value in enumerate(accs) if value >= threshold)
    best_epoch = None
    best_slope = float('-inf')
    for i in range(3, first_idx + 1):
        dt = steps[i] - steps[i - 1]
        if dt <= 0:
            continue
        slope = (accs[i] - accs[i - 1]) / dt
        if slope > best_slope:
            best_slope = slope
            best_epoch = int(steps[i])
    return best_epoch


def norm_peak_epoch(metrics: dict) -> tuple[int, float]:
    steps = np.asarray(metrics['log_steps'], dtype=int)
    norms = np.asarray(metrics['norms'], dtype=float)
    idx = int(np.argmax(norms))
    return int(steps[idx]), float(norms[idx])


def stable_turning_epoch(
    series: Iterable[float],
    epochs: Iterable[float],
    grok_epoch: int,
    *,
    smooth_window: int = 3,
    min_positive_slopes: int = 3,
) -> int:
    epochs_arr = np.asarray(list(epochs), dtype=float)
    values = np.asarray(list(series), dtype=float)
    pre_n = int(np.sum(epochs_arr <= grok_epoch))
    if pre_n < 2:
        return int(epochs_arr[0])

    pre_values = values[:pre_n]
    pre_epochs = epochs_arr[:pre_n]
    slopes = np.diff(pre_values)
    for idx in range(len(slopes) - min_positive_slopes + 1):
        if np.all(slopes[idx: idx + min_positive_slopes] > 0):
            return int(pre_epochs[idx])
    return int(pre_epochs[int(np.argmin(pre_values))])


def detect_mlp_u_shape(
    series: Iterable[float],
    epochs: Iterable[float],
    grok_epoch: int,
    *,
    rel_thresh: float = 0.10,
    smooth_window: int = 3,
) -> tuple[bool, float]:
    epochs_arr = np.asarray(list(epochs), dtype=float)
    values = np.asarray(list(series), dtype=float)
    pre_n = int(np.sum(epochs_arr <= grok_epoch))
    if pre_n < 3:
        return False, 0.0

    full_range = float(values.max() - values.min())
    if full_range <= 0:
        return False, 0.0
    threshold = rel_thresh * full_range

    pre_values = values[:pre_n]
    initial = float(pre_values[0])
    trough = float(pre_values.min())
    depth = initial - trough
    if depth < threshold:
        return False, depth
    trough_idx = int(np.argmin(pre_values))
    recovery = float(pre_values[trough_idx:].max() - trough)
    return recovery > threshold, depth


def detect_attention_post_grok_dip(
    series: Iterable[float],
    epochs: Iterable[float],
    grok_epoch: int,
    *,
    rel_thresh: float = 0.10,
    smooth_window: int = 3,
) -> tuple[bool, float]:
    epochs_arr = np.asarray(list(epochs), dtype=float)
    values = np.asarray(list(series), dtype=float)
    pre_n = int(np.sum(epochs_arr <= grok_epoch))
    if pre_n >= values.size:
        return False, 0.0

    full_range = float(values.max() - values.min())
    if full_range <= 0:
        return False, 0.0
    threshold = rel_thresh * full_range
    at_grok = float(values[pre_n - 1] if pre_n > 0 else values[0])
    post_min = float(values[pre_n:].min())
    dip = at_grok - post_min
    return dip > threshold, dip


def detect_attention_peak_then_dip(
    series: Iterable[float],
    epochs: Iterable[float],
    grok_epoch: int,
    *,
    rel_thresh: float = 0.05,
    peak_window_lo: float = 0.9,
    peak_window_hi: float = 1.1,
) -> tuple[bool, float, float, Optional[int]]:
    epochs_arr = np.asarray(list(epochs), dtype=float)
    values = np.asarray(list(series), dtype=float)
    if values.size == 0:
        return False, 0.0, 0.0, None

    full_range = float(values.max() - values.min())
    if full_range <= 0:
        return False, 0.0, 0.0, None
    threshold = rel_thresh * full_range

    x = epochs_arr / max(grok_epoch, 1)
    peak_candidates = np.where((x >= peak_window_lo) & (x <= peak_window_hi))[0]
    if peak_candidates.size == 0:
        return False, 0.0, 0.0, None

    peak_idx = int(peak_candidates[np.argmax(values[peak_candidates])])
    trough_idx = int(np.argmin(values[: peak_idx + 1]))
    rise = float(values[peak_idx] - values[trough_idx])
    dip = float(values[peak_idx] - values[peak_idx:].min())
    return bool(rise >= threshold and dip >= threshold), rise, dip, int(epochs_arr[peak_idx])


def iter_depth_runs(depth: int, tasks: Optional[Iterable[str]] = None):
    if depth == 1:
        runs = RUNS_1L
        if tasks is None:
            return list(runs)
        task_set = set(tasks)
        return [run for run in runs if run[1] in task_set]

    registry = RUNS_2L if depth == 2 else RUNS_3L
    selected_tasks = list(tasks) if tasks is not None else list(registry.keys())
    runs = []
    for task in selected_tasks:
        runs.extend(registry.get(task, []))
    return runs
