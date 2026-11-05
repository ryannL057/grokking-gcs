"""Original schedule/null rules, extracted without filesystem defaults."""
import numpy as np
AT,MT=.05,.10
IAAFT_ITERS=100

def attention_pass(values, epochs, anchor, threshold=AT):
    rel = epochs / max(anchor, 1)
    cand = np.flatnonzero((rel >= .9) & (rel <= 1.1))
    if not len(cand):
        return None
    span = np.ptp(values)
    if span <= 0:
        return False
    peak = int(cand[np.argmax(values[cand])])
    rise = values[peak] - values[:peak + 1].min()
    dip = values[peak] - values[peak:].min()
    return bool(rise >= threshold * span and dip >= threshold * span)

def mlp_pass(values, epochs, anchor, threshold=MT):
    pre = np.flatnonzero(epochs <= anchor)
    if len(pre) < 3:
        return None
    span = np.ptp(values)
    if span <= 0:
        return False
    v = values[pre]
    trough = int(np.argmin(v))
    fall = v[0] - v[trough]
    recovery = v[trough:].max() - v[trough]
    return bool(fall >= threshold * span and recovery > threshold * span)

def iaaft(x, rng, iters=IAAFT_ITERS):
    x = np.asarray(x, float)
    amplitudes = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    y = rng.permutation(x)
    for _ in range(iters):
        phases = np.angle(np.fft.rfft(y))
        y = np.fft.irfft(amplitudes * np.exp(1j * phases), n=len(x))
        y = sorted_x[np.argsort(np.argsort(y))]
    return y

def feasible_anchors(epochs):
    out = []
    for t in epochs:
        rel = epochs / max(t, 1)
        if np.sum(epochs <= t) >= 3 and np.any((rel >= .9) & (rel <= 1.1)):
            out.append(t)
    return np.asarray(out)

def summarize(null_counts, observed):
    arr = np.asarray(null_counts)
    return dict(observed=int(observed), null_mean=float(arr.mean()),
                null_p05=float(np.percentile(arr, 5)), null_p95=float(np.percentile(arr, 95)),
                null_max=int(arr.max()),
                p_value=float((1 + np.sum(arr >= observed)) / (1 + len(arr))),
                observed_exceeds_p95=bool(observed > np.percentile(arr, 95)))

from typing import Optional
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
