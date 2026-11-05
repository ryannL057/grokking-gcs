#!/usr/bin/env python3
"""Method-neutral component-order audit, v2.

See
submission5240_discussion_addendum_20260724/07_METHOD_NEUTRAL_ORDER/
PROTOCOL_V2_20260726.md. Read-only over cached model-analysis artifacts.

Corrects the two estimator errors identified in v1: a three-segment
direction-agnostic fit replaces the single-breakpoint fit, and the order
outcome compares the attention turnover with the MLP recovery *completion*
rather than the MLP trough. No t_grok is used at detection time.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import json
import pickle
import sys
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

from gcs.legacy.diagnostics_common import (  # noqa: E402
    detect_grok_epoch,
    load_gcs,
    load_metrics,
)

MANIFEST_DIR = (
    ROOT / "experiments" / "rebuttal_direct_jacobian_graph_20260723" / "trajectories"
)
AUX_NAME = "workflow_selector_aux_baselines_wsel_aux_pathalign20260502.pkl"
FAMILIES = ("gcs", "direct_j", "cka", "oriented_l2")
COMPONENTS = ("attention", "mlp")

MIN_SEG = 3
N_SHIFTS = 1000
N_BOOT = 10000
SEED = 20260726
PC_MIN_COUNT = 12          # positive control: GCS S3' >= 12/23
PC_NULL_MULTIPLE = 2.0     # positive control: >= 2x own median null rate

_GRID: dict[int, tuple] = {}


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def rank_unit(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values)
    lo, hi = float(ranks.min()), float(ranks.max())
    return np.zeros_like(ranks) if hi <= lo else (ranks - lo) / (hi - lo)


def grid(n: int):
    """Valid (b1, b2) breakpoint pairs with >= MIN_SEG points in each segment."""
    if n in _GRID:
        return _GRID[n]
    b1s, b2s = [], []
    for b1 in range(MIN_SEG, n - 2 * MIN_SEG + 2):
        for b2 in range(b1 + MIN_SEG - 1, n - MIN_SEG + 2):
            b1s.append(b1)
            b2s.append(b2)
    out = (np.asarray(b1s, dtype=int), np.asarray(b2s, dtype=int))
    _GRID[n] = out
    return out


def _prefix(x: np.ndarray, y: np.ndarray):
    z = lambda a: np.concatenate(([0.0], np.cumsum(a)))  # noqa: E731
    return z(np.ones_like(x)), z(x), z(y), z(x * x), z(x * y), z(y * y)


def _seg_sse_slope(pref, i: np.ndarray, j: np.ndarray):
    """Exact OLS SSE and slope on index ranges [i, j), vectorised."""
    Cn, Cx, Cy, Cxx, Cxy, Cyy = pref
    n = Cn[j] - Cn[i]
    Sx = Cx[j] - Cx[i]
    Sy = Cy[j] - Cy[i]
    Sxx = Cxx[j] - Cxx[i]
    Sxy = Cxy[j] - Cxy[i]
    Syy = Cyy[j] - Cyy[i]
    den = n * Sxx - Sx * Sx
    num = n * Sxy - Sx * Sy
    syy_c = Syy - Sy * Sy / n
    safe = den > 1e-12
    slope = np.where(safe, num / np.where(safe, den, 1.0), 0.0)
    sse = np.where(safe, syy_c - num * num / np.where(safe, n * den, 1.0), syy_c)
    return np.maximum(sse, 0.0), slope


def fit_three_segment(y: np.ndarray) -> dict[str, Any]:
    n = y.size
    x = np.arange(n, dtype=float)
    pref = _prefix(x, y)
    b1, b2 = grid(n)
    zero = np.zeros_like(b1)
    sse1, s1 = _seg_sse_slope(pref, zero, b1)
    sse2, s2 = _seg_sse_slope(pref, b1 - 1, b2)
    sse3, s3 = _seg_sse_slope(pref, b2 - 1, zero + n)
    total = sse1 + sse2 + sse3
    k = int(np.argmin(total))
    return {
        "i1": int(b1[k] - 1), "i2": int(b2[k] - 1),
        "s1": float(s1[k]), "s2": float(s2[k]), "s3": float(s3[k]),
        "b1": int(b1[k]),
    }


def events(y: np.ndarray, epochs: np.ndarray) -> dict[str, Any]:
    """Protocol event definitions, identical for every family and component."""
    f = fit_three_segment(y)
    s1, s2, s3 = f["s1"], f["s2"], f["s3"]
    i1, i2 = f["i1"], f["i2"]

    # attention turnover: last positive-to-negative slope change
    if s2 > 0 > s3:
        turn_idx = i2
    elif s1 > 0 > s2:
        turn_idx = i1
    else:
        turn_idx = None

    # MLP recovery onset: first negative-to-positive slope change
    if s1 < 0 < s2:
        onset_idx, fall_start = i1, 0
    elif s2 < 0 < s3:
        onset_idx, fall_start = i2, f["b1"] - 1
    else:
        onset_idx, fall_start = None, None

    comp_idx = None
    if onset_idx is not None:
        level = y[fall_start]
        after = np.arange(onset_idx, y.size)
        hit = after[y[onset_idx:] >= level]
        comp_idx = int(hit[0]) if hit.size else None

    return {
        "turnover_epoch": int(epochs[turn_idx]) if turn_idx is not None else None,
        "onset_epoch": int(epochs[onset_idx]) if onset_idx is not None else None,
        "completion_epoch": int(epochs[comp_idx]) if comp_idx is not None else None,
        "s1": s1, "s2": s2, "s3": s3,
        "e1": int(epochs[i1]), "e2": int(epochs[i2]),
    }


def outcomes(attn: dict, mlp: dict) -> dict[str, bool]:
    s1p = attn["turnover_epoch"] is not None
    s2p = mlp["onset_epoch"] is not None
    p1p = (
        s1p
        and mlp["completion_epoch"] is not None
        and attn["turnover_epoch"] < mlp["completion_epoch"]
    )
    return {"S1": s1p, "S2": s2p, "P1": p1p, "S3": bool(s1p and s2p and p1p)}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    diff = a.astype(float) - b.astype(float)
    idx = rng.integers(0, diff.size, size=(N_BOOT, diff.size))
    means = diff[idx].mean(axis=1)
    return (
        float(diff.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "rebuttal_method_neutral_order_v2_20260726",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted(MANIFEST_DIR.glob("*.pkl"))
    if len(manifest_paths) != 23:
        raise RuntimeError(f"Expected frozen 23-run manifest, found {len(manifest_paths)}")

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []

    for manifest_path in manifest_paths:
        manifest = load_pickle(manifest_path)
        exp_dir = Path(manifest["source_exp_dir"])
        if not exp_dir.is_absolute():
            exp_dir = ROOT / exp_dir
        aux = load_pickle(exp_dir / AUX_NAME)
        gcs_epochs_array, gcs_scores = load_gcs(exp_dir)
        t_grok = detect_grok_epoch(load_metrics(exp_dir))
        if t_grok is None:
            raise RuntimeError(f"No solved t_grok for {exp_dir}")

        gcs_epochs = [int(e) for e in gcs_epochs_array]
        aux_epochs = [int(e) for e in aux["epochs"]]
        dj_epochs = [int(e) for e in manifest["epochs"]]
        common = sorted(set(gcs_epochs) & set(aux_epochs) & set(dj_epochs))
        if len(common) < 3 * MIN_SEG:
            raise RuntimeError(f"Too few common epochs for {manifest_path.name}")

        gi = {e: i for i, e in enumerate(gcs_epochs)}
        ai = {e: i for i, e in enumerate(aux_epochs)}
        di = {e: i for i, e in enumerate(dj_epochs)}
        aux_series, dj_scores = aux["series"], manifest["path_scores"]

        def pick(src, imap, key, sign=1.0):
            return sign * np.asarray(
                [float(src[key][imap[e]]) for e in common], dtype=float
            )

        series = {
            "gcs": {"attention": pick(gcs_scores, gi, "attention_path"),
                    "mlp": pick(gcs_scores, gi, "mlp_path")},
            "direct_j": {"attention": pick(dj_scores, di, "attention_path"),
                         "mlp": pick(dj_scores, di, "mlp_path")},
            "cka": {"attention": pick(aux_series, ai, "path_cka_attention"),
                    "mlp": pick(aux_series, ai, "path_cka_mlp")},
            "oriented_l2": {"attention": pick(aux_series, ai, "path_l2_attention", -1.0),
                            "mlp": pick(aux_series, ai, "path_l2_mlp", -1.0)},
        }

        epochs_arr = np.asarray(common, dtype=int)
        row: dict[str, Any] = {
            "slug": manifest_path.stem,
            "act": aux.get("act", manifest.get("act")),
            "task": aux.get("task", manifest.get("task")),
            "seed": int(aux.get("seed", manifest.get("seed"))),
            "t_grok": int(t_grok),
            "n_common_epochs": len(common),
        }

        for family in FAMILIES:
            ya = rank_unit(series[family]["attention"])
            ym = rank_unit(series[family]["mlp"])
            ea, em = events(ya, epochs_arr), events(ym, epochs_arr)
            oc = outcomes(ea, em)
            for key, val in oc.items():
                row[f"{family}_{key}"] = val
            row[f"{family}_attn_turnover_epoch"] = ea["turnover_epoch"]
            row[f"{family}_mlp_onset_epoch"] = em["onset_epoch"]
            row[f"{family}_mlp_completion_epoch"] = em["completion_epoch"]
            for label, ev, key in (
                ("attn_turnover", ea, "turnover_epoch"),
                ("mlp_completion", em, "completion_epoch"),
            ):
                row[f"{family}_{label}_rel_tgrok"] = (
                    round(ev[key] / max(int(t_grok), 1), 4)
                    if ev[key] is not None else None
                )

            n_p1 = n_s3 = 0
            for _ in range(N_SHIFTS):
                ka = int(rng.integers(1, ya.size))
                km = int(rng.integers(1, ym.size))
                noc = outcomes(
                    events(np.roll(ya, ka), epochs_arr),
                    events(np.roll(ym, km), epochs_arr),
                )
                n_p1 += int(noc["P1"])
                n_s3 += int(noc["S3"])
            null_rows.append({
                "slug": manifest_path.stem, "family": family,
                "null_P1_rate": round(n_p1 / N_SHIFTS, 4),
                "null_S3_rate": round(n_s3 / N_SHIFTS, 4),
            })

        rows.append(row)

    write_csv(args.output_dir / "per_run.csv", rows)
    write_csv(args.output_dir / "per_run_null.csv", null_rows)

    n = len(rows)
    boot_rng = np.random.default_rng(SEED)
    arr = {
        f: {k: np.array([r[f"{f}_{k}"] for r in rows], dtype=float)
            for k in ("S1", "S2", "P1", "S3")}
        for f in FAMILIES
    }
    summary: dict[str, Any] = {
        "protocol": "07_METHOD_NEUTRAL_ORDER/PROTOCOL_V2_20260726.md",
        "n_runs": n, "min_seg": MIN_SEG, "n_shifts": N_SHIFTS,
        "n_boot": N_BOOT, "seed": SEED, "families": {},
    }
    for family in FAMILIES:
        nulls = [x for x in null_rows if x["family"] == family]
        entry: dict[str, Any] = {
            k: f"{int(arr[family][k].sum())}/{n}" for k in ("S1", "S2", "P1", "S3")
        }
        entry["null_P1_rate_median"] = round(
            float(np.median([x["null_P1_rate"] for x in nulls])), 4)
        entry["null_S3_rate_median"] = round(
            float(np.median([x["null_S3_rate"] for x in nulls])), 4)
        if family != "gcs":
            for k in ("S1", "S3"):
                d, lo, hi = paired_bootstrap(arr["gcs"][k], arr[family][k], boot_rng)
                b = int(np.sum((arr["gcs"][k] == 1) & (arr[family][k] == 0)))
                c = int(np.sum((arr["gcs"][k] == 0) & (arr[family][k] == 1)))
                p = mcnemar_exact(b, c)
                entry[f"{k}_gcs_minus_family"] = round(d, 4)
                entry[f"{k}_boot95"] = [round(lo, 4), round(hi, 4)]
                entry[f"{k}_mcnemar_p"] = round(p, 4)
                entry[f"{k}_separates"] = bool((lo > 0 or hi < 0) and p < 0.05)
        summary["families"][family] = entry

    gcs_s3 = int(arr["gcs"]["S3"].sum())
    gcs_null = summary["families"]["gcs"]["null_S3_rate_median"]
    pc_count = gcs_s3 >= PC_MIN_COUNT
    pc_null = (gcs_s3 / n) >= PC_NULL_MULTIPLE * gcs_null
    summary["positive_control"] = {
        "gcs_S3": f"{gcs_s3}/{n}",
        "required_min_count": PC_MIN_COUNT,
        "gcs_null_S3_rate_median": gcs_null,
        "passes_count_bar": bool(pc_count),
        "passes_null_bar": bool(pc_null),
        "PASSES": bool(pc_count and pc_null),
    }
    summary["uniqueness_licensed"] = bool(
        summary["positive_control"]["PASSES"]
        and all(summary["families"][f].get("S3_separates") for f in FAMILIES if f != "gcs")
    )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
