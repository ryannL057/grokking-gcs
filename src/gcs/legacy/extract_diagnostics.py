#!/usr/bin/env python3
"""
Extract quantitative diagnostics from GCS trajectories using a manifest CSV.

Outputs:
- Per-run diagnostics CSV
- Summary CSV (per mechanism + pooled)
- Optional LaTeX snippets for table insertion
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    from scipy.stats import binomtest as _scipy_binomtest
except Exception:
    _scipy_binomtest = None

PATH_ALIASES = {
    "overall": "overall",
    "overall_path": "overall",
    "attention": "attention",
    "attention_path": "attention",
    "mlp": "mlp",
    "mlp_path": "mlp",
}


def binomtest_greater_pvalue(k: int, n: int, p: float = 0.5) -> float:
    """Exact one-sided binomial p-value P[X>=k] for X~Bin(n,p)."""
    if n <= 0:
        return float("nan")
    if _scipy_binomtest is not None:
        return float(_scipy_binomtest(k, n, p, alternative="greater").pvalue)

    # SciPy-free fallback (exact tail sum).
    from math import comb

    tail = 0.0
    for i in range(k, n + 1):
        tail += comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return float(tail)


def load_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict pickle at {path}, got {type(obj)}")
    return obj


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "na", "nan"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def to_int(x: Any) -> Optional[int]:
    v = to_float(x)
    if v is None:
        return None
    return int(v)


def normalize_token(x: Any) -> str:
    s = str(x if x is not None else "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def normalize_layer_token(x: Any) -> str:
    v = to_float(x)
    if v is not None and float(v).is_integer():
        return str(int(v))
    return str(x if x is not None else "").strip().lower()


def _extract_training(metrics: Dict[str, Any]) -> Tuple[List[int], List[float], List[float]]:
    steps = metrics.get("log_steps", [])
    test_acc = metrics.get("test_accuracies", [])
    norms = metrics.get("norms", [])
    return list(steps), list(test_acc), list(norms)


def _extract_gcs_from_embedded(metrics: Dict[str, Any]) -> Tuple[List[int], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    if "gcs_steps_final_geo" in metrics and "path_gcs_scores_final_geo" in metrics:
        epochs = list(metrics.get("gcs_steps_final_geo", []))
        ps = metrics.get("path_gcs_scores_final_geo", {})
        return epochs, list(ps.get("overall_path", [])), list(ps.get("attention_path", [])), list(ps.get("mlp_path", []))

    if "gcs_steps" in metrics and "path_gcs_scores" in metrics:
        epochs = list(metrics.get("gcs_steps", []))
        ps = metrics.get("path_gcs_scores", {})
        return epochs, list(ps.get("overall_path", [])), list(ps.get("attention_path", [])), list(ps.get("mlp_path", []))

    if "epochs" in metrics and "path_gcs_scores" in metrics:
        epochs = list(metrics.get("epochs", []))
        ps = metrics.get("path_gcs_scores", {})
        return epochs, list(ps.get("overall_path", [])), list(ps.get("attention_path", [])), list(ps.get("mlp_path", []))

    if "epochs" in metrics and {"overall_path", "attention_path", "mlp_path"}.issubset(metrics.keys()):
        epochs = list(metrics.get("epochs", []))
        return epochs, list(metrics.get("overall_path", [])), list(metrics.get("attention_path", [])), list(metrics.get("mlp_path", []))

    return [], [], [], []


def load_trajectory(row: Dict[str, Any]) -> Dict[str, Any]:
    metrics_path = Path(row["metrics_path"])
    metrics = load_pickle(metrics_path)
    log_steps, test_acc, norms = _extract_training(metrics)

    gcs_source = (row.get("gcs_source") or "").strip()
    gcs_path_str = (row.get("gcs_path") or "").strip()

    gcs_epochs: List[int] = []
    overall: List[Optional[float]] = []
    attention: List[Optional[float]] = []
    mlp: List[Optional[float]] = []

    if gcs_source == "separate_gcs_results" and gcs_path_str:
        g = load_pickle(Path(gcs_path_str))
        gcs_epochs, overall, attention, mlp = _extract_gcs_from_embedded(g)
    else:
        gcs_epochs, overall, attention, mlp = _extract_gcs_from_embedded(metrics)
        if not gcs_epochs and gcs_path_str:
            g = load_pickle(Path(gcs_path_str))
            gcs_epochs, overall, attention, mlp = _extract_gcs_from_embedded(g)

    return {
        "log_steps": log_steps,
        "test_accuracies": test_acc,
        "norms": norms,
        "gcs_epochs": gcs_epochs,
        "overall": overall,
        "attention": attention,
        "mlp": mlp,
    }


def smooth(series: List[Optional[float]], window: int = 3) -> np.ndarray:
    arr = np.array([np.nan if v is None else float(v) for v in series], dtype=float)
    if len(arr) == 0:
        return arr
    if len(arr) < window:
        return np.nan_to_num(arr, nan=np.nanmedian(arr))

    if np.isnan(arr).any():
        med = np.nanmedian(arr)
        arr = np.where(np.isnan(arr), med, arr)

    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def detect_grokking_epoch(log_steps: List[int], test_acc: List[float]) -> Optional[int]:
    if len(log_steps) < 4 or len(test_acc) < 4:
        return None
    best_epoch, best_slope = None, float("-inf")
    for i in range(3, min(len(log_steps), len(test_acc))):
        dt = log_steps[i] - log_steps[i - 1]
        if dt <= 0:
            continue
        slope = (test_acc[i] - test_acc[i - 1]) / dt
        if slope > best_slope:
            best_slope = slope
            best_epoch = log_steps[i]
    return best_epoch


def first_crossing_epoch(log_steps: List[int], test_acc: List[float], threshold: float = 0.95) -> Optional[int]:
    for e, a in zip(log_steps, test_acc):
        if a >= threshold:
            return e
    return None


def find_turning_point(epochs: List[int], values: List[Optional[float]], K: int = 3) -> Tuple[Optional[int], Optional[str]]:
    if not epochs or not values:
        return None, "empty"

    sm = smooth(values, window=3)
    if len(sm) < K + 1:
        return None, "short"

    slopes = np.diff(sm)

    if np.all(slopes >= -1e-12):
        return None, "monotonic rise"
    if np.all(slopes <= 1e-12):
        return None, "monotonic decline"

    for i in range(len(slopes)):
        if slopes[i] > 0 and (i + K) <= len(slopes):
            if np.all(slopes[i : i + K] > 0):
                idx = min(i + 1, len(epochs) - 1)
                return int(epochs[idx]), None

    return None, "no stable neg-to-pos"


def check_diversification(values: List[Optional[float]], delta_pct: float) -> bool:
    if not values:
        return False
    sm = smooth(values, window=3)
    if len(sm) == 0:
        return False
    initial = sm[0]
    if abs(initial) < 1e-12:
        return False
    trough = np.min(sm)
    drop_pct = (initial - trough) / abs(initial) * 100
    return bool(drop_pct >= delta_pct)


def check_unification(epochs: List[int], values: List[Optional[float]], delta_pct: float, K: int = 3) -> bool:
    if not epochs or not values:
        return False
    sm = smooth(values, window=3)
    t_turn, _ = find_turning_point(epochs, values, K=K)
    if t_turn is None:
        return False

    try:
        turn_idx = epochs.index(t_turn)
    except ValueError:
        return False

    trough = np.min(sm[: turn_idx + 1])
    if abs(trough) < 1e-12:
        return False
    post_max = np.max(sm[turn_idx:])
    rise_pct = (post_max - trough) / abs(trough) * 100

    slopes_post = np.diff(sm[turn_idx:])
    sustained = False
    for i in range(max(0, len(slopes_post) - K + 1)):
        if np.all(slopes_post[i : i + K] >= -1e-12):
            sustained = True
            break

    return bool(rise_pct >= delta_pct and sustained)


def compute_per_run(rows: List[Dict[str, Any]], grok_threshold: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        traj = load_trajectory(row)

        t_grok_slope = detect_grokking_epoch(traj["log_steps"], traj["test_accuracies"])
        t_grok_thresh = first_crossing_epoch(traj["log_steps"], traj["test_accuracies"], threshold=grok_threshold)

        t_turn, turn_reason = find_turning_point(traj["gcs_epochs"], traj["overall"])
        t_attn, attn_reason = find_turning_point(traj["gcs_epochs"], traj["attention"])
        t_mlp, mlp_reason = find_turning_point(traj["gcs_epochs"], traj["mlp"])

        lead = (t_grok_slope - t_turn) if (t_grok_slope is not None and t_turn is not None) else None
        async_val = (t_attn - t_mlp) if (t_attn is not None and t_mlp is not None) else None

        if async_val is None:
            async_dir = "NA"
        elif async_val < 0:
            async_dir = "attn leads"
        elif async_val > 0:
            async_dir = "mlp leads"
        else:
            async_dir = "simultaneous"

        max_epoch = to_int(row.get("max_recorded_epoch"))
        if max_epoch is None and traj["log_steps"]:
            max_epoch = int(traj["log_steps"][-1])

        grok_observed = t_grok_thresh is not None
        grok_censored = (not grok_observed) and (max_epoch is not None)

        run_uid = (row.get("run_uid") or "").strip()
        if not run_uid:
            exp_dir = (row.get("exp_dir") or "").strip()
            run_uid = f"exp__{normalize_token(exp_dir)}" if exp_dir else f"run__{normalize_token(row.get('run_id'))}"

        out.append({
            **row,
            "run_uid": run_uid,
            "t_grok_slope": t_grok_slope,
            "t_grok_thresh": t_grok_thresh,
            "t_turn": t_turn,
            "turn_reason": turn_reason,
            "lead": lead,
            "t_attn": t_attn,
            "attn_reason": attn_reason,
            "t_mlp": t_mlp,
            "mlp_reason": mlp_reason,
            "delta_async": async_val,
            "async_dir": async_dir,
            "grok_observed": grok_observed,
            "grok_censored": grok_censored,
            "max_epoch": max_epoch,
            "_traj": traj,
        })

    return out


def summarize_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_evaluable = 0
    for r in group:
        traj = r.get("_traj", {})
        if isinstance(traj, dict) and traj.get("gcs_epochs") and traj.get("overall"):
            n_evaluable += 1

    leads = [r["lead"] for r in group if r.get("lead") is not None]
    pos = sum(1 for x in leads if x > 0)
    n = len(leads)

    median_lead = float(np.median(leads)) if leads else None
    iqr = None
    if leads:
        q1, q3 = np.percentile(np.array(leads), [25, 75])
        iqr = (float(q1), float(q3))

    sign_p = None
    if n > 0:
        sign_p = binomtest_greater_pvalue(pos, n, 0.5)

    async_vals = [r["delta_async"] for r in group if r.get("delta_async") is not None]
    async_attn_lead = sum(1 for v in async_vals if v < 0)

    grok_obs = sum(1 for r in group if r.get("grok_observed"))
    grok_cen = sum(1 for r in group if r.get("grok_censored"))

    return {
        "n_total": len(group),
        "n_runs": len(group),
        "n_evaluable": n_evaluable,
        "n_unevaluable": len(group) - n_evaluable,
        "n_with_lead": n,
        "n_positive_lead": pos,
        "median_lead": median_lead,
        "iqr_q1": iqr[0] if iqr else None,
        "iqr_q3": iqr[1] if iqr else None,
        "sign_test_p_exploratory": sign_p,
        "n_async_defined": len(async_vals),
        "n_async_attn_leads": async_attn_lead,
        "n_grok_observed": grok_obs,
        "n_grok_censored": grok_cen,
    }


def run_sensitivity(rows: List[Dict[str, Any]], deltas: List[float]) -> List[Dict[str, Any]]:
    out = []
    for delta in deltas:
        div_hits = 0
        uni_hits = 0
        async_hits = 0
        async_total = 0

        for r in rows:
            traj = r["_traj"]
            if check_diversification(traj["overall"], delta):
                div_hits += 1
            if check_unification(traj["gcs_epochs"], traj["overall"], delta):
                uni_hits += 1
            if r.get("delta_async") is not None:
                async_total += 1
                if abs(r["delta_async"]) > 200:
                    async_hits += 1

        out.append({
            "delta_pct": delta,
            "diversification_hits": div_hits,
            "unification_hits": uni_hits,
            "asynchrony_hits": async_hits,
            "n_runs": len(rows),
            "n_async_defined": async_total,
        })
    return out


def normalize_path_name(name: str) -> str:
    key = name.strip().lower()
    if key not in PATH_ALIASES:
        raise ValueError(f"Unknown path name '{name}'. Expected one of: {sorted(PATH_ALIASES.keys())}")
    return PATH_ALIASES[key]


def path_values(traj: Dict[str, Any], path: str) -> List[Optional[float]]:
    if path == "overall":
        return traj.get("overall", [])
    if path == "attention":
        return traj.get("attention", [])
    if path == "mlp":
        return traj.get("mlp", [])
    raise ValueError(path)


def summarize_path_hits(
    rows: List[Dict[str, Any]],
    path: str,
    delta_pct: float,
    *,
    K: int = 3,
) -> Dict[str, Any]:
    n_path_evaluable = 0
    div_hits = 0
    uni_hits = 0
    for r in rows:
        traj = r.get("_traj", {})
        if not isinstance(traj, dict):
            continue
        values = path_values(traj, path)
        epochs = traj.get("gcs_epochs", [])
        if not values or not epochs:
            continue
        n_path_evaluable += 1
        if check_diversification(values, delta_pct):
            div_hits += 1
        if check_unification(epochs, values, delta_pct, K=K):
            uni_hits += 1

    div_rate = (div_hits / n_path_evaluable) if n_path_evaluable else None
    uni_rate = (uni_hits / n_path_evaluable) if n_path_evaluable else None
    return {
        "n_path_evaluable": n_path_evaluable,
        "diversification_hits": div_hits,
        "unification_hits": uni_hits,
        "diversification_rate": div_rate,
        "unification_rate": uni_rate,
    }


def run_path_sensitivity(
    rows: List[Dict[str, Any]],
    deltas: List[float],
    paths: List[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in paths:
        for delta in deltas:
            s = summarize_path_hits(rows, path=path, delta_pct=delta, K=3)
            out.append(
                {
                    "path": path,
                    "delta_pct": delta,
                    "diversification_hits": s["diversification_hits"],
                    "unification_hits": s["unification_hits"],
                    "n_path_evaluable": s["n_path_evaluable"],
                    "diversification_rate": s["diversification_rate"],
                    "unification_rate": s["unification_rate"],
                }
            )
    return out


def group_key(row: Dict[str, Any], fields: List[str]) -> Tuple[str, ...]:
    vals = []
    for f in fields:
        vals.append(str(row.get(f, "NA")))
    return tuple(vals)


def build_group_matrix(
    rows: List[Dict[str, Any]],
    group_fields: List[str],
    paths: List[str],
    *,
    delta_pct: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        k = group_key(r, group_fields)
        grouped.setdefault(k, []).append(r)

    matrix_rows: List[Dict[str, Any]] = []
    for k in sorted(grouped.keys()):
        grp = grouped[k]
        base = summarize_group(grp)

        row: Dict[str, Any] = {
            "group": " | ".join(f"{f}={v}" for f, v in zip(group_fields, k)),
            "group_fields": ",".join(group_fields),
            "delta_pct": delta_pct,
            "n_unique_seeds": len({str(x.get("seed", "NA")) for x in grp}),
            **base,
        }
        for field, value in zip(group_fields, k):
            row[field] = value

        for path in paths:
            s = summarize_path_hits(grp, path=path, delta_pct=delta_pct, K=3)
            row[f"{path}_n"] = s["n_path_evaluable"]
            row[f"{path}_div_hits"] = s["diversification_hits"]
            row[f"{path}_uni_hits"] = s["unification_hits"]
            row[f"{path}_div_rate"] = s["diversification_rate"]
            row[f"{path}_uni_rate"] = s["unification_rate"]

        # Quick descriptive diagnostics for the selected path delta threshold.
        leads = [x.get("lead") for x in grp if x.get("lead") is not None]
        row["lead_mean"] = float(statistics.mean(leads)) if leads else None
        matrix_rows.append(row)

    return matrix_rows


def drop_internal(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned = []
    for r in rows:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        cleaned.append(rr)
    return cleaned


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def print_latex_main_table(rows: List[Dict[str, Any]]) -> None:
    print("\n% --- Diagnostics table (manifest-driven) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{llrrrrrrl}")
    print(r"\toprule")
    print(r"Config & Seed & $t_\text{grok}$ & $t_\text{turn}$ & Lead & $t_\text{attn}$ & $t_\text{mlp}$ & $\Delta_{\text{async}}$ & Async dir. \\")
    print(r"\midrule")

    for r in rows:
        cfg = r.get("mechanism", "NA")
        seed = r.get("seed", "NA")

        def f(v: Any) -> str:
            if v is None:
                return "---"
            if isinstance(v, bool):
                return "1" if v else "0"
            if isinstance(v, float) and math.isnan(v):
                return "---"
            return str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v)

        print(
            f"{cfg} & {seed} & {f(r.get('t_grok_slope'))} & {f(r.get('t_turn'))} & "
            f"{f(r.get('lead'))} & {f(r.get('t_attn'))} & {f(r.get('t_mlp'))} & "
            f"{f(r.get('delta_async'))} & {r.get('async_dir', 'NA')} \\\\" 
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract diagnostics from manifest-defined experiments")
    parser.add_argument("--manifest", type=str, default="scripts/experiment_manifest.csv")
    parser.add_argument("--output-csv", type=str, default="scripts/diagnostics_results.csv")
    parser.add_argument("--summary-csv", type=str, default="scripts/diagnostics_summary.csv")
    parser.add_argument("--sensitivity-csv", type=str, default="scripts/diagnostics_sensitivity.csv")
    parser.add_argument(
        "--path-sensitivity-csv",
        type=str,
        default=None,
        help="Optional CSV with path-specific sensitivity rows",
    )
    parser.add_argument(
        "--matrix-csv",
        type=str,
        default=None,
        help="Optional grouped matrix CSV (task×activation×mechanism etc.)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="mechanism",
        help="Comma-separated group fields for matrix output, e.g. task,activation,mechanism",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default="overall,attention,mlp",
        help="Comma-separated paths to summarize (overall,attention,mlp)",
    )
    parser.add_argument(
        "--matrix-delta",
        type=float,
        default=5.0,
        help="Delta threshold used by grouped matrix path hit statistics",
    )
    parser.add_argument("--grok-threshold", type=float, default=0.95)
    parser.add_argument("--deltas", type=str, default="3,5,8")
    parser.add_argument("--task", type=str, default="add")
    parser.add_argument(
        "--tasks-filter",
        type=str,
        default=None,
        help="Optional comma-separated task whitelist, e.g. add,sub,mul,div",
    )
    parser.add_argument("--mechanisms", type=str, default="Clock,Pizza")
    parser.add_argument(
        "--activation-filter",
        type=str,
        default=None,
        help="Optional comma-separated activation whitelist, e.g. SiLU",
    )
    parser.add_argument(
        "--num-layers-filter",
        type=str,
        default=None,
        help="Optional comma-separated num_layers whitelist, e.g. 1 or 1,2",
    )
    parser.add_argument("--min-p", type=int, default=None)
    parser.add_argument("--max-p", type=int, default=None)
    parser.add_argument("--print-latex", action="store_true")
    parser.add_argument("--summary-json", type=str, default="scripts/diagnostics_summary.json")
    args = parser.parse_args()

    manifest_rows = load_manifest(Path(args.manifest))
    allowed_mechs = {m.strip() for m in args.mechanisms.split(",") if m.strip()}
    allowed_tasks = None
    if args.tasks_filter:
        allowed_tasks = {x.strip() for x in args.tasks_filter.split(",") if x.strip()}
    allowed_acts = None
    if args.activation_filter:
        allowed_acts = {normalize_token(x) for x in args.activation_filter.split(",") if x.strip()}

    allowed_layers = None
    if args.num_layers_filter:
        allowed_layers = {normalize_layer_token(x) for x in args.num_layers_filter.split(",") if x.strip()}

    filtered = []
    for r in manifest_rows:
        if allowed_tasks is not None and r.get("task") not in allowed_tasks:
            continue
        if args.task and r.get("task") != args.task:
            continue
        if allowed_mechs and r.get("mechanism") not in allowed_mechs:
            continue
        if allowed_acts is not None and normalize_token(r.get("activation")) not in allowed_acts:
            continue
        if allowed_layers is not None and normalize_layer_token(r.get("num_layers")) not in allowed_layers:
            continue
        p = to_int(r.get("p"))
        if args.min_p is not None and p is not None and p < args.min_p:
            continue
        if args.max_p is not None and p is not None and p > args.max_p:
            continue
        if (r.get("status") or "").strip() == "missing_gcs":
            continue
        filtered.append(r)

    if not filtered:
        raise SystemExit("No manifest rows after filtering")

    per_run = compute_per_run(filtered, grok_threshold=args.grok_threshold)

    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    sens_rows = run_sensitivity(per_run, deltas)
    paths = [normalize_path_name(x) for x in args.paths.split(",") if x.strip()]
    path_sens_rows = run_path_sensitivity(per_run, deltas, paths)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_run:
        mech = r.get("mechanism", "Unknown")
        grouped.setdefault(mech, []).append(r)

    summary_rows = []
    for mech, grp in sorted(grouped.items()):
        s = summarize_group(grp)
        summary_rows.append({"group": mech, **s})

    pooled = summarize_group(per_run)
    summary_rows.append({"group": "Pooled", **pooled})

    write_csv(Path(args.output_csv), drop_internal(per_run))
    write_csv(Path(args.summary_csv), summary_rows)
    write_csv(Path(args.sensitivity_csv), sens_rows)
    if args.path_sensitivity_csv:
        write_csv(Path(args.path_sensitivity_csv), path_sens_rows)

    matrix_rows = None
    if args.matrix_csv:
        group_fields = [x.strip() for x in args.group_by.split(",") if x.strip()]
        if not group_fields:
            group_fields = ["mechanism"]
        matrix_rows = build_group_matrix(
            per_run,
            group_fields=group_fields,
            paths=paths,
            delta_pct=args.matrix_delta,
        )
        write_csv(Path(args.matrix_csv), matrix_rows)

    summary_json = {
        "n_manifest_rows": len(manifest_rows),
        "n_filtered_rows": len(filtered),
        "group_summary": summary_rows,
        "sensitivity": sens_rows,
        "path_sensitivity": path_sens_rows,
        "paths": paths,
        "matrix_group_by": [x.strip() for x in args.group_by.split(",") if x.strip()],
        "matrix_delta_pct": args.matrix_delta,
        "matrix_rows": len(matrix_rows) if matrix_rows is not None else None,
        "notes": [
            "Lead sign test is exploratory/descriptive, not confirmatory.",
            "Variable training horizons are retained; grok_observed/grok_censored reported per run.",
        ],
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Loaded manifest rows: {len(manifest_rows)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Per-run CSV: {args.output_csv}")
    print(f"Summary CSV: {args.summary_csv}")
    print(f"Sensitivity CSV: {args.sensitivity_csv}")
    if args.path_sensitivity_csv:
        print(f"Path sensitivity CSV: {args.path_sensitivity_csv}")
    if args.matrix_csv:
        print(f"Grouped matrix CSV: {args.matrix_csv}")
    print(f"Summary JSON: {args.summary_json}")

    if args.print_latex:
        print_latex_main_table(drop_internal(per_run))


if __name__ == "__main__":
    main()
