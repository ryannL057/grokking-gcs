#!/usr/bin/env python3
"""Measure attention-output swap equivariance for runs listed in a manifest.

This is the manifest-based companion to scripts/measure_equivariance.py.  It is
used for rebuttal seed extensions whose runs are not registered in RUNS_1L.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

pass  # Imports use the installed package.

from gcs.legacy.diagnostics_common import detect_grok_epoch, load_metrics, resolve_path  # noqa: E402
from gcs.legacy.measure_equivariance import process_run  # noqa: E402


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({
            "act": row["act"],
            "task": row["task"],
            "seed": int(row["seed"]),
            "path": row["path"],
        })
    return out


def load_pickle_dict(path: Path | None) -> dict[Any, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict pickle at {path}, got {type(payload)!r}")
    return payload


def write_pickle(path: Path, payload: dict[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_filter(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV with act,task,seed,path columns.")
    parser.add_argument("--output-cache", required=True, help="Pickle cache for manifest-only results.")
    parser.add_argument("--base-cache", default="", help="Existing equivariance cache to merge with.")
    parser.add_argument("--merged-output", default="", help="Optional merged base+manifest cache path.")
    parser.add_argument("--status-csv", default="", help="Optional per-run status CSV.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth checkpoint.")
    parser.add_argument("--tasks", default="", help="Optional comma-separated task filter.")
    parser.add_argument("--activations", default="", help="Optional comma-separated activation filter.")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed filter.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    output_cache = resolve_path(args.output_cache)
    base_cache = resolve_path(args.base_cache) if args.base_cache else None
    merged_output = resolve_path(args.merged_output) if args.merged_output else None
    status_csv = resolve_path(args.status_csv) if args.status_csv else output_cache.with_suffix(".status.csv")

    rows = read_manifest(manifest_path)
    task_filter = parse_csv_filter(args.tasks)
    act_filter = parse_csv_filter(args.activations)
    seed_filter = {int(x) for x in parse_csv_filter(args.seeds)}

    base = load_pickle_dict(base_cache)
    results = load_pickle_dict(output_cache)
    status_rows: list[dict[str, Any]] = []

    for row in rows:
        act = str(row["act"])
        task = str(row["task"])
        seed = int(row["seed"])
        key = (act, task, seed)
        exp_dir = resolve_path(str(row["path"]))

        if task_filter and task not in task_filter:
            continue
        if act_filter and act not in act_filter:
            continue
        if seed_filter and seed not in seed_filter:
            continue

        if key in results and not args.overwrite:
            status_rows.append({
                "act": act,
                "task": task,
                "seed": seed,
                "path": str(exp_dir),
                "status": "skipped_existing",
                "n_epochs": len(results[key].get("epochs", [])),
                "final_relative_dequiv": "",
                "error": "",
            })
            continue

        if not (exp_dir / "training_metrics.pkl").exists():
            status_rows.append({
                "act": act,
                "task": task,
                "seed": seed,
                "path": str(exp_dir),
                "status": "missing_training_metrics",
                "n_epochs": 0,
                "final_relative_dequiv": "",
                "error": "",
            })
            continue
        if not (exp_dir / "gcs_results.pkl").exists():
            status_rows.append({
                "act": act,
                "task": task,
                "seed": seed,
                "path": str(exp_dir),
                "status": "missing_gcs_results",
                "n_epochs": 0,
                "final_relative_dequiv": "",
                "error": "",
            })
            continue

        print(f"Processing {act}/{task}/s{seed}: {exp_dir}", flush=True)
        try:
            result = process_run(str(exp_dir), device=args.device, checkpoint_stride=int(args.stride))
            if result is None:
                raise RuntimeError("process_run returned None")
            try:
                metrics = load_metrics(str(exp_dir))
                result["t_grok"] = detect_grok_epoch(metrics)
            except Exception:
                result["t_grok"] = None

            results[key] = result
            rel = np.asarray(result["equi_dist"], dtype=float) / np.where(
                np.asarray(result["norm_scale"], dtype=float) > 1e-8,
                np.asarray(result["norm_scale"], dtype=float),
                1.0,
            )
            status_rows.append({
                "act": act,
                "task": task,
                "seed": seed,
                "path": str(exp_dir),
                "status": "ok",
                "n_epochs": len(result.get("epochs", [])),
                "final_relative_dequiv": float(rel[-1]) if rel.size else "",
                "error": "",
            })
            write_pickle(output_cache, results)
            write_csv(status_csv, status_rows)
        except Exception as exc:
            status_rows.append({
                "act": act,
                "task": task,
                "seed": seed,
                "path": str(exp_dir),
                "status": "error",
                "n_epochs": 0,
                "final_relative_dequiv": "",
                "error": repr(exc),
            })
            write_csv(status_csv, status_rows)
            print(f"Failed {act}/{task}/s{seed}: {exc!r}", flush=True)

    write_pickle(output_cache, results)
    write_csv(status_csv, status_rows)

    if merged_output is not None:
        merged = dict(base)
        merged.update(results)
        write_pickle(merged_output, merged)
        print(f"Saved merged cache with {len(merged)} runs: {merged_output}")

    print(f"Saved manifest equivariance cache with {len(results)} runs: {output_cache}")
    print(f"Saved status CSV: {status_csv}")


if __name__ == "__main__":
    main()
