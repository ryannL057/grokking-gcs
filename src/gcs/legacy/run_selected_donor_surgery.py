#!/usr/bin/env python3
"""Run final-base surgery with donors tied to selected inspection windows.

This is a lightweight donor-link check for the paper revision: it compares
GCS-selected window donors against the original equivariance-drop donor and a
deterministic random equal-width window donor on the same 17 commutative runs.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


ROOT = _gcs_data_root()
DEFAULT_WINDOWS = ROOT / "experiments/mechanistic_yield_cohort_core_heads_20260503/per_window.csv"


def read_window_rows(path: Path) -> tuple[list[dict], dict[tuple[str, str, int], dict]]:
    gcs_rows: list[dict] = []
    random_rows: dict[tuple[str, str, int], dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["act"], row["task"], int(row["seed"]))
            if row["selector"] == "gcs_attention":
                gcs_rows.append(row)
            elif row["selector"] == "random" and row.get("random_trial") == "0":
                random_rows[key] = row
    if not gcs_rows:
        raise SystemExit(f"No gcs_attention rows found in {path}")
    missing_random = [
        (row["act"], row["task"], int(row["seed"]))
        for row in gcs_rows
        if (row["act"], row["task"], int(row["seed"])) not in random_rows
    ]
    if missing_random:
        raise SystemExit(f"Missing random_trial=0 rows for {missing_random}")
    return gcs_rows, random_rows


def midpoint_epoch(row: dict) -> int:
    start = int(float(row["window_start_epoch"]))
    end = int(float(row["window_end_epoch"]))
    return int(round((start + end) / 2.0))


def donor_specs(gcs_row: dict, random_row: dict) -> list[tuple[str, str, int | None]]:
    return [
        ("gcs_window_start", "explicit_epoch", int(float(gcs_row["window_start_epoch"]))),
        ("gcs_window_center", "explicit_epoch", midpoint_epoch(gcs_row)),
        ("equivariance_drop", "equiv_drop", None),
        ("random_window_start", "explicit_epoch", int(float(random_row["window_start_epoch"]))),
    ]


def run_one(
    *,
    python_bin: str,
    device: str,
    output_dir: Path,
    act: str,
    task: str,
    seed: int,
    donor_label: str,
    donor_mode: str,
    donor_epoch: int | None,
    force: bool,
) -> None:
    donor_dir = output_dir / donor_label
    csv_path = donor_dir / f"{act.lower()}_{task}_s{seed}_surgery.csv"
    if csv_path.exists() and not force:
        return
    donor_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        "-m", "gcs.legacy.checkpoint_surgery_modular",
        "--act",
        act,
        "--task",
        task,
        "--seed",
        str(seed),
        "--device",
        device,
        "--base-labels",
        "final",
        "--conditions",
        "donor_pre,base,reset_attn,reset_mlp",
        "--donor-mode",
        donor_mode,
        "--skip-plot",
        "--output",
        str(donor_dir),
    ]
    if donor_epoch is not None:
        cmd.extend(["--donor-epoch", str(donor_epoch)])
    print(f"[{donor_label}] {act}/{task}/s{seed}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_surgery_rows(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for donor_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        donor_label = donor_dir.name
        for csv_path in sorted(donor_dir.glob("*_surgery.csv")):
            with csv_path.open(newline="") as f:
                by_condition = {row["condition"]: row for row in csv.DictReader(f)}
            if not {"base", "reset_attn", "reset_mlp"} <= set(by_condition):
                continue
            base = by_condition["base"]
            attn = by_condition["reset_attn"]
            mlp = by_condition["reset_mlp"]
            base_acc = float(base["test_acc"])
            attn_acc = float(attn["test_acc"])
            mlp_acc = float(mlp["test_acc"])
            base_eq = float(base["attn_rel_equi"])
            attn_eq = float(attn["attn_rel_equi"])
            rows.append(
                {
                    "donor_label": donor_label,
                    "act": base["act"],
                    "task": base["task"],
                    "seed": int(base["seed"]),
                    "t_grok": int(float(base["t_grok"])),
                    "donor_mode": base["donor_mode"],
                    "donor_target_epoch": int(float(base["donor_target_epoch"])),
                    "donor_epoch": int(float(base["donor_epoch"])),
                    "donor_minus_t_grok": int(float(base["donor_epoch"])) - int(float(base["t_grok"])),
                    "base_test_acc": base_acc,
                    "reset_attn_test_acc": attn_acc,
                    "reset_mlp_test_acc": mlp_acc,
                    "delta_attn": base_acc - attn_acc,
                    "delta_mlp": base_acc - mlp_acc,
                    "attn_reset_hurts_more": (base_acc - attn_acc) > (base_acc - mlp_acc),
                    "base_attn_rel_equi": base_eq,
                    "reset_attn_rel_equi": attn_eq,
                    "delta_attn_rel_equi": attn_eq - base_eq,
                }
            )
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def sign_test(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / float(2**n)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    order = [
        "gcs_window_start",
        "gcs_window_center",
        "equivariance_drop",
        "random_window_start",
    ]
    summary = []
    for donor_label in order:
        donor_rows = [row for row in rows if row["donor_label"] == donor_label]
        if not donor_rows:
            continue
        n = len(donor_rows)
        k = sum(bool(row["attn_reset_hurts_more"]) for row in donor_rows)
        summary.append(
            {
                "donor_label": donor_label,
                "n_runs": n,
                "attn_gt_mlp": f"{k}/{n}",
                "sign_test_p": sign_test(k, n),
                "mean_delta_attn": mean([float(row["delta_attn"]) for row in donor_rows]),
                "mean_delta_mlp": mean([float(row["delta_mlp"]) for row in donor_rows]),
                "median_delta_attn": median([float(row["delta_attn"]) for row in donor_rows]),
                "median_delta_mlp": median([float(row["delta_mlp"]) for row in donor_rows]),
                "median_donor_minus_t_grok": median([float(row["donor_minus_t_grok"]) for row in donor_rows]),
                "mean_delta_attn_rel_equi": mean([float(row["delta_attn_rel_equi"]) for row in donor_rows]),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/selected_donor_surgery_20260505")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    gcs_rows, random_rows = read_window_rows(args.windows)
    for gcs_row in gcs_rows:
        act = gcs_row["act"]
        task = gcs_row["task"]
        seed = int(gcs_row["seed"])
        random_row = random_rows[(act, task, seed)]
        for donor_label, donor_mode, donor_epoch in donor_specs(gcs_row, random_row):
            run_one(
                python_bin=args.python_bin,
                device=args.device,
                output_dir=args.output_dir,
                act=act,
                task=task,
                seed=seed,
                donor_label=donor_label,
                donor_mode=donor_mode,
                donor_epoch=donor_epoch,
                force=args.force,
            )

    rows = load_surgery_rows(args.output_dir)
    if not rows:
        raise SystemExit(f"No surgery rows found under {args.output_dir}")
    per_run_path = args.output_dir / "selected_donor_surgery_per_run.csv"
    summary_path = args.output_dir / "selected_donor_surgery_summary.csv"
    write_csv(per_run_path, rows)
    write_csv(summary_path, summarize(rows))
    print(f"Saved {per_run_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
