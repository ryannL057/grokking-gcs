#!/usr/bin/env python3
"""Run rebuttal donor-yield surgery for GCS and CKA/L2 windows.

This extends the selected-donor surgery audit by using the same final-base
reset-attention/reset-MLP protocol for windows selected by Attention-GCS and
endpoint/path-proximity baselines.  The intended rebuttal question is whether
GCS-selected donors have mechanistic follow-up value relative to strong CKA/L2
windows, not whether GCS dominates every scalar diagnostic.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = _gcs_data_root()
DEFAULT_WINDOWS = ROOT / "experiments/mechanistic_yield_cohort_core_heads_20260503/per_window.csv"
DEFAULT_REUSE_DIR = ROOT / "experiments/selected_donor_surgery_20260505"


LABEL_SPECS: dict[str, dict[str, str | None]] = {
    "gcs_window_start": {
        "selector": "gcs_attention",
        "epoch_choice": "start",
        "label": "Attention-GCS start",
    },
    "gcs_window_center": {
        "selector": "gcs_attention",
        "epoch_choice": "center",
        "label": "Attention-GCS center",
    },
    "attn_cka_window_start": {
        "selector": "path_cka_attention_rise",
        "epoch_choice": "start",
        "label": "Attn-output CKA start",
    },
    "attn_cka_window_center": {
        "selector": "path_cka_attention_rise",
        "epoch_choice": "center",
        "label": "Attn-output CKA center",
    },
    "attn_l2_window_start": {
        "selector": "path_l2_attention_drop",
        "epoch_choice": "start",
        "label": "Attn-output L2 start",
    },
    "attn_l2_window_center": {
        "selector": "path_l2_attention_drop",
        "epoch_choice": "center",
        "label": "Attn-output L2 center",
    },
    "final_cka_window_start": {
        "selector": "final_repr_cka_rise",
        "epoch_choice": "start",
        "label": "Final-repr CKA start",
    },
    "final_cka_window_center": {
        "selector": "final_repr_cka_rise",
        "epoch_choice": "center",
        "label": "Final-repr CKA center",
    },
    "equivariance_drop": {
        "selector": None,
        "epoch_choice": "equiv_drop",
        "label": "Independent equivariance-drop donor",
    },
    "random_window_start": {
        "selector": "random",
        "epoch_choice": "start",
        "label": "Random legal window start",
    },
}

DEFAULT_LABEL_ORDER = [
    "gcs_window_start",
    "gcs_window_center",
    "attn_cka_window_start",
    "attn_cka_window_center",
    "attn_l2_window_start",
    "attn_l2_window_center",
    "final_cka_window_start",
    "final_cka_window_center",
    "equivariance_drop",
    "random_window_start",
]


def parse_labels(text: str) -> list[str]:
    labels = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [label for label in labels if label not in LABEL_SPECS]
    if unknown:
        raise SystemExit(
            "Unknown donor labels: "
            + ", ".join(unknown)
            + f". Known labels: {', '.join(DEFAULT_LABEL_ORDER)}"
        )
    return labels


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: str | float | int) -> int:
    return int(round(float(value)))


def as_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def midpoint_epoch(row: dict[str, str]) -> int:
    return int(round((as_int(row["window_start_epoch"]) + as_int(row["window_end_epoch"])) / 2.0))


def run_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (row["act"], row["task"], int(row["seed"]))


def build_window_lookup(
    path: Path,
    *,
    split: str,
    donor_labels: list[str],
) -> tuple[list[tuple[str, str, int]], dict[tuple[str, str, int], dict[str, dict[str, str]]]]:
    requested_splits = {part.strip() for part in split.split(",") if part.strip()}
    use_all_splits = not requested_splits or "all" in requested_splits
    required_selectors = {
        str(LABEL_SPECS[label]["selector"])
        for label in donor_labels
        if LABEL_SPECS[label]["selector"] not in {None, "random"}
    }
    rows_by_run: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    random_trial0: dict[tuple[str, str, int], dict[str, str]] = {}

    for row in read_csv(path):
        if not use_all_splits and row.get("split") not in requested_splits:
            continue
        key = run_key(row)
        selector = row["selector"]
        if selector == "random":
            if row.get("random_trial") == "0":
                random_trial0[key] = row
            continue
        if selector in required_selectors:
            rows_by_run[key][selector] = row

    cohort_keys = sorted(rows_by_run)
    if not cohort_keys:
        raise SystemExit(f"No selector rows for split={split} in {path}")

    missing: list[str] = []
    for key in cohort_keys:
        for selector in sorted(required_selectors):
            if selector not in rows_by_run[key]:
                missing.append(f"{key}:{selector}")
        if "random_window_start" in donor_labels and key not in random_trial0:
            missing.append(f"{key}:random_trial0")
    if missing:
        preview = ", ".join(missing[:10])
        raise SystemExit(f"Missing required window rows ({len(missing)}): {preview}")

    for key, row in random_trial0.items():
        if key in rows_by_run:
            rows_by_run[key]["random"] = row

    return cohort_keys, rows_by_run


def donor_epoch_from_row(row: dict[str, str], epoch_choice: str) -> int:
    if epoch_choice == "start":
        return as_int(row["window_start_epoch"])
    if epoch_choice == "center":
        return midpoint_epoch(row)
    if epoch_choice == "end":
        return as_int(row["window_end_epoch"])
    raise ValueError(f"Unsupported epoch choice for window row: {epoch_choice}")


def surgery_csv_path(output_dir: Path, donor_label: str, act: str, task: str, seed: int) -> Path:
    return output_dir / donor_label / f"{act.lower()}_{task}_s{seed}_surgery.csv"


def maybe_reuse_csv(
    *,
    reuse_dir: Path | None,
    output_dir: Path,
    donor_label: str,
    act: str,
    task: str,
    seed: int,
) -> bool:
    if reuse_dir is None:
        return False
    src = surgery_csv_path(reuse_dir, donor_label, act, task, seed)
    dst = surgery_csv_path(output_dir, donor_label, act, task, seed)
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def run_one(
    *,
    python_bin: str,
    device: str,
    output_dir: Path,
    reuse_dir: Path | None,
    act: str,
    task: str,
    seed: int,
    donor_label: str,
    donor_mode: str,
    donor_epoch: int | None,
    force: bool,
    dry_run: bool,
) -> None:
    csv_path = surgery_csv_path(output_dir, donor_label, act, task, seed)
    if csv_path.exists() and not force:
        return
    if dry_run:
        if reuse_dir is not None and surgery_csv_path(reuse_dir, donor_label, act, task, seed).exists():
            print(f"[would-reuse:{donor_label}] {act}/{task}/s{seed}", flush=True)
            return
    if not force and maybe_reuse_csv(
        reuse_dir=reuse_dir,
        output_dir=output_dir,
        donor_label=donor_label,
        act=act,
        task=task,
        seed=seed,
    ):
        print(f"[reuse:{donor_label}] {act}/{task}/s{seed}", flush=True)
        return

    donor_dir = output_dir / donor_label
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
    print(f"[run:{donor_label}] {act}/{task}/s{seed}", flush=True)
    if dry_run:
        print(" ".join(cmd), flush=True)
        return
    env = os.environ.copy()
    if "MPLCONFIGDIR" not in env:
        mpl_cache = Path("/tmp/gcs_core_matplotlib")
        mpl_cache.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_cache)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def load_one_surgery(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["condition"]: row for row in rows}


def window_metadata(
    *,
    label: str,
    key: tuple[str, str, int],
    windows: dict[tuple[str, str, int], dict[str, dict[str, str]]],
) -> dict[str, object]:
    spec = LABEL_SPECS[label]
    selector = spec["selector"]
    epoch_choice = str(spec["epoch_choice"])
    if selector is None:
        return {
            "selector": "",
            "selector_label": spec["label"],
            "window_split": "",
            "window_epoch_choice": epoch_choice,
            "window_start_epoch": "",
            "window_end_epoch": "",
            "window_start_minus_t_grok": "",
            "window_start_test_accuracy": "",
            "window_end_test_accuracy": "",
            "window_delta_test_accuracy": "",
            "window_equivariance_percentile": "",
        }

    row = windows[key][str(selector)]
    return {
        "selector": selector,
        "selector_label": row.get("selector_label", spec["label"]),
        "window_split": row.get("split", ""),
        "window_epoch_choice": epoch_choice,
        "window_start_epoch": as_int(row["window_start_epoch"]),
        "window_end_epoch": as_int(row["window_end_epoch"]),
        "window_start_minus_t_grok": as_int(row["start_minus_t_grok"]),
        "window_start_test_accuracy": as_float(row["start_test_accuracy"]),
        "window_end_test_accuracy": as_float(row["end_test_accuracy"]),
        "window_delta_test_accuracy": as_float(row["delta_test_accuracy"]),
        "window_equivariance_percentile": as_float(row["equivariance_window_percentile"]),
    }


def load_surgery_rows(
    *,
    output_dir: Path,
    donor_labels: list[str],
    cohort_keys: list[tuple[str, str, int]],
    windows: dict[tuple[str, str, int], dict[str, dict[str, str]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for donor_label in donor_labels:
        for key in cohort_keys:
            act, task, seed = key
            csv_path = surgery_csv_path(output_dir, donor_label, act, task, seed)
            if not csv_path.exists():
                continue
            by_condition = load_one_surgery(csv_path)
            if not {"donor_pre", "base", "reset_attn", "reset_mlp"} <= set(by_condition):
                continue
            donor = by_condition["donor_pre"]
            base = by_condition["base"]
            attn = by_condition["reset_attn"]
            mlp = by_condition["reset_mlp"]

            base_acc = as_float(base["test_acc"])
            attn_acc = as_float(attn["test_acc"])
            mlp_acc = as_float(mlp["test_acc"])
            delta_attn = base_acc - attn_acc
            delta_mlp = base_acc - mlp_acc
            donor_epoch = as_int(base["donor_epoch"])
            t_grok = as_int(base["t_grok"])
            row: dict[str, object] = {
                "donor_label": donor_label,
                "donor_display_label": LABEL_SPECS[donor_label]["label"],
                "act": act,
                "task": task,
                "seed": seed,
                "t_grok": t_grok,
                "donor_mode": base["donor_mode"],
                "donor_target_epoch": as_int(base["donor_target_epoch"]),
                "donor_epoch": donor_epoch,
                "donor_minus_t_grok": donor_epoch - t_grok,
                "donor_pre_test_acc": as_float(donor["test_acc"]),
                "donor_pre_attn_rel_equi": as_float(donor["attn_rel_equi"]),
                "base_test_acc": base_acc,
                "reset_attn_test_acc": attn_acc,
                "reset_mlp_test_acc": mlp_acc,
                "delta_attn": delta_attn,
                "delta_mlp": delta_mlp,
                "attn_minus_mlp_delta": delta_attn - delta_mlp,
                "attn_reset_hurts_more": int(delta_attn > delta_mlp),
                "base_attn_rel_equi": as_float(base["attn_rel_equi"]),
                "reset_attn_rel_equi": as_float(attn["attn_rel_equi"]),
                "reset_mlp_rel_equi": as_float(mlp["attn_rel_equi"]),
                "delta_attn_rel_equi": as_float(attn["attn_rel_equi"]) - as_float(base["attn_rel_equi"]),
                "delta_mlp_rel_equi": as_float(mlp["attn_rel_equi"]) - as_float(base["attn_rel_equi"]),
            }
            row.update(window_metadata(label=donor_label, key=key, windows=windows))
            rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return float("nan")
    ordered = sorted(finite)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def sign_test_upper_tail(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / float(2**n)


def summarize(rows: list[dict[str, object]], donor_labels: list[str]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for donor_label in donor_labels:
        donor_rows = [row for row in rows if row["donor_label"] == donor_label]
        if not donor_rows:
            continue
        n = len(donor_rows)
        k = sum(int(row["attn_reset_hurts_more"]) for row in donor_rows)
        summary.append(
            {
                "donor_label": donor_label,
                "donor_display_label": LABEL_SPECS[donor_label]["label"],
                "selector": LABEL_SPECS[donor_label]["selector"] or "",
                "epoch_choice": LABEL_SPECS[donor_label]["epoch_choice"],
                "n_runs": n,
                "attn_gt_mlp": f"{k}/{n}",
                "sign_test_p": sign_test_upper_tail(k, n),
                "median_donor_minus_t_grok": median([float(row["donor_minus_t_grok"]) for row in donor_rows]),
                "median_donor_pre_test_acc": median([float(row["donor_pre_test_acc"]) for row in donor_rows]),
                "median_delta_attn": median([float(row["delta_attn"]) for row in donor_rows]),
                "median_delta_mlp": median([float(row["delta_mlp"]) for row in donor_rows]),
                "median_attn_minus_mlp_delta": median(
                    [float(row["attn_minus_mlp_delta"]) for row in donor_rows]
                ),
                "mean_delta_attn": mean([float(row["delta_attn"]) for row in donor_rows]),
                "mean_delta_mlp": mean([float(row["delta_mlp"]) for row in donor_rows]),
                "median_reset_attn_test_acc": median(
                    [float(row["reset_attn_test_acc"]) for row in donor_rows]
                ),
                "median_reset_mlp_test_acc": median(
                    [float(row["reset_mlp_test_acc"]) for row in donor_rows]
                ),
                "median_delta_attn_rel_equi": median(
                    [float(row["delta_attn_rel_equi"]) for row in donor_rows]
                ),
                "median_window_delta_test_accuracy": median(
                    [as_float(row["window_delta_test_accuracy"]) for row in donor_rows]
                ),
                "median_window_equivariance_percentile": median(
                    [as_float(row["window_equivariance_percentile"]) for row in donor_rows]
                ),
            }
        )
    return summary


def summarize_pairwise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key_label: dict[tuple[str, str, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (str(row["act"]), str(row["task"]), int(row["seed"]))
        by_key_label[key][str(row["donor_label"])] = row

    pairs = [
        ("gcs_window_start", "attn_cka_window_start"),
        ("gcs_window_start", "attn_l2_window_start"),
        ("gcs_window_start", "final_cka_window_start"),
        ("gcs_window_center", "attn_cka_window_center"),
        ("gcs_window_center", "attn_l2_window_center"),
        ("gcs_window_center", "final_cka_window_center"),
        ("attn_cka_window_start", "attn_l2_window_start"),
    ]
    out: list[dict[str, object]] = []
    for left, right in pairs:
        paired = [(labels[left], labels[right]) for labels in by_key_label.values() if left in labels and right in labels]
        if not paired:
            continue
        out.append(
            {
                "left_label": left,
                "right_label": right,
                "n_runs": len(paired),
                "left_earlier": sum(int(l["donor_epoch"]) < int(r["donor_epoch"]) for l, r in paired),
                "left_same_or_earlier": sum(
                    int(l["donor_epoch"]) <= int(r["donor_epoch"]) for l, r in paired
                ),
                "median_left_minus_right_epoch": median(
                    [float(l["donor_epoch"]) - float(r["donor_epoch"]) for l, r in paired]
                ),
                "median_left_minus_right_donor_acc": median(
                    [float(l["donor_pre_test_acc"]) - float(r["donor_pre_test_acc"]) for l, r in paired]
                ),
                "median_left_minus_right_delta_attn": median(
                    [float(l["delta_attn"]) - float(r["delta_attn"]) for l, r in paired]
                ),
                "median_left_minus_right_delta_mlp": median(
                    [float(l["delta_mlp"]) - float(r["delta_mlp"]) for l, r in paired]
                ),
                "median_left_minus_right_attn_minus_mlp_delta": median(
                    [
                        float(l["attn_minus_mlp_delta"]) - float(r["attn_minus_mlp_delta"])
                        for l, r in paired
                    ]
                ),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--split",
        default="heldout_seed123_777",
        help="Window split to include. Use a comma-separated list or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/rebuttal_donor_yield_gcs_cka_l2_20260706",
    )
    parser.add_argument(
        "--reuse-dir",
        type=Path,
        default=DEFAULT_REUSE_DIR,
        help="Optional directory containing compatible surgery CSVs to copy before rerunning.",
    )
    parser.add_argument("--donor-labels", default=",".join(DEFAULT_LABEL_ORDER))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    donor_labels = parse_labels(args.donor_labels)
    reuse_dir = args.reuse_dir if args.reuse_dir and args.reuse_dir.exists() else None
    cohort_keys, windows = build_window_lookup(
        args.windows,
        split=args.split,
        donor_labels=donor_labels,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in cohort_keys:
        act, task, seed = key
        for donor_label in donor_labels:
            spec = LABEL_SPECS[donor_label]
            selector = spec["selector"]
            epoch_choice = str(spec["epoch_choice"])
            if selector is None:
                donor_mode = "equiv_drop"
                donor_epoch = None
            else:
                donor_mode = "explicit_epoch"
                donor_epoch = donor_epoch_from_row(windows[key][str(selector)], epoch_choice)
            run_one(
                python_bin=args.python_bin,
                device=args.device,
                output_dir=args.output_dir,
                reuse_dir=reuse_dir,
                act=act,
                task=task,
                seed=seed,
                donor_label=donor_label,
                donor_mode=donor_mode,
                donor_epoch=donor_epoch,
                force=args.force,
                dry_run=args.dry_run,
            )

    if args.dry_run:
        return

    rows = load_surgery_rows(
        output_dir=args.output_dir,
        donor_labels=donor_labels,
        cohort_keys=cohort_keys,
        windows=windows,
    )
    if not rows:
        raise SystemExit(f"No surgery rows found under {args.output_dir}")
    write_csv(args.output_dir / "donor_yield_surgery_per_run.csv", rows)
    write_csv(args.output_dir / "donor_yield_surgery_summary.csv", summarize(rows, donor_labels))
    write_csv(args.output_dir / "donor_yield_surgery_pairwise.csv", summarize_pairwise(rows))
    print(f"Saved {args.output_dir / 'donor_yield_surgery_per_run.csv'}")
    print(f"Saved {args.output_dir / 'donor_yield_surgery_summary.csv'}")
    print(f"Saved {args.output_dir / 'donor_yield_surgery_pairwise.csv'}")


if __name__ == "__main__":
    main()
