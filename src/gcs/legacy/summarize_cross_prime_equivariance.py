#!/usr/bin/env python3
"""Summarize final attention-output swap equivariance across available primes.

This is a cached-artifact evaluator for Claim 2.  It deliberately separates
"commutative-only evidence" from a true zero-overlap replication, which
requires both commutative and non-commutative tasks at the same prime.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

from gcs.runtime import data_root as _gcs_data_root

ROOT = _gcs_data_root()
if str(ROOT) not in sys.path:
    pass  # Package imports.
if str(ROOT / "scripts") not in sys.path:
    pass  # Package imports.

from gcs.core.data import create_data  # noqa: E402
from gcs.core.models import Transformer  # noqa: E402
from gcs.legacy.measure_equivariance import build_swap_pairs, measure_attn_equivariance  # noqa: E402
from gcs.legacy.diagnostics_common import RUNS_1L, resolve_path  # noqa: E402

COMMUTATIVE = {"add", "mul"}
NONCOMMUTATIVE = {"sub", "div"}


def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.pkl"
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)
        if isinstance(cfg, dict):
            return cfg
        if hasattr(cfg, "__dict__"):
            return dict(cfg.__dict__)
    ckpts = sorted((run_dir / "checkpoints").glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No config.pkl or checkpoints found in {run_dir}")
    ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    return ckpt.get("config") or ckpt.get("model_config") or {}


def infer_from_path(run_dir: Path, cfg: dict) -> dict:
    name = run_dir.name
    path_text = str(run_dir)
    out = dict(cfg)
    if "task" not in out:
        m = re.search(r"1layer_(add|mul|sub|div)", name)
        if not m:
            m = re.search(r"/(add|mul|sub|div)(?:/)?$", path_text)
        if m:
            out["task"] = m.group(1)
    if "p" not in out:
        m = re.search(r"_p(\d+)", name)
        if m:
            out["p"] = int(m.group(1))
        else:
            out["p"] = 113
    if "seed" not in out:
        m = re.search(r"_s(\d+)", name)
        if m:
            out["seed"] = int(m.group(1))
        else:
            out["seed"] = 777
    if "act_type" not in out:
        prefix = name.split("_", 1)[0]
        mapping = {"relu": "ReLU", "gelu": "GeLU", "silu": "SiLU", "selu": "SeLU"}
        act_from_path = next((act for act in ["ReLU", "GeLU", "SiLU", "SeLU"] if f"/{act}/" in path_text), None)
        out["act_type"] = act_from_path or mapping.get(prefix.lower(), prefix)
    out.setdefault("num_layers", 1)
    out.setdefault("d_model", 128)
    out.setdefault("d_mlp", 512)
    out.setdefault("d_head", 32)
    out.setdefault("num_heads", 4)
    out.setdefault("n_ctx", 3)
    out.setdefault("attention_rate", 1.0)
    out.setdefault("fraction", 0.3)
    out.setdefault("d_vocab", int(out["p"]) + 1)
    return out


def final_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted((run_dir / "checkpoints").glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    def epoch_of(path: Path) -> int:
        m = re.search(r"epoch_(\d+)", path.name)
        return int(m.group(1)) if m else -1

    return max(ckpts, key=epoch_of)


def load_model(run_dir: Path, cfg: dict, device: str) -> tuple[Transformer, int]:
    ckpt_path = final_checkpoint(run_dir)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config") or ckpt.get("model_config") or {}
    merged = dict(cfg)
    merged.update({k: v for k, v in ckpt_cfg.items() if v is not None})

    model = Transformer(
        num_layers=int(merged.get("num_layers", 1)),
        d_vocab=int(merged.get("d_vocab", int(merged["p"]) + 1)),
        d_model=int(merged.get("d_model", 128)),
        d_mlp=int(merged.get("d_mlp", 512)),
        d_head=int(merged.get("d_head", 32)),
        num_heads=int(merged.get("num_heads", 4)),
        n_ctx=int(merged.get("n_ctx", 3)),
        act_type=str(merged.get("act_type", "SiLU")),
        attention_rate=float(merged.get("attention_rate", 1.0)),
        use_ln=bool(merged.get("use_ln", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, int(ckpt.get("epoch", -1))


def evaluate_accuracy(model: Transformer, cfg: dict, device: str, batch_size: int) -> tuple[float, float]:
    train_data, test_data = create_data(
        p=int(cfg["p"]),
        fraction=float(cfg.get("fraction", 0.3)),
        seed=int(cfg.get("seed", 777)),
        device=device,
        task=str(cfg["task"]),
    )

    def acc(dataset) -> float:
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                end = min(start + batch_size, len(dataset))
                prompts = torch.stack([dataset[i][0] for i in range(start, end)]).to(device)
                labels = torch.stack([dataset[i][1] for i in range(start, end)]).to(device)
                logits = model(prompts)[:, -1, : int(cfg["p"])]
                pred = logits.argmax(dim=-1)
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())
        return correct / total if total else float("nan")

    return acc(train_data), acc(test_data)


def discover_runs(include_p113: bool, explicit_dirs: list[str]) -> list[Path]:
    runs: list[Path] = []
    if include_p113:
        for _act, _task, _seed, path in RUNS_1L:
            runs.append(resolve_path(path))
    patterns = [
        "experiments/*_1layer_add_p97_*",
        "experiments/*_1layer_mul_p97_*",
        "experiments/*_1layer_sub_p97_*",
        "experiments/*_1layer_div_p97_*",
        "experiments/maintrack_minimal/phaseB_crossp/*_1layer_*_p97_*",
        "experiments/maintrack_minimal/phaseB_crossp/*_1layer_*_p127_*",
        "experiments/cross_prime_noncomm_silu_20260426/*_1layer_*_p97_*",
        "experiments/cross_prime_noncomm_silu_20260426/*_1layer_*_p127_*",
        "experiments/cross_prime_comm_silu_20260426/*_1layer_*_p97_*",
        "experiments/cross_prime_comm_silu_20260426/*_1layer_*_p127_*",
    ]
    for pattern in patterns:
        runs.extend(ROOT.glob(pattern))
    for item in explicit_dirs:
        runs.append(resolve_path(item))

    seen = set()
    unique = []
    for run in runs:
        run = run.resolve()
        if run in seen or not (run / "checkpoints").exists():
            continue
        seen.add(run)
        unique.append(run)
    return sorted(unique, key=lambda p: str(p))


def evaluate_run(run_dir: Path, device: str, batch_size: int) -> dict:
    cfg = infer_from_path(run_dir, load_config(run_dir))
    p = int(cfg["p"])
    task = str(cfg["task"])
    model, epoch = load_model(run_dir, cfg, device)
    prompts_ab, prompts_ba = build_swap_pairs(p, device=device)
    diff, norm = measure_attn_equivariance(
        model, prompts_ab, prompts_ba, device=device, batch_size=batch_size
    )
    train_acc, test_acc = evaluate_accuracy(model, cfg, device=device, batch_size=batch_size)
    group = "commutative" if task in COMMUTATIVE else "noncommutative" if task in NONCOMMUTATIVE else "other"
    return {
        "run_dir": str(run_dir),
        "p": p,
        "task": task,
        "group": group,
        "act": str(cfg.get("act_type", "")),
        "seed": int(cfg.get("seed", -1)),
        "attention_rate": float(cfg.get("attention_rate", 1.0)),
        "weight_decay": float(cfg.get("weight_decay", float("nan"))),
        "final_epoch": epoch,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "attn_swap_diff": float(diff),
        "attn_norm": float(norm),
        "relative_dequiv": float(diff / norm) if norm > 1e-8 else float("nan"),
    }


def apply_filters(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    filtered = rows
    if args.activation:
        keep = {item.strip() for item in args.activation.split(",") if item.strip()}
        filtered = [row for row in filtered if row["act"] in keep]
    if args.attention_rate is not None:
        filtered = [
            row for row in filtered if np.isclose(row["attention_rate"], args.attention_rate, rtol=0, atol=1e-9)
        ]
    if args.weight_decay is not None:
        filtered = [
            row for row in filtered if np.isclose(row["weight_decay"], args.weight_decay, rtol=0, atol=1e-9)
        ]
    return filtered


def summarize(rows: list[dict], min_test_acc: float) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    zero_rows: list[dict] = []
    if not rows:
        return summary_rows, zero_rows
    df = {}
    for row in rows:
        df.setdefault((row["p"], row["group"]), []).append(row)
    for (p, group), items in sorted(df.items()):
        vals = np.array([r["relative_dequiv"] for r in items], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q_items = [r for r in items if r["test_acc"] >= min_test_acc]
        q_vals = np.array([r["relative_dequiv"] for r in q_items], dtype=float)
        q_vals = q_vals[np.isfinite(q_vals)]
        test_accs = np.array([r["test_acc"] for r in items], dtype=float)
        summary_rows.append(
            {
                "p": p,
                "group": group,
                "n": int(vals.size),
                "n_qualified": int(q_vals.size),
                "median_relative_dequiv": float(np.median(vals)),
                "min_relative_dequiv": float(np.min(vals)),
                "max_relative_dequiv": float(np.max(vals)),
                "median_relative_dequiv_qualified": float(np.median(q_vals)) if q_vals.size else "",
                "median_test_acc": float(np.median(test_accs)),
                "min_test_acc": float(np.min(test_accs)),
            }
        )

    primes = sorted({r["p"] for r in rows})
    for p in primes:
        p_rows = [r for r in rows if r["p"] == p]
        q_rows = [r for r in p_rows if r["test_acc"] >= min_test_acc]
        comm = [r["relative_dequiv"] for r in q_rows if r["group"] == "commutative"]
        nonc = [r["relative_dequiv"] for r in q_rows if r["group"] == "noncommutative"]
        excluded = len(p_rows) - len(q_rows)
        if not comm or not nonc:
            zero_rows.append(
                {
                    "p": p,
                    "min_test_acc": min_test_acc,
                    "has_both_groups": False,
                    "zero_overlap": "",
                    "comm_max": max(comm) if comm else "",
                    "noncomm_min": min(nonc) if nonc else "",
                    "qualified_comm_n": len(comm),
                    "qualified_noncomm_n": len(nonc),
                    "excluded_low_acc_n": excluded,
                    "note": "missing qualified commutative or noncommutative group",
                }
            )
            continue
        zero_rows.append(
            {
                "p": p,
                "min_test_acc": min_test_acc,
                "has_both_groups": True,
                "zero_overlap": bool(max(comm) < min(nonc)),
                "comm_max": float(max(comm)),
                "noncomm_min": float(min(nonc)),
                "qualified_comm_n": len(comm),
                "qualified_noncomm_n": len(nonc),
                "excluded_low_acc_n": excluded,
                "note": "",
            }
        )
    return summary_rows, zero_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--include-p113", action="store_true", help="Include RUNS_1L p=113 baseline")
    parser.add_argument(
        "--min-test-acc",
        type=float,
        default=0.99,
        help="Minimum final test accuracy for qualified zero-overlap summaries",
    )
    parser.add_argument("--run-dirs", default="", help="Comma-separated additional run directories")
    parser.add_argument("--output", default=str(ROOT / "experiments" / "cross_prime_equivariance"))
    parser.add_argument("--activation", default="", help="Optional comma-separated activation filter")
    parser.add_argument("--attention-rate", type=float, default=None, help="Optional attention-rate filter")
    parser.add_argument("--weight-decay", type=float, default=None, help="Optional weight-decay filter")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    explicit = [x for x in args.run_dirs.split(",") if x]
    runs = discover_runs(args.include_p113, explicit)
    rows: list[dict] = []
    failures: list[dict] = []
    for run_dir in runs:
        try:
            row = evaluate_run(run_dir, args.device, args.batch_size)
            rows.append(row)
            print(
                f"{row['p']} {row['act']} {row['task']} s{row['seed']} "
                f"acc={row['test_acc']:.3f} dequiv={row['relative_dequiv']:.4f}"
            )
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "error": repr(exc)})
            print(f"FAIL {run_dir}: {exc}")

    rows = apply_filters(rows, args)
    summary_rows, zero_rows = summarize(rows, min_test_acc=args.min_test_acc)
    write_csv(out_dir / "final_equivariance.csv", rows)
    write_csv(out_dir / "summary_by_prime_group.csv", summary_rows)
    write_csv(out_dir / "zero_overlap_by_prime.csv", zero_rows)
    if failures:
        write_csv(out_dir / "failures.csv", failures)
    else:
        stale_failures = out_dir / "failures.csv"
        if stale_failures.exists():
            stale_failures.unlink()
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "n_runs": len(rows),
                "n_failures": len(failures),
                "min_test_acc": args.min_test_acc,
                "filters": {
                    "activation": args.activation,
                    "attention_rate": args.attention_rate,
                    "weight_decay": args.weight_decay,
                },
                "summary_by_prime_group": summary_rows,
                "zero_overlap_by_prime": zero_rows,
                "outputs": {
                    "final_equivariance_csv": str(out_dir / "final_equivariance.csv"),
                    "summary_csv": str(out_dir / "summary_by_prime_group.csv"),
                    "zero_overlap_csv": str(out_dir / "zero_overlap_by_prime.csv"),
                },
            },
            f,
            indent=2,
        )
    print(f"Saved {out_dir / 'final_equivariance.csv'}")
    print(f"Saved {out_dir / 'summary_by_prime_group.csv'}")
    print(f"Saved {out_dir / 'zero_overlap_by_prime.csv'}")


if __name__ == "__main__":
    main()
