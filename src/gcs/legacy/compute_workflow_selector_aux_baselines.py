#!/usr/bin/env python3
"""Auxiliary scalar baselines for the held-out workflow selector benchmark.

The main GCS selector is a fixed-final-reference cross-input coherence
diagnostic.  This script computes skeptical alternatives that a reviewer might
ask about: distance to final weights, distance/similarity to final
representations, and ordinary path-output scalar summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

pass  # Imports use the installed package.

from gcs.core.models import Transformer  # noqa: E402
from gcs.legacy.diagnostics_common import RUNS_1L, load_gcs, resolve_path  # noqa: E402


def extract_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Cannot parse epoch from {path}")
    return int(match.group(1))


def checkpoint_map(exp_dir: Path) -> dict[int, Path]:
    ckpt_dir = exp_dir / "checkpoints"
    return {extract_epoch(path): path for path in ckpt_dir.glob("*.pth")}


def load_payload(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "model_state_dict" not in payload:
        payload = {"model_state_dict": payload, "config": {}}
    return payload


def build_model_from_payload(payload: dict[str, Any], device: str) -> Transformer:
    state = payload["model_state_dict"]
    cfg = payload.get("config") or payload.get("model_config") or {}
    d_model, d_vocab = state["embed.W_E"].shape
    d_mlp, _ = state["blocks.0.mlp.W_in"].shape
    n_heads, d_head, _ = state["blocks.0.attn.W_Q"].shape
    num_layers = sum(1 for key in state if key.endswith(".attn.W_Q"))
    model = Transformer(
        num_layers=int(cfg.get("num_layers", num_layers)),
        d_vocab=int(cfg.get("d_vocab", d_vocab)),
        d_model=int(cfg.get("d_model", d_model)),
        d_mlp=int(cfg.get("d_mlp", d_mlp)),
        d_head=int(cfg.get("d_head", d_head)),
        num_heads=int(cfg.get("num_heads", n_heads)),
        n_ctx=int(cfg.get("n_ctx", 3)),
        act_type=str(cfg.get("act_type", "SiLU")),
        use_cache=False,
        use_ln=bool(cfg.get("use_ln", False)),
        attention_rate=float(cfg.get("attention_rate", 1.0) or 1.0),
    ).to(device)
    return model


def flat_l2_distance(state: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> float:
    sq = 0.0
    ref_sq = 0.0
    for name, ref_tensor in ref.items():
        if name not in state:
            continue
        a = state[name].detach().float().cpu()
        b = ref_tensor.detach().float().cpu()
        sq += float(torch.sum((a - b) ** 2).item())
        ref_sq += float(torch.sum(b ** 2).item())
    return float(np.sqrt(sq) / max(np.sqrt(ref_sq), 1e-12))


def participation_ratio(x: torch.Tensor) -> float:
    x = x.float()
    if x.ndim != 2 or x.shape[0] <= 1:
        return float("nan")
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    tr = torch.trace(cov)
    tr2 = torch.sum(cov * cov)
    if float(tr2.item()) <= 1e-20:
        return float("nan")
    return float((tr * tr / tr2).item())


def trace_variance(x: torch.Tensor) -> float:
    x = x.float()
    if x.ndim != 2 or x.shape[0] <= 1:
        return float("nan")
    xc = x - x.mean(dim=0, keepdim=True)
    return float(torch.mean(torch.sum(xc * xc, dim=1)).item())


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    numerator = torch.sum(xy * xy)
    denom = torch.sqrt(torch.sum(xx * xx) * torch.sum(yy * yy))
    if float(denom.item()) <= 1e-20:
        return float("nan")
    return float((numerator / denom).item())


def collect_sites(model: Transformer, prompts: torch.Tensor, batch_size: int) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {"attention": [], "mlp": [], "overall": []}

    def save_attention(tensor: torch.Tensor, name: str) -> torch.Tensor:
        chunks["attention"].append(tensor[:, -1, :].detach().cpu())
        return tensor

    def save_mlp(tensor: torch.Tensor, name: str) -> torch.Tensor:
        chunks["mlp"].append(tensor[:, -1, :].detach().cpu())
        return tensor

    def save_overall(tensor: torch.Tensor, name: str) -> torch.Tensor:
        chunks["overall"].append(tensor[:, -1, :].detach().cpu())
        return tensor

    model.blocks[0].hook_attn_out.add_hook(save_attention, dir="fwd")
    model.blocks[0].hook_mlp_out.add_hook(save_mlp, dir="fwd")
    model.blocks[0].hook_resid_post.add_hook(save_overall, dir="fwd")
    model.eval()
    with torch.no_grad():
        for start in range(0, prompts.shape[0], batch_size):
            _ = model(prompts[start:start + batch_size])
    model.remove_all_hooks()
    return {key: torch.cat(vals, dim=0) for key, vals in chunks.items() if vals}


def build_probe_prompts(
    *,
    p: int,
    device: str,
    max_prompts: int | None,
    seed: int = 0,
) -> torch.Tensor:
    """Build a deterministic input probe set without reading geometry tensors.

    Some archived final-geometry files contain CUDA device ids from the
    original machine.  For these scalar controls we only need a fixed input set,
    so we sample operand pairs directly from the task domain.
    """
    xs, ys = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    eq = torch.full_like(xs.reshape(-1), fill_value=p)
    prompts = torch.stack([xs.reshape(-1), ys.reshape(-1), eq], dim=1).long()
    if max_prompts is not None and 0 < max_prompts < prompts.shape[0]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        idx = torch.randperm(prompts.shape[0], generator=gen)[:max_prompts]
        prompts = prompts[idx]
    return prompts.to(device)


def compute_for_run(
    exp_dir: Path,
    *,
    device: str,
    batch_size: int,
    max_prompts: int | None,
) -> dict[str, Any]:
    epochs, _ = load_gcs(exp_dir)
    epochs = [int(e) for e in epochs]
    ckpts = checkpoint_map(exp_dir)
    missing = [e for e in epochs if e not in ckpts]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints for epochs {missing[:5]} in {exp_dir}")

    final_epoch = max(ckpts)
    final_payload_cpu = load_payload(ckpts[final_epoch], map_location="cpu")
    final_state_cpu = final_payload_cpu["model_state_dict"]

    final_payload_dev = load_payload(ckpts[final_epoch], map_location=device)
    final_cfg = final_payload_dev.get("config") or final_payload_dev.get("model_config") or {}
    final_state = final_payload_dev["model_state_dict"]
    p = int(final_cfg.get("p", final_state["embed.W_E"].shape[1] - 1))
    prompts = build_probe_prompts(
        p=p,
        device=device,
        max_prompts=max_prompts,
        seed=0,
    )
    model = build_model_from_payload(final_payload_dev, device=device)
    model.load_state_dict(final_payload_dev["model_state_dict"])
    final_sites = collect_sites(model, prompts, batch_size=batch_size)
    final_norms = {
        key: float(torch.linalg.vector_norm(value).item())
        for key, value in final_sites.items()
    }
    final_overall = final_sites["overall"]

    out: dict[str, list[float]] = {
        "weight_final_distance": [],
        "final_repr_l2_distance": [],
        "final_repr_cka": [],
        "path_l2_attention": [],
        "path_l2_mlp": [],
        "path_l2_overall": [],
        "path_cka_attention": [],
        "path_cka_mlp": [],
        "path_cka_overall": [],
        "path_pr_attention": [],
        "path_pr_mlp": [],
        "path_pr_overall": [],
        "attention_output_variance": [],
    }

    for epoch in epochs:
        payload_cpu = load_payload(ckpts[epoch], map_location="cpu")
        out["weight_final_distance"].append(
            flat_l2_distance(payload_cpu["model_state_dict"], final_state_cpu)
        )

        payload_dev = load_payload(ckpts[epoch], map_location=device)
        model.load_state_dict(payload_dev["model_state_dict"])
        sites = collect_sites(model, prompts, batch_size=batch_size)
        overall = sites["overall"]
        out["final_repr_l2_distance"].append(
            float(torch.linalg.vector_norm(overall - final_overall).item() / max(final_norms["overall"], 1e-12))
        )
        out["final_repr_cka"].append(linear_cka(overall, final_overall))
        for site in ("attention", "mlp", "overall"):
            out[f"path_l2_{site}"].append(
                float(torch.linalg.vector_norm(sites[site] - final_sites[site]).item() / max(final_norms[site], 1e-12))
            )
            out[f"path_cka_{site}"].append(linear_cka(sites[site], final_sites[site]))
        out["path_pr_attention"].append(participation_ratio(sites["attention"]))
        out["path_pr_mlp"].append(participation_ratio(sites["mlp"]))
        out["path_pr_overall"].append(participation_ratio(overall))
        out["attention_output_variance"].append(trace_variance(sites["attention"]))

    return {
        "epochs": epochs,
        "final_epoch": int(final_epoch),
        "n_prompts": int(prompts.shape[0]),
        "series": out,
    }


def load_extra_runs(path: str) -> list[tuple[str, str, int, Path]]:
    if not path:
        return []
    rows: list[tuple[str, str, int, Path]] = []
    with open(resolve_path(path), newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["act"], row["task"], int(row["seed"]), resolve_path(row["path"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="_wsel_aux20260502")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-prompts", type=int, default=200)
    parser.add_argument("--tasks", default="add,sub,mul,div")
    parser.add_argument("--activations", default="ReLU,GeLU,SiLU")
    parser.add_argument("--seeds", default="42,123,777")
    parser.add_argument("--extra-runs-csv", default="")
    parser.add_argument("--summary-dir", default="experiments/workflow_selection_aux_baselines_20260502")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    activations = {x.strip() for x in args.activations.split(",") if x.strip()}
    seeds = {int(x.strip()) for x in args.seeds.split(",") if x.strip()}
    registry = [
        (act, task, seed, resolve_path(base))
        for act, task, seed, base in RUNS_1L
    ]
    registry.extend(load_extra_runs(args.extra_runs_csv))
    selected = [
        (act, task, seed, exp_dir)
        for act, task, seed, exp_dir in registry
        if act in activations and task in tasks and int(seed) in seeds
    ]
    if args.limit > 0:
        selected = selected[: args.limit]

    summary = []
    for idx, (act, task, seed, exp_dir) in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] {act}/{task}/s{seed}: {exp_dir}")
        out_path = exp_dir / f"workflow_selector_aux_baselines{args.output_suffix}.pkl"
        if args.skip_existing and out_path.exists():
            print(f"  [SKIP] {out_path}")
            summary.append({
                "act": act,
                "task": task,
                "seed": int(seed),
                "exp_dir": str(exp_dir),
                "output": str(out_path),
                "n_epochs": "",
                "n_prompts": "",
                "skipped_existing": True,
            })
            continue
        payload = compute_for_run(
            exp_dir,
            device=args.device,
            batch_size=int(args.batch_size),
            max_prompts=int(args.max_prompts) if args.max_prompts else None,
        )
        payload.update({"act": act, "task": task, "seed": int(seed), "exp_dir": str(exp_dir)})
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        summary.append({
            "act": act,
            "task": task,
            "seed": int(seed),
            "exp_dir": str(exp_dir),
            "output": str(out_path),
            "n_epochs": len(payload["epochs"]),
            "n_prompts": payload["n_prompts"],
        })

    out_dir = resolve_path(args.summary_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"runs": summary, "output_suffix": args.output_suffix}, f, indent=2)
    print(f"Saved summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
