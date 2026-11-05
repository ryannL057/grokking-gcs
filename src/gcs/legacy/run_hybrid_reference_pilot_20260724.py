#!/usr/bin/env python3
"""Frozen hybrid-reference pilot: H1 midpoint and H2 hybrid graph.

Companion to ``run_earlier_layer_reference_pilot.py``, which measured the
preceding-block reference alone and placed a hybrid reference out of scope.
This driver reuses that script's manifest, provenance, legacy gate, run loop,
and audit surface unchanged, and replaces only the construction of the
reference k-NN graph and tangent bases.

Two dimension-preserving variants are frozen in
``rebuttal_2026/HYBRID_REFERENCE_PILOT_PROTOCOL_20260724.md``:

``h1_midpoint``
    ``x = (h_{L-1} + h_L) / 2``; graph and tangent bases both built from ``x``.

``h2_hybrid_graph``
    neighbours from the sum of the two pairwise Euclidean distance matrices,
    each divided by its own median off-diagonal entry; tangent bases built
    from ``h_L`` using those neighbours.

Feature-space concatenation is illegal here: ``analysis.batched_gcs`` requires
tangent bases of shape ``[N, d, d_model]`` with ``d_model`` equal to the
embedding dimension, and concatenation would double it.
"""

from __future__ import annotations

from gcs.runtime import data_root as _gcs_data_root

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = _gcs_data_root()
pass  # Package imports do not search the experiment-data directory.

import gcs.legacy.run_earlier_layer_reference_pilot as p3  # noqa: E402
from gcs.analysis.geometry import build_knn_graph  # noqa: E402
from gcs.analysis.reference_geometry import (  # noqa: E402
    _compute_tangent_bases,
    _validate_dimensions,
    count_knn_components,
    extract_block_outputs,
)

MODES = ("h1_midpoint", "h2_hybrid_graph")
DEFAULT_PROTOCOL = ROOT / "rebuttal_2026" / "HYBRID_REFERENCE_PILOT_PROTOCOL_20260724.md"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "rebuttal_hybrid_reference_pilot_20260724"
HOOK_POINT = "hook_resid_post"


def _extract(model: Any, prompts: torch.Tensor, layer_idx: int, device: str) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    return extract_block_outputs(
        model,
        prompts,
        layer_idx=int(layer_idx),
        device=device,
        hook_point=HOOK_POINT,
    ).to(dtype=dtype)


def _median_offdiag(distance: torch.Tensor) -> float:
    n = int(distance.shape[0])
    mask = ~torch.eye(n, dtype=torch.bool, device=distance.device)
    value = float(torch.median(distance[mask]).item())
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"Degenerate median off-diagonal distance: {value!r}")
    return value


def hybrid_knn(
    h_ref: torch.Tensor,
    h_tgt: torch.Tensor,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Neighbours from scale-normalised summed distances at both layers."""
    d_ref = torch.cdist(h_ref, h_ref, p=2)
    d_tgt = torch.cdist(h_tgt, h_tgt, p=2)
    scale_ref = _median_offdiag(d_ref)
    scale_tgt = _median_offdiag(d_tgt)
    combined = d_ref / scale_ref + d_tgt / scale_tgt
    n = int(combined.shape[0])
    combined = combined.clone()
    combined[torch.arange(n), torch.arange(n)] = float("inf")
    distances, indices = torch.topk(combined, k=int(k), dim=1, largest=False)
    return (
        indices.detach().cpu().numpy(),
        distances.detach().cpu().numpy(),
        {"median_offdiag_reference": scale_ref, "median_offdiag_target": scale_tgt},
    )


def build_hybrid_reference_geometry(
    final_model: Any,
    anchor_geometry: dict[str, Any],
    *,
    reference_layer: int,
    device: str,
    mode: str,
    force_degenerate: bool = False,
) -> tuple[dict[str, Any], float]:
    """Drop-in replacement for ``p3.build_paired_reference_geometry``."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
    model_dtype = next(final_model.parameters()).dtype
    if model_dtype != p3.FROZEN_DTYPE:
        raise RuntimeError(
            f"Frozen pilot requires {p3.FROZEN_DTYPE_NAME}, got {model_dtype}"
        )

    reference_layer = int(reference_layer)
    target_layer = reference_layer + 1
    prompts = anchor_geometry["X_prompts"].detach().to(device="cpu", dtype=torch.int64)
    selected = prompts[: p3.FROZEN_N].to(device=device)

    h_tgt = _extract(final_model, selected, target_layer, device)
    # The degenerate gate replaces the earlier layer with the target layer, so
    # both variants must collapse onto the canonical single-layer geometry.
    h_ref = h_tgt if force_degenerate else _extract(
        final_model, selected, reference_layer, device
    )

    d_model = int(h_tgt.shape[1])
    if int(h_ref.shape[1]) != d_model:
        raise RuntimeError(
            f"Residual widths differ: reference={int(h_ref.shape[1])}, target={d_model}"
        )
    _validate_dimensions(
        available_n=int(prompts.shape[0]),
        N=p3.FROZEN_N,
        k=p3.FROZEN_K,
        d=p3.FROZEN_D,
        d_model=d_model,
    )

    scales: dict[str, float] = {}
    if mode == "h1_midpoint":
        x_sample = 0.5 * (h_ref + h_tgt)
        knn_indices, knn_distances = build_knn_graph(x_sample, k=p3.FROZEN_K)
        knn_indices = np.asarray(knn_indices)
        knn_distances = np.asarray(knn_distances)
    else:
        x_sample = h_tgt
        knn_indices, knn_distances, scales = hybrid_knn(h_ref, h_tgt, k=p3.FROZEN_K)

    components = count_knn_components(knn_indices)
    if components != 1:
        raise RuntimeError(f"Hybrid k-NN graph is disconnected: components={components}")

    tangent_bases = _compute_tangent_bases(x_sample, knn_indices, p3.FROZEN_D)
    for basis in tangent_bases:
        if int(basis.shape[-1]) != d_model:
            raise RuntimeError(
                f"Tangent basis width {int(basis.shape[-1])} != d_model {d_model}"
            )

    with torch.no_grad():
        embedded = final_model.embed(selected.to(dtype=torch.int64)).detach()

    anchor_embeddings = [
        value.detach().to(device=device, dtype=model_dtype)
        for value in anchor_geometry["X_embeddings"]
    ]
    differences = [
        float(torch.max(torch.abs(rebuilt.to(device=device) - anchor)).item())
        for rebuilt, anchor in zip(list(embedded), anchor_embeddings)
    ]
    max_embedding_difference = max(differences, default=0.0)
    if max_embedding_difference > 1e-12:
        raise RuntimeError(
            "Anchor embeddings do not match the final checkpoint embedding table: "
            f"max_abs={max_embedding_difference:.3g}"
        )

    geometry = {
        "X_sample": x_sample,
        "X_prompts": selected,
        "knn_indices": knn_indices,
        "knn_distances": knn_distances,
        "N": p3.FROZEN_N,
        "k": p3.FROZEN_K,
        "d": p3.FROZEN_D,
        "d_model": d_model,
        # Retain the exact serialised canonical inputs so that the paired
        # comparison changes only the graph and the tangent directions.
        "X_embeddings": anchor_embeddings,
        "all_input_tangent_bases": tangent_bases,
        "layer_idx": reference_layer,
        "hook_point": HOOK_POINT,
        "model_type": "transformer",
        "embedding_type": None,
        "prompt_subset_rule": "caller_saved_order_prefix",
        "knn_components": components,
        "reference_construction": (
            f"hybrid:{mode}"
            f":blocks.{reference_layer}+blocks.{target_layer}.{HOOK_POINT}"
            + (":degenerate" if force_degenerate else "")
        ),
        "hybrid_mode": mode,
        "hybrid_reference_layer": reference_layer,
        "hybrid_target_layer": target_layer,
        "hybrid_distance_scales": scales,
        "anchor_embedding_max_abs_difference": max_embedding_difference,
    }
    return geometry, max_embedding_difference


def _tensor_max_abs_diff(left: Any, right: Any) -> float:
    a = torch.as_tensor(np.asarray(left), dtype=torch.float64)
    b = torch.as_tensor(np.asarray(right), dtype=torch.float64)
    if a.shape != b.shape:
        raise RuntimeError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float(torch.max(torch.abs(a - b)).item())


def run_degenerate_gates(
    rows: list[dict[str, Any]],
    *,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Both variants must collapse onto the canonical geometry when h_{L-1}=h_L."""
    # ``p3.run_one`` sets this before loading; the gate must match it or the
    # checkpoint materialises in float32 and the frozen dtype check trips.
    torch.set_default_dtype(p3.FROZEN_DTYPE)
    report: dict[str, Any] = {"schema_version": 1, "checks": []}
    row = rows[0]
    exp_dir = p3.resolve_path(row["exp_dir"])
    anchor_path = exp_dir / str(row["anchor_geometry"])
    anchor = p3.load_pickle_cpu(anchor_path)
    checkpoints = sorted((exp_dir / "checkpoints").glob("*.pth"), key=p3.checkpoint_epoch)
    model, _ = p3.load_checkpoint(checkpoints[-1], device=device)

    for mode in MODES:
        geometry, _ = build_hybrid_reference_geometry(
            model,
            anchor,
            reference_layer=int(row["reference_layer"]),
            device=device,
            mode=mode,
            force_degenerate=True,
        )
        x_diff = _tensor_max_abs_diff(
            geometry["X_sample"].detach().cpu(), anchor["X_sample"]
        )
        knn_diff = int(
            np.max(
                np.abs(
                    np.asarray(geometry["knn_indices"], dtype=np.int64)
                    - np.asarray(anchor["knn_indices"], dtype=np.int64)
                )
            )
        )
        tangent_diff = max(
            _tensor_max_abs_diff(rebuilt.detach().cpu(), stored)
            for rebuilt, stored in zip(
                geometry["all_input_tangent_bases"], anchor["all_input_tangent_bases"]
            )
        )
        check = {
            "mode": mode,
            "run_slug": p3.run_slug(row),
            "x_sample_max_abs_diff": x_diff,
            "knn_index_max_abs_diff": knn_diff,
            "tangent_basis_max_abs_diff": tangent_diff,
            "passed": bool(x_diff <= 1e-12 and knn_diff == 0 and tangent_diff <= 1e-10),
        }
        report["checks"].append(check)

    report["status"] = "PASS" if all(c["passed"] for c in report["checks"]) else "FAIL"
    p3.save_json_atomic(output_dir / "degenerate_gates.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--manifest", type=Path, default=p3.DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--gates-only", action="store_true")
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = p3.read_manifest(args.manifest)

    if not args.skip_gates:
        report = run_degenerate_gates(rows, device=args.device, output_dir=output_dir)
        print(json.dumps(report, indent=2))
        if report["status"] != "PASS":
            raise SystemExit("Degenerate gate failed; refusing to run the pilot.")
    if args.gates_only:
        return

    p3.build_paired_reference_geometry = (
        lambda final_model, anchor_geometry, *, reference_layer, device: (
            build_hybrid_reference_geometry(
                final_model,
                anchor_geometry,
                reference_layer=reference_layer,
                device=device,
                mode=args.mode,
            )
        )
    )
    p3.save_json_atomic(
        output_dir / "hybrid_mode.json",
        {
            "mode": args.mode,
            "protocol_path": str(args.protocol),
            "protocol_sha256": p3.sha256_file(args.protocol),
            "manifest_path": str(args.manifest),
            "manifest_sha256": p3.sha256_file(args.manifest),
            "driver": str(Path(__file__).resolve()),
            "driver_sha256": p3.sha256_file(Path(__file__).resolve()),
            "note": "reference geometry replaced; all other stages inherited from P3",
        },
    )

    argv = [
        "run_earlier_layer_reference_pilot.py",
        "--manifest", str(args.manifest),
        "--protocol", str(args.protocol),
        "--output-dir", str(output_dir),
        "--device", str(args.device),
    ]
    if args.max_checkpoints is not None:
        argv += ["--max-checkpoints", str(args.max_checkpoints)]
    if args.overwrite:
        argv.append("--overwrite")
    sys.argv = argv
    p3.main()


if __name__ == "__main__":
    main()
