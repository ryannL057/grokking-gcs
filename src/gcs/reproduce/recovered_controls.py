"""Cached checks of saved neighbor statistics and the eight-run attention control."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _runs(ref: Path, folder: str) -> list[dict]:
    data = json.loads((ref / "evidence" / folder / "records.json").read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("runs"), list):
        raise ValueError(f"Invalid recovered-record schema: {folder}")
    if not all(isinstance(row, dict) for row in data["runs"]):
        raise ValueError(f"Invalid run records: {folder}")
    return data["runs"]


def _series(values, name: str, *, length: int | None = None, fractions: bool = False) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric series: {name}") from error
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"Empty, nonfinite or non-vector series: {name}")
    if length is not None and len(array) != length:
        raise ValueError(f"Length mismatch: {name}")
    if fractions and ((array < 0).any() or (array > 1).any()):
        raise ValueError(f"Fractions outside [0, 1]: {name}")
    return array


def _epochs(values, name: str) -> np.ndarray:
    epochs = _series(values, name)
    if (epochs < 0).any() or not (epochs == np.floor(epochs)).all() or (np.diff(epochs) <= 0).any():
        raise ValueError(f"Epochs must be nonnegative integers in strictly increasing order: {name}")
    return epochs.astype(np.int64)


def _integer(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Expected a nonnegative integer: {name}")
    return value


def neighbor_enrichment(ref: Path) -> dict:
    """Recompute means from original per-prompt fractions; label the saved baseline."""
    runs = _runs(ref, "reference_neighbors")
    expected = {"ReLU", "GeLU", "SiLU"}
    seen = set()
    rows = []
    for run in runs:
        activation = run.get("activation")
        config = run["config"]
        if activation not in expected or activation in seen:
            raise ValueError("Missing, duplicate or unexpected neighbor-audit activation")
        seen.add(activation)
        if (config.get("act_type") != activation or config.get("task") != "add"
                or config.get("seed") != 777 or config.get("p") != 113 or config.get("fraction") != .3
                or run.get("sample_count") != 3830 or run.get("k") != 15):
            raise ValueError(f"Unexpected neighbor-audit configuration: {activation}")
        fractions = _series(run["same_answer_fractions"], activation, length=3830, fractions=True)
        counts = fractions * 15
        if not np.allclose(counts, np.round(counts), rtol=0, atol=1e-10):
            raise ValueError(f"Same-answer fractions must correspond to k=15 neighbors: {activation}")
        baseline = float(run["saved_random_baseline"])
        if not np.isfinite(baseline) or not 0 < baseline <= 1:
            raise ValueError(f"Invalid saved random baseline: {activation}")
        mean = float(fractions.mean())
        rows.append({"activation": activation, "seed": 777, "n": len(fractions), "k": 15,
                     "mean_same_answer_fraction": mean, "minimum_same_answer_fraction": float(fractions.min()),
                     "all_neighbors_same_answer_count": int((fractions == 1).sum()),
                     "saved_random_baseline": baseline, "enrichment_ratio": mean / baseline,
                     "source": run["source"]})
    if seen != expected:
        raise ValueError("Neighbor audit requires all three activations")
    return {"n_models": len(rows), "total_prompts": sum(r["n"] for r in rows),
            "measurement_evidence": "saved_per_prompt_same_answer_fractions",
            "baseline_evidence": "saved_scalar", "original_neighbor_indices_available": False,
            "original_random_draws_available": False,
            "scope": "Complete training-set kNN audit; distinct from the N=200 GCS graph.",
            "per_run": rows}


def uniform_factorial(ref: Path) -> dict:
    """Summarize actual saved schedules without reconstructing missing early logs."""
    expected = {(1., 1., 777), (1., .1, 777), (0., .1, 777), (0., 1., 777),
                (1., .1, 42), (1., .1, 123), (0., 1., 42), (0., 1., 123)}
    expected_without_gcs = {(1., .1, 42), (1., .1, 123)}
    seen = set()
    rows = []
    for run in _runs(ref, "uniform_factorial"):
        config = run["config"]
        key = (config.get("attention_rate"), config.get("weight_decay"), config.get("seed"))
        if key not in expected or key in seen:
            raise ValueError("Missing, duplicate or unexpected factorial condition")
        seen.add(key)
        alpha, wd, seed = key
        ident = f"a{alpha:g}_wd{wd:g}_s{seed}"
        if (run.get("id") != ident or config.get("p") != 113 or config.get("act_type") != "SiLU"
                or config.get("task") != "add" or config.get("num_layers") != 1):
            raise ValueError(f"Unexpected factorial configuration: {ident}")
        configured = _integer(config["total_epochs"], "configured epochs")
        checkpoint_epoch = _integer(run["last_saved_checkpoint"]["epoch"], "last checkpoint epoch")
        metrics = run["metrics"]
        epochs = _epochs(metrics["log_steps"], ident + " accuracy")
        accuracy = _series(metrics["test_accuracies"], ident + " test accuracy", length=len(epochs), fractions=True)
        _series(metrics["train_accuracies"], ident + " train accuracy", length=len(epochs), fractions=True)
        if epochs[-1] > checkpoint_epoch:
            raise ValueError(f"Accuracy log extends beyond the last saved checkpoint: {ident}")
        above = np.flatnonzero(accuracy >= .95)
        first_above = int(epochs[above[0]]) if len(above) else None
        complete_start = bool(epochs[0] == 0)
        if not len(above):
            threshold_status = "not_observed_in_saved_log" if complete_start else "not_observed_in_partial_log"
        elif not complete_start:
            threshold_status = "left_censored" if above[0] == 0 else "observed_in_partial_log"
        else:
            threshold_status = "observed_on_saved_grid"
        gcs_summary = None
        gcs = run.get("gcs")
        if gcs is None:
            if key not in expected_without_gcs:
                raise ValueError(f"Missing expected GCS trajectory: {ident}")
        else:
            if key in expected_without_gcs:
                raise ValueError(f"Unexpected GCS trajectory outside the recovered source set: {ident}")
            gcs_epochs = _epochs(gcs["epochs"], ident + " GCS")
            if gcs_epochs[-1] > checkpoint_epoch:
                raise ValueError(f"GCS extends beyond the last saved checkpoint: {ident}")
            if set(gcs["scores"]) != {"overall_path", "attention_path", "mlp_path"}:
                raise ValueError(f"Incomplete GCS paths: {ident}")
            paths = {}
            for path, values in gcs["scores"].items():
                values = _series(values, ident + " " + path, length=len(gcs_epochs))
                paths[path] = {"first": float(values[0]), "last": float(values[-1]),
                               "minimum": float(values.min()), "maximum": float(values.max()),
                               "range": float(np.ptp(values))}
            gcs_summary = {"n": len(gcs_epochs), "first_epoch": int(gcs_epochs[0]),
                           "last_epoch": int(gcs_epochs[-1]), "paths": paths}
        rows.append({"id": ident, "attention_rate": alpha, "weight_decay": wd, "seed": seed,
                     "configured_epochs": configured, "last_saved_checkpoint_epoch": checkpoint_epoch,
                     "continuation": run["continuation"],
                     "accuracy_log": {"n": len(epochs), "first_epoch": int(epochs[0]),
                                      "last_epoch": int(epochs[-1]), "starts_at_zero": complete_start},
                     "last_logged_test_accuracy": float(accuracy[-1]), "maximum_logged_test_accuracy": float(accuracy.max()),
                     "first_observed_ge_threshold_epoch": first_above,
                     "threshold_epoch": first_above if complete_start else None,
                     "threshold_status": threshold_status,
                     "gcs_summary_eligible": bool(len(above) and gcs_summary is not None),
                     "gcs": gcs_summary, "directory": run["directory"]})
    if seen != expected:
        raise ValueError("Factorial control requires all eight reported runs")
    cells = []
    for alpha, wd in sorted({(r["attention_rate"], r["weight_decay"]) for r in rows}):
        group = [r for r in rows if (r["attention_rate"], r["weight_decay"]) == (alpha, wd)]
        cells.append({"attention_rate": alpha, "weight_decay": wd, "n_runs": len(group),
                      "n_observed_ge_threshold": sum(r["first_observed_ge_threshold_epoch"] is not None for r in group),
                      "last_logged_accuracy_range": [min(r["last_logged_test_accuracy"] for r in group),
                                                     max(r["last_logged_test_accuracy"] for r in group)],
                      "n_partial_accuracy_logs": sum(not r["accuracy_log"]["starts_at_zero"] for r in group)})
    return {"n_runs": len(rows), "threshold": .95, "n_with_gcs": sum(r["gcs"] is not None for r in rows),
            "n_eligible_gcs_summary_runs": sum(r["gcs_summary_eligible"] for r in rows),
            "cohort": "Four seed-777 figure runs and four targeted extensions; unequal saved horizons.",
            "limitations": ["First threshold observations in partial logs are not full-run onset estimates.",
                            "The old onset-by-1600 statement is not verified for the two 5k uniform-attention extensions.",
                            "Last logged accuracy is not a new final-checkpoint evaluation. GCS and accuracy schedules remain separate."],
            "by_cell": cells, "per_run": rows}
