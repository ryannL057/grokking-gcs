"""Guard recovered measurements, cohort completeness, and partial-log semantics."""
import importlib
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/reference"


def controls():
    try:
        return importlib.import_module("gcs.reproduce.recovered_controls")
    except ModuleNotFoundError:
        pytest.fail("Recovered control reducers have not been implemented")


def editable_records(tmp_path, folder):
    destination = tmp_path / "evidence" / folder
    shutil.copytree(REF / "evidence" / folder, destination)
    path = destination / "records.json"
    return path, json.loads(path.read_text())


def test_neighbor_claim_uses_all_three_saved_per_prompt_records():
    result = controls().neighbor_enrichment(REF)
    assert result["n_models"] == 3
    assert result["total_prompts"] == 11490
    assert {r["activation"]: r["mean_same_answer_fraction"] for r in result["per_run"]} == {
        "ReLU": 1.0, "GeLU": 1.0, "SiLU": 1.0}
    assert all(r["saved_random_baseline"] == .0088 for r in result["per_run"])
    assert all(r["enrichment_ratio"] == pytest.approx(113.63636363636363) for r in result["per_run"])
    assert result["baseline_evidence"] == "saved_scalar"
    assert result["original_neighbor_indices_available"] is False


def test_neighbor_mean_changes_with_saved_per_prompt_measurement(tmp_path):
    path, data = editable_records(tmp_path, "reference_neighbors")
    relu = next(r for r in data["runs"] if r["activation"] == "ReLU")
    relu["same_answer_fractions"][0] = 0.0
    path.write_text(json.dumps(data))
    rows = {r["activation"]: r for r in controls().neighbor_enrichment(tmp_path)["per_run"]}
    assert rows["ReLU"]["mean_same_answer_fraction"] == pytest.approx(3829 / 3830)
    assert rows["ReLU"]["all_neighbors_same_answer_count"] == 3829
    assert rows["SiLU"]["mean_same_answer_fraction"] == 1.0


@pytest.mark.parametrize("damage", ["missing", "duplicate", "truncated", "nonfinite", "out_of_range", "zero_baseline"])
def test_neighbor_rejects_invalid_or_incomplete_measurements(tmp_path, damage):
    path, data = editable_records(tmp_path, "reference_neighbors")
    runs = data["runs"]
    if damage == "missing":
        runs.pop()
    elif damage == "duplicate":
        runs[-1] = runs[0]
    elif damage == "truncated":
        runs[0]["same_answer_fractions"].pop()
    elif damage == "zero_baseline":
        runs[0]["saved_random_baseline"] = 0.0
    else:
        runs[0]["same_answer_fractions"][0] = float("nan") if damage == "nonfinite" else 1.1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        controls().neighbor_enrichment(tmp_path)


def test_uniform_controls_keep_real_horizons_and_censored_onsets():
    result = controls().uniform_factorial(REF)
    assert result["n_runs"] == 8
    assert result["n_with_gcs"] == 6
    rows = {r["id"]: r for r in result["per_run"]}
    for seed in (42, 123):
        row = rows[f"a0_wd1_s{seed}"]
        assert row["configured_epochs"] == 30000
        assert row["last_saved_checkpoint_epoch"] == 4999
        assert row["accuracy_log"] == {"n": 2, "first_epoch": 4600, "last_epoch": 4800, "starts_at_zero": False}
        assert row["gcs"]["n"] == 25
        assert row["gcs"]["first_epoch"] == 0
        assert row["gcs"]["last_epoch"] == 4800
        assert row["first_observed_ge_threshold_epoch"] == 4600
        assert row["threshold_epoch"] is None
        assert row["threshold_status"] == "left_censored"
    assert rows["a1_wd0.1_s777"]["last_logged_test_accuracy"] == pytest.approx(.015773576462691577)
    assert rows["a1_wd0.1_s42"]["last_logged_test_accuracy"] == pytest.approx(.004362904127978521)
    assert rows["a1_wd0.1_s123"]["last_logged_test_accuracy"] == pytest.approx(.016668531155610247)
    assert rows["a1_wd0.1_s42"]["threshold_status"] == "not_observed_in_partial_log"


def test_uniform_summary_recomputes_outcomes_from_accuracy_logs(tmp_path):
    path, data = editable_records(tmp_path, "uniform_factorial")
    row = next(r for r in data["runs"] if r["id"] == "a0_wd1_s42")
    row["metrics"]["test_accuracies"] = [.4, .5]
    path.write_text(json.dumps(data))
    result = controls().uniform_factorial(tmp_path)
    row = next(r for r in result["per_run"] if r["id"] == "a0_wd1_s42")
    assert row["last_logged_test_accuracy"] == .5
    assert row["first_observed_ge_threshold_epoch"] is None
    cell = next(c for c in result["by_cell"] if c["attention_rate"] == 0 and c["weight_decay"] == 1)
    assert cell["n_runs"] == 3
    assert cell["n_observed_ge_threshold"] == 2


@pytest.mark.parametrize("damage", ["missing", "duplicate", "unordered", "length", "nonfinite", "future_epoch", "missing_gcs"])
def test_uniform_rejects_incomplete_cohorts_and_malformed_curves(tmp_path, damage):
    path, data = editable_records(tmp_path, "uniform_factorial")
    runs = data["runs"]
    row = next(r for r in runs if r["id"] == "a0_wd1_s42")
    if damage == "missing":
        runs.pop()
    elif damage == "duplicate":
        runs[-1] = runs[0]
    elif damage == "unordered":
        row["metrics"]["log_steps"] = [4800, 4600]
    elif damage == "length":
        row["gcs"]["scores"]["attention_path"].pop()
    elif damage == "nonfinite":
        row["gcs"]["scores"]["attention_path"][0] = float("inf")
    elif damage == "future_epoch":
        row["metrics"]["log_steps"][-1] = 30000
    else:
        row["gcs"] = None
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        controls().uniform_factorial(tmp_path)


def test_cli_runs_recovered_controls_but_keeps_response_norms_incomplete(tmp_path):
    from gcs.reproduce.cli import main
    out = tmp_path / "controls"
    assert main(ROOT, ["--target", "gap_reference_sampling", "--target", "gap_uniform_factorial", "--output", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["complete"] is True
    assert {r["target"]: r["status"] for r in report["targets"]} == {
        "gap_reference_sampling": "passed", "gap_uniform_factorial": "passed"}
    assert json.loads((out / "gap_reference_sampling.json").read_text())["total_prompts"] == 11490
    assert json.loads((out / "gap_uniform_factorial.json").read_text())["n_runs"] == 8
    missing = tmp_path / "norms"
    assert main(ROOT, ["--target", "gap_jvp_norms", "--output", str(missing)]) == 1
    assert json.loads((missing / "report.json").read_text())["targets"][0]["status"] == "incomplete"


def test_cli_rejects_corruption_of_recovered_input(tmp_path):
    from gcs.reproduce.cli import main
    shutil.copytree(ROOT / "manifests", tmp_path / "manifests")
    destination = tmp_path / "results/reference/evidence/reference_neighbors"
    shutil.copytree(REF / "evidence/reference_neighbors", destination)
    path = destination / "records.json"
    data = json.loads(path.read_text())
    data["runs"][0]["same_answer_fractions"][0] = 0.0
    path.write_text(json.dumps(data))
    out = tmp_path / "outputs"
    assert main(tmp_path, ["--target", "gap_reference_sampling", "--output", str(out)]) == 1
    record = json.loads((out / "report.json").read_text())["targets"][0]
    assert record["status"] == "failed"
    assert "checksum mismatch" in record["error"]
    assert not (out / "gap_reference_sampling.json").exists()
