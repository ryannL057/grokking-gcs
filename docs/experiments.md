# Running experiments

This guide covers training runs, ablations and component interventions. For figures and tables from the saved results, start with the [reproduction guide](reproduction.md). Install the [model dependencies](../README.md#measure-gcs-and-reset-a-component) before running experiments.

## Train a paper configuration

```bash
python scripts/train.py --list
python scripts/train.py --run 1l_gelu_add_s42 --dry-run
python scripts/train.py --run 1l_gelu_add_s42 --device cuda:0 \
  --output data/experiments/gelu_1layer_add_p113_s42_20260301_041540
```

`--list` shows all 104 run IDs; `--dry-run` prints a run's configuration and checkpoint/GCS schedules. Named runs use the paper's integrated trainer, including post-training reference construction without an extra seed reset. In uninterrupted runs, multi-layer geometries are built in layer order and the canonical outputs measure the last block. The five-epoch `configs/demo.json` example provides a shorter execution check.

For cohort experiments, place each run under `data/` followed by its `historical_directory` in `manifests/primary_runs.json`. This snippet prints the commands for all runs:

```python
import json, shlex
from pathlib import Path
for run in json.loads(Path('manifests/primary_runs.json').read_text()):
    args = ['python', 'scripts/train.py', '--run', run['run_id'], '--device', 'cuda:0',
            '--output', str(Path('data') / run['historical_directory'])]
    print(shlex.join(args))
```

Seven extended runs train for 20k updates, then resume with fresh AdamW to 40k or 50k. Their `training_stages` preserve the post-update 19999 checkpoint and pre-update 20000 checkpoint. The resume phase reseeds and reconstructs the model before loading that state, then builds the final reference for the last block. Reproducing these runs requires the optimizer restart.

Use the recorded per-run horizons; a uniform 15k or 20k budget does not reproduce the cohort. The 17 reconstructed configurations are marked explicitly. Even with the same settings, new trajectories can change event selections and eligibility. Report their new outcomes instead of forcing archived donors or grok times onto them.

## Ablations and interventions

The experiment launcher lists 43 scripts. Select a script and use `--help` to see its arguments:

```bash
python scripts/experiment.py --list
python scripts/experiment.py --data-root data checkpoint_surgery_modular --help
```

Unless a command's help says otherwise, its input paths are relative to the data root. Use absolute paths for files elsewhere. The table lists the inputs and execution order for each study.

| Study | Entrypoints, in dependency order | Inputs / current-paper choices |
|---|---|---|
| Primary trajectory | `train.py --run`; `recompute_gcs_existing_geometry` | Weights, saved geometry, exact recorded epoch schedule |
| Geometry N/d/k | `ablation_n`, `ablation_d`, `ablation_k` | Trained run; use `reproduce.py --target fig06 --target fig07` to replot the recovered historical arrays |
| Reference epoch / seed / task | `ablation_reference`, `cross_seed_reference`, `evaluate_task_reference_ablation` | Explicit reference epochs or paired trained runs |
| Shuffled / answer graph | `run_shuffled_reference_control`, `run_answer_only_graph_control`, `analyze_reference_knn_structure` | Trained runs, reference inputs, frozen cohort manifest |
| Earlier-layer / hybrid | `run_earlier_layer_reference_pilot`, `run_hybrid_reference_pilot_20260724`, `evaluate_earlier_layer_reference_pilot` | Eight-run manifest and original protocol; preserve provenance checks |
| Path and metric controls | `analyze_path_purity_controls`, `gcs_method_sanity_checks`, `recompute_gcs_aggregation_baselines` | Saved geometry and checkpoint trajectories; Procrustes uses the reviewed replicate-8 inputs |
| Window selection | `measure_equivariance`, `compute_workflow_selector_aux_baselines`, `evaluate_workflow_selection_heldout` | Primary/extension weights and event caches; use 11-checkpoint windows |
| Direct Jacobian / GrokAlign | `run_direct_jacobian_graph_baseline`, `evaluate_direct_jacobian_selector_sensitivity`, `systematic_grokalign_comparison --cache-only`, `evaluate_grokalign_selector_benchmark` | Direct-J trajectories / auxiliary metric caches / frozen selector protocol |
| NKd / grok-free timing | `run_gcs_nkd_selector_sensitivity`, `run_method_neutral_order_check_v2_20260726` | 23-run Direct-J manifest, geometry variants, auxiliary trajectories |
| Reset / submodule / matched | `checkpoint_surgery_modular` | Final hosts and selected donors; 32 neuron subsets, eight closest-norm subsets |
| Task algebra / sitewise symmetry | `measure_equivariance`, `analyze_swap_symmetry_sites`, `sitewise_temporal_symmetry_acquisition` | All 36 planned endpoints; explicit p97 checkpoint for the counterexample; 17-run temporal sites |
| Activation patch | `activation_patch_commutative` | Commutative 17-run cohort, matching donor events |
| Alternative donors / recovery | `run_selected_donor_surgery`, `continuation_surgery_recovery` | Four donor rules; include budgets 0,1,5,10,20,50,100,200,500,1000 |
| Mechanistic benchmark | `run_all_window_intervention_yield`, `run_rebuttal_donor_yield_surgery`, `analyze_gcs_dynamic_proximity_controls` | Legal all-window donors, CKA/L2/GCS and intervention rows |
| Fourier / single-run example | `compare_gcs_nanda_measures`, `circuit_case_study_addition` | Five specified addition runs; per-checkpoint dynamic frequencies |
| Truncated endpoints | `select_near_solved_endpoints_20260724`, `run_near_solved_endpoint_queue_20260724`, `evaluate_near_solved_endpoint_queue_20260724` | Generate and verify selection manifest before measurement |
| LayerNorm / group / attention-rate pilots | `train_transformer`, `train_group_comp` | Set `--use-ln`, `--group`, or `--attention-rate` explicitly; pilot cohorts remain separate |

Small frozen protocol inputs are under `configs/archive_protocols`. Copy their contents into an empty data root to obtain the original relative layout:

```bash
mkdir -p data
cp -Rn configs/archive_protocols/. data/
```

Keep the NKd cohort's `run_manifest.csv` and adjacent `run_manifest.paths.json` together. The CSV retains its original bytes; the JSON records its original data root and SHA-256 so the loader can relocate the experiment paths to your data root. A changed CSV is rejected. For new experiments, write a separate cohort CSV with `exp_dir` paths relative to your data root and pass it with `--cohort-manifest`; do not reuse the archived path metadata for a different CSV.

The earlier-layer and Direct-J manifests contain hashes of the original experiment files. New training will produce different files, so record a new manifest and protocol for those runs. Keep the hash checks enabled to catch accidental mixing of experiments. The input table above lists what each script needs; the repository includes only the small protocol files and example weights.

Examples after the required weights and upstream measurements exist:

```bash
python scripts/experiment.py --data-root data evaluate_workflow_selection_heldout \
  --window-lens 11 --output-dir experiments/workflow_reproduction
python scripts/experiment.py --data-root data continuation_surgery_recovery \
  --device cuda:0 --budgets 0,1,5,10,20,50,100,200,500,1000 \
  --output-dir experiments/recovery_reproduction
python scripts/experiment.py --data-root data compare_gcs_nanda_measures \
  --experiment checkpoints/1layer/SiLU/add --device cuda:0 \
  --checkpoint-freq 200 --output-subdir nanda_progress_comparison --skip-plots
```

The full training and intervention cohorts were not rerun for this release, and full-run GPU time has not been measured here. The [verification record](../verification/REPORT.md) lists the checks completed on small examples.

## Fixed-component continuation

```bash
python scripts/freeze.py --arm VL \
  --trough examples/checkpoints/trough.pth --late examples/checkpoints/t_minus.pth \
  --final-epoch 19999 --t-minus 7600 --config examples/checkpoints/config.json \
  --device cuda:0 --output outputs/freeze-VL
```

This command runs 12,400 updates. Other arms are `VV`, `LV`, `LL`, `0L`, `0V`, `0V_long`, `V_unfrozen`, and `L_attn_frozen`. Initialization arms require `--initial` with true pre-update weights. The output records the matched budget and frozen component; each branch starts with a fresh optimizer.

For the GrokAlign table component, run `python scripts/experiment.py --data-root data systematic_grokalign_comparison --cache-only --device cuda:0` before `evaluate_grokalign_selector_benchmark`. It writes `data/experiments/grokalign_systematic/grokalign_all_runs.pkl` in the evaluator's expected schema. The adapter fails if a requested trajectory is missing or partial. The N ablation now builds one final-model reference per N, saves it as `geometry_N<N>.npz`, and reuses it across all checkpoints. Standalone legacy protocols explicitly use float64.
