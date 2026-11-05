# Reproducing paper results

Run these commands from the repository root after following the [installation instructions](../README.md#installation). Write each run to a new output directory so that earlier results are preserved.

## Choose a figure or table

```bash
python scripts/reproduce.py --list
python scripts/reproduce.py --target fig01 --target tab01 --output outputs/figure-table-check
```

Repeat `--target` to select several results. The [result index](coverage.md) lists every figure, table and appendix analysis. Inputs are read from `results/reference/` after their sizes and SHA-256 checksums are checked; `scripts/verify_artifacts.py` also checks the configs and example weights.

To check every registered result:

```bash
python scripts/reproduce.py --all --output outputs/paper
```

The expected outcome is **40 passed, one incomplete**. The response-norm control lacks its original inputs, so this full check returns exit code `1`. Commands selecting only supported targets return `0` when successful. See [coverage and limitations](known_gaps.md) for the missing material.

## What is recalculated

These commands use saved trajectories, measurements or metadata. The calculation depends on the result:

| Saved input | Calculation |
|---|---|
| Primary trajectories | Recomputes the grok anchor from accuracy logs and shape rules from GCS trajectories for all 104 runs |
| Figures 1/2 | Recomputes medians/IQR from individual curves and checks them against saved summaries |
| Figures 6/7 | Recomputes per-curve min-max scaling from all 33 recovered trajectories, preserving their individual epoch grids |
| Other supported figures | Replots the current manuscript's saved numerical arrays; relevant stacks/normalizations are recomputed |
| Freeze experiment | Recomputes nine-arm successes from 153 raw branch logs, censoring at each arm's specified budget |
| Mechanistic selection | Reselects cached donors and reruns the 20,000-sample bootstrap |
| Threshold / Procrustes shape | Recomputes the nine amplitude-threshold combinations from 104 trajectories and matched-epoch Procrustes decisions from eight runs |
| Selector uncertainty | Reruns the 10,000-resample six-seed attention intervals and 20,000-resample NKd intervals; recomputes paired window gaps/Jaccards and median null rates |
| Task algebra | Aggregates all 36 planned endpoint measurements and the p97 reset/sitewise counterexample |
| Fourier control | Recomputes level, difference, and time-adjusted correlations from five trajectories |
| Null controls | Aggregates 1,000 saved null counts; does not rerun every IAAFT draw |
| Other reference/scope pilots | Aggregates saved per-run measurements and flags; does not recompute every Jacobian |
| Prompt-source pilot | Recomputes grok anchors and path counts from 36 recovered trajectories and 12 training-accuracy logs in `app_reference_variants` |
| Neighbor-label audit | Recomputes means and enrichment ratios from 11,490 original per-prompt fractions; uses the labeled saved random baseline |
| Uniform-attention control | Summarizes eight runs from their actual accuracy and GCS logs; exposes different horizons and missing early logs |
| Tables 5/7 | Exports recorded configuration and timing metadata; does not benchmark the current machine |

The order statistic is recomputed from all 104 trajectories. Its order-null comparison uses the saved summary because individual draws were not retained. Reference-variant reports give each path's positive count and their intersection separately.

## Additional controls

Replot the dimension and sample-size sweeps with `python scripts/reproduce.py --target fig06 --target fig07 --output outputs/ablations`. Each figure is saved as PDF, PNG and SVG. The original arrays and conversion records are bundled in `results/reference/evidence/ablations`.

Check the neighbor-label and uniform-attention records with:

```bash
python scripts/reproduce.py --target gap_reference_sampling \
  --target gap_uniform_factorial --output outputs/recovered-controls
```

Both commands return exit code `0` when successful; the `gap_` names are retained for compatibility. The neighbor report separates recomputed per-prompt means from the saved random baseline. The uniform-attention report lists each run's configured horizon, last checkpoint, accuracy and GCS intervals, and threshold observations. For logs that begin above 0.95, `threshold_status` is `left_censored` and `threshold_epoch` is null: the original crossing time is unknown.

## Reading the outputs

Table 12 exports unrounded accuracy endpoints. For the CKA window, `0.9423873424530028 - 0.1315583437681198 = 0.810828998684883`, or `0.811` to three decimals. The current manuscript reports `0.810`, the difference between its displayed endpoints `0.942` and `0.132`. Compute differences before rounding when using the exported values; the archived endpoints are unchanged.

Each run writes `report.json` with per-target status, output hashes and elapsed time. `passed` means the calculation listed above completed successfully. Missing inputs are marked `incomplete`; corrupt inputs or runtime errors are marked `failed`. Either gives a nonzero process exit code. Model analysis and new training are separate steps, described below and in the [experiment guide](experiments.md).

## Analyze a checkpoint

The [checkpoint example](../README.md#measure-gcs-and-reset-a-component) reuses the saved reference inputs, graph and tangent directions. Compare its output with [expected.json](../examples/checkpoints/expected.json): the absolute tolerances are `1e-5` for GCS and `1e-12` for accuracy. The model and reference use float64. Device and library versions can cause small numerical differences.

Use `analyze.py --reference CHECKPOINT` to build a new reference with a fixed sampling seed. To compare with the archived scores, use `--geometry` with the saved reference. Named training runs (`train.py --run`) construct their reference as part of the original training protocol. Load `.pth` and legacy pickle files only from sources you trust.

## Run the tests

```bash
python -m pip install -r requirements/test.txt
python -m pytest -q
```

The tests cover small numerical examples, batched/scalar agreement, input integrity, paper counts, output paths, and training, reset and freeze behavior.

Figures use Matplotlib's noninteractive Agg backend and portable fonts. Set `GCS_FIGURE_FONT_DIR` to use matching fonts you have permission to use. Typography can differ across machines; the numerical checks are independent of it.

The Procrustes post-grok recovery count uses the first saved value exceeding the global trough by strictly more than 10% of the full span, after a fall of at least 10%; its epoch must follow the grok anchor. This descriptive full-trajectory check is separate from the pre-grok shape criterion.
