# Reproduction coverage and limitations

The saved results support 40 figure, table and appendix-analysis targets. One additional control remains incomplete. The [result index](coverage.md) lists each target; the [reproduction guide](reproduction.md) explains what its command calculates.

## Response norms

The original 19,200 per-response values and inline generating code for the reported minimum norm of 0.135 have not been located. We include the unchanged [historical summary and provenance](../results/reference/evidence/response_norms/provenance.json), but cannot independently recompute its minimum or threshold counts from this release.

The summary records four model regimes, three checkpoints per regime, and fixed geometry with N=200, k=15 and d=8. Those twelve weights and four geometries remain in the research archive and are not bundled. The summary's checksum verifies the file's integrity; it does not verify the original calculation. Accordingly, `gap_jvp_norms` stays `incomplete`, and the full check reports 40 passed, one incomplete and zero failed.

## Controls with partial records

The neighbor-label records contain 3,830 per-prompt same-answer fractions for each of three addition models: ReLU, GeLU and SiLU, with seed 777 and k=15. All three recomputed means are 1.000. The random baseline of 0.0088 is a saved scalar; original neighbor indices and random draws were not retained. A deterministic replay of the baseline procedure matched that scalar. The bundled calculation uses the saved fractions and baseline.

The uniform-attention records cover four seed-777 figure runs and four targeted seed extensions. The learned-attention / wd=0.1 runs have last logged accuracies of 0.0157735765 for the 40k seed-777 run, and 0.0043629041 and 0.0166685312 for the seed-42/123 30k continuations. These agree with the manuscript's reported 0.004–0.017 range.

The two uniform-attention / wd=1 seed-42/123 extensions have checkpoints through epoch 4,999, accuracy entries only at 4,600 and 4,800, and 25 GCS points through 4,800. Their directory and config labels say 30k. Reports use the actual saved horizons and leave the threshold-crossing time null because the earlier accuracy log is missing. The historical onset-by-1,600 claim remains unverified. Both learned-attention extensions also lack early accuracy logs. A last logged accuracy is not a new evaluation of the final checkpoint.

The commands retain the IDs `gap_reference_sampling` and `gap_uniform_factorial` and return `passed` when these calculations succeed. [Data provenance](data.md#recovered-neighbor-and-uniform-attention-records) describes the original files and export boundaries. Figure 6/7 inputs contain all 33 original d/N curves; the prompt-source pilot includes all 36 trajectories. No new training was used to reconstruct these records.

## Interpreting counts and rounding

In the one-layer LayerNorm pilot, each path criterion has 10/11 positives, while their intersection has 9/11. In the shuffled-reference pilot, the corresponding counts are 10/12 per path and 8/12 jointly. Reports expose both per-path and joint counts.

The CKA window gain rounds to `0.811` when calculated from unrounded endpoints. The manuscript gives `0.810`, using its displayed Table 12 endpoints. The [reproduction guide](reproduction.md#reading-the-outputs) shows both calculations; the saved values have not been adjusted to match the rounded text.

## New training runs

The 104 primary configurations contain 87 saved configs and 17 reconstructions from model metadata, logs and Appendix A settings. Their source is recorded in `manifests/primary_runs.json`. Seven extended runs also retain the 20k fresh-optimizer restart. New trajectories can differ, particularly when a configuration or reference draw was reconstructed.

Full training, the complete intervention cohorts and every archived pilot protocol were not rerun for this release. The [experiment guide](experiments.md) lists their inputs and execution order. Some protocols check hashes of the original cohorts; new experiments need their own recorded manifests with the checks enabled.
