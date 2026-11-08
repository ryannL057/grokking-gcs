# Paper result index

Choose an ID below and run `python scripts/reproduce.py --target ID --output outputs/ID`. Use a new output directory for each run. These commands work from the bundled results; the [reproduction guide](reproduction.md#what-is-recalculated) explains each calculation.

Forty targets are supported. The response-norm control remains incomplete: its historical summary is included, but the original per-response values and generating code are unavailable. See [coverage and limitations](known_gaps.md).

| ID | Paper result | Evidence level | Status |
|---|---|---|---|
| `fig01` | The score and its trajectories | numeric recomputation | passed |
| `fig02` | Attention and MLP trajectories across depths | numeric recomputation | passed |
| `fig03` | Accuracy after activation patching and parameter resets | numeric recomputation | passed |
| `fig04` | Changing the probed token position changes the measured response | numeric recomputation | passed |
| `fig05` | Coherence trajectories with three references from the same run | numeric recomputation | passed |
| `fig06` | Number of probe directions | numeric recomputation | passed |
| `fig07` | Number of sampled inputs | numeric recomputation | passed |
| `fig08` | Number of neighbors | numeric recomputation | passed |
| `fig09` | Removing the derivative through attention | numeric recomputation | passed |
| `fig10` | MLP trajectories by activation | numeric recomputation | passed |
| `fig11` | Attention trajectories by task | numeric recomputation | passed |
| `fig12` | MLP coherence recovery with learned or uniform attention | numeric recomputation | passed |
| `fig13` | Attention patterns at five checkpoints | numeric recomputation | passed |
| `fig14` | Parameter-count-matched MLP resets for each run | numeric recomputation | passed |
| `tab01` | Attention and MLP trajectory counts across model depths | numeric recomputation | passed |
| `tab02` | Trajectory counts under different references | numeric recomputation | passed |
| `tab03` | Continued learning with a fixed MLP | numeric recomputation | passed |
| `tab04` | Changes captured by selected windows on held-out runs | numeric recomputation | passed |
| `tab05` | Evidence sets, their scope and where they are reported | descriptive metadata | passed |
| `tab06` | Recorded ranges for the 104 primary runs | numeric recomputation | passed |
| `tab07` | Representative wall-clock times (single RTX 4090) | descriptive metadata | passed |
| `tab08` | Alternative aggregations across 35 one-layer runs | numeric recomputation | passed |
| `tab09` | Trajectory criteria by operation, pooled over one to three layers | numeric recomputation | passed |
| `tab10` | Null baselines for the 104-run schedule criteria (post hoc) | numeric recomputation | passed |
| `tab11` | Difference between attention and MLP reset costs for selected donors | numeric recomputation | passed |
| `tab12` | Three selected windows in the same addition run | numeric recomputation | passed |
| `tab13` | Operand-swap distances within the coherence window | numeric recomputation | passed |
| `tab14` | Attention GCS and restricted Fourier progress before grokking | numeric recomputation | passed |
| `tab15` | Full and partial resets in 17 addition/multiplication runs | numeric recomputation | passed |
| `app_scope` | Architecture and group scope pilots | aggregate per run metrics | passed |
| `app_endpoints` | Truncated trajectories and non-grokking controls | aggregate per run metrics | passed |
| `app_measurement` | Probe, SVD, Procrustes and detached-gradient checks | numeric recomputation | passed |
| `app_order` | Signed order of attention and MLP upturns | numeric recomputation | passed |
| `app_robustness` | Cluster bootstrap and leave-one-factor-out | numeric recomputation | passed |
| `app_selectors` | Six-seed selection and reference/geometry controls | aggregate per run metrics | passed |
| `app_interventions` | Recovery, alternative donors, reverse-patching gap closure and nine fixed-component arms | numeric recomputation | passed |
| `app_uniform_graphs` | Uniform edge-weight sweeps over 35 saved graphs | numeric recomputation | passed |
| `app_reference_variants` | Input-source, other-task, shuffled, earlier-layer and hybrid reference controls | aggregate per run metrics | passed |
| `gap_jvp_norms` | 19,200 response-norm audit | missing original inputs | incomplete |
| `gap_reference_sampling` | Reference-neighbor label enrichment | means from 11,490 saved per-input fractions; archived random baseline | passed |
| `gap_uniform_factorial` | Uniform-attention factorial control | eight runs; accuracy/GCS summaries with actual horizons and partial-log flags | passed |

The machine-readable [registry](../manifests/paper_map.json) gives input paths and commands. Table 7 uses the recorded experiment timings; it does not benchmark your machine. The following keys locate additional paper results in the output JSON:

| Reported claim | Target / JSON output key |
|---|---|
| Threshold-halving/doubling 88–97 matches | `app_robustness.threshold_sweep` |
| Procrustes 6/8 vs 0/8 pre-grok; 6/8 post-grok recovery | `app_measurement.procrustes_shape` |
| Six-seed 10k-bootstrap intervals | `app_selectors.six_seed_attention_ci` |
| 71-run Jaccard and absolute start gaps | `app_selectors.six_seed_proximity_windows` |
| NKd center shifts and bootstrap intervals | `app_selectors.geometry_sensitivity` |
| Grok-free circular-shift null rates | `app_selectors.grok_free.null_rates` |
| 36 planned endpoints, p97 counterexample and downstream sites | `app_interventions.task_algebra` |

Figure and table numbers, titles, pages and values follow the NeurIPS 2026 camera-ready PDF.
