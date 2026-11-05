# Paper result index

Choose an ID below and run `python scripts/reproduce.py --target ID --output outputs/ID`. Use a new output directory for each run. These commands work from the bundled results; the [reproduction guide](reproduction.md#what-is-recalculated) explains each calculation.

Forty targets are supported. The response-norm control remains incomplete: its historical summary is included, but the original per-response values and generating code are unavailable. See [coverage and limitations](known_gaps.md).

| ID | Paper result | Evidence level | Status |
|---|---|---|---|
| `fig01` | Path coherence and workflow overview | numeric recomputation | passed |
| `fig02` | depth trajectories | numeric recomputation | passed |
| `fig03` | checkpoint surgery | numeric recomputation | passed |
| `fig04` | probe positions | numeric recomputation | passed |
| `fig05` | reference trajectories | numeric recomputation | passed |
| `fig06` | probe dimension | numeric recomputation | passed |
| `fig07` | sample size | numeric recomputation | passed |
| `fig08` | neighbor count | numeric recomputation | passed |
| `fig09` | detached attention | numeric recomputation | passed |
| `fig10` | cohort mlp | numeric recomputation | passed |
| `fig11` | cohort attention | numeric recomputation | passed |
| `fig12` | uniform attention | numeric recomputation | passed |
| `fig13` | attention maps | numeric recomputation | passed |
| `fig14` | matched mlp | numeric recomputation | passed |
| `tab01` | diagnostics | numeric recomputation | passed |
| `tab02` | joint reference | numeric recomputation | passed |
| `tab03` | fixed mlp continuation | numeric recomputation | passed |
| `tab04` | workflow selection | numeric recomputation | passed |
| `tab05` | experiment configs | descriptive metadata | passed |
| `tab06` | primary saved ranges | numeric recomputation | passed |
| `tab07` | compute cost | descriptive metadata | passed |
| `tab08` | aggregation cohort | numeric recomputation | passed |
| `tab09` | trajectory by task | numeric recomputation | passed |
| `tab10` | null baselines | numeric recomputation | passed |
| `tab11` | selector replacement | numeric recomputation | passed |
| `tab12` | window example | numeric recomputation | passed |
| `tab13` | window sites | numeric recomputation | passed |
| `tab14` | fourier trend | numeric recomputation | passed |
| `tab15` | intervention summary appendix | numeric recomputation | passed |
| `app_scope` | Architecture and group scope pilots | aggregate per run metrics | passed |
| `app_endpoints` | Truncated trajectories and non-grokking controls | aggregate per run metrics | passed |
| `app_measurement` | Probe, SVD, Procrustes and detached-gradient checks | numeric recomputation | passed |
| `app_order` | Signed order of attention and MLP upturns | numeric recomputation | passed |
| `app_robustness` | Cluster bootstrap and leave-one-factor-out | numeric recomputation | passed |
| `app_selectors` | Six-seed selection and reference/geometry controls | aggregate per run metrics | passed |
| `app_interventions` | Recovery, donor, rescue-share and nine freeze arms | numeric recomputation | passed |
| `app_uniform_graphs` | Uniform edge-weight sweeps over 35 saved graphs | numeric recomputation | passed |
| `app_reference_variants` | Prompt-source, other-task, shuffled, earlier-layer and hybrid reference controls | aggregate per run metrics | passed |
| `gap_jvp_norms` | 19,200 response-norm audit | missing original inputs | incomplete |
| `gap_reference_sampling` | Reference-neighbor label enrichment | means from 11,490 saved per-prompt fractions; archived random baseline | passed |
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

The registry is tied to manuscript SHA-256 `68e89c480607d5b6c9e7497800e145ae7805f4136d9ca0ffbb4736a4518ebd1a`. Each reproduction report records that identity.
