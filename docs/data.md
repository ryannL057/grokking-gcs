# Data and checkpoints

Modular-arithmetic inputs and labels are generated from the chosen operation and modulus. Finite-group tasks use the included group utilities. Neither requires an external dataset download.

Saved results are in `results/reference/` (about 28 MiB). The three example checkpoints, geometry and metadata are in `examples/checkpoints/` (about 7.24 MiB). The [artifact manifest](../manifests/artifacts.json) records their exact sizes and SHA-256 checksums. Run `python scripts/verify_artifacts.py` to check the bundle.

Numerical results and example weights use CC BY 4.0. Original code, configs and documentation use MIT, including `examples/checkpoints/config.json`. See [license scope](../LICENSING.md) and the asset-directory notices.

Write new experiments under `data/` and generated figures or reports under `outputs/`. These directories, virtual environments and new checkpoints are ignored by Git.

## Experiment cohorts

| Cohort | Membership source | Use |
|---|---|---|
| Primary 104 | `manifests/primary_runs.json` | One/two/three-layer signature counts; actual observed horizons and GCS schedules |
| One-layer 35 | Primary records filtered to depth 1 | Reference and aggregation controls |
| Commutative 17 | `primary/surgery_stats.csv` and intervention per-run records | Resets, patching, recovery, nine freeze arms |
| Held-out workflow 23/11 | `evidence/workflow/joined.csv.gz` | All eligible runs / commutative subset; seed 42 tuning is excluded |
| Extended workflow 71/35 | `evidence/workflow/six_seed.csv.gz` | Six-seed all / commutative held-out cohorts |
| Mechanistic 11 | `evidence/mechanistic/windows.csv` | All-window donor replacement benchmark |
| Fourier 5 | `evidence/fourier/0..4` | Includes the extended SiLU/add/seed-123 run, which is not a primary/intervention run |
| Prompt-source pilot 12 × 3 | `evidence/prompt_sources` | Complete train/test/full-grid GCS trajectories, with per-run accuracy logs for the grok anchors |
| Neighbor-label audit 3 | `evidence/reference_neighbors` | ReLU/GeLU/SiLU addition seed 777; 3,830 per-prompt fractions per model and saved random baselines |
| Uniform-attention control 8 | `evidence/uniform_factorial` | Four seed-777 figure runs plus four targeted extensions; actual accuracy/GCS schedules and continuation metadata |
| SiLU dimension/sample-size sweeps | `evidence/ablations` | Original Figure 6/7 trajectories, with six dimension settings and five sample sizes |
| Architecture/group pilots | `evidence/narrative` | Separate LayerNorm, S5, cyclic, and dihedral cohorts |

Paths in this table are relative to `results/reference` unless starting with `manifests`. Use each cohort's recorded membership when comparing results; a shared seed or task name does not imply a shared cohort. The 17-run freeze experiment has 153 raw records, one per arm and run.

## Source records

`export_provenance.json` maps curated files to original archive paths and hashes. `primary_runs.json` records saved versus reconstructed configuration sources, observed checkpoint epochs, GCS epochs, and the measured block. `code_provenance.json` maps retained scientific modules to the original source hashes. Historical path strings inside raw records are provenance labels; cached reproduction never resolves them against the author's workspace.

Most CSV, JSON and NPZ files are copies of the original results. Training accuracy logs were exported to `primary/training_metrics.csv.gz`, and the example geometry was converted from pickle to numeric NPZ without changing its arrays. `svd_provenance.json` records the neighborhoods used for the independent SVD check. The checkpoint example's expected values come from the original scores and intervention record.

The prompt-source inputs contain all 36 original trajectories and accuracy fields from 12 training logs. Their provenance file identifies the original inputs and generating scripts by hash. The reducer recomputes the grok anchors and path counts; historical directory strings join the exported records.

`configs/archive_protocols` preserves small original protocol documents and cohort manifests at their historical relative paths. The NKd manifest has an added `run_manifest.paths.json` that binds its original root to the unchanged CSV's SHA-256; the loader uses this metadata to relocate paths. These documents may describe archived experiments outside the verified execution set; they do not broaden this repository's verification claims.

## Recovered dimension and sample-size sweeps

Figures 6 and 7 use lossless JSON exports from `pretrained/SiLU/silu_20251103_212053/` in the historical `ryannL057/geometric-grokking-rebuild` repository, commit `2518b6cbaac03f0de3a85a9f140baba3fcde2dee`. The historical repository is private; the bundled exports require no GitHub access. [The provenance record](../results/reference/evidence/ablations/provenance.json) records original Git blob IDs, SHA-256 values, source paths, conversion checks, and sample-size settings.

| Figure | Parameter values | Saved points per path | Last epoch |
|---|---|---|---|
| 6 | d = 2, 4, 6, 10, 12, 14 | 75 each | 14800 |
| 7 | N = 100, 400, 750 | 69 each | 13600 |
| 7 | N = 200 | 76 | 14999 |
| 7 | N = 1850 | 75 | 14800 |

All 2,424 GCS values match their original pickle values after JSON conversion. The dimension results also match the union of three original GPU shards; the N=100/400/750 curves match the original merged file, and N=200 matches the final-geometry training record. N=100/400/750 retain the original four-decimal scores. There is no d=8 curve in the Figure 6 input. Each path and parameter setting is independently scaled by its own minimum and maximum, without extending shorter trajectories. The original pickle files remain in the authors' archive; the cached commands read only the bundled JSON.

## Recovered neighbor and uniform-attention records

The [neighbor records](../results/reference/evidence/reference_neighbors/records.json) contain all 11,490 original per-prompt same-answer fractions, their configs, the saved random baseline and original per-epoch summaries. Means and enrichment ratios are recomputed; the baseline is explicitly labeled as a saved scalar. The [provenance record](../results/reference/evidence/reference_neighbors/provenance.json) identifies the three source PKLs and the original generator. No neighbor indices or old random draws are represented as recovered.

The [uniform-attention records](../results/reference/evidence/uniform_factorial/records.json) preserve the eight original configs, selected accuracy fields, continuation metadata, all six available GCS trajectories and final saved checkpoint identities. The [provenance record](../results/reference/evidence/uniform_factorial/provenance.json) gives the original plotting and cohort specifications, source hashes, inclusion rules and missing-log boundaries. This is four seed-777 figure runs plus four targeted extensions, not a balanced twelve-run design. No extra checkpoint weights are bundled.

The uniform/wd=1 seed-42/123 configs specify 30k epochs, while their continuations target 5k and final saved checkpoints are at 4,999. Their two saved accuracy entries and 25-point GCS schedules remain separate. The reducer reports partial-log threshold observations with null full-run onset estimates. It does not interpolate missing points, extend trajectories, or use the configured horizon as the observed horizon. Only runs with observed test accuracy at least 0.95 are eligible for the original grok-only GCS summaries; the seed-777 low-wd failure's saved GCS remains available as an illustration.

The four JSON files preserve the selected original fields and their provenance. The original PKLs remain in the research archive; the reproduction commands read the bundled JSON.

## Historical response-norm summary

The [response-norm summary](../results/reference/evidence/response_norms/historical_summary.json) is a byte-for-byte copy of the archived report for four model regimes at three checkpoints each. It records fixed geometry with N=200/k=15/d=8 and the historical aggregate of 19,200 responses, minimum 0.135 and threshold counts. The [provenance record](../results/reference/evidence/response_norms/provenance.json) identifies the source and its checksum.

The original per-response arrays and inline dynamic execution code have not been located. This release preserves the reported aggregate without independently recomputing it. File-integrity verification does not establish numerical reproducibility; `gap_jvp_norms` remains incomplete. Source paths describe the original workspace, and the twelve weights and four geometries are not bundled here. This auxiliary control is a documented limitation of the first release.

## Adding new results

Keep `results/reference/` unchanged during normal use. To publish revised reference results, record their sources and cohort membership, update the artifact manifest, rerun the affected calculations, and retain the previous release. Checksum updates alone do not validate new scientific content.

Larger checkpoint studies require training the relevant configurations or supplying their weights. The included example runs with the bundled files and does not download additional weights.

## Manuscript version

The [paper map](../manifests/paper_map.json) records manuscript SHA-256 `68e89c480607d5b6c9e7497800e145ae7805f4136d9ca0ffbb4736a4518ebd1a`, the text revision dated 5 November 2026. Its figure files, numerical inputs, and all 29 figure/table numbers and page references match the preceding final-round revision. Every reproduction report records the manuscript identity used for that run.
