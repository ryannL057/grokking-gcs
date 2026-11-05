# Frozen Cohort-Level N/k/d Selector Sensitivity Protocol

Date frozen: 2026-07-24, before computing any cohort-level variant trajectory.

## Status and prior knowledge

The submitted response already reports single-run trajectory correlations for
several values of \(N\), \(k\), and \(d\). We also know the submitted
Attention-GCS held-out results and have inspected the earlier single-run
ablation artifacts. We have not inspected window centers, target percentiles,
or top-20 hits for the cohort-level variants below.

This is a discussion-preparation audit, not an external preregistration.

## Question and cohort

Does the submitted Attention-GCS window selector retain its held-out target
yield when one geometry hyperparameter is changed at a time?

- Use the exact submitted held-out one-layer cohort: the 23 registered
  seed-123/777 runs used by the frozen Direct-J audit.
- Expected denominators are 23 test-accuracy-rise runs and 11 commutative
  attention-equivariance-drop runs.
- The run is the unit of analysis. The 11 equivariance cases are a subset of
  the 23 runs, not additional computations.

## Frozen variants

The submitted default is \((N,k,d)=(200,15,8)\). Evaluate exactly five
one-factor-at-a-time variants:

| Variant | N | k | d |
|---|---:|---:|---:|
| `k10` | 200 | 10 | 8 |
| `k20` | 200 | 20 | 8 |
| `d4` | 200 | 15 | 4 |
| `d12` | 200 | 15 | 12 |
| `n100` | 100 | 15 | 8 |

No Cartesian-product or target-specific variant is allowed.

For `k10`, `k20`, `d4`, and `d12`, retain the exact ordered prompts,
final-checkpoint representations, and checkpoint-local embeddings stored in
each run's submitted `final_geometry.pkl`; rebuild only the k-NN graph and
local tangent bases required by the stated variant. For `n100`, take the first
100 entries in that canonical saved order and rebuild \(k=15,d=8\). Record
prompt, representation, neighbor, and parameter hashes.

## Frozen trajectory and selector

- Compute only the output-adjacent attention path, the path used by the
  submitted primary selector.
- Reuse each run's exact submitted GCS checkpoint schedule.
- Use the submitted row-wise absolute-cosine alignment and spectral-entropy
  aggregation.
- Use the frozen three-point smoothing, first-prominent-post-valley rule, and
  centered \(W=11\) window without retuning.
- Evaluate the same accuracy-rise and commutative equivariance-drop targets.
- A percentile is the fraction of all legal within-run windows whose target
  score is no greater than the selected score. A top-20 hit is percentile
  at least 0.8.

The implementation may batch independent samples within each tangent rank.
Before the formal variants, it must:

1. match the legacy per-sample JVP score on a real checkpoint within
   \(10^{-10}\); and
2. recompute a \((200,15,8)\) control on all 23 runs and exactly reproduce the
   submitted selector centers and target percentiles.

If the second gate fails, stop before interpreting the variants and reconcile
the code-path difference.

## Frozen reporting

For every variant and target report:

- run count;
- median percentile with a 20,000-resample run bootstrap interval;
- top-20 count and rate;
- mean percentile;
- median absolute center shift from the canonical selector.

Also report the within-run standard deviation and range across the five
variant percentiles, summarized across runs. Report every variant; do not
select a favorable \(N\), \(k\), or \(d\) after seeing a target.

## Interpretation branches

- If all variants preserve high median yield and most centers remain close,
  describe cohort-level selector stability over this local range.
- If target yield is stable but centers move, distinguish target-level
  robustness from exact-window instability.
- If one or more variants materially degrade, report the boundary by variant
  and target; do not average it away.

No branch supports hyperparameter invariance, optimality, or generalization
beyond this fixed small-model cohort.

All response-facing values require a separate reconstruction audit from raw
trajectories and frozen targets.
