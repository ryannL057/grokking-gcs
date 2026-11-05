# Frozen Mechanistic-Yield Selector Audit Protocol

Date frozen: 2026-07-23

## Status and disclosure

This is a post-submission, post-result audit that uses cached scores and cached
checkpoint-surgery outcomes only. It is not an external preregistration.

Before freezing this protocol, we had already inspected:

- the aggregate 17-run selected-donor surgery table for GCS, CKA, L2,
  final-representation CKA, equivariance, and one random donor;
- the all-window residual analysis relating GCS to attention-specific
  replacement cost; and
- the Direct-J transition-target selector sensitivity.

We had not inspected the common within-run mechanistic-yield percentiles,
top-20 hits, or GCS-versus-selector discordance tables specified below,
including the Direct-J and accuracy-oracle joins. No selector will be chosen
or removed after those outputs are viewed.

## Question

When every selector is evaluated by the same downstream intervention target,
which selectors nominate donors with high attention-specific replacement cost?

This audit tests mechanistic follow-up yield. It does not test whether any
selector uniquely discovers a mechanism, and it does not identify a minimal
circuit.

## Frozen cohort

The primary cohort is the exact 11-run held-out commutative cohort shared by:

- the cached all-window intervention grid;
- the submitted held-out selector trajectories; and
- the Direct-J attention trajectories.

It contains seeds 123/777 for one-layer addition and multiplication runs,
subject to the submitted successful-run inclusion rule. Development seed 42
is excluded from the primary audit. The script must fail unless the shared
cohort contains exactly 11 runs.

The run is the summary and pairing unit.

## Frozen intervention target

For legal donor checkpoint \(t\),

\[
M_t =
\Delta\mathrm{acc}_{\mathrm{reset\text{-}attn},t}
-
\Delta\mathrm{acc}_{\mathrm{reset\text{-}MLP},t}.
\]

Higher \(M_t\) means that replacing final attention with the donor attention
is more damaging than replacing the final MLP with the donor MLP under the
same final-base surgery protocol.

The primary donor for every window selector is its window center. Window
starts and ends will not be searched.

## Frozen selectors

All available selectors below are reported:

1. Attention-GCS, using the submitted post-valley-peak window rule.
2. Attention-output CKA, using its submitted proximity rule.
3. Attention-output L2, using its submitted proximity rule.
4. Final-representation CKA, using its submitted proximity rule.
5. Accuracy oracle: the center of the legal \(W=11\) window with the largest
   test-accuracy rise; earliest center breaks exact ties.
6. Attention-equivariance oracle: the center of the legal \(W=11\) window
   with the largest attention-equivariance improvement; earliest center
   breaks exact ties.
7. Direct-J attention under every previously frozen rule:
   post-valley peak, largest absolute change, largest rise, and largest drop.
8. Uniform-random legal donor expectation, computed exactly over all legal
   donors rather than from a favorable random draw.

The two oracle selectors are task- or outcome-informed upper references, not
deployable task-signal-free baselines. Direct-J rules are reported as a
sensitivity family; no Direct-J rule is selected using the intervention
target.

## Primary metrics

For each non-random selector and run:

1. selected \(M_t\);
2. within-run percentile
   \[
   P_t = |\{u:M_u\leq M_t\}| / |\{u\}|;
   \]
3. top-20 hit, defined as \(P_t\geq0.8\);
4. whether \(M_t>0\);
5. donor epoch minus \(t_{\mathrm{grok}}\);
6. donor test accuracy.

For every selector, report the median percentile, top-20 count/rate,
positive-\(M_t\) count/rate, median \(M_t\), median donor timing, and median
donor accuracy. Report a 20,000-resample run bootstrap interval for the median
percentile using seed 20260723.

For each non-GCS selector, report paired GCS-minus-selector percentile
differences and the top-20 table:

- both hit;
- GCS only;
- comparator only;
- neither.

Exact McNemar values are descriptive because many selector comparisons are
reported; no multiplicity-adjusted superiority claim is planned.

For the uniform-random reference, average the same quantities exactly over
all legal donors within each run, then summarize runs. It has no single donor
epoch and is excluded from paired donor comparisons.

## Interpretation branches

- If GCS has clearly higher yield than accuracy and proximity selectors, state
  the matched downstream advantage without claiming attention sufficiency.
- If GCS has comparable yield at a different stage, claim temporal or
  hypothesis complementarity, not unique recovery.
- If GCS is not stronger, state that the downstream intervention validates the
  GCS-motivated hypothesis but is not exclusively discoverable from GCS.

Under every branch, the Direct-J transition sensitivity remains binding:
the full solution-frame construction is not claimed necessary for generic
transition localization.

