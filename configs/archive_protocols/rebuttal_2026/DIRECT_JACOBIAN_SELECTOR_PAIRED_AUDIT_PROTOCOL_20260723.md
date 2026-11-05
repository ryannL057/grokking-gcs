# Frozen Direct-Jacobian Selector Sensitivity and Paired Audit Protocol

Frozen on 2026-07-23 before executing this cached-score audit. This is a
post-result red-team sensitivity prompted by the concern that a direct
Jacobian trajectory may encode a rise, drop, or abrupt change rather than the
post-valley peak used by the submitted GCS selector. It is not described as an
external preregistration.

## Fixed scope

- Reuse the completed direct-Jacobian trajectories from
  `experiments/rebuttal_direct_jacobian_graph_20260723/`; do not recompute JVPs.
- Keep the exact submitted held-out cohort: 23 accuracy runs and 11
  commutative attention-equivariance runs from seeds 123/777.
- Audit only the direct-Jacobian **attention path**, because selector-rule
  fairness was raised for the pre-specified attention-to-attention comparison.
- Keep the cached score, `W=11` window budget, target construction, percentile
  definition, top-20 threshold, and all graph/projection choices unchanged.
- Evaluate all four rules below on held-out data and report all four. Do not
  select a winner by either held-out target.

## Four selector rules

The first rule is the submitted rule:

1. `post_valley_peak`: the exact three-point-smoothed
   first-prominent-post-valley selector used in the frozen main evaluation.

For the three change rules, first apply the same edge-padded three-point
smoothing. Let `h=5` be the half-width of the `W=11` window and restrict
candidate center indices to `h <= c < n-h`, so every selected window has the
full budget without boundary clipping. Define the incoming checkpoint slope

```text
delta_c = (s_c - s_{c-1}) / (epoch_c - epoch_{c-1}).
```

Break ties by the earliest center. The remaining rules are:

2. `largest_absolute_change`: maximize `abs(delta_c)`;
3. `largest_rise`: maximize `delta_c`;
4. `largest_drop`: maximize `-delta_c`.

The selected center for a rule depends only on the direct-Jacobian trajectory,
not on accuracy or equivariance. The same selected center is evaluated against
every available target for that run.

## Paired primary audit

Pair only the already-frozen `post_valley_peak` Attention-GCS and
direct-Jacobian-attention results within the same run.

- Define the percentile difference as
  `Attention-GCS percentile - Direct-J attention percentile`.
- Report the median paired difference and its 95% run-bootstrap interval using
  20,000 paired resamples with seed `20260723`.
- Also report the mean paired difference and the number of positive, tied, and
  negative run-level differences as descriptive checks.
- Define a top-20 hit as percentile `>=0.8`. Report the full paired 2x2 table:
  both hit, GCS-only hit, Direct-J-only hit, and neither hit.
- Apply the conventional two-sided exact McNemar test by conditioning on the
  discordant count and using an exact `Binomial(n, 0.5)` test.
- Keep accuracy (`n=23`) and attention equivariance (`n=11`) separate.

No unpaired test, target-wise rule selection, projection-seed search, window
change, or multiple-comparison claim will be introduced.

## Interpretation boundary

- If all three change rules remain below Attention-GCS, this addresses the
  narrow concern that the main result is solely an artifact of imposing a
  post-valley-peak rule on Direct-J.
- If a change rule materially recovers a target, report that rule and narrow
  the conclusion: the solution-frame construction supports the submitted
  interpretable non-monotone attention peak, while direct Jacobian changes can
  provide a different transition locator.
- Regardless of outcome, this sensitivity does not estimate the optimized
  performance of the best possible Direct-J method and does not establish
  uniform dominance, causal necessity, or generalization.
