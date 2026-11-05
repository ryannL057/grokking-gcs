# Frozen Near-Solved Endpoint Queue Protocol

Date frozen: 2026-07-24, after checkpoint-availability inspection and before
computing any new near-solved-reference GCS trajectory.

## Status and availability disclosure

The response's "35 one-layer runs" refers to the registered `RUNS_1L` cohort.
Before GCS computation, a checkpoint-only availability audit found:

- 24/35 runs contain a saved checkpoint whose logged test accuracy is in the
  inclusive interval \([0.90,0.95]\);
- 11/35 jump from below 0.90 to above 0.95 between adjacent saved checkpoints;
- the checkpoints do not contain optimizer state, so an exact intermediate
  training state cannot be reconstructed.

The primary estimand is therefore the 24-run exact-band cohort. It must never
be described as 35 runs. A secondary operational queue uses the first saved
checkpoint with accuracy at least 0.90 for every run; its 11 out-of-band
references must be identified and summarized separately.

The reference selection and inclusion flags were fixed using accuracy before
any new GCS result was viewed.

## Reference and trajectory construction

For every run:

1. Anchor the sample cohort to the exact 200 prompts stored in the submitted
   final geometry.
2. Use the selected checkpoint as the reference model.
3. Build the one-layer `hook_resid_post` reference geometry with
   \(N=200,k=15,d=8\) on those fixed prompts.
4. Analyze only the saved canonical GCS epochs at or before the reference
   epoch, including the reference checkpoint.
5. Compute the submitted overall, attention, and MLP paths with the original
   final-token probe, row-wise absolute-cosine alignment, and spectral-entropy
   aggregation.

No final checkpoint representation, neighbor graph, tangent basis, or
embedding is imported into the near-solved geometry. Only the ordered prompts
and frozen analysis conventions are shared.

The implementation may use the batch-equivalent JVP path only after a real
checkpoint agrees with the legacy implementation within \(10^{-10}\).

## Frozen endpoint-relative signature

The primary outcome is the same locator-like combined signature used in the
existing endpoint-boundary controls:

1. smooth each path trajectory with a three-point moving average;
2. normalize each path independently to \([0,1]\) over the available prefix;
3. define normalized time \(x=t/t_{\mathrm{reference}}\);
4. choose the attention maximum over \(x\in[0.30,0.95]\) and call it late if
   its location is greater than 0.45;
5. find the MLP minimum and its first subsequent recovery to at least 0.75;
   call recovery early if its location is at most 0.70;
6. call the combined signature present only when both the late-attention and
   early-MLP conditions hold.

Report the two component indicators and the combined indicator for every run.
Do not alter the thresholds after viewing the cohort.

As a paired diagnostic, truncate the existing solved-final-reference
trajectory to the identical epoch prefix and apply the same signature. Also
report pathwise Spearman correlations and signature agreement between the
near-solved and solved-final reference on that common prefix. This diagnostic
separates reference quality from the shorter observation horizon; it is not a
second primary outcome.

## Frozen reporting

Primary, exact-band cohort:

- exact reference epoch and logged train/test accuracy for all 24 runs;
- late-attention, early-MLP, and combined-signature counts/rates;
- exact two-sided Clopper-Pearson intervals for the three rates;
- task and activation breakdowns, labeled descriptive.

Secondary, full operational queue:

- the same counts for all 35 first-at-or-above-0.90 references;
- separate results for the 24 in-band and 11 overshoot references;
- the achieved reference-accuracy range and median.

## Interpretation branches

- A high exact-band combined rate supports graceful degradation to the tested
  near-solved range.
- Mixed component or combined rates support partial rather than graceful
  preservation.
- A low rate keeps the solved-endpoint requirement binding.

Under every branch, the exact-band availability denominator is 24, the
11 overshoot references are sensitivity cases, and the experiment does not
establish online prediction or behavior below 90% accuracy.

All response-facing values require an independent reconstruction of reference
selection, signature booleans, counts, and intervals.
