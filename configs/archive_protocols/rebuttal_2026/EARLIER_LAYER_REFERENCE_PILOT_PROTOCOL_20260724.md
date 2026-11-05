# Frozen Earlier-Layer Reference Pilot Protocol

Date frozen: 2026-07-24, before computing pilot trajectories.

## Status and question

This is an exploratory discussion-preparation pilot, not a powered validation
study. It asks whether an output-adjacent path schedule remains legible when
its reference graph and local directions are built in the preceding block's
final-checkpoint residual space rather than the output-adjacent block's space.

## Frozen cohort

- Eight successful registered runs: four two-layer and four three-layer
  Transformers.
- Within each depth, include add, mul, sub, and div exactly once.
- Fix the run manifest before trajectory computation, using final accuracy and
  artifact availability only, not any earlier-reference result.
- Prefer held-out seed-123/777 runs and balance activations across the eight
  cases as closely as the registered artifacts allow.

The run is the unit. Results are reported individually as well as in aggregate.

## Paired prompt and layer construction

For a model with output-adjacent target block \(L\):

- take the exact ordered prompts from the canonical
  `final_geometry_layer{L}.pkl`;
- at the same final checkpoint, extract
  `blocks.{L-1}.hook_resid_post` for those prompts;
- build a new \(N=200,k=15,d=8\) k-NN/tangent geometry from that preceding
  block output;
- compute overall, attention, and MLP GCS for target block \(L\), not block
  \(L-1\), over the exact canonical checkpoint schedule.

This is a preceding-block reference applied to output-adjacent paths. It is
not a layer-selection benchmark. Existing per-layer geometry files may not be
used directly because they were built from independently sampled prompts and
are not row-paired.

The comparator is the existing canonical output-adjacent-reference trajectory
on the same prompts, target block, paths, and epochs. The batch-equivalent JVP
implementation must first agree with the legacy score within \(10^{-10}\) on
a real multilayer checkpoint.

## Frozen diagnostics

For every run and path report:

- Spearman correlation with the canonical trajectory on common epochs;
- normalized root-mean-square trajectory difference;
- the canonical and preceding-reference range.

For the attention path, apply the unchanged three-point-smoothed
first-prominent-post-valley selector and \(W=11\) window. Report center shift
and any available accuracy-rise or task-equivariance target percentile using
the existing target definitions.

Using the submitted detectors and the run's solved-reference \(t_{\mathrm{grok}}\),
report:

- strict attention peak-then-dip;
- MLP U-shape;
- positive overall-path lead;
- agreement of each boolean with the canonical reference.

No detector or threshold may be selected after viewing the pilot.

## Interpretation

- Consistent trajectories and selector yield would motivate a larger
  preceding-layer-reference study.
- Mixed path- or depth-dependent results define where endpoint-layer choice
  matters.
- Disorganized results are a valid negative pilot and should be disclosed as
  exploratory.

No outcome supports coordinate invariance, absence of answer-class bias,
hybrid-reference validation, or general layerwise routing. A hybrid reference
is explicitly outside this pilot.

All response-facing summaries require a separate reconstruction audit.
