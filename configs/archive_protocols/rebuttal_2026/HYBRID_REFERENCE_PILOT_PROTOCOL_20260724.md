# Frozen Hybrid-Reference Pilot Protocol

Date frozen: 2026-07-24, before computing any hybrid trajectory.

Companion to `EARLIER_LAYER_REFERENCE_PILOT_PROTOCOL_20260724.md`, which
explicitly placed a hybrid reference outside its scope. This protocol closes
that gap and nothing else.

## Status and question

Exploratory discussion-preparation pilot, not a powered validation study. It
asks whether an output-adjacent path schedule remains legible when its
reference graph and local directions are built from a combination of the
target block and the preceding block, rather than from either alone.

The two existing endpoints of this comparison are already measured on the
identical cohort:

- target block \(L\) alone: the canonical submitted reference;
- preceding block \(L-1\) alone: the frozen earlier-layer pilot
  (median Spearman \(\rho\) 0.702 attention, 0.229 MLP, 0.104 overall).

This pilot fills in between them.

## Dimensional constraint (binding)

`analysis/batched_gcs.py` requires `tangent_bases` of shape
\([N, d, d_{\text{model}}]\) with \(d_{\text{model}}\) equal to the embedding
dimension, and rejects any mismatch. Feature-space concatenation of two
residual streams yields \(2 d_{\text{model}}\) and is therefore not a legal
construction. Both variants below are dimension preserving by design. No
projection, PCA, or learned reduction may be introduced to work around this,
because each would add a free parameter that this pilot cannot calibrate.

## Frozen cohort

The identical eight-run manifest used by the earlier-layer pilot:
`scripts/manifests/earlier_layer_reference_pilot_20260724.csv`, SHA-256
verified at run time. Four two-layer and four three-layer Transformers, with
add, mul, sub, and div once per depth. Prompts, target block, path
definitions, and checkpoint schedule are taken unchanged from each run's
canonical `final_geometry_layer{L}.pkl`. The run is the unit.

## Frozen variants

Let \(h_{L}\) and \(h_{L-1}\) be final-checkpoint `hook_resid_post`
representations at the prediction position, for the exact ordered canonical
prompts, \(N=200\).

**H1, midpoint representation.**
\(x = (h_{L-1} + h_{L}) / 2\). Both the \(k\)-NN graph and the tangent bases
are built from \(x\). No per-layer rescaling is applied: the two tensors
inhabit the same residual stream, and rescaling would introduce an
uncalibrated free parameter. Because
\(h_{L} = h_{L-1} + \text{attn}_L + \text{mlp}_L\), H1 is exactly the
reference that retains half of the target block's own contribution.

**H2, hybrid graph with endpoint tangents.**
Neighbors are chosen from the sum of the two pairwise Euclidean distance
matrices, each divided by its own median off-diagonal entry before summing.
Tangent bases are computed from \(h_{L}\) using those hybrid neighbor
indices. H2 isolates the question of which inputs count as neighbors from the
question of which local directions are probed.

\(N=200\), \(k=15\), \(d=8\), float64, exactly as in the canonical analysis.
Both variants are pre-registered and both will be reported, whatever they
show. No third variant may be added after seeing either result.

## Validation gates

1. The batch-equivalent JVP implementation must agree with the legacy score
   within \(10^{-10}\) on a real multilayer checkpoint.
2. Rebuilt embeddings must match the canonical anchor embedding table to
   within \(10^{-12}\); the serialized canonical prompts and embeddings are
   retained so that only the graph and tangent directions change.
3. The \(k\)-NN graph must be connected, and the component count is recorded.
4. Setting \(h_{L-1} = h_{L}\) in H1 must reproduce the canonical trajectory
   bitwise, and H2 with the distance matrix of \(h_L\) alone must likewise
   reproduce it. Both degenerate checks run before the pilot.

## Frozen diagnostics

Identical to the earlier-layer pilot, computed per run and per path against
the canonical trajectory on common epochs:

- Spearman correlation and normalized root-mean-square difference;
- canonical and hybrid range;
- strict attention peak-then-dip, MLP U-shape, and positive overall-path
  lead, each with agreement against the canonical reference, using the
  submitted detectors and the run's solved-reference \(t_{\mathrm{grok}}\);
- attention selector center shift under the unchanged three-point-smoothed
  first-prominent-post-valley rule with \(W=11\);
- accuracy-rise and task-equivariance target percentiles under the existing
  target definitions.

No detector, threshold, or summary statistic may be chosen after viewing any
hybrid result.

## Interpretation, fixed in advance

- If H1 and H2 both track the canonical trajectory closely, the honest
  reading is limited: H1 contains half of the target block by construction,
  so a high correlation is partly definitional and must be reported as such.
- If H2 tracks the canonical trajectory while the earlier-layer pilot did
  not, the informative content is that neighbor identity is more robust than
  the choice of tangent space.
- If both behave like the earlier-layer pilot, the conclusion is that
  detailed trajectories depend on the reference layer, and a hybrid does not
  repair it.
- A disorganized or mixed result is a valid negative pilot and is disclosed
  as exploratory.

No outcome supports coordinate invariance, absence of answer-class
conditioning, general layerwise routing, or any broadening of the paper's
claims. Eight runs cannot establish a depth effect.

## Reporting discipline

Response-facing summaries require a separate reconstruction audit, as for P1,
P2, and P3. The result is reported to Reviewer 5tW8 only if the hybrid
question is raised, or as a correction if any frozen response text claims the
hybrid reference is untested.
