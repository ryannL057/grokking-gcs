# Frozen Direct-Jacobian Graph Baseline Protocol

Frozen before the full computation on 2026-07-23. This protocol isolates the
Reviewer 2 question: whether a graph built directly from cross-input
path-Jacobian similarity performs as well as the submitted construction without
using final-solution neighborhoods or local SVD directions.

## Cohort and evaluation

- Use the exact submitted held-out selector cohort: seeds 123 and 777 from the
  registered 1-layer factorial runs.
- Expected evaluation denominators are 23 test-accuracy runs and 11
  commutative attention-equivariance runs.
- Reuse the exact checkpoint schedule and the same `N=200` sampled prompts
  stored with each run's submitted GCS geometry.
- Use all three submitted paths: attention output, MLP output, and full block
  output at layer 0.
- Use the frozen 11-checkpoint centered window budget, three-point smoothing,
  and first-prominent-post-valley selector from the submitted held-out
  benchmark.
- Report the same four headline quantities for the attention-path baseline:
  median accuracy-rise percentile, accuracy top-20% hit rate, median
  equivariance-drop percentile, and equivariance top-20% hit rate.

## Reference-free Jacobian sketch

At each checkpoint, embed the saved prompts with that checkpoint's own
embedding matrix. This avoids importing the final checkpoint representation
into the baseline. For every sample and path, compute

```text
z_i(t,p) = concat_l normalize(J_{t,p}(x_i) r_l)
```

with the following frozen choices:

- `d=8` shared random directions;
- directions are unit-normalized Gaussian vectors generated once with seed
  `20260723`;
- the same ordered directions are reused across every sample, checkpoint,
  path, and run;
- directions are injected at the final token, matching the submitted probe
  location;
- every individual JVP is normalized before concatenation.

## Graph and scalar aggregation

- Compute `S_ij = abs(cos(z_i, z_j))`.
- Exclude self-similarity when selecting neighbors.
- Retain the top `k=15` neighbors per node, then symmetrize by the union of the
  directed neighbor relations.
- Put similarity weights on retained edges, zero elsewhere, and keep unit
  self-loops.
- Apply the submitted spectral-entropy aggregation unchanged:
  `score = N - 2^H`, where `H` is the entropy of normalized absolute
  eigenvalues above the existing numerical threshold.

The resulting scalar is called the **direct-Jacobian graph score**, not GCS,
because it omits both final-solution neighborhoods and local tangent bases.

## Interpretation locked before results

- If clearly weaker than Attention-GCS on both targets, the final-solution
  neighborhoods/local directions add value under this protocol.
- If comparable, narrow the contribution to a broader cross-input path
  coherence family; do not claim the full construction is necessary.
- If mixed, report the target/path dependence and retain the complementarity
  framing.

No direction seed, graph budget, path definition, selector rule, smoothing, or
window length may be changed after inspecting the full results.
