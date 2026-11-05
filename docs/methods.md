# Numerical conventions

Models and GCS calculations use float64. Training uses full-batch AdamW. This page describes the reference graph, path definitions, timing rules and intervention budgets used in the paper. Original source records are in `manifests/code_provenance.json`.

The main modular tasks use modulus 113, three tokens `[a,b,=]`, a 30% training split, width 128, four heads, MLP width 512, learning rate 0.001, weight decay 1, and AdamW betas `(0.9,0.98)`. Division excludes zero divisors. Primary configurations span one to three layers and ReLU/GELU/SiLU. The finite manifest, rather than a complete Cartesian product, defines the 104 grok-observed runs.

## Geometry and paths

The standard reference has `N=200`, `k=15`, and `d=8`. Euclidean nearest neighbors and centered-neighborhood SVD define a fixed graph and local tangent bases. A graph edge exists if **either** directed neighbor relation exists; each unordered edge is evaluated once. The connectivity matrix is symmetric with unit diagonal.

The named attention and MLP paths are derivatives from the embedded input to their respective block outputs. The standard MLP path includes upstream attention; the detached-attention control is a separate analysis. The primary multi-layer diagnostic uses the last block. Named primary runs expose the measured last block in both canonical scores and `final_geometry.pkl`, while retaining all per-layer fields. Some archived legacy files have a layer-zero compatibility alias; use their explicit layer fields when interpreting those original files.

The main probe injects directions at the final token and reads the final-token output. Operand/all-token controls average the **JVP responses before computing alignment**. Alignment uses rank-matched local directions and absolute cosine agreement. Procrustes and rank-permuted variants are separate controls.

For eigenvalues `lambda` of the connectivity matrix, normalize `abs(lambda)` to probabilities `p`. The score is `N - 2**H(p)`, with base-two Shannon entropy. The implementation retains the original small-value thresholds. Mean edge coherence and participation rank are separate baselines.

## Timing and shape rules

The grok anchor is the epoch of the largest finite-difference test-accuracy increase, starting at the fourth record and ending at the first record attaining 99%. No smoothed anchor is substituted.

Attention rise/fall selects its peak in `[0.9,1.1] × t_grok` and requires both changes to be at least 5% of the full trajectory span. MLP fall/recovery uses the pre-grok trough, requires a fall of at least 10%, and a recovery **strictly greater than** 10% of the full span. Display smoothing and normalization do not enter these numerical decisions. Joint counts mean intersection of positives, not agreement of Boolean verdicts.

Regular checkpoints precede that iteration's update. The final checkpoint is saved after all updates and is labeled `total_epochs - 1`. This convention is preserved; treating the label as an update count can introduce an off-by-one budget error.

## Interventions and freeze budgets

Attention resets replace `W_Q/W_K/W_V/W_O`; MLP resets replace its input/output weights and biases. Other parameters stay at the host state. For the matched MLP null, the reported statistic is the mean over runs of each run's maximum drop among 32 candidate subsets.

For a saved final epoch `F` and late checkpoint `t_minus`, the continuation budget is `B=F+1-t_minus`. In `VV`, `VL`, `LV`, and `LL`, the first letter denotes the fixed MLP donor and the second the trainable host (`V`: trough, `L`: late). `0L/0V` use the true pre-update MLP initialization. `0V_long` permits `3B` updates, with the original extra-1,000-update stopping rule after observed success. `V_unfrozen` uses a fresh optimizer and the matched budget; `L_attn_frozen` freezes attention instead. Frozen parameters receive neither gradients nor AdamW decay.

The table uses observed 99% attainment within the allowed budget. A success after `B` is censored in the matched comparison, including archived unfrozen continuations that ran longer.

Seven primary trajectories contain an archived optimizer restart at update 20,000. Their manifest stages and the preserved `resume_training` entrypoint retain reseeding, fresh AdamW, and last-block-only reference reconstruction. The source batch script and equality of the 19999/20000 checkpoint tensors establish the boundary; a small staged regression checks that its result differs from uninterrupted AdamW.
