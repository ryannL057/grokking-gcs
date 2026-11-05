# Source and distribution

The repository contains the scientific code, experiment configurations, saved numerical results and three example checkpoints. The Python wheel and software source distribution contain the software package; the results and weights are distributed with the repository. [LICENSING.md](../LICENSING.md) gives the terms for each part.

## Model and analysis implementations

The Transformer backend is [TransformerLens 2.16.1](https://github.com/TransformerLensOrg/TransformerLens/tree/a634e5757cd2ae43add9beb492be7a2dfb443a30), with adapters for the paper's checkpoint names and axes, initialization, activations, LayerNorm placement, attention interpolation and hooks. The MLP variants use the same hook infrastructure.

The Fourier implementation follows the [MIT-licensed Grokking Demo](https://github.com/TransformerLensOrg/TransformerLens/blob/a634e5757cd2ae43add9beb492be7a2dfb443a30/demos/Grokking_Demo.ipynb), with the paper's measurement conventions. The GrokAlign comparison uses our PyTorch implementation of centroid-gradient alignment. Its method citation is retained; no GrokAlign source checkout is required. See [third-party credits](../THIRD_PARTY.md) for the pinned sources and notices.

Compatibility checks include archived checkpoints and short training runs. They do not establish identical long-run optimization trajectories across hardware. The [verification record](../verification/REPORT.md) describes the checks performed.

## Included research results

The bundled inputs support 40 figure, table and appendix-analysis targets. The response-norm control includes its historical summary, but remains incomplete because its original individual values and generating code are unavailable. The [coverage notes](known_gaps.md) describe this and the other saved-record limits.

Three example checkpoints are included. Larger checkpoint studies require training the relevant configurations or supplying the required weights. The full historical checkpoint collection is not part of this distribution.

## Source history

The public repository starts from a verified source snapshot. Previous development archives remain private, including earlier implementations that were replaced before publication. Historical commit identifiers in the bundled verification reports identify those development snapshots. The current source, inputs and reports can be used without that history.
