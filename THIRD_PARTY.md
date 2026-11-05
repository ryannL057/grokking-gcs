# Attribution and third-party licenses

Our training code originally used Neel Nanda's [TransformerLens Grokking Demo notebook](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Grokking_Demo.ipynb). The implementations below build on its MIT-licensed source and TransformerLens 2.16.1.

| Component | Source and terms |
|---|---|
| Transformer execution and hook infrastructure | [TransformerLens 2.16.1](https://github.com/TransformerLensOrg/TransformerLens/tree/a634e5757cd2ae43add9beb492be7a2dfb443a30), MIT; installed dependency |
| `src/gcs/core/lens_model.py`, `src/gcs/core/hooks.py` | Local adapters for checkpoint axes, initialization, path access and callbacks around TransformerLens |
| `src/gcs/analysis/fourier.py` | Fourier projection/loss design from the [MIT Grokking Demo](https://github.com/TransformerLensOrg/TransformerLens/blob/a634e5757cd2ae43add9beb492be7a2dfb443a30/demos/Grokking_Demo.ipynb), with the paper's logits projection, frequency-selection and precision conventions |
| `src/gcs/analysis/grokalign.py` | Locally implemented centroid-gradient cosine using PyTorch; retains the [GrokAlign method attribution](https://github.com/ThomasWalker1/GrokAlign), without importing or bundling its upstream source |
| MLP variants and remaining scientific modules | Curated from the authors' GCS research archive; source records and changes are in `manifests/code_provenance.json` |

The TransformerLens source is pinned to commit `a634e5757cd2ae43add9beb492be7a2dfb443a30`. Its full copyright and MIT permission notice is included in [src/gcs/licenses/TransformerLens-LICENSE](src/gcs/licenses/TransformerLens-LICENSE) and in the installed package.

Earlier extracted Fourier helpers and model implementations were replaced before publication. The old development archives remain private and are outside this repository's license grants. [Distribution details](docs/distribution.md) describe the implementations included here.

PyTorch, NumPy, SciPy, pandas, Matplotlib, scikit-learn, einops, tqdm, Transformers, wandb and test dependencies are installed packages; their own distribution licenses apply. Manuscript-machine fonts are not bundled.

Our original code, configurations and documentation use MIT; numerical results and example weights use CC BY 4.0. [LICENSING.md](LICENSING.md) defines the scope. These grants preserve the third-party terms and notices above.
