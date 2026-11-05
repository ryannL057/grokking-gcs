# Verification

The repository is checked with Python 3.11. The [GitHub workflow](../.github/workflows/verify.yml) runs two jobs: saved-result reproduction and model tests.

| Check | Result |
|---|---|
| Test suite | 100 passed locally and in GitHub Actions |
| Bundled inputs | 519 files verified by size and SHA-256 |
| All paper targets | 40 passed, one incomplete, zero failed |
| README examples | Figure/table output, checkpoint GCS, component resets and the five-epoch training demo completed |
| Package installation | Built and installed the wheel; the examples imported that installation |

The response-norm target is the one incomplete result. The full reproduction command returns `1` for it; the workflow checks that this is the only incomplete target and that every supported target passes. Details are in [coverage and limitations](../docs/known_gaps.md).

## Run the checks

```bash
python -m pip install -r requirements/test.txt
python scripts/verify_artifacts.py
python -m pytest -q
python scripts/reproduce.py --all --output outputs/verification
```

Use a new output directory for each run. The [reproduction report](reproduction.json) records the saved-result checks, and [results.json](results.json) records the environment and example checks. Current hosted runs are listed in [GitHub Actions](https://github.com/ryannL057/grokking-gcs/actions/workflows/verify.yml).

## Model compatibility

The TransformerLens adapter was compared across 72 combinations of depth, activation, LayerNorm and attention interpolation, including checkpoint loading, forward outputs and path Jacobian-vector products. Additional checks covered the MLP variants, short training runs, Fourier measurements and centroid alignment. A checkpoint example was also tested on an RTX 4090 with PyTorch 2.7.1+cu126. These earlier model checks are retained in [the compatibility record](model_compatibility.json).

The model environment pins `datasets==4.0.0` and `pyarrow==20.0.0` for compatibility with NumPy 1.26.4. Other tested versions are listed in `requirements/`.

These checks cover the bundled results, implementations and examples. The full historical training and intervention cohorts were not rerun for this release.
