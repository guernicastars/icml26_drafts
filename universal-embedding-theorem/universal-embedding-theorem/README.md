# Universal Embedding Theorem

This folder is the first experiment scaffold for the draft in
`/Users/meuge/Downloads/files (4)/universal_embedding_theorem_v3.tex`.

The current implementation focuses on the synthetic validation track from the
paper. It turns the draft's abstract claims into runnable experiments around:

- PCA recovery of a known causal subspace
- sparse gradient recovery with `O(k log(d / k))` measurements
- double descent / benign overfitting in a sparse latent regime
- noise-dimension augmentation studies
- superposition with sparse activation and random feature dictionaries

The real-data tracks in the draft (Polymarket and language-model embeddings) are
still TODO. This scaffold is designed so we can add those datasets without
rewriting the core metrics.

## Layout

- `experiments/universal_embedding/`: shared synthetic data generators and metrics
- `experiments/run_synthetic_validation.py`: end-to-end sweep runner
- `experiments/render_empirical_note.py`: render a LaTeX note from experiment summaries
- `paper/synthetic_validation_note.tex`: standalone LaTeX report entry point
- `tests/test_synthetic_validation.py`: small regression tests for the math utilities

## Run

```bash
cd /Users/meuge/coding/maynard/ICML\ Sprint/universal-embedding-theorem
python3 experiments/run_synthetic_validation.py --output-dir experiments/artifacts
python3 experiments/render_empirical_note.py --artifacts-dir experiments/artifacts --output-tex paper/generated_results.tex
cd paper && pdflatex synthetic_validation_note.tex
python3 -m pytest tests/test_synthetic_validation.py
```

## Outputs

The runner writes:

- `pca_alignment.csv`
- `sparse_recovery.csv`
- `double_descent.csv`
- `noise_augmentation.csv`
- `superposition.csv`
- grouped summaries for each table
- `synthetic_validation_overview.png`
- `paper/generated_results.tex`
- `paper/synthetic_validation_note.pdf` after compilation

## Notes

- The current PCA experiment works directly with synthetic embeddings whose
  covariance is constructed to have a known causal subspace.
- The current double-descent study uses minimum-norm linear regression as a
  controlled stand-in for the broader benign-overfitting story in the draft.
- The noise-dimension study currently tests literal augmentation by appending
  independent Gaussian coordinates, which is enough to exercise the pipeline but
  should still be treated as exploratory.
- The superposition study measures decoding error against the theorem's
  `s sqrt(log m / d)` rate using random feature dictionaries and sparse binary
  activations.
- The tests are written for `pytest`; if that module is not installed in the
  active environment, run the experiment smoke test first and install `pytest`
  before using the test target.
