# Generated artifacts

On a fresh checkout, run `bash scripts/run_release.sh` to verify/extract the
bundled `archive.zip`, install missing dependencies, and run the prediction-only
training stage. It also saves the CLI JSON result to `training_summary.json`.

`python -m safemask_knee train --release data/release --config configs/default.yaml`
needs **no expert truth** and fits grouped acceptance propensities, writes
`schema.json`, `data_manifest.json`, the label-free `audit/`,
`debiased_training_weights.csv`, `propensity_diagnostics.json`, and a **partial**
`reports/summary.md`. It does **not** create a risk mask or expert-label
accuracy. Run this on the supplied prediction-only archive.

Only with genuine separately licensed official expert answers,
`python -m safemask_knee all --release <RELEASE_DIR> --config configs/default.yaml`
writes `schema.json`, `data_manifest.json`, `audit/`, the two downstream CSVs,
`loo_conformal_scores.csv` (study-excluded calibration diagnostics),
`evaluation/`, and `reports/summary.md` under this directory. Generated data and
reports are intentionally ignored by Git; only this README (not generated
schema or input-derived artifacts) should be committed. A hash-only copy of the input manifest is
also written to `data/manifest.json` (ignored by Git).

For research only; not for clinical use; requires expert radiologist oversight; statistical guarantees are not clinical guarantees.
