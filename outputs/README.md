# Generated artifacts

On a fresh checkout, run `bash scripts/run_release.sh` to verify/extract the
bundled `archive.zip`, install missing dependencies, and run the prediction-only
training stage. It saves the CLI JSON results to `doctor_cli_summary.json`
and `training_summary.json`. The local read-only aggregate dashboard uses
`quality/data_quality.json`, `propensity_diagnostics.json` and
`sensitivity/sensitivity_overview.csv`; it cannot serve raw release files.

`python -m safemask_knee train --release data/release --config configs/default.yaml`
needs **no expert truth** and fits grouped acceptance propensities, writes
`schema.json`, `data_manifest.json`, `run_provenance.json`,
aggregate `quality/` and label-free `audit/`, `sensitivity/`,
`debiased_training_weights.csv`, `propensity_diagnostics.json`, and a **partial**
`reports/summary.md`. It does **not** create a risk mask or expert-label
accuracy. Run this on the supplied prediction-only archive.

After training, `python -m safemask_knee review-plan --release data/release
--groups 25` uses the saved weights to write a **private**, targeted annotation
worklist to `review_plan/review_queue.csv`, a blank 11-finding
`review_plan/expert_label_template.csv` and an ID-free explanatory
`review_plan/review_plan.md`. It excludes entire known development/patient/
report groups. The queue contains study IDs: it is not served by the dashboard
or uploaded by CI, and must stay private and out of Git. It is neither a random
sample nor a clinical mask, and needs independent expert review of separately
licensed MRI images before any truth labels exist.

`python -m safemask_knee verify --release data/release --archive archive.zip
--recompute` performs a read-only source/archive/weights audit and an optional
study-grouped numeric refit. `verification/verification.md` and
`verification/verification.json` contain only aggregate counts, finding names,
check results and SHA-256 fingerprints—not study IDs or row data. A mismatch
fails loudly; no clinical truth or MRI diagnosis is certified. Other
researchers must obtain and verify their own appropriately licensed data.

Only with genuine separately licensed official expert answers,
`python -m safemask_knee all --release <RELEASE_DIR> --config configs/default.yaml`
writes `schema.json`, `data_manifest.json`, `audit/`, the two downstream CSVs,
`loo_conformal_scores.csv` (study-excluded calibration diagnostics),
`evaluation/`, and `reports/summary.md` under this directory. Generated data and
reports are intentionally ignored by Git; only this README (not generated
schema or input-derived artifacts) should be committed. A hash-only copy of the input manifest is
also written to `data/manifest.json` (ignored by Git).

For research only; not for clinical use; requires expert radiologist oversight; statistical guarantees are not clinical guarantees.
