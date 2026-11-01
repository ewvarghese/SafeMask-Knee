# SafeMask-Knee

**Research-only, CPU-only toolkit for auditing selective RSNA knee MRI labels and
producing two alternative training CSVs.** No DICOM files, image model, GPU,
remote service, download, or network access is needed at **runtime**. Bring your
own locally licensed release directory; original competition data are **not**
redistributed by this repository.

For research only; not for clinical use; requires expert radiologist oversight; statistical guarantees are not clinical guarantees.

> **Not a medical device, clinical decision support system, or replacement for
> expert review.** A statistically selective mask is not a radiologist's diagnosis.
> In particular, a rejected finding is **unknown**, not a verified negative.

## Run the bundled archive now (CPU, no expert truth)

From the repository root:

```bash
bash scripts/run_release.sh
```

On a fresh checkout, this one command creates a local `.venv`, installs pinned
requirements if missing (installation may access a package index), checks the
**bundled** `archive.zip` SHA-256 and all **26** member hashes against
`FILES.json`, extracts only required inputs into Git-ignored `data/release/`,
and runs label-free training. Subsequent runs reuse the verified local inputs.
It writes preflight `outputs/doctor.json`, JSON status
`outputs/training_summary.json`, research weights
`outputs/debiased_training_weights.csv`, aggregate quality/sensitivity
reports, and `reports/summary.md`. Runtime verification and training use **no
network**.

**Windows / VS Code:** Running `bash` at a PowerShell prompt usually launches
**WSL**, whose Linux Python is separate from Windows Python. The pinned stack
supports **Python 3.10–3.13 (3.11 recommended)**, not WSL Python 3.14;
installing `python3.14-venv` alone would leave you with incompatible wheels.
The easiest route is to remain in **PowerShell** and use native Windows
Python 3.11, **without bash**:

```powershell
py -3.11 -m venv --clear .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -e .
.\.venv\Scripts\python.exe scripts\bootstrap_release.py --archive archive.zip --out data\release
.\.venv\Scripts\python.exe -m safemask_knee doctor --release data\release --archive archive.zip --config configs\default.yaml
.\.venv\Scripts\python.exe -m safemask_knee train --release data\release --config configs\default.yaml
```

`--clear` removes the failed/mixed virtual environment **only**, not the
repository or source data; do not reuse one `.venv` across WSL and Windows.
If you prefer WSL, install Python 3.11 and `python3.11-venv` **inside your WSL
distribution** (if available from its package manager), then run
`SAFEMASK_PYTHON=python3.11 bash scripts/run_release.sh` from a WSL terminal.
The runner will rebuild a half-created `.venv`; it now rejects Python 3.14
with an explicit explanation instead of proceeding to an incompatible pip
install. To view aggregate results in a **local read-only dashboard**:

```bash
.venv/bin/python -m safemask_knee dashboard --output-root outputs
# Open http://127.0.0.1:8765/ in your browser; Ctrl-C stops the server.
```

Only `/` and aggregated `/api/summary` and `/healthz` are served; raw release
CSVs, study IDs, weights, expert labels and images cannot be downloaded from
this server. If forwarding a Codespaces port, keep it **private**. For a remote
preview, explicitly bind `--host 0.0.0.0 --port 8765`; the default is loopback
only. If dependencies are already installed and you prefer separate steps:

```bash
python scripts/bootstrap_release.py --archive archive.zip --out data/release
python -m safemask_knee train --release data/release --config configs/default.yaml
```

The bundled archive's `expert_oof.csv` contains 58-study **predictions, not
expert labels**; its dataset card explicitly says the official expert answer
key is **not redistributed**. Curated predictions are **CC BY-NC 4.0**, while
original Kaggle/RSNA data have separate terms. The MIT license applies to
toolkit code, not to source or derived label artifacts.

This actually fits five study/duplicate-report-group-held-out Platt-calibrated
acceptance models, estimates IPCW, and writes
`outputs/debiased_training_weights.csv` (48,477 pairs), a **label-free** audit,
`outputs/data_manifest.json`, `reports/summary.md`, and copied notices in
`ATTRIBUTION.md`. No source CSV is checked into Git. `expert_label`/`fold`
remain blank. **No accepted accuracy, disease-erasure verdict, or conformal
risk mask is estimable** without real per-study expert outcomes; `all`,
`masks`, and `evaluate` refuse to fabricate them. Because the source accepts a
nearly deterministic confidence-ranked subset, check the propensity floor/
ceiling rates and overlap warning in the partial report before using weights.

### Run from GitHub, safely

- **Codespaces:** GitHub → **Code → Codespaces → Create codespace**. The
  `.devcontainer/devcontainer.json` installs pinned Python 3.11 dependencies,
  verifies/extracts the bundled archive, and runs `doctor` on creation. In the
  Codespace terminal run `bash scripts/run_release.sh`; optionally start the
  read-only dashboard with `python -m safemask_knee dashboard --host 0.0.0.0`.
  Keep its forwarded port private. Download your private results from the
  Codespace; Git never includes generated weights/release inputs.
- **Actions:** Once the workflow is on the default branch, open **Actions →
  Research toolkit checks → Run workflow** with `run_training` enabled. PRs and
  pushes run tests; a manual dispatch additionally verifies the archive and
  runs label-free training. It uploads **only**
  `outputs/ci/public_summary.json`, a small allowlisted, ID-free aggregate
  summary. **No raw data, per-study outputs, weights, expert answers or masks
  are published as CI artifacts.** The source's CC BY-NC 4.0 and other data
  licence terms still apply.

If you have the *separately licensed official* competition `train.csv` locally,
convert only the release's 58 development studies without copying the source
file into this repository:

```bash
python scripts/prepare_expert.py --official-train /secure/path/train.csv --release data/release --out data/release/expert_results.csv
python -m safemask_knee all --release data/release --config configs/default.yaml
```

The second run can then generate the risk-mask CSV and outer-fold expert-panel
reports. The source used that panel in development: even with grouped CV,
results are not independent clinical validation. If no official train.csv is
available, the **partial training artifact is the honest endpoint**.

### Additional, disjoint expert-labeled cohort (requires independent inputs)

After a complete `all` run using the **official** 58 development studies, an
additional outcome audit can apply the **frozen development-panel cutoffs** to
new studies. Supply a *different* release-style directory containing all three
score/confidence/accepted CSVs with 11 findings per new study, plus a separate
licensed long truth CSV (`StudyInstanceUID,finding,expert_label`). External
scores must be generated without consulting those external answers; their study
IDs must not appear **anywhere** in the source release. Known patient/report
cluster overlap is refused.

```bash
.venv/bin/python -m safemask_knee external-evaluate \
  --release data/release --external-release /secure/external_scores \
  --external-labels /secure/external_results.csv --config configs/default.yaml
```

The resulting aggregate files are in `outputs/external_evaluation/`; they
contain no external IDs or labels. The tool **cannot verify** a user's upstream
score provenance or unknown repeated patients, and cannot claim independent
clinical validation, risk certification, or 1% error control. Without the
official development-panel key, this command deliberately fails. It never
uses external labels to fit cutoffs. No new MRI model can be trained from the
bundled CSV-only archive.

## Three-command quickstart (with expert labels)

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -e .
python scripts/make_synthetic.py --out /tmp/safemask-synthetic  # optional offline smoke test; NO patient data
python -m safemask_knee all --release "<RELEASE_DIR>" --config configs/default.yaml
```

Replace `<RELEASE_DIR>` with the **local** path to the actual release; the
synthetic generator is only a demonstration. For tests first, run
`scripts/run_all.sh "<RELEASE_DIR>"` (it runs `pytest -q` before the pipeline).
Requirements are pinned direct dependencies; initial *installation* may require
a package index, but the program itself never accesses the network. Set
`OPENBLAS_NUM_THREADS=1` / `OMP_NUM_THREADS=1` for strict CPU thread control.

The `train` path needs `accuracy_soft_labels.csv`, `accuracy_confidence.csv`,
`accuracy_accepted60.csv`, and optionally metadata/notices. The **full** `all`
path additionally requires an expert evaluation CSV/JSON with actual truth on
the 58-study panel. A prediction-only `expert_oof.csv` does NOT qualify.
The toolkit expects exactly these **11** canonical findings: ACL, MCL, Medial
Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Baker's cyst,
Contusion, Fracture. **Synovitis is excluded.** Validation requires all 11 per
study and identical `(study_id, finding)` keys across the three release files.
Metadata fairness strata (age, sex, site/scanner) are skipped with a warning
when missing; expert evaluation is needed to make the requested audit and masks.

### Commands

```bash
python -m safemask_knee doctor      --release "<RELEASE_DIR>" --archive archive.zip
python -m safemask_knee schema      --release "<RELEASE_DIR>" --out outputs/schema.json
python -m safemask_knee quality     --release "<RELEASE_DIR>" --out outputs/quality
python -m safemask_knee train       --release "<RELEASE_DIR>" --config configs/default.yaml  # without expert truth
python -m safemask_knee audit       --release "<RELEASE_DIR>" --config configs/default.yaml --out outputs/audit
python -m safemask_knee weights     --release "<RELEASE_DIR>" --config configs/default.yaml --out outputs/debiased_training_weights.csv
python -m safemask_knee sensitivity --release "<RELEASE_DIR>" --config configs/default.yaml --out outputs/sensitivity
python -m safemask_knee masks       --release "<RELEASE_DIR>" --config configs/default.yaml --out outputs/clinical_risk_masks.csv
python -m safemask_knee evaluate    --release "<RELEASE_DIR>" --config configs/default.yaml --out outputs/evaluation
python -m safemask_knee report      --release "<RELEASE_DIR>" --out reports/summary.md
python -m safemask_knee all         --release "<RELEASE_DIR>" --config configs/default.yaml
python -m safemask_knee dashboard   --output-root outputs  # read-only local viewer
```

Each invocation prints a **source → canonical column-mapping table to stderr**
and a compact JSON result to stdout; INFO logs go to stderr and DEBUG logs to
`outputs/logs/pipeline.log`. `evaluate` regenerates weights/masks if called
alone so it cannot silently read stale artifacts; `report` verifies source
SHA-256 hashes against the preceding run. `all` writes both
`reports/summary.md` and `outputs/reports/summary.md`. The SHA-256-only input
manifest is in `data/manifest.json` and `outputs/data_manifest.json`; source
CSVs themselves are never copied. Notices are copied **verbatim** into a
run-specific `ATTRIBUTION.md` in the current working directory. Run in a fresh
working directory if you do not want to replace the repository's attribution
template. Generated data and reports are Git-ignored.

## Downstream artifacts: **exact column order**

`outputs/debiased_training_weights.csv` (one row per study/finding):

```text
study_id, finding, soft_score, confidence, accepted60, expert_label, propensity_accepted, weight_raw, weight_clipped, weight_final, fold, notes
```

`expert_label` and outer `fold` are **blank for other studies**, and **blank for
every study when the input has no expert truth**. Rejected records
have weight 0; they are **not** a supervised source of verified negatives.
Accepted weights use a study-held-out, Platt-calibrated acceptance propensity,
are clipped to `[0.1, 10]`, and are normalized to mean 1 **within each finding**.
Propensity is stored for rejected records too, but their weight stays zero.

`outputs/clinical_risk_masks.csv` (one row per study/finding):

```text
study_id, finding, tier, risk_bound, conformal_threshold, soft_score, confidence, accepted60, accepted_clinical, reason, fold
```

`tier` is 1, 2, or 3; `accepted_clinical` is exactly 0 or 1. The filename is a
compatibility requirement, **not a clinical endorsement**. `risk_bound` is the
nominal per-finding **marginal class-conditional** accepted-wrong-singleton risk
for true positives (FNR among *all positives*), **not the error conditional on
acceptance**. Tiers 2/3 also have the same configured false-positive bound;
Tier 1 uses the separately configured `false_positive_bound`. The threshold
is the score-side cutoff for the score's binary sign; **do not apply it alone**:
use `accepted_clinical`, which also checks the other class's rank p-value.
The threshold is blank where finite sample size makes that decision impossible.
For expert studies, the mask is calibrated on the **other outer study folds**;
for other studies it uses the full expert panel (and is *not* scored on that
panel). `fold` is blank for studies without expert truth. A selected finding's
working pseudo-label is `soft_score >= 0.5` unless an *external* expert-labeled
training set is separately supplied; `accepted_clinical` does **not** make that
pseudo-label an expert diagnosis.

Audit/evaluation CSV tables have their own explanatory columns (including
`outputs/evaluation/finding_risk_checks.csv`, which checks the per-finding
rather than only pooled tier risks); markdown reports include explicit
study-CV, bias, uncertainty, and licence caveats. See
`outputs/schema.json` for the exact observed input mapping and validation
counts. `outputs/quality/data_quality.md` has aggregate duplicate/report,
metadata, and upstream-exposure diagnostics; `outputs/sensitivity/` compares
IPCW floors/caps on the **same clipped OOF probabilities** (never proof of
bias correction). `outputs/run_provenance.json` records input hashes,
config/seed, git revision and installed versions without source IDs.
`outputs/README.md` documents the artifact paths.

### Example: *optional* downstream PyTorch training

PyTorch is **not** an install/runtime dependency. This example assumes your own
image-model logits and `study_id`/`finding` join are aligned with the CSV; the
example is **not** a clinical model. Never train on expert rows and then report
metrics on those same expert studies. All splits must be at least by study (and
by patient when repeated patient IDs are available).

```python
import pandas as pd
import torch
import torch.nn.functional as F

w = pd.read_csv("outputs/debiased_training_weights.csv")
m = pd.read_csv("outputs/clinical_risk_masks.csv")
assert w[["study_id", "finding"]].equals(m[["study_id", "finding"]])
# logits: float32 vector from YOUR downstream model, in the CSV's row order.
logits = torch.as_tensor(your_logits, dtype=torch.float32)
working_label = torch.as_tensor((w.soft_score.to_numpy() >= 0.5).astype("float32"))
per_pair = F.binary_cross_entropy_with_logits(logits, working_label, reduction="none")

# Option 1: only train on selected pseudo-labels (not verified ground truth).
selected = torch.as_tensor(m.accepted_clinical.to_numpy(dtype="float32"))
masked_loss = (per_pair * selected).sum() / selected.sum().clamp_min(1)

# Option 2: train on accepted60 pseudo-labels, using stabilized IPCW.
weighted = torch.as_tensor(w.weight_final.to_numpy(dtype="float32"))
weighted_bce = (per_pair * weighted).sum() / weighted.sum().clamp_min(1)
```

**Do not** infer model recall/precision from weights alone; run a prospective
expert evaluation with held-out patients. The two objectives are alternatives,
not a licence to treat newly unmasked pseudo-labels as clinical ground truth.

## What is estimated, and what is *not*

1. **Audit:** compare expert prevalence among accepted/rejected labels, a
   positive/negative acceptance ratio, false negatives **inside accepted**, and
   released errors by finding; tabulate all-study acceptance by finding,
   soft/confidence bin and available metadata. Tests of metadata differences
   compare study/patient clusters, not 11 independent finding rows. A ratio
   <0.8 is a **descriptive flag**, not causal identification of MNAR.
2. **IPCW:** cross-fit `P(accepted60=1 | soft, confidence, other findings' soft
   and accepted flags, metadata)` using study-grouped logistic/held-out Platt;
   compute stabilized, bounded weights **only for accepted pseudo-labels**.
   This relies on selection ignorability conditional on measured features and
   positivity; latent outcome-dependent selection cannot be fixed by weights.
3. **Mask:** fixed-score class-conditional conformal ranks, a leave-one-study-
   out diagnostic and outer study-level 5-fold evaluation. Tier 1 targets 1%
   *marginal* accepted-negative FNR among all positives. With at most 58
   positives per finding, the smallest conformal p-value exceeds 1%; the
   honest option is to **abstain** from negative Tier 1 decisions rather than
   assert 1% safety with high coverage. Tier 2 defaults to 12% and Tier 3 to
   20% marginal per-class wrong-singleton bounds (not conditional accepted
   accuracy); both values are configurable. Tier 3 attempts higher coverage,
   subject to measured feasibility, without post-hoc bound relaxation.
   These are per-finding nominal bounds, not a simultaneous guarantee over all
   11. Repeated patients also require careful cluster-level exchangeability.
4. **Evaluation:** an outer study/patient-grouped K-fold compares tabular A
   (naive rejected-as-negative), B (soft), C (confidence-weighted soft), D
   (accepted IPCW) proxies with the release baseline and held-out-fold mask.
   A–D are fitted **only on non-expert patient/study groups**; D recomputes
   IPCW within that training pool so expert holdouts cannot leak into its
   weights. Frozen proxies are assessed over the outer expert folds.
   Proxies are **not MRI models**. The expert panel helped develop the release:
   outer CV prevents *this toolkit* from evaluating a threshold/model on its
   own fitted expert labels, but does **not** undo upstream reuse. An independent
   external expert cohort is required for final performance claims.

The program **never guarantees** 60%+ coverage, 97% accepted accuracy, or a
finite-sample clinically meaningful 1% FNR on arbitrary releases. If the
configured bounds and coverage target conflict, it reports the **unmet goal**
rather than using an uncertified acceptance fallback. `METHODOLOGY.md` derives
the risk definitions, rank tests, bootstrap, IPCW assumptions, and limitations.

## Assumptions and column mapping

- A row's soft score >=0.5 supplies the working pseudo-positive label. If the
  expert evaluation includes its own released prediction, the **release sanity
  check uses that field instead**. A mismatch is flagged in `schema.json` and
  explained in `outputs/evaluation/eval_report.md`. Confidence is required in
  [0,1] but is **not** assumed to be a calibrated probability.
- CSVs can be **wide** (`exam_id,ACL,MCL,...`) or **long** (`study_id,finding,value`).
  IDs such as `exam_id`, `StudyInstanceUID`, `study_id` are mapped to `study_id`.
  Finding aliases like `accepted60_ACL` map to `accepted60[ACL]`, and
  `ACL_confidence` to `confidence[ACL]`. E.g. on the synthetic fixture:
  `exam_id -> study_id`, `accepted60_Medial OA -> accepted60[Medial OA]`,
  `expert_label -> expert_label`. Auto-mapping is **printed** and saved to
  `outputs/schema.json`; unknown/ambiguous IDs, findings, extra/missing pairs,
  invalid values, and duplicate rows fail with a clear error. If the actual
  format differs, set `columns.study_id`, `.finding`, `.soft_score`,
  `.confidence`, `.accepted60`, `.expert_label`, `.predicted_label`, or
  `columns.findings: {ACL: name_of_wide_column, ...}` in the YAML. When the
  source name differs by file, use a role map such as
  `columns.findings: {ACL: {soft_score: acl_soft, confidence: acl_conf, accepted60: acl_mask}}`.
  A single global study-ID override must exist in **each** file; if IDs
  differ across files, use their auto-detected aliases. Set `paths.metadata`
  or `paths.evaluation` when auto-discovery is ambiguous.
- A missing metadata CSV is allowed; available strata only are reported.
  `patient_id`, when present, keeps all a patient's studies in the same fold
  and bootstrap cluster; otherwise study_id is the clustering unit. No
  demographic stratum is imputed as fair. Expert CSV must have all 11 findings
  per included study; full expert-dependent `all` requires exactly 58 studies
  / 638 pairs and verifies IDs against the development-panel metadata when
  that marker is supplied. A different panel size may be inspected by
  `schema` but is refused for a full run.
- The published release may use a different binarization policy than `>=0.5`.
  Neither comparison nor reporting silently changes the published mask.
- Library seeds and input hashes help reproduce runs on the same software/CPU;
  cross-platform floating-point/BLAS differences can change threshold ties.

## Citation and attribution

If you use this code, cite the toolkit and **separately cite the actual RSNA
competition and the authors of the label release you received**, using the
release's supplied citation and licence notices (not guessed here).

```bibtex
@software{safemask_knee_2026,
  title = {SafeMask-Knee: grouped selective-label audit and training artifacts},
  author = {SafeMask-Knee contributors},
  year = {2026},
  url = {https://github.com/ewvarghese/SafeMask-Knee}
}
```

**Licensing nuance:** Code is MIT; a binding research-only restriction would
make it *not* permissive OSS. `LICENSE` therefore carries a **nonbinding**
research-use/safety notice, not a new license restriction. Source data retain
their own terms and must be checked independently. This toolkit does not
license original competition data or authorize clinical use.

For research only; not for clinical use; requires expert radiologist oversight; statistical guarantees are not clinical guarantees.
