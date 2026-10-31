# Attribution for this run

Derived from the user-supplied RSNA knee MRI competition label release. The competition data and third-party labels remain under their own terms. The MIT license applies ONLY to SafeMask-Knee code; no images or original release tables are redistributed.

For research only; not for clinical use; requires expert radiologist oversight; statistical guarantees are not clinical guarantees.

## Notices copied from the release

### `DATASET_CARD.txt`

```text
RSNA Knee 11-Label Expert-Stressed Soft Targets

4,407 training study identifiers, 11 findings. Synovitis is deliberately excluded. This is a pseudo-label target dataset, not a new expert annotation set or a competition submission.

Version2 adds Perplexity Decider1.2 via the native OpenRouter Decisions API: eleven related finding questions per report in one joint request, independent reports sent concurrently. The expert pilot's candidate pool and expert prediction values are frozen and verified unchanged after the complete weak-report pass; newly available Decider rank alternatives are excluded from this release's scored comparison. Baseline version1 remains separately recoverable.

Matched version1 comparison: accuracy delta +0.26%; paired conditional study-bootstrap95percent interval [-0.52%, +1.04%]. This is a development candidate; a positive point difference does not establish independent superiority. Both versions remain available.

Development benchmark: AUC-oriented shrunken fusion macro ROC-AUC=0.9287, conditional study-bootstrap 95% interval [0.9037, 0.9516]. The separate accuracy-oriented selective97::stack_interactions variant reaches 97.13% accepted accuracy at 60.03% study-finding coverage; conditional study-bootstrap accuracy interval [95.30%, 98.69%]. These numbers describe DIFFERENT variants. Complete per-label and calibrated scores are in benchmark_results.json. The intervals resample studies with all 11 findings together and do not capture all model/source-selection uncertainty.

The 58 official image-expert studies are the same familiar public development panel used repeatedly in this project. Several public sources were also selected/tuned on that panel. Our fusion/calibration/acceptance stage excludes each outer expert group during its fitting, but this does not undo public-source or historical selection exposure. These are not independent clinical validation or hidden Kaggle leaderboard results. Family selection after development comparison is disclosed.

Files: soft_labels.csv has AUC-oriented continuous ranking targets for all 4,407 studies; its 58 expert rows contain actual outer-held predictions, not copied answers. These ranking targets are not calibrated clinical probabilities. weak_labels.csv contains only 4,349 weak rows. calibrated_soft_labels.csv and accuracy_soft_labels.csv are separate calibrated options. label_weights.csv contains estimated training reliability, not certified medical confidence. strict_mri_oof_soft_labels.csv excludes the weak-training-exposed full MRI ensemble. study_metadata.csv provides folds, duplicate-report groups, missing MRI and ambiguity flags. expert_oof.csv preserves the unpatched benchmark outputs. Source score/evidence matrices and benchmark_core.py support reproduction. Assistant review codes describe report evidence, not image-expert ground truth.

accuracy_confidence.csv ranks the accuracy-oriented predictions by estimated correctness. accuracy_accepted60.csv selects the top60percent of study-finding cells separately within the weak population and expert panel, using confidence only. The expert panel therefore accepts383/638 findings (60.03percent), not60percent of complete MRI studies. This fixed-coverage rank mask differs from a learned-threshold acceptance mask. Every model still outputs all11 continuous scores. A97percent point estimate on this familiar small development panel does not certify97percent population accuracy.

The full five-reader MRI ensemble never fitted official expert answers. Its weak-row training exposure counts are verified against the original fit audits and recorded per study: up to 4 of 5 members saw a weak row's earlier report-derived targets. Counts for this release: {'0': 49, '4': 4296}. Strict MRI-OOF scores use the row's excluded training fold and are supplied separately. This is case-UID exclusion, not verified patient independence. Four weak studies lack verified MRI teacher outputs and use report-source fallbacks. Exact report duplicates and multi-report ambiguity are flagged.

No raw radiology reports, DICOM images, OAI records, official answer keys, credentials or model weights are distributed. Released numeric components reproduce fusion and scores; regenerating MRI components from raw images requires the original trained weights, which are not bundled. Download and accept the original competition data terms to join images and reproduce expert scores: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection . Requirements: numpy, scipy, scikit-learn. Example: python reproduce_public.py --train-csv /path/to/official/train.csv --dataset .

New curated predictions are released under CC BY-NC 4.0; original source licences/attribution remain as listed in source_provenance.json and THIRD_PARTY_NOTICES.txt. Code and labels from other authors are credited, not claimed as independent new annotations. No image model trained on this release has a new Kaggle leaderboard score.

```

### `THIRD_PARTY_NOTICES.txt`

```text
Third-party source credits and licences

{"name": "JEV pure and hybrid", "ref": "https://www.kaggle.com/datasets/riadmohamed42/jev-knee-labels", "license": "CC0-1.0", "overlap": "Hybrid is95pct selected Steven/Pilkwang plus5pct JEV; authors reused expert panel."}
{"name": "Steven v2", "ref": "https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels", "license": "CC0-1.0"}
{"name": "Pilkwang", "ref": "https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels", "license": "CC0-1.0"}
{"name": "Lixin GPT5.6", "ref": "https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56", "license": "CC0-1.0"}
{"name": "TypeSafe JEV", "ref": "https://www.kaggle.com/datasets/hosseini7798/rsna-knee-label-extraction-jev", "license": "MIT"}
{"name": "Qwen2.5-32B raw report labels", "ref": "https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "license": "CC0-1.0"}
{"name": "Fresh local OpenJev", "ref": "https://huggingface.co/openjev/openjev-FP8", "runtime": "Local MLX8bit reconstructed from FP8, full original-language report, no external API"}
{"name": "Assistant-authored report safeguards", "reference": "https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee", "reference_license": "Apache-2.0", "scope": "Deterministic rules and638 direct report-evidence reviews, not new image-expert ground truth"}
{"name": "Owned ConvNeXt-Small MRI readers", "scope": "Five report-only readers; expert study UIDs excluded from fitting. Up to4of5 members saw a weak row earlier; actual per-study fit-audit exposure counts supplied. StrictMRI-OOF alternative supplied."}
{"name": "Perplexity Decider1.2 joint report labels", "ref": "https://huggingface.co/perplexity-ai/pplx-decider-v1.2", "gateway": "OpenRouter", "model": "perplexity/pplx-decider-v1.2-20261016", "scope": "All11findingquestions share one native request per report. Original report text only; no expert answers or MRI images transmitted. API probabilities retained without extra temperature."}

Original Kaggle/RSNA data remain subject to their own terms. No raw original reports/images or official answer key are included.

MIT notice for TypeSafe JEV source, credited to hosseini7798 (Mahdi Hosseini), per its published dataset licence:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

```
