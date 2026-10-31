# Methods, estimands, assumptions, and failure conditions

For research only; not for clinical use; requires expert radiologist oversight; statistical guarantees are not clinical guarantees.

Source attribution: this toolkit processes a user-supplied **RSNA knee MRI
competition label release**. It does not contain images, original release
CSVs, or a reference radiologist model. See `ATTRIBUTION.md` and copied notices
from the user's release. The 58 expert development studies are **not an
independent external test cohort**.

## Units, labels, and leakage prevention

Let `i` index studies, `f` the 11 findings (no Synovitis), `S_if` the released
continuous soft score, `C_if` confidence, `A_if` the released 60%-acceptance
indicator, and `Y_if` expert truth observed **only** for the small panel. The
release gives all 11 pairs for each study. Its published binary prediction,
when included in expert results, is the *descriptive baseline* `B_if`; the
mask's working prediction is `1{S_if >= 0.5}`. These two can differ; the schema
and evaluation report count the discrepancy explicitly.

`grouped_study_folds` partitions unique study IDs; if a patient ID is supplied,
all studies for a patient stay in one group. When normalized-report hashes are
supplied, identical reports also share a fold; connected patient/report
clusters are not divided. Without patient IDs (as in the supplied archive),
**patient independence cannot be verified** merely by grouping report hashes.
Every data transformation that *fits* a classifier, imputer, calibrator, or
cutoff uses training groups only.
Propensity OOF folds are independent of the 5 outer expert folds. The latter
fit class-conditional rank cutoffs on other folds and report OOF predictions
for **all** expert outcomes. A LOO diagnostic excludes the entire study (or
patient group) from its own nonconformity calibration. `outputs/loo_conformal_scores.csv` records the true-class p-value with each
study/patient omitted; it is a diagnostic, not a claim of independent test
accuracy. The final full-panel cutoffs apply **only to other, expert-unlabeled
studies**, so final metrics are never computed on that same fitted panel.

## 1. Selection audit

For each finding, the expert-panel accepted and rejected prevalences are
`sum(A*Y)/sum(A)` and `sum((1-A)*Y)/sum(1-A)`. The descriptive selection ratio
is

```text
SBR_f = [sum_i A_if Y_if / sum_i Y_if] /
        [sum_i A_if (1-Y_if) / sum_i (1-Y_if)].
```

`SBR_f < 0.8` flags disproportionately rejected expert positives. The
**within-accepted false-negative proportion** is
`sum_i A_if Y_if (1-B_if) / sum_i A_if Y_if`. It is different from the
**marginal accepted-negative false-negative risk among all positives** used
for conformal calibration below. `bias_table.csv` includes exact counts of
accepted mistakes by finding and positive/negative error type. A Fisher
2x2 test compares selection in positives vs negatives *within a finding*;
BH-adjusted q-values are exploratory (58 studies, low power).

Across all studies, acceptance rates are stratified by finding, deciles of
score and confidence, and available metadata. Score/confidence bins are
**descriptive** because repeated finding rows from one study are correlated.
Metadata contrasts use per-study acceptance fractions, and per-patient
cluster means when a patient spans multiple studies, with exploratory Welch
comparisons, SMDs and BH q-values. Finding-level comparisons use paired
study-level differences. Small/empty strata give no p-value. This does not
prove MNAR *conditional on unmeasured variables*, causation, or population
fairness; data missing by site/scanner may conceal disparate errors.

## 2. Out-of-fold stabilized inverse-probability weighting

`X_if` contains own soft score, confidence, margin, peer soft/acceptance
findings (with **own acceptance entry removed**), age/sex/site/scanner when
available, and a categorical finding. It excludes study/patient IDs, expert
truth, and own `A_if`. In outer propensity fold `k` a logistic classifier is
fit on other study groups, with a **disjoint inner study-group holdout** for
Platt calibration of logits. Prediction `e_hat_if^(-k)` on the omitted group is
out-of-fold. The finding-specific numerator `p_hat_f^(-k)` is computed on the
outer training groups. For each accepted pair:

```text
raw_if     = p_hat_f^(-k) / max(e_hat_if^(-k), e_min)
clipped_if = clip(raw_if, w_min, w_max)
final_if   = clip(c_f * clipped_if, w_min, w_max)
mean(final_if | A_if=1, finding=f) = 1
```

The scale `c_f` is solved by bisection **inside** the clipping bounds; merely
normalizing *after* clipping can violate the maximum. Rejected pairs get
weight zero, not a fake negative target. Defaults are `e_min=0.02` and
`[w_min,w_max]=[0.1,10]`. OOF propensities and fold study IDs are recorded
under `outputs/propensity_diagnostics.json`.

IPCW approximates an inverse-observation-probability correction only if
`Y ⟂ A | X`, there is positivity in X, released pseudo-labels are accurate
enough, and the propensity model is reasonable. **True unmeasured-label MNAR
violates ignorability**. The expert-panel post-weighting SBR is a diagnostic,
not proof of causal bias repair. Weight truncation trades bias for variance;
check extreme weights/ESS and per-finding disagreement before using them.
The supplied archive's acceptance is a near-deterministic ranking of
confidence: many OOF propensities hit the 0.02/0.98 floor/ceiling. This is a
**positivity/overlap warning**; an outcome-independent fit to A does not
establish `Y ⟂ A | X`, and no IPCW claim can recover erased expert outcomes.

### Prediction-only archive: honest partial training

The new `data/release/` archive contains 4,407 studies and metadata marking a
58-study development panel, but `expert_oof.csv` is a *score/prediction* table
with **no official expert answer key**. Its dataset card says predictions are
CC BY-NC 4.0; original competition data remain separately licensed.
`python -m safemask_knee train --release data/release` therefore runs schema,
label-free acceptance/operational metadata audit and grouped OOF propensity,
then writes IPCW weights and a report conspicuously marking all expert
prevalence/error/risk fields as unavailable. It does **not** generate
`clinical_risk_masks.csv`, certify 1% risk, or train an MRI image model. The
source's benchmark JSON reports aggregate development performance but cannot
supply per-study expert truth. With a separately licensed official `train.csv`,
run `scripts/prepare_expert.py` to extract only the 58 required study/finding
labels into ignored local `expert_results.csv`, then run `all`. Upstream reuse
of that panel remains a reason against external-validation claims.

## 3. Fixed-score, class-conditional conformal ranks

This release contains no image model to refit, so **a genuine Jackknife+
construction cannot be implemented**. The implemented procedure is
leave-one-study-out *diagnostic* plus class-conditional (Mondrian) conformal
ranks of a **fixed** released score; describing it as Jackknife+ would be
incorrect. For calibration expert pairs in finding `f`, define candidate-
class nonconformities `R_if(1)=1-S_if` on expert positives and `R_if(0)=S_if`
on expert negatives. Given a new score `S`, for `y` in `{0,1}`:

```text
p_f(y; S) = [1 + sum_{j in calibration, Y_jf=y}
                    1{ R_jf(y) >= R_f(y; S) }] / (n_f,y + 1).
C_f(S) = {y in {0,1}: p_f(y; S) > alpha_f,y}.
```

The leading 1 and `>=` ties yield conservative discrete ranks. Accept only
if `C_f(S)` is **one singleton** agreeing with `1{S>=0.5}`. For tier 1,
`alpha_f,1 = 0.01` (positive-class exclusion, so accepted-negative FNR among
*all expert positives*) and `alpha_f,0 = 0.10` (accepted-positive FPR among
*all negatives*). Tier 2 defaults to 0.12/0.12; Tier 3 to 0.20/0.20. Tier 3
uses the least conservative **preconfigured** nominal bound; it is never
increased after looking at expert-panel accuracy to hit a coverage target.
The output `risk_bound` always records `alpha_f,1`. Output score cutoffs are
only a human-readable side of the rank test; `accepted_clinical` includes
**both** p-values and the score-sign check.

**What the mathematical statement actually is:** if calibration and *future*
studies are exchangeable **within class** for each finding and the scoring
function is fixed independently of calibration labels, the true-class
conformal p-value is super-uniform. An accepted wrong singleton implies
`Y not in C`, so

```text
P(accepted negative | Y=1, finding=f) <= alpha_f,1,
P(accepted positive | Y=0, finding=f) <= alpha_f,0.
```

These are **marginal class-conditional risks across all new studies**, not
`P(error | accepted)`, not `P(error | accepted positive)`, not a bound on
clinical harm, and not a distribution-free guarantee under score-development
reuse. Bounds are **per finding**, not simultaneous over all 11; a joint
family-wise guarantee requires a multiplicity adjustment and less coverage.
The released scores/mask were developed using the 58 expert studies:
that upstream dependence **may invalidate the independent-fixed-score
assumption** even after this toolkit's LOO/outer CV. External calibration and
independent testing would be required for a prospective claim. Likewise,
selecting alphas/score functions on the same panel after observing results
invalidates the formal rank statement; the config must be fixed in advance.

**Infeasibility at 1%:** with `n_pos` positive calibrations, the smallest
p-value is `1/(n_pos+1)`. A 1% rule cannot exclude the positive class until
at least 99 *positive* calibrations per finding; there are at most 58 panel
studies (about 46 in an outer training fold). Thus Tier 1 typically cannot
accept any predicted negative without contradicting its nominal bound; it
abstains. Similarly, zero observed mistakes among 58 independent cases has
one-sided 95% binomial upper error bound about 5%, not 1%, even **before**
per-finding positive scarcity, selection or multiple testing. Aggressive
coverage and very low per-finding risk are not simultaneously promised.

## 4. Evaluation and uncertainty

The expert panel is split into 5 outer patient/study-grouped folds. Strategies
A–D use the **same** tabular features, pseudo-label training studies, folds,
logistic learner and nominal 60%-coverage selection rule; they differ only in
pseudo-label/weight: A treats rejected release pseudo labels as negative,
B uses soft targets on all studies, C multiplies soft-label loss by confidence,
D uses only accepted binary pseudo labels weighted by OOF IPCW. Fractional
labels are optimized by duplicating each record into positive/negative
binomial losses with fractional sample weights. Models are fitted **once on
non-expert patient/study groups only**, excluding even non-expert studies
sharing a patient with the expert panel. D recomputes its propensity OOF
**inside that non-expert training pool**; using global weights for proxy
training would indirectly leak held-out expert acceptance through propensity
fits and normalization. The published full-release weight CSV remains a
separate downstream artifact. A–D use no expert truth and their frozen models
are evaluated over the five outer expert folds; only the mask's rank cutoffs
need to change with each outer expert fold. A proxy comparison requires at
least eight non-expert study/patient groups.
Training-set (not expert-truth-optimized) confidence quantiles set proxy
acceptance, so held-out coverage can differ from the nominal 60%. The proxy
has access to the release's soft scores, so it is **not a blind image model**;
these figures only diagnose training targets. The descriptive released mask
uses its expert evaluation predictions when supplied. The expert-panel mask
uses cutoffs calibrated on *other* outer folds; its `fold` column records
which fold generated each decision.

For each strategy compute selected accuracy, selected recall = accepted true
positives / **all** positives, unrestricted score recall/specificity, AUPRC,
10-bin ECE, selection ratio and accepted-positive precision. A separate
`finding_risk_checks.csv` checks both class-conditional risks **per finding**;
a pooled tier-average can otherwise hide an outlier. A study-cluster
bootstrap resamples entire studies (patient clusters when available), not
rows, for 95% percentile intervals. The bootstrap expresses uncertainty
about this **nonindependent development panel**, not corrected release-
development reuse. Per-finding and per-tier tables include actual counts;
undefined rates are blank/NaN rather than invented as zero. Risk/coverage
curves at other cutoffs are **descriptive only**, not conformal guarantees.

The source description's sanity target is 638 evaluated findings, 383 accepted
(60.03%) and 372 correct (97.13%). If counts differ, the evaluation report
names the actual counts and possible release/prediction-semantics differences,
never forces the target. **Accepted accuracy is not accepted-positive
precision.** The latter can be much lower even if most accepted negatives
are easy. A-D relative proxy recall cannot prove a future CNN's gains; if D
fails to preserve 97% empirical accepted-positive precision or improve a
finding, the report says so. When a CV mask cannot attain >= baseline
coverage at its chosen alphas, it reports the failure rather than relaxing
risk silently.

## Limits requiring independent oversight

Expert panel used during development, only 58 cases, unknown upstream score
calibration, unmeasured MNAR, site/scanner shifts, prevalence change,
non-random expert cohort, binary-score semantics, missing demographic fields,
strong class imbalance and pseudo-label error all limit inference. A
statistical guarantee (when assumptions hold) is **not** a clinical guarantee.

For research only; not for clinical use; requires expert radiologist oversight; statistical guarantees are not clinical guarantees.
