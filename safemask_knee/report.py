"""Final research summary using previously generated audited artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DISCLAIMER, FINDINGS
from .io import ReleaseData

LOG = logging.getLogger(__name__)


def _percentage(value: Any) -> str:
    return f"{float(value):.2%}" if pd.notna(value) else "not estimable"


def write_summary(data: ReleaseData, output_root: Path,
                  out: Path) -> dict[str, Any]:
    """Render headlines, findings, disparities, cutoffs and failure modes.

    Parameters
    ----------
    data : ReleaseData
        Loaded only to confirm the same release/study counts.
    output_root : Path
        Directory containing audit, weight, mask and evaluation outputs.
    out : Path
        User-requested report path, often reports/summary.md.

    Raises
    ------
    FileNotFoundError
        If the pipeline has not yet generated all required artifacts.
    """
    names = {
        "bias": output_root / "audit/bias_table.csv",
        "fairness": output_root / "audit/fairness_table.csv",
        "weights": output_root / "debiased_training_weights.csv",
        "masks": output_root / "clinical_risk_masks.csv",
        "metrics": output_root / "evaluation/metrics_by_strategy.csv",
        "tier": output_root / "evaluation/tier_metrics.csv",
        "per_finding": output_root / "evaluation/per_finding_metrics.csv",
        "finding_risk": output_root / "evaluation/finding_risk_checks.csv",
        "diagnostics": output_root / "conformal_diagnostics.json",
        "manifest": output_root / "data_manifest.json",
    }
    missing = [str(path) for path in names.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Run `all` first; missing report inputs: " + ", ".join(missing))
    bias = pd.read_csv(names["bias"])
    fairness = pd.read_csv(names["fairness"])
    weights = pd.read_csv(names["weights"])
    masks = pd.read_csv(names["masks"])
    metrics = pd.read_csv(names["metrics"])
    tier = pd.read_csv(names["tier"])
    per_finding = pd.read_csv(names["per_finding"])
    finding_risk = pd.read_csv(names["finding_risk"])
    diagnostics = json.loads(names["diagnostics"].read_text(encoding="utf-8"))
    if len(weights) != len(data.frame) or len(masks) != len(data.frame):
        raise ValueError("Saved artifact row count does not match the release; rerun all")
    baseline = metrics.set_index("strategy").loc["released_accepted60"]
    clinical = metrics.set_index("strategy").loc["clinical_risk_mask"]
    naive = metrics.set_index("strategy").loc["A_naive_accepted_binary"]
    ours = metrics.set_index("strategy").loc["D_IPCW_weighted"]
    issues = bias.loc[bias.mnar_flag.eq("positive_underselected"), "finding"].tolist()
    flagged = fairness.loc[fairness.flag.ne("not_flagged") & fairness.attribute.isin(
        ["age_band", "sex", "site", "scanner"])]
    risk_violations = finding_risk.loc[~finding_risk.empirical_bounds_met, "finding"].tolist()
    accepted = weights.loc[weights.accepted60.eq(1)]
    gain = float(clinical.coverage - baseline.coverage)
    ci_low = baseline.accuracy_at_coverage_ci_low
    ci_high = baseline.accuracy_at_coverage_ci_high
    lines = ["# SafeMask-Knee: research summary", "", DISCLAIMER, "",
             "**Source and licensing:** Derived from the user-supplied RSNA knee "
             "MRI competition label release. Source CSVs and images are not "
             "redistributed. Review generated `ATTRIBUTION.md` for copied "
             "source notices; the toolkit code is MIT licensed.", "",
             "## Headline results (58-study expert *development* panel)", "",
             "| Strategy | Accepted / evaluated | Coverage | Accepted accuracy | "
             "Selected recall | Accepted-positive precision |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in (baseline, naive, ours, clinical):
        lines.append(f"| {row.name} | {int(row.n_accepted)} / {int(row.n_evaluated)} | "
                     f"{_percentage(row.coverage)} | "
                     f"{_percentage(row.accuracy_at_coverage)} | "
                     f"{_percentage(row.selected_recall)} | "
                     f"{_percentage(row.accepted_positive_precision)} |")
    lines += ["", f"Released accepted correctness: {int(baseline.n_accepted - baseline.n_errors_accepted)}"
              f"/{int(baseline.n_accepted)}, with "
              f"{_percentage(ci_low)}–{_percentage(ci_high)} 95% "
              "**study-cluster bootstrap** interval for accepted accuracy. "
              "This is a descriptive check, **not independent validation**: "
              "the 58 studies were used in release development.",
              f"Clinical-mask outer-CV coverage change: {gain:+.2%} vs released mask. "
              "If negative, the configured risk constraint makes the requested "
              "coverage increase infeasible in these data; no unsafe fallback "
              "was applied. The aspirational 70–85% overall range is " +
              ("reached at its lower edge." if clinical.coverage >= .70 else
               "not reached; Tier 1's discrete 1% rule sacrifices negative coverage."),
              "",
              "## Eleven released-mask mistakes and possible silent erasure", "",
              f"Observed accepted mistakes: **{int(bias.accepted_errors.sum())}**. "
              "False negatives/positives below describe the released accepted mask; "
              "OOF-mask residual mistakes are separately tabulated (may differ from "
              "the described 11 if the "
              "source/evaluation semantics differ).", "",
             "| Finding | Released mistakes | OOF mask mistakes | Accepted false negatives | "
             "Accepted false positives | Positive/negative selection ratio |",
             "|---|---:|---:|---:|---:|---:|"]
    mask_error = finding_risk.set_index("finding").clinical_errors_accepted
    for row in bias.itertuples():
        lines.append(f"| {row.finding} | {row.accepted_errors} | "
                     f"{int(mask_error.loc[row.finding])} | "
                     f"{row.accepted_false_negatives} | {row.accepted_false_positives} | "
                     f"{row.selection_bias_ratio:.3f} |")
    lines += ["", "**Under-selected positives** (descriptive ratio <0.80): " +
              (", ".join(issues) or "none flagged") + ". This cannot establish "
              "MNAR conditional on unobserved factors. An unaccepted finding is "
              "never treated as a confirmed negative in the downstream masks.", "",
              "**Metadata under-acceptance flags** (exploratory): " +
              ("; ".join(f"{r.attribute}={r.stratum}: {r.flag} "
                         f"({r.rate_difference:+.1%} vs other strata)" for r in
                         flagged.itertuples()) if not flagged.empty else
               "none, or the corresponding metadata were not supplied") + ". "
              "Missing metadata fields are skipped, not asserted to be fair.", "",
              "## IPCW and 60%-coverage proxy comparison", "",
              "OOF propensities are grouped by study/patient, with a separate "
              "study-grouped Platt holdout. Accepted weights are "
              "P(accepted for finding)/P(accepted|X), clipped and "
              "**normalized to mean 1 within finding**; rejected weights are 0 "
              "(they have no verified release training label). "
              f"Accepted weight range: {accepted.weight_final.min():.3f}–"
              f"{accepted.weight_final.max():.3f}, median "
              f"{accepted.weight_final.median():.3f}. "
              "This corrects only selection explainable by observed X, not "
              "unmeasured MNAR or classifier label error.", "",
              f"Selection ratios: released {baseline.selection_bias_ratio:.3f}; "
              f"OOF-IPCW-adjusted expert estimate "
              f"{ours.ipcw_adjusted_selection_ratio:.3f}. "
              "A-D are **tabular proxy models trained only on non-expert "
              "patient/study groups using released pseudo-labels**; D refits "
              "training-only IPCW. They are not MRI models; a gain in proxy recall "
              "does not imply an image model "
              "will improve. D vs A selected-recall difference: "
              f"{ours.selected_recall - naive.selected_recall:+.2%}. "
              f"D accepted-positive precision: {_percentage(ours.accepted_positive_precision)}; "
              "see the evaluation report if the empirical 97% target was missed.", "",
              "| Finding | IPCW accepted weight mean | 95th percentile | Maximum | "
              "Effective sample size | D−A selected recall | Weighted selection ratio |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    a_find = per_finding.loc[per_finding.strategy.eq("A_naive_accepted_binary")].set_index("finding")
    d_find = per_finding.loc[per_finding.strategy.eq("D_IPCW_weighted")].set_index("finding")
    for name in FINDINGS:
        sub = accepted.loc[accepted.finding == name, "weight_final"]
        delta = d_find.loc[name, "selected_recall"] - a_find.loc[name, "selected_recall"]
        ess = float(sub.sum() ** 2 / (sub ** 2).sum()) if len(sub) else float("nan")
        ratio = d_find.loc[name, "ipcw_adjusted_selection_ratio"]
        lines.append(f"| {name} | {sub.mean():.3f} | {sub.quantile(.95):.3f} | "
                     f"{sub.max():.3f} | {ess:.1f} | {delta:+.2%} | "
                     f"{ratio:.3f} |")
    lines += ["", "## Outer-fold tiered mask and final non-expert cutoffs", "",
              "Conformal *class-conditional marginal wrong-singleton* bounds assume "
              "fixed scores and exchangeability; they do **not** bound the error "
              "conditional on acceptance or certify 97% accepted accuracy. "
              "Tier 1 bounds accepted-negative FNR among all positives at 1%; "
              "with <99 positive calibrations/finding, discrete ranks typically "
              "forbid accepting predicted negatives. This is not Jackknife+ "
              "(which requires underlying model refits).", "",
              "| Tier | Released coverage | Outer-CV coverage | Accepted accuracy | "
              "Observed marginal FNR / bound | Observed marginal FPR / bound |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in tier.itertuples():
        lines.append(f"| {r.tier} | {_percentage(r.baseline_coverage)} | "
                     f"{_percentage(r.clinical_coverage_outer_oof)} | "
                     f"{_percentage(r.conditional_accuracy_accepted)} | "
                     f"{_percentage(r.observed_marginal_fnr)} / {_percentage(r.nominal_fnr_bound)} "
                     f"| {_percentage(r.observed_marginal_fpr)} / {_percentage(r.nominal_fpr_bound)} |")
    lines += ["", "Per-finding OOF risk violations/undefined classes: " +
              (", ".join(risk_violations) if risk_violations else "none observed") +
              ". See `evaluation/finding_risk_checks.csv`; a pooled tier rate "
              "cannot certify every finding. Even an empty violation list is "
              "not independent or clinical validation.", "",
              "Final (non-expert only) score cutoffs below are descriptive. "
              "They are **not** evaluated on the same panel that calibrated them.", "",
              "| Finding | Tier | Accept negative if score <= | "
              "Accept positive if score >= | Positive calibration n |",
              "|---|---:|---:|---:|---:|"]
    for name, stats in diagnostics["full_panel_cutoffs_for_nonexpert_only"].items():
        low = stats["negative_score_ceiling"]
        high = stats["positive_score_floor"]
        low_text = f"{low:.4f}" if low is not None else "unavailable"
        high_text = f"{high:.4f}" if high is not None else "unavailable"
        lines.append(f"| {name} | {stats['tier']} | {low_text} | {high_text} | "
                     f"{stats['n_positive']} |")
    lines += ["", "**Coverage/accuracy targets are empirical goals, not constraints "
              "that can always be met.** Read `evaluation/tier_metrics.csv` and "
              "`evaluation/eval_report.md` for unmet goals and source-count "
              "discrepancies. Scores, confidence, and released acceptances are "
              "derived label metadata only; no DICOM/image model is involved.", "",
              "## Three central limitations", "",
              "1. **Upstream reuse:** the expert development panel helped build "
              "the source release. Outer study folds and LOO prevent this toolkit "
              "from directly scoring its own fitted threshold, but cannot "
              "undo upstream overfitting; an independent external panel is required.",
              "2. **Finite calibration and shift:** 58 studies cannot give useful "
              "1% per-finding false-negative certification with negative "
              "acceptance. Exchangeability, stable prevalence/sites/scanners, "
              "and a fixed scoring function are unverified assumptions.",
              "3. **Missing-outcome assumptions:** even excellent observed-X IPCW "
              "propensity and tabular proxy recall cannot remove unmeasured "
              "MNAR, prove fairness, or certify a downstream image model.", "",
              "See `METHODOLOGY.md`, the generated audit/evaluation reports, "
              "`outputs/data_manifest.json` and `ATTRIBUTION.md`.", "", DISCLAIMER, ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    # The repo layout requests both paths; do not make relative links to one
    # another or include release CSV rows in either document.
    mirror = output_root / "reports/summary.md"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    if mirror.resolve() != out.resolve():
        mirror.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Research summary written: %s", out)
    return {"path": str(out), "mirror": str(mirror),
            "release_accepted_accuracy": float(baseline.accuracy_at_coverage),
            "mask_coverage_oof": float(clinical.coverage),
            "n_accepted_errors": int(bias.accepted_errors.sum())}


def write_training_summary(data: ReleaseData, output_root: Path,
                           out: Path) -> dict[str, Any]:
    """Document the fitted IPCW artifact when NO expert truth was supplied.

    No conformal mask, accepted-error count, or label-outcome metric is ever
    inferred from a prediction-only expert_oof.csv or aggregate benchmark JSON.
    """
    if data.frame.expert_label.notna().any():
        raise ValueError("Use the full evaluation report when expert truth is available")
    if (output_root / "clinical_risk_masks.csv").exists() or (
            output_root / "evaluation/metrics_by_strategy.csv").exists():
        raise ValueError("A stale clinical mask/evaluation is present without "
                         "expert truth; use a fresh output_root for partial training")
    bias_path = output_root / "audit/bias_table.csv"
    fairness_path = output_root / "audit/fairness_table.csv"
    weights_path = output_root / "debiased_training_weights.csv"
    if not all(p.is_file() for p in (bias_path, fairness_path, weights_path)):
        raise FileNotFoundError("Run `train` first; label-free audit/weights are missing")
    bias = pd.read_csv(bias_path)
    fairness = pd.read_csv(fairness_path)
    weights = pd.read_csv(weights_path)
    accepted = weights.loc[weights.accepted60.eq(1)]
    diagnostic = json.loads((output_root / "propensity_diagnostics.json").read_text(
        encoding="utf-8"))["summary"]
    source_studies = pd.DataFrame()
    if "official_expert_evaluation_row" in data.frame:
        source_studies = data.frame.loc[
            data.frame.official_expert_evaluation_row.astype(str).eq("1")]
    published: dict[str, Any] | None = None
    benchmark = data.files.get("context_benchmark_results.json")
    if benchmark:
        obj = json.loads(benchmark.read_text(encoding="utf-8"))
        published = obj.get("selected", {}).get("accuracy_score", {})
    operational = {"report_script_bucket", "report_length_bucket",
                   "multi_report_context_flag", "verified_mri_teacher_available",
                   "official_expert_evaluation_row",
                   "full_mri_ensemble_weak_target_training_exposure"}
    flags = fairness.loc[fairness.flag.ne("not_flagged") &
                         fairness.attribute.isin(operational)]
    source_count = int(source_studies.study_id.nunique())
    source_accepted = int(source_studies.accepted60.sum())
    missing_demographic = [name for name in ("age", "sex", "site", "scanner")
                           if name not in data.frame or
                           data.frame[name].notna().sum() == 0]
    metadata_note = (("No supplied " + ", ".join(missing_demographic) + ". ")
                     if missing_demographic else
                     "Demographic metadata are supplied, but acceptance alone "
                     "cannot quantify outcome fairness. ")
    patient_note = ("Patient IDs were absent; patient independence remains "
                    "unverified. " if "patient_id" not in data.frame else
                    "Known patient IDs were kept in the same CV group. ")
    duplicate_note = ("Normalized duplicate-report hashes are also kept "
                      "together in CV. " if "normalized_report_group" in data.frame
                      else "")
    prediction_note = ("The supplied `expert_oof.csv` contains out-of-fold "
                       "predictions, NOT an official answer key. " if
                       "context_expert_oof.csv" in data.files else "")
    card = data.files.get("context_DATASET_CARD.txt")
    source_licence_note = ("The source DATASET_CARD states **curated predictions "
                           "are CC BY-NC 4.0**; original competition data remain "
                           "under separate terms. " if card and
                           "CC BY-NC 4.0" in card.read_text(encoding="utf-8") else
                           "Verify the input-data licence independently. ")
    lines = ["# SafeMask-Knee: trained propensity; expert outcomes unavailable", "",
             DISCLAIMER, "",
             "**STATUS: PARTIAL, NOT AN EXPERT-LABEL AUDIT OR CONFORMAL MASK.** "
             + prediction_note + "We fitted the acceptance-propensity "
             "model on release acceptance indicators with grouped OOF Platt "
             "calibration and wrote IPCW weights. We did **not** generate "
             "`clinical_risk_masks.csv`, claim any accepted accuracy, or "
             "evaluate a clinical risk bound.", "",
             f"Release: **{data.frame.study_id.nunique()} studies / {len(data.frame)} "
             f"finding pairs**. Source metadata identify **{source_count}** "
             f"official development studies with **{source_accepted}/"
             f"{len(source_studies)}** accepted pairs (acceptance only; no "
             "expert correctness could be computed). All-study acceptance "
             f"{data.frame.accepted60.mean():.2%}.", "",
             "## Fitted acceptance-propensity / IPCW artifact", "",
             f"`{weights_path}` has **{len(weights)}** rows, of which "
             f"**{len(accepted)}** are accepted. Accepted `weight_final` "
             f"ranges {accepted.weight_final.min():.3f}–"
             f"{accepted.weight_final.max():.3f}; within each finding it "
             "is bounded and mean-normalized to 1. Rejected weights are 0. "
             "Expert labels and outcome-derived CV folds are blank. "
             "The trained logistic acceptance classifiers and held-out Platt "
             "calibrators yield **out-of-fold propensities**; this CSV is an "
             "artifact for downstream training, not a trained MRI model. "
             "The original 60% mask's rejected labels are NOT verified negatives.", "",
             f"**Positivity warning:** {diagnostic['propensity_lower_clip_fraction']:.1%} "
             f"of OOF propensities hit the {diagnostic['propensity_floor']:.2f} "
             f"floor and {diagnostic['propensity_upper_clip_fraction']:.1%} hit "
             f"the {1 - diagnostic['propensity_floor']:.2f} ceiling; "
             f"{diagnostic['weight_at_max_fraction_accepted']:.1%} of accepted "
             "weights hit the configured maximum after normalization. The "
             "source mask is a near-deterministic top-confidence rank, "
             "not random missingness. IPCW cannot recover outcomes in "
             "regions with no accepted labels; this file is *not* an "
             "empirically validated debiasing guarantee.", "",
             "| Finding | Acceptance | IPCW 95th percentile | IPCW effective sample size |",
             "|---|---:|---:|---:|"]
    for row in bias.itertuples():
        sub = accepted.loc[accepted.finding.eq(row.finding), "weight_final"]
        ess = sub.sum() ** 2 / (sub ** 2).sum() if len(sub) else float("nan")
        lines.append(f"| {row.finding} | {row.all_acceptance_rate:.2%} | "
                     f"{sub.quantile(.95):.3f} | {ess:.1f} |")
    lines += ["", "## What the source reports versus what was verified here", "",
              "We independently checked input hashes/schema, fitted "
              "propensities and acceptance coverage. **We cannot reproduce "
              "accuracy without official expert labels.**"]
    if published:
        accuracy = published.get("fixed60_accuracy")
        cells = published.get("fixed60_cells")
        errors = published.get("fixed60_errors")
        if all(v is not None for v in (accuracy, cells, errors)):
            lines += [f"The source's own `benchmark_results.json` reports "
                      f"{float(accuracy):.2%} correct among {cells} selected "
                      f"with {errors} mistakes. Those are **unverified, "
                      "development-panel source assertions**, not results "
                      "recomputed by this toolkit; no comparable per-finding "
                      "error or MNAR claim is possible."]
    lines += ["", "Acceptance disparities by available report-script/length "
              "and other operational metadata (NOT protected-group fairness): " +
              ("; ".join(f"{r.attribute}={r.stratum}: "
                         f"{r.acceptance_rate:.1%} ({r.rate_difference:+.1%} "
                         "vs rest)" for r in flags.itertuples()) if not flags.empty
               else "none flagged") + ".", "",
              metadata_note + patient_note + duplicate_note +
              "No MNAR verdict or outcome-based demographic fairness finding "
              "is possible without expert truth.", "",
              "## How to complete the expert-dependent stages", "",
              "Obtain the official, separately licensed competition `train.csv` "
              "under its own terms. Provide an `expert_results.csv` with "
              "`StudyInstanceUID,finding,expert_label` covering exactly the "
              "58 development studies and all 11 findings (no Synovitis), "
              "or use the offline adapter `scripts/prepare_expert.py` to join "
              "those 58 IDs from the official CSV. Then run `all --release "
              "data/release --config configs/default.yaml`. Without that "
              "file, `masks`, `evaluate`, and `all` fail loudly rather than "
              "fabricate a conformal certificate.", "",
              "## Additional diagnostics", "",
              "`outputs/quality/data_quality.md` checks duplicate reports, "
              "missing patient/demographic metadata and upstream exposure; "
              "`outputs/sensitivity/sensitivity_report.md` compares alternative "
              "IPCW floors and caps without using truth. "
              "`outputs/run_provenance.json` records code/config/runtime versions "
              "and source hashes without source rows. Launch the optional "
              "read-only aggregate viewer via `python -m safemask_knee dashboard "
              "--output-root outputs`; it does not serve raw inputs or weights.", "",
              "## Attribution and limits", "",
              source_licence_note + "Generated labels/weights may inherit "
              "source-data restrictions. MIT licenses the toolkit *code only*. "
              "See `ATTRIBUTION.md` for the source's copied dataset card and "
              "THIRD_PARTY_NOTICES; no source CSV or DICOM is committed.", "",
              "Three limitations: (1) truth withheld / the familiar 58 "
              "studies were reused upstream; (2) IPCW requires selection "
              "ignorable conditional on measured X and cannot remove true "
              "unmeasured MNAR; (3) no expert calibration means **no "
              "risk-certified acceptance mask or trained MRI image model**.", "",
              DISCLAIMER, ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    mirror = output_root / "reports/summary.md"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    if mirror.resolve() != out.resolve():
        mirror.write_text(text, encoding="utf-8")
    LOG.info("Partial research summary: %s (no expert truth; NO clinical mask)", out)
    return {"status": "incomplete_missing_expert_truth", "path": str(out),
            "mirror": str(mirror), "source_expert_studies": source_count,
            "source_expert_accepted": source_accepted,
            "accepted_accuracy": None, "mask_path": None}
