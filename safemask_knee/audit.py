"""Phase 1: expert-panel selection diagnostics and study-level fairness audit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind, ttest_rel

from .constants import DISCLAIMER, FINDINGS
from .io import ReleaseData, group_keys

LOG = logging.getLogger(__name__)


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return float("nan")
    scale = np.sqrt((np.var(a) + np.var(b)) / 2)
    return float((np.mean(a) - np.mean(b)) / scale) if scale > 0 else 0.0


def _fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini--Hochberg adjusted exploratory p-values, retaining NaNs."""
    out = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float).sort_values()
    if valid.empty:
        return out
    n = len(valid)
    ordered = np.minimum.accumulate(
        (valid.to_numpy()[::-1] * n / np.arange(n, 0, -1)))[::-1]
    out.loc[valid.index] = np.clip(ordered, 0, 1)
    return out


def bias_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize acceptance correlated with expert outcome; not a causal MNAR test.

    Parameters
    ----------
    frame : DataFrame
        Validated long-format release, with expert labels for a panel only.

    Returns
    -------
    DataFrame
        Per-finding positive/negative acceptance, prevalence, SBR and errors.
    """
    rows: list[dict[str, Any]] = []
    for finding in FINDINGS:
        full = frame.loc[frame.finding == finding]
        panel = full.loc[full.expert_label.notna()]
        if panel.empty:
            soft_a = full.loc[full.accepted60.eq(1), "soft_score"].to_numpy()
            soft_r = full.loc[full.accepted60.eq(0), "soft_score"].to_numpy()
            rows.append({"finding": finding, "all_n": len(full),
                         "all_acceptance_rate": float(full.accepted60.mean()),
                         "expert_n": 0, "expert_positive_n": np.nan,
                         "expert_negative_n": np.nan,
                         "accepted_n": np.nan, "rejected_n": np.nan,
                         "accepted_prevalence": np.nan, "rejected_prevalence": np.nan,
                         "positive_acceptance_rate": np.nan,
                         "negative_acceptance_rate": np.nan,
                         "selection_bias_ratio": np.nan,
                         "fnr_inside_accepted": np.nan, "accepted_errors": np.nan,
                         "accepted_false_negatives": np.nan,
                         "accepted_false_positives": np.nan,
                         "mean_soft_accepted": float(np.mean(soft_a)) if len(soft_a) else np.nan,
                         "mean_soft_rejected": float(np.mean(soft_r)) if len(soft_r) else np.nan,
                         "soft_score_smd": _smd(soft_a, soft_r),
                         "fisher_p": np.nan,
                         "mnar_flag": "not_estimable_no_expert_truth",
                         "disclaimer": DISCLAIMER})
            continue
        y = panel.expert_label.to_numpy(dtype=int)
        a = panel.accepted60.to_numpy(dtype=bool)
        p = panel.released_prediction.to_numpy(dtype=int)
        positive = y == 1
        neg = ~positive
        accepted_pos = int((a & positive).sum())
        rejected_pos = int((~a & positive).sum())
        accepted_neg = int((a & neg).sum())
        rejected_neg = int((~a & neg).sum())
        pos_rate = _ratio(accepted_pos, int(positive.sum()))
        neg_rate = _ratio(accepted_neg, int(neg.sum()))
        sbr = _ratio(pos_rate, neg_rate)
        if positive.sum() and neg.sum():
            p_value = float(fisher_exact([[accepted_pos, rejected_pos],
                                          [accepted_neg, rejected_neg]])[1])
        else:
            p_value = float("nan")
        soft_a = full.loc[full.accepted60.eq(1), "soft_score"].to_numpy()
        soft_r = full.loc[full.accepted60.eq(0), "soft_score"].to_numpy()
        rows.append({
            "finding": finding, "all_n": len(full),
            "all_acceptance_rate": float(full.accepted60.mean()),
            "expert_n": len(panel), "expert_positive_n": int(positive.sum()),
            "expert_negative_n": int(neg.sum()),
            "accepted_n": int(a.sum()), "rejected_n": int((~a).sum()),
            "accepted_prevalence": _ratio(accepted_pos, int(a.sum())),
            "rejected_prevalence": _ratio(rejected_pos, int((~a).sum())),
            "positive_acceptance_rate": pos_rate,
            "negative_acceptance_rate": neg_rate,
            "selection_bias_ratio": sbr,
            "fnr_inside_accepted": _ratio(int((a & positive & (p == 0)).sum()),
                                          accepted_pos),
            "accepted_errors": int((a & (p != y)).sum()),
            "accepted_false_negatives": int((a & positive & (p == 0)).sum()),
            "accepted_false_positives": int((a & neg & (p == 1)).sum()),
            "mean_soft_accepted": float(np.mean(soft_a)) if len(soft_a) else np.nan,
            "mean_soft_rejected": float(np.mean(soft_r)) if len(soft_r) else np.nan,
            "soft_score_smd": _smd(soft_a, soft_r),
            "fisher_p": p_value,
            "mnar_flag": "positive_underselected" if sbr < 0.8 else "not_flagged",
            "disclaimer": DISCLAIMER,
        })
    result = pd.DataFrame(rows)
    result["fdr_q"] = _fdr(result.fisher_p)
    return result


def _study_rates(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[["study_id", "accepted60"]].copy()
    keys["_cluster"] = group_keys(frame).to_numpy()
    rates = keys.groupby("study_id", observed=True, sort=True).agg(
        acceptance_rate=("accepted60", "mean"),
        accepted_n=("accepted60", "sum"), cluster=("_cluster", "first")).reset_index()
    for meta in ("age", "sex", "site", "scanner", "report_script_bucket",
                 "report_length_bucket", "multi_report_context_flag",
                 "verified_mri_teacher_available", "official_expert_evaluation_row",
                 "full_mri_ensemble_weak_target_training_exposure"):
        if meta in frame:
            rates[meta] = frame.groupby("study_id", observed=True, sort=True)[meta].first().to_numpy()
    if "age" in rates:
        rates["age_band"] = pd.cut(rates["age"], [0, 30, 45, 60, np.inf],
                                   labels=["<30", "30-44", "45-59", "60+"],
                                   right=False, include_lowest=True)
    return rates


def fairness_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Find descriptive disparities; inferential tests use studies, not 11 rows.

    Soft-score/confidence bins are descriptive only because their row-level
    observations are dependent within studies. Metadata tests use study-level
    acceptance rates (patient-cluster means when repeated patients exist).
    """
    base = _study_rates(frame)
    rows: list[dict[str, Any]] = []

    def add(attribute: str, stratum: str, sub: pd.DataFrame,
            reference: pd.DataFrame, method: str, p_value: float,
            study_level: bool) -> None:
        if sub.empty:
            return
        a = sub.acceptance_rate.to_numpy() if study_level else sub.accepted60.to_numpy()
        b = (reference.acceptance_rate.to_numpy() if study_level else
             reference.accepted60.to_numpy())
        rows.append({
            "attribute": attribute, "stratum": str(stratum),
            "n_studies": int(sub.study_id.nunique()), "n_pairs": int(len(sub) * 11)
            if study_level else len(sub),
            "accepted_n": int(sub.accepted_n.sum()) if study_level
            else int(sub.accepted60.sum()),
            "acceptance_rate": float(np.mean(a)),
            "reference_rate": float(np.mean(b)) if len(b) else np.nan,
            "rate_difference": float(np.mean(a) - np.mean(b)) if len(b) else np.nan,
            "standardized_difference": _smd(a, b),
            "p_value": p_value, "test": method,
            "note": ("patient/study-clustered exploratory comparison" if study_level
                     else "descriptive only; repeated measures within study"),
            "disclaimer": DISCLAIMER,
        })

    for finding in FINDINGS:
        sub = frame.loc[frame.finding == finding]
        other = frame.loc[frame.finding != finding].groupby("study_id", observed=True)[
            "accepted60"].mean().reindex(sub.study_id).to_numpy()
        p = float(ttest_rel(sub.accepted60.to_numpy(), other).pvalue)
        ref = sub[["study_id"]].copy()
        ref["accepted60"] = other
        add("finding", finding, sub, ref, "paired study t-test (exploratory)", p, False)
    for key in ("soft_score", "confidence"):
        binned = frame.assign(_bin=pd.cut(frame[key], bins=np.linspace(0, 1, 11),
                                          include_lowest=True))
        for bin_name, sub in binned.groupby("_bin", observed=True):
            ref = binned.loc[binned._bin != bin_name]
            add(f"{key}_bin", str(bin_name), sub, ref, "not tested (cluster dependence)",
                float("nan"), False)
    operational = ("report_script_bucket", "report_length_bucket",
                   "multi_report_context_flag", "verified_mri_teacher_available",
                   "official_expert_evaluation_row",
                   "full_mri_ensemble_weak_target_training_exposure")
    for attribute in ("age_band", "sex", "site", "scanner", *operational):
        if attribute not in base or base[attribute].notna().sum() == 0:
            continue
        groups = base.dropna(subset=[attribute]).copy()
        # A cluster spanning different metadata strata cannot be treated as two
        # independent units. Use only one deterministic stratum per cluster for
        # the exploratory p-value; the descriptive rates still use all studies.
        cluster = groups.groupby("cluster", observed=True, sort=True).agg(
            acceptance_rate=("acceptance_rate", "mean"),
            stratum=(attribute, "first")).reset_index()
        for label, sub in groups.groupby(attribute, observed=True):
            ref = groups.loc[groups[attribute] != label]
            a_cluster = cluster.loc[cluster.stratum == label, "acceptance_rate"]
            b_cluster = cluster.loc[cluster.stratum != label, "acceptance_rate"]
            p = float(ttest_ind(a_cluster, b_cluster, equal_var=False).pvalue) if (
                len(a_cluster) >= 2 and len(b_cluster) >= 2) else np.nan
            add(attribute, str(label), sub, ref,
                "Welch t-test on patient/study clusters (exploratory)", p, True)
    result = pd.DataFrame(rows)
    result["fdr_q"] = _fdr(result.p_value)
    result["flag"] = "not_flagged"
    meta = result.attribute.isin(["age_band", "sex", "site", "scanner", *operational])
    low = meta & result.rate_difference.lt(-0.05)
    result.loc[low, "flag"] = "possible_under_acceptance"
    result.loc[low & result.fdr_q.lt(0.10), "flag"] = "under_acceptance_exploratory"
    return result


def _figures(bias: pd.DataFrame, fairness: pd.DataFrame, out: Path,
             cfg: dict[str, Any]) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    bins = fairness.loc[fairness.attribute.eq("soft_score_bin")]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(bins)), bins.acceptance_rate, marker="o", label="Release acceptance")
    ax.set_xticks(range(len(bins)), bins.stratum, rotation=60, ha="right")
    ax.set_ylabel("Proportion accepted")
    ax.set_title("Acceptance vs soft-score bin (all studies)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figures / "acceptance_vs_soft_score.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    loc = np.arange(len(bias))
    if bias.expert_n.sum():
        ax.barh(loc - 0.18, bias.accepted_prevalence, 0.35, label="Accepted")
        ax.barh(loc + 0.18, bias.rejected_prevalence, 0.35, label="Rejected")
    else:
        ax.text(.5, .5, "Not estimable: expert ground truth absent",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_yticks(loc, bias.finding)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Expert-positive prevalence (58-study panel)")
    if bias.expert_n.sum():
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "prevalence_accepted_rejected.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if bias.expert_n.sum():
        ax.barh(bias.finding, bias.selection_bias_ratio)
        ax.axvline(1, color="black", ls="--", lw=1)
    else:
        ax.text(.5, .5, "Not estimable: expert ground truth absent",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("P(accepted|expert+) / P(accepted|expert-)")
    ax.set_title("Descriptive selection-bias ratio (not a causal MNAR proof)")
    fig.tight_layout()
    fig.savefig(figures / "selection_bias_by_finding.png", dpi=140)
    plt.close(fig)

    meta = fairness.loc[fairness.attribute.isin([
        "age_band", "sex", "site", "scanner", "report_script_bucket",
        "report_length_bucket", "multi_report_context_flag",
        "verified_mri_teacher_available", "official_expert_evaluation_row",
        "full_mri_ensemble_weak_target_training_exposure"])]
    fig, ax = plt.subplots(figsize=(9, max(3, min(11, 2 + .28 * len(meta)))))
    if not meta.empty:
        labels = meta.attribute.astype(str) + ": " + meta.stratum.astype(str)
        ax.barh(labels, meta.acceptance_rate)
        ax.axvline(float(cfg["coverage_target"]), c="gray", ls="--", lw=1)
    else:
        ax.text(0.5, 0.5, "No supplied metadata strata", ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Study-level acceptance fraction")
    fig.tight_layout()
    fig.savefig(figures / "metadata_acceptance_rates.png", dpi=140)
    plt.close(fig)


def _unlabeled_audit(data: ReleaseData, bias: pd.DataFrame,
                     fairness: pd.DataFrame, out: Path) -> dict[str, Any]:
    """Report score/acceptance disparities without inventing expert outcomes."""
    operational = {"report_script_bucket", "report_length_bucket",
                   "multi_report_context_flag", "verified_mri_teacher_available",
                   "official_expert_evaluation_row",
                   "full_mri_ensemble_weak_target_training_exposure"}
    flagged = fairness.loc[fairness.flag.ne("not_flagged") &
                           fairness.attribute.isin(operational)]
    strata = [f"{r.attribute}={r.stratum}: {r.acceptance_rate:.1%} accepted "
              f"({r.rate_difference:+.1%} vs other observed strata; {r.flag})"
              for r in flagged.itertuples()]
    source_expert = 0
    if "official_expert_evaluation_row" in data.frame:
        source_expert = int(data.frame.loc[
            data.frame.official_expert_evaluation_row.astype(str).eq("1"),
            "study_id"].nunique())
    missing_demographic = [name for name in ("age", "sex", "site", "scanner")
                           if name not in data.frame or
                           data.frame[name].notna().sum() == 0]
    demographic_note = ("Missing demographic fields: " + ", ".join(missing_demographic) +
                        ". ") if missing_demographic else ""
    group_note = ("Identical normalized reports are kept in one CV cluster "
                  "where a duplicate hash is supplied. " if
                  "normalized_report_group" in data.frame else "")
    group_note += ("Patient IDs were not supplied, so independence across "
                   "patients is unverified." if "patient_id" not in data.frame
                   else "Known patient IDs are kept in the same CV cluster.")
    prediction_note = ("The supplied expert_oof.csv contains PREDICTIONS ONLY. "
                       if "context_expert_oof.csv" in data.files else "")
    card = data.files.get("context_DATASET_CARD.txt")
    license_note = ("Source curated predictions are CC BY-NC 4.0 per the "
                    "supplied dataset card; " if card and
                    "CC BY-NC 4.0" in card.read_text(encoding="utf-8") else
                    "Review the supplied data's terms separately; ")
    lines = ["# Selection audit — expert truth unavailable", "", DISCLAIMER, "",
             f"**Input:** {data.frame.study_id.nunique()} studies / {len(data.frame)} "
             f"finding pairs. Source metadata mark {source_expert} development "
             "studies. " + prediction_note +
             "No per-finding expert truth was provided.", "",
             "**Verdict: MNAR erasure, accepted mistake counts, sensitivity, "
             "specificity, and clinical risk CANNOT be evaluated on this "
             "release alone.** Source-reported aggregate accuracy is not "
             "an expert label per study and is not independently reproduced here. "
             "Rejects are unknown, not expert negatives.", "",
             "## Label-free acceptance by finding", "",
             "| Finding | Studies | Accepted fraction | Accepted-vs-rejected soft-score SMD |",
             "|---|---:|---:|---:|"]
    for row in bias.itertuples():
        lines.append(f"| {row.finding} | {row.all_n} | "
                     f"{row.all_acceptance_rate:.2%} | {row.soft_score_smd:.3f} |")
    lines += ["", "The score contrast is NOT expert-positive prevalence or "
              "proof of outcome-dependent selection. The ground-truth-only "
              "columns of `bias_table.csv` are BLANK (NaN), never zero. "
              "`fairness_table.csv` tabulates acceptance by score/confidence "
              "bins and available operational metadata.", "",
              "## Available metadata strata (not demographic fairness)", "",
              "Under-accepted operational strata (< -5 pp vs the observed "
              "complement; cluster-aware exploratory BH tests): " +
              ("; ".join(strata) if strata else "none flagged") + ".", "",
              demographic_note + "Acceptance disparities alone do not establish "
              "demographic fairness in outcome errors. Report-script, "
              "report-length, duplicated-report, and MRI-teacher-exposure "
              "strata are operational covariates, not protected groups. "
              + group_note, "",
              "![Acceptance bins](figures/acceptance_vs_soft_score.png)", "",
              "![Operational acceptance](figures/metadata_acceptance_rates.png)", "",
              "The other two figures explicitly show that expert truth is "
              "not available. See supplied notices copied to ATTRIBUTION.md. "
              + license_note + "MIT covers toolkit code, NOT derived inputs "
              "or their downstream reuse.", "",
              DISCLAIMER, ""]
    (out / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    LOG.warning("No expert truth: label-free selection audit only; outcome-dependent "
                "MNAR and accepted-label error counts NOT estimated")
    return {"status": "unlabeled_audit_only", "expert_truth_available": False,
            "positive_underselected_findings": None,
            "accepted_expert_errors": None,
            "operational_flags": strata,
            "bias_table": str(out / "bias_table.csv")}


def run_audit(data: ReleaseData, out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Write per-finding/fairness CSVs, four PNGs, and a research-only audit."""
    out.mkdir(parents=True, exist_ok=True)
    bias = bias_table(data.frame)
    fairness = fairness_table(data.frame)
    bias.to_csv(out / "bias_table.csv", index=False, float_format="%.6g")
    fairness.to_csv(out / "fairness_table.csv", index=False, float_format="%.6g")
    _figures(bias, fairness, out, cfg)
    if not data.frame.expert_label.notna().any():
        return _unlabeled_audit(data, bias, fairness, out)
    flagged = bias.loc[bias.mnar_flag.eq("positive_underselected"), "finding"].tolist()
    low = fairness.loc[fairness.flag.ne("not_flagged") & fairness.attribute.isin(
        ["age_band", "sex", "site", "scanner"]), ["attribute", "stratum", "flag"]]
    groups = [f"{r.attribute}={r.stratum} ({r.flag})" for r in low.itertuples()]
    errors = bias.loc[bias.accepted_errors.gt(0),
                      ["finding", "accepted_errors", "accepted_false_negatives",
                       "accepted_false_positives"]]
    lines = ["# Selective-label audit", "", DISCLAIMER, "",
             f"**Scope:** {data.frame.study_id.nunique()} release studies / "
             f"{len(data.frame)} pairs; {data.frame.expert_label.notna().sum()} evaluated "
             "expert pairs. Expert studies were used in release development; this "
             "is a descriptive audit, not an independent validation set.", "",
             "## Verdict: possible silent disease erasure", "",
             "The following findings have a descriptive positive/negative acceptance "
             "ratio below 0.80: **" + (", ".join(flagged) or "none observed") + "**. "
             "This is evidence of outcome-associated selection on the small expert "
             "panel, not identification of MNAR conditional on unmeasured factors. "
             "Never turn an unaccepted finding into a verified negative.", "",
             "| Finding | Expert positives | P(accepted|+) | P(accepted|−) | Ratio | "
             "Accepted errors | Fisher p / FDR q |",
             "|---|---:|---:|---:|---:|---:|---:|" ]
    for r in bias.itertuples():
        lines.append(f"| {r.finding} | {r.expert_positive_n} | "
                     f"{r.positive_acceptance_rate:.3f} | {r.negative_acceptance_rate:.3f} | "
                     f"{r.selection_bias_ratio:.3f} | {r.accepted_errors} | "
                     f"{r.fisher_p:.3g} / {r.fdr_q:.3g} |")
    lines += ["", "## Released accepted-label errors by finding", "",
              f"Observed {int(bias.accepted_errors.sum())} accepted mistakes "
              "(the source description reports 11; a difference indicates different "
              "release/evaluation semantics).", "",
              "| Finding | Mistakes | Accepted false negatives | Accepted false positives |",
              "|---|---:|---:|---:|"]
    for r in errors.itertuples():
        lines.append(f"| {r.finding} | {r.accepted_errors} | "
                     f"{r.accepted_false_negatives} | {r.accepted_false_positives} |")
    lines += ["", "## Metadata disparities", "",
              "Under-accepted strata (difference < -5 percentage points vs other "
              "observed strata; exploratory cluster-level tests with BH adjustment): "
              + ("; ".join(groups) if groups else "none flagged, or metadata unavailable") + ".",
              "Missing metadata: " + ("; ".join(data.warnings) or "none reported") + ".",
              "Strata with no supplied data are omitted, not imputed as normal.", "",
              "## Interpretation and files", "",
              "`bias_table.csv` includes accepted/rejected expert prevalence, selection "
              "ratios and errors for all eleven findings. `fairness_table.csv` includes "
              "finding, score/confidence bins, and available metadata strata. "
              "Inferences on 11 rows per study would be pseudoreplication; metadata "
              "tests instead compare patient/study-cluster means. These exploratory "
              "p-values do not establish causation or equitable clinical performance.", "",
              "![Acceptance bins](figures/acceptance_vs_soft_score.png)", "",
              "![Expert prevalence](figures/prevalence_accepted_rejected.png)", "",
              "![Selection bias](figures/selection_bias_by_finding.png)", "",
              "![Metadata acceptance](figures/metadata_acceptance_rates.png)", "",
              "Source: user-supplied RSNA knee MRI release; see generated "
              "`ATTRIBUTION.md` for copied notices.", "", DISCLAIMER, ""]
    (out / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Audit: %s findings with positive under-acceptance; %s metadata flags; "
             "%s accepted expert errors", len(flagged), len(groups), int(bias.accepted_errors.sum()))
    return {"positive_underselected_findings": flagged, "metadata_flags": groups,
            "accepted_expert_errors": int(bias.accepted_errors.sum()),
            "bias_table": str(out / "bias_table.csv")}
