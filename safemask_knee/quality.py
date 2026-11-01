"""Aggregate input-quality diagnostics; never expose raw identifiers in reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DISCLAIMER, FINDINGS
from .io import ReleaseData, group_keys


def quality_record(data: ReleaseData) -> dict[str, Any]:
    """Summarize a strictly validated release without disclosing study IDs."""
    frame = data.frame
    studies = frame.drop_duplicates("study_id")
    n = int(frame.study_id.nunique())
    counts = frame.groupby("study_id", observed=True).finding.nunique()
    missing_metadata = {name: int(studies[name].isna().sum()) if name in studies else n
                        for name in ("age", "sex", "site", "scanner", "patient_id")}
    groups = group_keys(frame).groupby(frame.study_id, observed=True).first()
    cluster_size = groups.value_counts()
    duplicated_reports = 0
    affected_reports = 0
    crosses_expert = 0
    if "normalized_report_group" in studies:
        values = studies.normalized_report_group.dropna().astype(str)
        common = values.value_counts()
        duplicated_reports = int((common > 1).sum())
        affected_reports = int(common.loc[common > 1].sum())
        if "official_expert_evaluation_row" in studies:
            by_hash = studies.loc[studies.normalized_report_group.notna()].groupby(
                "normalized_report_group", observed=True)[
                    "official_expert_evaluation_row"].nunique()
            crosses_expert = int((by_hash > 1).sum())
    score_vectors = frame.pivot(index="study_id", columns="finding", values="soft_score")
    # Identical score vectors may reflect repeated text, NOT proven same patient.
    repeated_scores = int(score_vectors.duplicated(keep=False).sum())
    expert_ids = int(frame.loc[frame.expert_label.notna(), "study_id"].nunique())
    official = (int(studies.official_expert_evaluation_row.astype(str).eq("1").sum())
                if "official_expert_evaluation_row" in studies else None)
    findings = []
    for finding in FINDINGS:
        sub = frame.loc[frame.finding.eq(finding)]
        findings.append({
            "finding": finding, "n_studies": int(sub.study_id.nunique()),
            "n_accepted": int(sub.accepted60.sum()),
            "acceptance_rate": float(sub.accepted60.mean()),
            "score_min": float(sub.soft_score.min()),
            "score_max": float(sub.soft_score.max()),
            "confidence_min": float(sub.confidence.min()),
            "confidence_max": float(sub.confidence.max()),
            "nonfinite_score_or_confidence": int((~np.isfinite(
                sub[["soft_score", "confidence"]].to_numpy(dtype=float))).any(axis=1).sum()),
        })
    warnings = []
    if not data.frame.expert_label.notna().any():
        warnings.append("No official expert outcomes: error, MNAR and risk metrics unavailable")
    if "patient_id" not in studies or missing_metadata["patient_id"]:
        warnings.append("Patient independence cannot be established for missing patient IDs")
    if duplicated_reports:
        warnings.append("Duplicate normalized-report groups are clustered for CV; "
                        "this is not proof that all repeated patients were found")
    if crosses_expert:
        warnings.append("Identical report groups span official-development and weak "
                        "rows; group-level leakage must be considered")
    if "full_mri_ensemble_weak_target_training_exposure" in studies:
        exposed = pd.to_numeric(studies.full_mri_ensemble_weak_target_training_exposure,
                                errors="coerce").gt(0)
        if exposed.any():
            warnings.append("Some source MRI teacher components previously saw weak "
                            "targets; release-only CV cannot undo upstream exposure")
    record = {
        "disclaimer": DISCLAIMER,
        "status": "validated_input_quality_not_clinical_validation",
        "n_studies": n, "n_pairs": len(frame), "n_findings": len(FINDINGS),
        "all_studies_have_eleven_findings": bool(counts.eq(11).all()),
        "duplicate_study_finding_pairs": int(frame.duplicated(["study_id", "finding"]).sum()),
        "metadata_missing_studies": missing_metadata,
        "cv_clusters": int(groups.nunique()),
        "cv_largest_cluster_studies": int(cluster_size.max()),
        "normalized_report_duplicate_groups": duplicated_reports,
        "studies_in_duplicate_report_groups": affected_reports,
        "report_groups_crossing_development_weak": crosses_expert,
        "studies_with_identical_soft_vectors": repeated_scores,
        "official_development_studies_marked": official,
        "expert_truth_studies": expert_ids,
        "finding_quality": findings,
        "warnings": warnings,
    }
    return record


def write_quality(data: ReleaseData, out: Path) -> dict[str, Any]:
    """Write ID-free JSON and Markdown after strict loader/schema validation."""
    out.mkdir(parents=True, exist_ok=True)
    record = quality_record(data)
    (out / "data_quality.json").write_text(json.dumps(record, indent=2,
                                                      sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Release data quality (aggregate, not outcome validation)", "", DISCLAIMER, "",
             f"{record['n_studies']} studies × {record['n_findings']} findings = "
             f"{record['n_pairs']} pairs. Strict loader checks passed: complete "
             "study/finding keys, finite bounded scores, and matching release files.", "",
             f"CV: {record['cv_clusters']} connected study/patient/report groups "
             f"(largest contains {record['cv_largest_cluster_studies']} studies). "
             f"Repeated normalized reports: {record['normalized_report_duplicate_groups']} "
             f"groups / {record['studies_in_duplicate_report_groups']} studies; "
             f"{record['report_groups_crossing_development_weak']} groups cross "
             "the marked development/weak split. This does not establish "
             "independent patients if patient IDs are missing.", "",
             "| Finding | Studies | Accepted | Acceptance | Score range | Confidence range |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in record["finding_quality"]:
        lines.append(f"| {row['finding']} | {row['n_studies']} | {row['n_accepted']} | "
                     f"{row['acceptance_rate']:.1%} | "
                     f"{row['score_min']:.3f}–{row['score_max']:.3f} | "
                     f"{row['confidence_min']:.3f}–{row['confidence_max']:.3f} |")
    lines += ["", "## Metadata unavailable for these study counts", "",
              "; ".join(f"{k}: {v}" for k, v in record["metadata_missing_studies"].items()),
              "", "## Warnings", ""]
    lines += [f"- {item}" for item in record["warnings"]] or ["- None beyond "
                                                               "study-level caution."]
    lines += ["", "Only aggregates appear in this report; it does not disclose "
              "raw study IDs, expert labels, predictions or patient metadata. "
              "The 58 development cases cannot substitute for independent "
              "validation. See input hashes in `outputs/data_manifest.json`.", "",
              DISCLAIMER, ""]
    (out / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")
    return {"path": str(out / "data_quality.md"), "n_studies": record["n_studies"],
            "cv_clusters": record["cv_clusters"], "warnings": record["warnings"]}
