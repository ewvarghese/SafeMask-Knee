"""Private, targeted study-group review planning from label-free artifacts.

This is an annotation *worklist*, not a clinical prediction or an evaluation
sample. Only the locally saved queue/template contain study identifiers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DISCLAIMER, FINDINGS, WEIGHT_COLUMNS
from .io import ReleaseData, group_keys, sha256_file


def _aligned_weights(data: ReleaseData, weights_path: Path) -> pd.DataFrame:
    """Require a complete, untampered key/value match to the loaded release."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Run `train` or `all` first; no weights CSV: {weights_path}")
    weights = pd.read_csv(weights_path, dtype={"study_id": str, "finding": str})
    if list(weights.columns) != list(WEIGHT_COLUMNS):
        raise ValueError("Saved weights must have the exact downstream twelve-column schema")
    if weights[["study_id", "finding"]].isna().any().any() or (
            weights[["study_id", "finding"]].eq("").any().any()):
        raise ValueError("Saved weights contain blank study/finding keys")
    if weights.duplicated(["study_id", "finding"]).any():
        raise ValueError("Saved weights contain duplicate study/finding keys")
    frame = data.frame[["study_id", "finding", "soft_score", "confidence",
                        "accepted60"]].copy()
    frame["finding"] = frame.finding.astype(str)
    if frame.duplicated(["study_id", "finding"]).any() or (
            not frame.groupby("study_id").finding.nunique().eq(len(FINDINGS)).all()) or (
            set(frame.finding) != set(FINDINGS)):
        raise ValueError("Review planning requires all eleven findings for every study")
    joined = frame.merge(weights, on=["study_id", "finding"], how="outer",
                         validate="one_to_one", suffixes=("", "_saved"), indicator=True)
    if len(joined) != len(frame) or not joined._merge.eq("both").all():
        raise ValueError("Saved weights (study_id, finding) keys do not match the release")
    joined = joined.drop(columns="_merge")
    for name in ("soft_score", "confidence", "accepted60"):
        actual = pd.to_numeric(joined[name + "_saved"], errors="coerce").to_numpy(float)
        expected = joined[name].to_numpy(float)
        if not np.isfinite(actual).all() or not np.allclose(
                actual, expected, rtol=0, atol=5e-8 if name != "accepted60" else 0):
            raise ValueError(f"Saved weights {name} differs from the release; rerun training")
    propensity = pd.to_numeric(joined.propensity_accepted, errors="coerce").to_numpy(float)
    weight = pd.to_numeric(joined.weight_final, errors="coerce").to_numpy(float)
    accepted = joined.accepted60.to_numpy(dtype=bool)
    if not np.isfinite(propensity).all() or ((propensity <= 0) | (propensity >= 1)).any():
        raise ValueError("Saved OOF propensity must be finite and strictly between 0 and 1")
    if not np.isfinite(weight).all() or (weight[accepted] <= 0).any() or (
            weight[~accepted] != 0).any():
        raise ValueError("Saved accepted weights must be positive; rejected weights must be 0")
    return joined


def make_review_plan(data: ReleaseData, weights: pd.DataFrame, n_groups: int,
                     seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Rank *connected study groups* for targeted, explicitly biased review."""
    if n_groups < 1:
        raise ValueError("--groups must be at least 1")
    frame = data.frame
    # group_keys is the same patient-ID / identical-report transitive grouping
    # used for training CV. The group key itself is never exported.
    group_map = pd.DataFrame({"study_id": frame.study_id.to_numpy(),
                              "group_key": group_keys(frame).to_numpy()}).drop_duplicates("study_id")
    group_map = group_map.set_index("study_id").group_key
    expert_ids = set(frame.loc[frame.expert_label.notna(), "study_id"])
    marker_present = ("official_expert_evaluation_row" in frame and
                      frame.official_expert_evaluation_row.notna().any())
    if marker_present:
        marked = frame.official_expert_evaluation_row.astype(str).str.lower().isin(
            ("1", "1.0", "true"))
        expert_ids.update(frame.loc[marked, "study_id"])
    blocked_groups = set(group_map.loc[list(expert_ids)]) if expert_ids else set()

    values = weights[["study_id", "finding", "soft_score", "accepted60",
                      "propensity_accepted", "weight_final"]].copy()
    values["score_ambiguity"] = 1 - (2 * values.soft_score - 1).abs()
    values["rare_acceptance"] = values.accepted60 * (1 - values.propensity_accepted)
    studies = values.groupby("study_id", sort=False).agg(
        accepted_findings=("accepted60", "sum"),
        mean_score_ambiguity=("score_ambiguity", "mean"),
        mean_oof_rare_acceptance=("rare_acceptance", "mean"),
        sum_accepted_weights=("weight_final", "sum")).reset_index()
    studies["rejected_findings"] = len(FINDINGS) - studies.accepted_findings
    studies["mean_accepted_weight"] = (studies.sum_accepted_weights /
                                       studies.accepted_findings.replace(0, np.nan))
    studies["priority_score"] = (.50 * studies.rejected_findings / len(FINDINGS)
                                 + .30 * studies.mean_score_ambiguity
                                 + .20 * studies.mean_oof_rare_acceptance)
    studies["group_key"] = studies.study_id.map(group_map)
    if studies.group_key.isna().any():
        raise AssertionError("Study group mapping is incomplete")
    eligible = studies.loc[~studies.group_key.isin(blocked_groups)].copy()
    if eligible.empty:
        raise ValueError("No non-development study groups are available for review")
    group_scores = eligible.groupby("group_key").priority_score.max().to_dict()
    order = sorted(group_scores, key=lambda group: (
        -group_scores[group], hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()))
    selected = order[:n_groups]
    ranks = {group: i + 1 for i, group in enumerate(selected)}
    queue = eligible.loc[eligible.group_key.isin(ranks)].copy()
    queue["group_rank"] = queue.group_key.map(ranks)
    queue["studies_in_group"] = queue.groupby("group_key").study_id.transform("size")
    queue = queue.sort_values(["group_rank", "priority_score", "study_id"],
                              ascending=[True, False, True], kind="stable")
    queue = queue[["group_rank", "study_id", "studies_in_group", "accepted_findings",
                   "rejected_findings", "mean_score_ambiguity",
                   "mean_oof_rare_acceptance", "mean_accepted_weight", "priority_score"]]
    queue = queue.rename(columns={"priority_score": "research_priority_score"})
    queue["disclaimer"] = DISCLAIMER
    # No pseudo-labels, scores or acceptance flags on the annotation sheet:
    # avoid anchoring the reviewing expert to the existing predictions.
    template = pd.DataFrame({"study_id": queue.study_id.to_numpy()})
    for finding in FINDINGS:
        template[finding] = ""
    summary = {"status": "targeted_review_only_not_validation",
               "disclaimer": DISCLAIMER,
               "requested_groups": n_groups,
               "selected_groups": len(selected),
               "selected_studies": len(queue),
               "eligible_groups": len(group_scores),
               "eligible_studies": len(eligible),
               "excluded_development_groups": len(blocked_groups),
               "excluded_development_studies": len(studies) - len(eligible),
               "development_marker_present": bool(marker_present),
               "expert_truth_present": bool(frame.expert_label.notna().any())}
    return queue, template, summary


def write_review_plan(data: ReleaseData, weights_path: Path, out: Path,
                      n_groups: int, seed: int) -> dict[str, Any]:
    """Write only to a local directory; never overwrite a filled-in template."""
    out = Path(out)
    if n_groups < 1:
        raise ValueError("--groups must be at least 1")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing review plan at {out}; "
                              "it may contain expert annotations. Move it aside first.")
    weights = _aligned_weights(data, weights_path)
    queue, template, summary = make_review_plan(data, weights, n_groups, seed)
    summary["source_weights_sha256"] = sha256_file(weights_path)
    report = ["# Targeted research review plan", "", DISCLAIMER, "",
              f"**{summary['selected_groups']} complete patient/study/report groups** "
              f"({summary['selected_studies']} studies) selected from "
              f"{summary['eligible_groups']} eligible groups. "
              f"{summary['excluded_development_groups']} groups touching a marked "
              "development/expert-labeled study were excluded in full.", "",
              "**Purpose:** a local worklist to request new independent expert "
              "annotations *when separately licensed original images are available*. "
              "This repository contains no MRI images or expert answers.", "",
              "The study priority is a fixed heuristic: 0.50 × rejected-finding "
              "fraction + 0.30 × mean score ambiguity "
              "(1 − |2×soft_score−1|) + 0.20 × mean of "
              "accepted60×(1−OOF propensity). Groups are ranked by their "
              "highest study priority; the seed breaks ties. A score reflects "
              "label-selection/ambiguity signals, **not** disease probability, "
              "human-review urgency or a validated error risk.", "",
              "`review_queue.csv` contains local study IDs and aggregate per-study "
              "selection diagnostics for an authorized coordinator. "
              "`expert_label_template.csv` contains IDs and **blank** fields for "
              "all 11 findings; it intentionally hides weak predictions from "
              "the expert annotator. Fill a cell with 0/1 only after an independent "
              "radiologist review of separately licensed source images; never copy "
              "released soft scores or accepted flags into it.", "",
              "**Not a random sample:** this targeted shortlist cannot estimate "
              "accuracy, prevalence, fairness, risk or population performance. "
              "It is not an independent clinical validation set, even if labels "
              "are later added. Upstream source-model reuse and unknown repeated "
              "patients cannot be ruled out from the CSV release. Keep labeled "
              "rows out of Git/Actions and do not send this folder to the "
              "aggregate dashboard.", ""]
    if not summary["development_marker_present"]:
        report += ["**Caution:** official development markers are unavailable; "
                   "unlabeled development studies may remain in the queue. "
                   "Do not use the queue for independent validation.", ""]
    report += [DISCLAIMER, ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stage inside the configured output root, so an interrupted write leaves
    # no half-filled plan and no private file outside the ignored outputs tree.
    staged = Path(tempfile.mkdtemp(prefix=".review-plan-", dir=out.parent))
    try:
        queue.to_csv(staged / "review_queue.csv", index=False, float_format="%.9g")
        template.to_csv(staged / "expert_label_template.csv", index=False)
        (staged / "review_plan.md").write_text("\n".join(report), encoding="utf-8")
        (staged / "plan_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite existing review plan at {out}")
        staged.rename(out)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return {**summary, "queue": str(out / "review_queue.csv"),
            "template": str(out / "expert_label_template.csv"),
            "report": str(out / "review_plan.md")}
