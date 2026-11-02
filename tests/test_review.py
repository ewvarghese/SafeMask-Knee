"""Local research review worklists must not masquerade as clinical validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from safemask_knee.constants import DISCLAIMER, FINDINGS, WEIGHT_COLUMNS
from safemask_knee.io import ReleaseData
from safemask_knee.review import write_review_plan
from test_end_to_end import _run


def _toy(tmp_path: Path) -> tuple[ReleaseData, Path]:
    studies = ["SYN-DEV", "SYN-LINKED", "SYN-A", "SYN-B", "SYN-LOW"]
    rows = []
    for study in studies:
        for finding in FINDINGS:
            rows.append({"study_id": study, "finding": finding,
                         "soft_score": .5 if study in ("SYN-A", "SYN-B") else .1,
                         "confidence": .7,
                         "accepted60": 0 if study in ("SYN-A", "SYN-B") else 1,
                         "expert_label": pd.NA,
                         "official_expert_evaluation_row": 1 if study == "SYN-DEV" else 0,
                         "normalized_report_group": "COPY" if study in (
                             "SYN-DEV", "SYN-LINKED") else study,
                         "patient_id": "SAME-PATIENT" if study in (
                             "SYN-A", "SYN-B") else study})
    frame = pd.DataFrame(rows)
    frame["expert_label"] = frame.expert_label.astype("Int64")
    weights = frame[["study_id", "finding", "soft_score", "confidence",
                     "accepted60", "expert_label"]].copy()
    weights["propensity_accepted"] = .5
    weights["weight_raw"] = weights.accepted60.astype(float)
    weights["weight_clipped"] = weights.weight_raw
    weights["weight_final"] = weights.weight_raw
    weights["fold"] = pd.NA
    weights["notes"] = "synthetic"
    weights = weights.loc[:, list(WEIGHT_COLUMNS)]
    path = tmp_path / "weights.csv"
    weights.to_csv(path, index=False)
    data = ReleaseData(frame=frame, mapping=[], files={}, notices=(), warnings=(),
                       metadata_fields=("patient_id", "normalized_report_group",
                                        "official_expert_evaluation_row"),
                       release_dir=tmp_path, prediction_source="invented")
    return data, path


def test_review_plan_keeps_transitive_groups_and_blank_findings(tmp_path: Path) -> None:
    data, weights = _toy(tmp_path)
    out = tmp_path / "private-output/review_plan"
    summary = write_review_plan(data, weights, out, n_groups=1, seed=317)
    assert summary["selected_groups"] == 1
    assert summary["selected_studies"] == 2
    assert summary["excluded_development_groups"] == 1
    assert summary["excluded_development_studies"] == 2
    queue = pd.read_csv(summary["queue"])
    assert set(queue.study_id) == {"SYN-A", "SYN-B"}
    assert set(queue.group_rank) == {1}
    assert set(queue.studies_in_group) == {2}
    assert "research_priority_score" in queue and set(queue.disclaimer) == {DISCLAIMER}
    template = pd.read_csv(summary["template"])
    assert list(template.columns) == ["study_id", *FINDINGS]
    assert template[list(FINDINGS)].isna().all().all()
    assert set(template.study_id) == set(queue.study_id)
    report = Path(summary["report"]).read_text()
    assert DISCLAIMER in report and "Not a random sample" in report
    assert "SYN-A" not in report and "SYN-DEV" not in report
    aggregate = (out / "plan_summary.json").read_text()
    assert DISCLAIMER in aggregate and "SYN-A" not in aggregate
    assert "SYN-DEV" not in aggregate and "study_id" not in aggregate
    assert "Synovitis" not in Path(summary["template"]).read_text()
    # Existing annotations are never overwritten by a second command.
    template.loc[0, "ACL"] = 1
    template.to_csv(out / "expert_label_template.csv", index=False)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_review_plan(data, weights, out, n_groups=2, seed=317)
    assert pd.read_csv(out / "expert_label_template.csv").loc[0, "ACL"] == 1


def test_review_plan_rejects_stale_or_tampered_weight_rows(tmp_path: Path) -> None:
    data, weights = _toy(tmp_path)
    frame = pd.read_csv(weights)
    frame.loc[0, "soft_score"] = .345
    frame.to_csv(weights, index=False)
    with pytest.raises(ValueError, match="soft_score differs"):
        write_review_plan(data, weights, tmp_path / "bad", n_groups=1, seed=317)
    assert not (tmp_path / "bad").exists()
    frame.loc[0, "soft_score"] = .1
    frame = frame.iloc[:-1]
    frame.to_csv(weights, index=False)
    with pytest.raises(ValueError, match="keys do not match"):
        write_review_plan(data, weights, tmp_path / "bad", n_groups=1, seed=317)
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="--groups"):
        write_review_plan(data, weights, tmp_path / "bad", n_groups=0, seed=317)


def test_cli_label_free_review_is_private_and_manifest_bound(
        release_dir: Path, config_path: Path, tmp_path: Path) -> None:
    release = tmp_path / "prediction-only"
    shutil.copytree(release_dir, release)
    expert = pd.read_csv(release / "expert_results.csv")
    (release / "expert_results.csv").unlink()
    marked = set(expert.exam_id)
    metadata = pd.read_csv(release / "study_metadata.csv")
    metadata["official_expert_evaluation_row"] = metadata.exam_id.isin(marked).astype(int)
    # A non-panel patient that is connected to a panel study is blocked too.
    dev = next(iter(sorted(marked)))
    weak = next(iter(sorted(set(metadata.exam_id) - marked)))
    metadata.loc[metadata.exam_id.eq(weak), "patient_id"] = metadata.loc[
        metadata.exam_id.eq(dev), "patient_id"].iloc[0]
    metadata.to_csv(release / "study_metadata.csv", index=False)
    work = tmp_path / "work"
    work.mkdir()
    command = ["review-plan", "--release", str(release),
               "--config", str(config_path), "--groups", "15"]
    first = _run(command, work)
    assert first.returncode == 2 and "Run `train` or `all` first" in first.stderr
    trained = _run(["train", "--release", str(release),
                    "--config", str(config_path)], work)
    assert trained.returncode == 0, trained.stderr[-2500:]
    result = _run(command, work)
    assert result.returncode == 0, result.stderr[-2500:]
    assert "COLUMN MAPPING" in result.stderr
    summary = json.loads(result.stdout)
    assert summary["selected_groups"] == 15
    assert summary["selected_studies"] >= 15
    assert summary["development_marker_present"] is True
    assert summary["expert_truth_present"] is False
    assert "SYN-" not in result.stdout
    root = work / "outputs/review_plan"
    rows = pd.read_csv(root / "review_queue.csv")
    assert set(rows.study_id).isdisjoint(marked | {weak})
    assert list(pd.read_csv(root / "expert_label_template.csv")) == ["study_id", *FINDINGS]
    assert "SYN-" not in (root / "review_plan.md").read_text()
    again = _run(command, work)
    assert again.returncode == 2 and "Refusing to overwrite" in again.stderr
    metadata.loc[metadata.exam_id.eq(weak), "sex"] = "X"
    metadata.to_csv(release / "study_metadata.csv", index=False)
    stale = _run(command, work)
    assert stale.returncode == 2 and "Release inputs changed" in stale.stderr
