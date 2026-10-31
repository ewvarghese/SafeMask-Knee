"""Prediction-only release: fit acceptance only and refuse invented expert truth."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from safemask_knee.config import load_config
from safemask_knee.constants import DISCLAIMER, FINDINGS, WEIGHT_COLUMNS
from safemask_knee.io import SchemaError, grouped_study_folds, load_release
from scripts.prepare_expert import prepare_expert
from test_end_to_end import _run


def _prediction_only(release_dir: Path, tmp_path: Path) -> Path:
    dest = tmp_path / "prediction-only"
    shutil.copytree(release_dir, dest)
    panel = pd.read_csv(dest / "expert_results.csv")
    (dest / "expert_results.csv").unlink()
    metadata = pd.read_csv(dest / "study_metadata.csv")
    expert_ids = set(panel.exam_id)
    metadata["official_expert_evaluation_row"] = metadata.exam_id.isin(expert_ids).astype(int)
    metadata["normalized_report_group"] = metadata.exam_id.map(lambda s: s + "-hash")
    metadata.loc[:1, "normalized_report_group"] = "SAME-REPORT-HASH"
    metadata.to_csv(dest / "study_metadata.csv", index=False)
    predictions = panel.loc[panel.finding.eq("ACL"), ["exam_id", "predicted_label"]]
    predictions.rename(columns={"exam_id": "StudyInstanceUID",
                                "predicted_label": "accuracy_calibrated__ACL"}).to_csv(
                                    dest / "expert_oof.csv", index=False)
    # No gold, no benchmark JSON; only a file named expert_oof with predictions.
    return dest


def test_prediction_only_train_refuses_expert_metrics(release_dir: Path,
                                                    config_path: Path,
                                                    tmp_path: Path) -> None:
    release = _prediction_only(release_dir, tmp_path)
    working = tmp_path / "workspace"
    working.mkdir()
    command = ["train", "--release", str(release), "--config", str(config_path)]
    result = _run(command, working)
    assert result.returncode == 0, result.stderr[-5000:]
    assert "COLUMN MAPPING" in result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "incomplete_missing_expert_truth"
    assert summary["masks"] is summary["evaluation"] is None
    assert summary["audit"]["accepted_expert_errors"] is None
    assert summary["weights"]["expert_truth_available"] is False
    assert summary["weights"]["propensity_heldout"] is True
    root = working / "outputs"
    assert not (root / "clinical_risk_masks.csv").exists()
    assert not (root / "evaluation/metrics_by_strategy.csv").exists()
    w = pd.read_csv(root / "debiased_training_weights.csv")
    assert list(w) == list(WEIGHT_COLUMNS)
    assert len(w) == 160 * 11
    assert w.expert_label.isna().all() and w.fold.isna().all()
    assert np.isfinite(w.propensity_accepted).all()
    assert w.loc[w.accepted60.eq(0), "weight_final"].eq(0).all()
    assert (w.loc[w.accepted60.eq(1)].groupby("finding").weight_final.mean() - 1
            ).abs().max() < 1e-8
    bias = pd.read_csv(root / "audit/bias_table.csv")
    assert bias.expert_n.eq(0).all() and bias.accepted_errors.isna().all()
    for path in (root / "audit/audit_report.md", working / "reports/summary.md",
                 root / "audit/bias_table.csv", root / "audit/fairness_table.csv"):
        assert DISCLAIMER in path.read_text(encoding="utf-8")
    narrative = (working / "reports/summary.md").read_text(encoding="utf-8")
    assert "NOT an expert" in narrative or "NOT AN EXPERT" in narrative
    assert "No MNAR verdict" in narrative
    assert "Positivity warning" in narrative
    assert "patient" in narrative.lower()
    assert "97.13%" not in narrative  # no benchmark JSON in invented fixture
    diag = json.loads((root / "propensity_diagnostics.json").read_text())
    assert all(set(item["test_studies"]).isdisjoint(item["base_fit_studies"])
               and set(item["test_studies"]).isdisjoint(item["platt_studies"])
               and set(item["base_fit_studies"]).isdisjoint(item["platt_studies"])
               for item in diag["folds"])
    assert len(diag["folds"]) == 5
    report = _run(["report", "--release", str(release),
                   "--config", str(config_path)], working)
    assert report.returncode == 0, report.stderr[-2000:]
    for stage in ("masks", "evaluate", "all"):
        fail = _run([stage, "--release", str(release),
                     "--config", str(config_path)], working)
        assert fail.returncode != 0
        assert "contains predictions, NOT expert ground truth" in fail.stderr
        assert not (root / "clinical_risk_masks.csv").exists()
    stale_mask = root / "clinical_risk_masks.csv"
    stale_mask.write_text("stale mask from a different source\n", encoding="utf-8")
    assert "stale expert-dependent" in _run(command, working).stderr
    assert "stale clinical mask" in _run(
        ["report", "--release", str(release), "--config", str(config_path)],
        working).stderr


def test_duplicate_report_groups_are_kept_with_patient_groups(release_dir: Path,
                                                               config_path: Path) -> None:
    frame = load_release(release_dir, load_config(config_path),
                         print_mapping=False).frame.copy()
    ids = sorted(frame.study_id.unique())[:3]
    frame["normalized_report_group"] = frame.study_id.astype(str)
    frame.loc[frame.study_id.isin(ids[:2]), "normalized_report_group"] = "SHARED-REPORT"
    p1 = frame.loc[frame.study_id.eq(ids[1]), "patient_id"].iloc[0]
    frame.loc[frame.study_id.eq(ids[2]), "patient_id"] = p1
    covered: set[str] = set()
    for train, test in grouped_study_folds(frame, 5, 8):
        assert train.isdisjoint(test)
        assert len(set(ids) & test) in (0, 3)
        assert covered.isdisjoint(test)
        covered.update(test)
    assert covered == set(frame.study_id)


def test_offline_expert_adapter_uses_only_verified_official_labels(
        release_dir: Path, config_path: Path, tmp_path: Path) -> None:
    dest = tmp_path / "adapter-release"
    shutil.copytree(release_dir, dest)
    panel = pd.read_csv(dest / "expert_results.csv")
    (dest / "expert_results.csv").unlink()
    for name in ("accuracy_soft_labels.csv", "accuracy_accepted60.csv",
                 "study_metadata.csv"):
        path = dest / name
        table = pd.read_csv(path)
        table = table.rename(columns={"exam_id": "StudyInstanceUID",
                                      **{f"accepted60_{f}": f for f in FINDINGS}})
        if name == "study_metadata.csv":
            table["official_expert_evaluation_row"] = table.StudyInstanceUID.isin(
                panel.exam_id).astype(int)
        table.to_csv(path, index=False)
    official = panel.pivot(index="exam_id", columns="finding", values="expert_label")
    official["Synovitis"] = 1  # deliberately ignored by adapter
    official.index.name = "StudyInstanceUID"
    licensed = tmp_path / "secure" / "train.csv"
    licensed.parent.mkdir()
    official.to_csv(licensed)
    result = prepare_expert(licensed, dest, dest / "expert_results.csv")
    assert result["expert_pairs"] == 638
    assert result["accepted_pairs"] == 383
    assert result["correct_accepted"] == 372
    assert result["source_count_sanity_match"] is True
    extracted = pd.read_csv(dest / "expert_results.csv")
    assert extracted.StudyInstanceUID.nunique() == 58
    assert set(extracted.finding) == set(FINDINGS)
    assert len(extracted) == 638
    assert "Synovitis" not in extracted.finding.values
    assert (dest / "train.csv").exists() is False
    loaded = load_release(dest, load_config(config_path), print_mapping=False,
                          require_expert=True)
    assert loaded.frame.expert_label.notna().sum() == 638
    metadata_path = dest / "study_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    old = metadata.official_expert_evaluation_row.eq(1)
    # A copied official file joined to the WRONG development IDs must fail.
    metadata.loc[old.idxmax(), "official_expert_evaluation_row"] = 0
    metadata.loc[(~old).idxmax(), "official_expert_evaluation_row"] = 1
    metadata.to_csv(metadata_path, index=False)
    with pytest.raises(SchemaError, match="do not match metadata"):
        load_release(dest, load_config(config_path), print_mapping=False,
                     require_expert=True)
    (dest / "expert_oof.csv").write_text("StudyInstanceUID,accuracy_calibrated__ACL\n")
    with pytest.raises(ValueError, match="Predictions are not"):
        prepare_expert(dest / "expert_oof.csv", dest, dest / "expert_results.csv")
