"""New independently scored IDs: no external answer key can enter calibration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from safemask_knee.config import load_config
from safemask_knee.constants import DISCLAIMER, FINDINGS
from safemask_knee.external_validation import evaluate_external
from safemask_knee.io import SchemaError, load_release, write_manifest
from safemask_knee.masks import write_masks
from safemask_knee.provenance import write_provenance
from scripts.make_synthetic import generate_release
from test_end_to_end import _run


def _external_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "invented-external-generator"
    generate_release(raw, studies=70, expert_studies=10, seed=17)
    labels = pd.read_csv(raw / "expert_results.csv")
    selected = set(labels.exam_id)
    renamed = {study: "EXT-" + study for study in selected}
    ext = tmp_path / "external-score-bundle"
    ext.mkdir()
    for name in ("accuracy_soft_labels.csv", "accuracy_confidence.csv",
                 "accuracy_accepted60.csv", "study_metadata.csv"):
        frame = pd.read_csv(raw / name)
        sid = "exam_id" if "exam_id" in frame else "study_id"
        frame = frame.loc[frame[sid].isin(selected)].copy()
        frame[sid] = frame[sid].map(renamed)
        if "patient_id" in frame:
            frame["patient_id"] = "EXT-PATIENT-" + frame.patient_id.astype(str)
        frame.to_csv(ext / name, index=False)
    (ext / "NOTICE.txt").write_text("Invented external fixture, CC0; NOT real MRI.\n")
    labels.exam_id = labels.exam_id.map(renamed)
    gold = tmp_path / "licensed-external-expert-results.csv"
    labels.to_csv(gold, index=False)
    return ext, gold


def test_external_cohort_frozen_cutoffs_and_aggregate_metrics(
        release_dir: Path, config_path: Path, tmp_path: Path) -> None:
    cfg = load_config(config_path)
    source = load_release(release_dir, cfg, print_mapping=False, require_expert=True)
    root = tmp_path / "outputs"
    write_masks(source, root / "clinical_risk_masks.csv", cfg)
    snapshot = (root / "conformal_diagnostics.json").read_bytes()
    ext_path, truth_path = _external_fixture(tmp_path)
    external = load_release(ext_path, cfg, print_mapping=False, require_expert=False)
    out = tmp_path / "external-evaluation"
    result = evaluate_external(source, external, truth_path, root, out, cfg)
    assert result["n_external_studies"] == 10
    assert result["source_model_independence_verified"] is False
    assert (root / "conformal_diagnostics.json").read_bytes() == snapshot
    summary = json.loads((out / "external_summary.json").read_text())
    assert summary["n_external_pairs"] == 10 * len(FINDINGS)
    assert summary["external_labels_never_used_in_cutoffs"] is True
    assert summary["known_id_and_group_overlap"] is False
    assert set(summary["outcomes"]) == {"source_accepted60", "frozen_expert_panel_mask"}
    metrics = pd.read_csv(out / "per_finding_metrics.csv")
    assert len(metrics) == 2 * len(FINDINGS)
    assert metrics.disclaimer.eq(DISCLAIMER).all()
    assert DISCLAIMER in (out / "external_report.md").read_text()
    first_id = sorted(external.frame.study_id.unique())[0]
    text = (out / "external_summary.json").read_text() + (
        out / "external_report.md").read_text()
    assert first_id not in text and "EXT-PATIENT-" not in text
    assert "expert_label" not in metrics.columns
    # The public CLI checks the saved input/config manifest before scoring.
    write_manifest(source, root, tmp_path / "data")
    write_provenance(source, root, cfg, "all", config_path)
    cli = _run(["external-evaluate", "--release", str(release_dir),
                "--external-release", str(ext_path), "--external-labels",
                str(truth_path), "--config", str(config_path)], tmp_path)
    assert cli.returncode == 0, cli.stderr[-3000:]
    assert json.loads(cli.stdout)["n_external_studies"] == 10
    # A known cross-cohort patient identifier invalidates the external check.
    dev_patient = source.frame.loc[source.frame.expert_label.notna(), "patient_id"].iloc[0]
    modified = external.frame.copy()
    modified.loc[modified.study_id.eq(first_id), "patient_id"] = dev_patient
    with pytest.raises(SchemaError, match="share a known patient_id"):
        evaluate_external(source, replace(external, frame=modified), truth_path,
                          root, tmp_path / "reject-patient", cfg)
    ids = external.frame.copy()
    ids.loc[ids.study_id.eq(first_id), "study_id"] = source.frame.study_id.iloc[0]
    with pytest.raises(SchemaError, match="disjoint from ALL source"):
        evaluate_external(source, replace(external, frame=ids), truth_path,
                          root, tmp_path / "reject-id", cfg)
    malformed = tmp_path / "incomplete_external_truth.csv"
    pd.read_csv(truth_path).iloc[1:].to_csv(malformed, index=False)
    with pytest.raises(SchemaError, match="each study needs all 11 findings"):
        evaluate_external(source, external, malformed, root,
                          tmp_path / "reject-missing", cfg)
    with pytest.raises(SchemaError, match="Prediction-only expert_oof"):
        evaluate_external(source, external, ext_path / "expert_oof.csv", root,
                          tmp_path / "reject-predictions", cfg)
    tampered = json.loads(snapshot)
    tampered["full_panel_cutoffs_for_nonexpert_only"]["ACL"]["n_positive"] += 1
    (root / "conformal_diagnostics.json").write_text(json.dumps(tampered))
    with pytest.raises(SchemaError, match="Saved expert-panel conformal cutoffs"):
        evaluate_external(source, external, truth_path, root,
                          tmp_path / "reject-cutoff", cfg)
