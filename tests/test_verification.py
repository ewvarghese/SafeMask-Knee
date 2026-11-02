"""A third party can check local integrity without receiving private rows."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from safemask_knee.constants import DISCLAIMER, FINDINGS
from safemask_knee.io import sha256_file
from safemask_knee import verification
from test_bootstrap_release import _archive
from test_end_to_end import _run


def _trained(release_dir: Path, config_path: Path, tmp_path: Path
             ) -> tuple[Path, Path, list[str]]:
    source = tmp_path / "release"
    shutil.copytree(release_dir, source)
    panel = pd.read_csv(source / "expert_results.csv")
    (source / "expert_results.csv").unlink()
    metadata = source / "study_metadata.csv"
    table = pd.read_csv(metadata)
    table["official_expert_evaluation_row"] = table.exam_id.isin(panel.exam_id).astype(int)
    table.to_csv(metadata, index=False)
    work = tmp_path / "work"
    work.mkdir()
    base = ["--release", str(source), "--config", str(config_path)]
    trained = _run(["train", *base], work)
    assert trained.returncode == 0, trained.stderr[-2500:]
    return work, source, base


def test_verify_grouped_recomputation_is_id_free_and_read_only(
        release_dir: Path, config_path: Path, tmp_path: Path) -> None:
    work, source, base = _trained(release_dir, config_path, tmp_path)
    help_result = _run(["verify", "--help"], work)
    assert help_result.returncode == 0 and DISCLAIMER in help_result.stdout
    weights = work / "outputs/debiased_training_weights.csv"
    before = sha256_file(weights)
    outcome = _run(["verify", *base, "--recompute"], work)
    assert outcome.returncode == 0, outcome.stderr[-2500:]
    assert "COLUMN MAPPING" in outcome.stderr
    summary = json.loads(outcome.stdout)
    assert summary["command"] == "verify" and summary["checks_failed"] == 0
    assert summary["status"] == "local_consistency_only_not_clinical_validation"
    report = (work / "outputs/verification/verification.md").read_text()
    record_text = (work / "outputs/verification/verification.json").read_text()
    record = json.loads(record_text)
    assert record["n_studies"] == 160 and record["n_pairs"] == 160 * 11
    assert record["n_findings"] == len(FINDINGS) == 11
    assert len(record["finding_checks"]) == 11
    assert record["outcome_accuracy"] is None
    assert record["clinical_validation"] == "not_established"
    assert record["weights_sha256"] == before == sha256_file(weights)
    assert [c["result"] for c in record["checks"] if c["check"] == "pinned_archive"] == [
        "not_checked"]
    assert [c["result"] for c in record["checks"] if c["check"] ==
            "numeric_recomputation"] == ["pass"]
    assert [c["result"] for c in record["checks"] if c["check"] ==
            "clinical_outcomes"] == ["not_evaluable"]
    for text in (report, record_text, outcome.stdout):
        assert DISCLAIMER in text and "SYN-" not in text and "PX-" not in text
    assert not (work / "outputs/clinical_risk_masks.csv").exists()
    assert (source / "accuracy_soft_labels.csv").exists()
    manifest = work / "outputs/data_manifest.json"
    saved = json.loads(manifest.read_text())
    manifest.write_bytes((json.dumps(saved, separators=(",", ":")) + "\r\n").encode())
    reformatted = _run(["verify", *base], work)
    assert reformatted.returncode == 0, reformatted.stderr[-2000:]
    result = json.loads((work / "outputs/verification/verification.json").read_text())
    assert result["source_manifest_sha256"] == record["source_manifest_sha256"]


def test_verify_fails_loudly_for_weight_and_manifest_tampering(
        release_dir: Path, config_path: Path, tmp_path: Path) -> None:
    work, source, base = _trained(release_dir, config_path, tmp_path)
    weights_path = work / "outputs/debiased_training_weights.csv"
    original = weights_path.read_bytes()
    weights = pd.read_csv(weights_path)
    idx = weights.index[weights.accepted60.eq(1)][0]
    weights.loc[idx, "weight_final"] = 999
    weights.to_csv(weights_path, index=False)
    broken = _run(["verify", *base], work)
    assert broken.returncode == 2 and "integrity checks failed" in broken.stderr.lower()
    record = json.loads((work / "outputs/verification/verification.json").read_text())
    assert record["status"] == "integrity_checks_failed"
    assert [c["result"] for c in record["checks"] if c["check"] == "saved_weights"] == [
        "fail"]
    assert [c["result"] for c in record["checks"] if c["check"] ==
            "numeric_recomputation"] == ["not_checked"]
    assert "SYN-" not in json.dumps(record)
    assert weights_path.read_bytes() != original  # verifier did not silently repair it
    weights_path.write_bytes(original)
    metadata = source / "study_metadata.csv"
    table = pd.read_csv(metadata)
    table.loc[0, "sex"] = "M" if table.loc[0, "sex"] == "F" else "F"
    table.to_csv(metadata, index=False)
    changed = _run(["verify", *base], work)
    assert changed.returncode == 2
    checks = {c["check"]: c["result"] for c in json.loads((
        work / "outputs/verification/verification.json").read_text())["checks"]}
    assert checks["source_manifest"] == "fail"
    assert checks["run_provenance"] == "fail"
    assert "SYN-" not in (work / "outputs/verification/verification.md").read_text()
    (work / "outputs/clinical_risk_masks.csv").write_text("stale expert mask")
    stale = _run(["verify", *base], work)
    assert stale.returncode == 2
    stale_checks = {c["check"]: c["result"] for c in json.loads((
        work / "outputs/verification/verification.json").read_text())["checks"]}
    assert stale_checks["no_stale_expert_artifacts"] == "fail"
    provenance = work / "outputs/run_provenance.json"
    record = json.loads(provenance.read_text())
    record["git_revision"] = "SYN-PRIVATE-ID-001"
    provenance.write_text(json.dumps(record))
    secret = _run(["verify", *base], work)
    assert secret.returncode == 2
    assert "SYN-PRIVATE-ID-001" not in (
        work / "outputs/verification/verification.json").read_text()
    missing = _run(["verify", *base, "--archive", str(tmp_path / "absent.zip")], work)
    assert missing.returncode == 2
    checks = {c["check"]: c["result"] for c in json.loads((
        work / "outputs/verification/verification.json").read_text())["checks"]}
    assert checks["pinned_archive"] == "fail"


def test_archive_checker_hashes_every_member_and_extracted_input(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "invented.zip"
    digest = _archive(archive)
    release = tmp_path / "release"
    release.mkdir()
    (release / "accuracy_soft_labels.csv").write_bytes(
        b"fixture accuracy_soft_labels.csv\n")
    monkeypatch.setattr(verification, "KNOWN_ARCHIVE_SHA256", digest)
    passed = verification._archive_check(archive, release)
    assert passed["members_checked"] >= 10
    assert passed["extracted_inputs_checked"] == 1
    (release / "accuracy_soft_labels.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="extracted input"):
        verification._archive_check(archive, release)
    bad_digest = _archive(archive, tamper_member="unneeded.txt")
    monkeypatch.setattr(verification, "KNOWN_ARCHIVE_SHA256", bad_digest)
    with pytest.raises(ValueError, match="archive member"):
        verification._archive_check(archive, release)
    unsafe_digest = _archive(archive, extra="../outside.txt")
    monkeypatch.setattr(verification, "KNOWN_ARCHIVE_SHA256", unsafe_digest)
    with pytest.raises(ValueError, match="unsafe"):
        verification._archive_check(archive, release)
    assert not (tmp_path / "outside.txt").exists()


def test_unmarked_prediction_release_does_not_invent_independence(
        release_dir: Path, config_path: Path, tmp_path: Path) -> None:
    source = tmp_path / "unmarked-release"
    shutil.copytree(release_dir, source)
    (source / "expert_results.csv").unlink()
    work = tmp_path / "work"
    work.mkdir()
    args = ["--release", str(source), "--config", str(config_path)]
    fitted = _run(["train", *args], work)
    assert fitted.returncode == 0, fitted.stderr[-2000:]
    report = (work / "reports/summary.md").read_text()
    assert "zero *identified* studies does not establish independence" in report
    checked = _run(["verify", *args], work)
    assert checked.returncode == 0, checked.stderr[-2000:]
    record = json.loads((work / "outputs/verification/verification.json").read_text())
    assert record["clinical_validation"] == "not_established"
    assert record["status"] == "local_consistency_only_not_clinical_validation"
