"""All added workflows remain aggregate, reproducible and research-only."""

from __future__ import annotations

import json
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from safemask_knee.config import load_config
from safemask_knee.constants import DISCLAIMER, FINDINGS
from safemask_knee.dashboard import dashboard_snapshot, make_handler
from safemask_knee.io import load_release, write_manifest
from safemask_knee.provenance import config_hash, write_provenance
from safemask_knee.quality import quality_record, write_quality
from safemask_knee.sensitivity import sensitivity_tables, write_sensitivity
from safemask_knee.weights import make_weight_table
from scripts.summarize_ci import public_summary, write_public_summary
from test_end_to_end import _run


def test_quality_sensitivity_provenance_have_no_raw_study_ids(
        release_dir: Path, config_path: Path, tmp_path: Path) -> None:
    cfg = load_config(config_path)
    data = load_release(release_dir, cfg, print_mapping=False)
    quality = quality_record(data)
    assert quality["n_studies"] == 160 and quality["n_pairs"] == 1760
    assert quality["all_studies_have_eleven_findings"] is True
    assert quality["duplicate_study_finding_pairs"] == 0
    assert len(quality["finding_quality"]) == len(FINDINGS) == 11
    assert quality["metadata_missing_studies"]["patient_id"] == 0
    root = tmp_path / "outputs"
    quality_result = write_quality(data, root / "quality")
    assert DISCLAIMER in Path(quality_result["path"]).read_text()
    weights, _ = make_weight_table(data, cfg)
    per, overall = sensitivity_tables(weights, cfg)
    assert len(overall) == 9 and len(per) == 99
    baseline = overall.loc[overall.is_configured_baseline].iloc[0]
    accepted = weights.loc[weights.accepted60.eq(1)].weight_final
    assert abs(baseline.effective_sample_size -
               accepted.sum() ** 2 / (accepted ** 2).sum()) < 1e-6
    assert per.mean_weight.sub(1).abs().max() < 1e-8
    with pytest.raises(ValueError, match="lower values cannot undo clipping"):
        sensitivity_tables(weights, cfg, floors=[.001])
    result = write_sensitivity(weights, root / "sensitivity", cfg)
    assert result["n_variants"] == 9
    assert DISCLAIMER in Path(result["path"]).read_text()
    assert not any("study_id" in name for name in per.columns)
    write_manifest(data, root, tmp_path / "manifest")
    provenance = write_provenance(data, root, cfg, "train", config_path)
    record = json.loads(Path(provenance["path"]).read_text())
    assert record["config_sha256_canonical"] == config_hash(cfg)
    assert record["random_seed"] == cfg["seed"]
    assert len(record["input_files"]) == 6
    assert "SYN-00000" not in Path(provenance["path"]).read_text()
    assert DISCLAIMER in Path(provenance["path"]).read_text()


def test_doctor_preflight_and_label_gate(release_dir: Path, config_path: Path,
                                         tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    command = ["doctor", "--release", str(release_dir),
               "--config", str(config_path), "--archive", str(tmp_path / "missing.zip")]
    result = _run(command, work)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout)["status"] == "ready_full"
    assert DISCLAIMER in (work / "outputs/doctor.md").read_text()
    assert "SYN-" not in (work / "outputs/doctor.json").read_text()
    broken = tmp_path / "not-the-known-archive.zip"
    broken.write_bytes(b"unsafe replaced archive")
    fail = _run([*command[:-1], str(broken)], work)
    assert fail.returncode != 0 and "SHA-256 mismatch" in fail.stderr
    no_gold = tmp_path / "no-gold"
    shutil.copytree(release_dir, no_gold)
    (no_gold / "expert_results.csv").unlink()
    partial = _run(["doctor", "--release", str(no_gold),
                    "--config", str(config_path), "--archive", str(tmp_path / "missing.zip")], work)
    assert partial.returncode == 0
    assert json.loads(partial.stdout)["status"] == "ready_label_free_only"
    strict = _run(["doctor", "--release", str(no_gold), "--require-expert",
                   "--config", str(config_path)], work)
    assert strict.returncode != 0 and "Expert truth is required" in strict.stderr
    incomplete = tmp_path / "partial-panel"
    shutil.copytree(release_dir, incomplete)
    truth = pd.read_csv(incomplete / "expert_results.csv")
    selected_ids = set(truth.exam_id.unique()[:10])
    truth.loc[truth.exam_id.isin(selected_ids)].to_csv(
        incomplete / "expert_results.csv", index=False)
    partial_panel = _run(["doctor", "--release", str(incomplete),
                          "--config", str(config_path)], work)
    assert partial_panel.returncode == 0, partial_panel.stderr[-2500:]
    assert json.loads(partial_panel.stdout)["status"] == "expert_panel_incomplete"


def _dashboard_fixture(root: Path) -> None:
    (root / "quality").mkdir(parents=True)
    (root / "sensitivity").mkdir()
    (root / "audit").mkdir()
    quality = {"n_studies": 2, "n_pairs": 22, "finding_quality": [
        {"finding": name, "acceptance_rate": .5} for name in FINDINGS],
        "PRIVATE_STUDY_ID": "SECRET-UID-001"}
    (root / "quality/data_quality.json").write_text(json.dumps(quality))
    diagnostics = {"summary": {"n_accepted": 11, "effective_sample_size_accepted": 9.5,
                               "expert_truth_available": False,
                               "propensity_lower_clip_fraction": .3,
                               "propensity_upper_clip_fraction": .4,
                               "PRIVATE_STUDY_ID": "SECRET-UID-001"}}
    (root / "propensity_diagnostics.json").write_text(json.dumps(diagnostics))
    pd.DataFrame([{"propensity_floor": .02, "weight_cap": 10,
                   "effective_sample_size": 9.5, "fraction_at_cap": 0.,
                   "is_configured_baseline": True}]).to_csv(
                       root / "sensitivity/sensitivity_overview.csv", index=False)
    (root / "sensitivity/sensitivity_summary.json").write_text(
        json.dumps({"n_variants": 1}))
    (root / "run_provenance.json").write_text(json.dumps({
        "git_revision": "ABC", "config_sha256_canonical": "HASH",
        "PRIVATE_STUDY_ID": "SECRET-UID-001"}))
    pd.DataFrame([{"attribute": "report_script_bucket", "stratum": "A",
                   "n_studies": 8, "acceptance_rate": .5},
                  {"attribute": "report_script_bucket", "stratum": "SMALL",
                   "n_studies": 2, "acceptance_rate": .2},
                  {"attribute": "site", "stratum": "SENSITIVE",
                   "n_studies": 10, "acceptance_rate": .9}]).to_csv(
                       root / "audit/fairness_table.csv", index=False)


def test_dashboard_http_only_serves_aggregate_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _dashboard_fixture(root)
    snapshot = dashboard_snapshot(root)
    serialized = json.dumps(snapshot, allow_nan=False)
    assert len(snapshot["findings"]) == 11
    assert "SECRET-UID-001" not in serialized
    assert "SENSITIVE" not in serialized and "SMALL" not in serialized
    assert len(snapshot["operational_strata"]) == 1
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as response:
            page = response.read().decode()
            assert DISCLAIMER in page
            assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(base + "/api/summary") as response:
            assert json.loads(response.read())["n_pairs"] == 22
        for path in ("/outputs/debiased_training_weights.csv", "/data/release/expert_oof.csv"):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base + "/", data=b"x",
                                                       method="POST"))
        assert error.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_safe_ci_export_and_repository_configs(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    _dashboard_fixture(root)
    record = write_public_summary(root, root / "ci/public_summary.json")
    text = (root / "ci/public_summary.json").read_text()
    assert "SECRET-UID-001" not in text and "study_id" not in text
    assert record["clinical_mask_generated"] is False
    assert DISCLAIMER in text and DISCLAIMER in (root / "ci/job_summary.md").read_text()
    with pytest.raises(ValueError, match="inside ignored outputs"):
        write_public_summary(root, tmp_path / "public.json")
    (root / "clinical_risk_masks.csv").write_text("not permitted")
    with pytest.raises(ValueError, match="limited to label-free"):
        public_summary(root)
    repo = Path(__file__).resolve().parents[1]
    workflow = (repo / ".github/workflows/research.yml").read_text()
    devcontainer = json.loads((repo / ".devcontainer/devcontainer.json").read_text())
    assert "workflow_dispatch" in workflow and "run_training" in workflow
    assert ".venv/bin/python scripts/summarize_ci.py" in workflow
    assert "outputs/ci/public_summary.json" in workflow
    assert "path: outputs/debiased_training_weights.csv" not in workflow
    assert devcontainer["forwardPorts"] == [8765]
    assert "bootstrap_release.py" in devcontainer["postCreateCommand"]
