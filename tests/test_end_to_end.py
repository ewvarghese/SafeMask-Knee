"""CLI artifacts, exact schemas, attribution, disclaimers, reproducible output."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from safemask_knee.constants import DISCLAIMER, MASK_COLUMNS, WEIGHT_COLUMNS


def _run(command: list[str], working: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "20261101"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run([sys.executable, "-m", "safemask_knee", *command],
                          cwd=working, env=env, text=True, capture_output=True,
                          check=False, timeout=180)


def test_cli_all_generates_exact_schemas_and_reports(release_dir: Path,
                                                     config_path: Path,
                                                     tmp_path: Path) -> None:
    command = ["all", "--release", str(release_dir), "--config", str(config_path)]
    result = _run(command, tmp_path)
    assert result.returncode == 0, result.stderr[-5000:]
    assert "COLUMN MAPPING" in result.stderr
    summary = json.loads(result.stdout.strip())
    assert summary["command"] == "all"
    assert summary["evaluation"]["sanity_check_counts_match"] is True
    assert summary["evaluation"]["coverage_goal_met"] is True
    assert summary["evaluation"]["empirical_tier_bounds_met"] is True
    assert summary["evaluation"]["ipcw_accepted_positive_precision"] >= .97
    assert summary["evaluation"]["ipcw_adjusted_sbr"] > .28
    root = tmp_path / "outputs"
    weights_path = root / "debiased_training_weights.csv"
    masks_path = root / "clinical_risk_masks.csv"
    assert weights_path.exists() and masks_path.exists()
    weights = pd.read_csv(weights_path)
    masks = pd.read_csv(masks_path)
    assert list(weights) == list(WEIGHT_COLUMNS)
    assert list(masks) == list(MASK_COLUMNS)
    assert len(weights) == len(masks) == 1760
    assert weights[["study_id", "finding"]].equals(masks[["study_id", "finding"]])
    assert weights.expert_label.notna().sum() == 638
    assert weights.fold.notna().sum() == masks.fold.notna().sum() == 638
    assert masks.accepted_clinical.isin([0, 1]).all()
    assert masks.loc[masks.tier.eq(1) & masks.soft_score.lt(.5),
                     "accepted_clinical"].eq(0).all()
    assert (tmp_path / "reports/summary.md").exists()
    for path in (root / "schema.json", root / "data_manifest.json",
                 tmp_path / "data/manifest.json", root / "audit/bias_table.csv",
                 root / "audit/fairness_table.csv", root / "audit/audit_report.md",
                 root / "evaluation/metrics_by_strategy.csv",
                 root / "evaluation/per_finding_metrics.csv",
                 root / "evaluation/tier_metrics.csv",
                 root / "evaluation/finding_risk_checks.csv",
                 root / "evaluation/proxy_training_diagnostics.json",
                 root / "loo_conformal_scores.csv",
                 root / "evaluation/risk_coverage_curves.png",
                 root / "evaluation/calibration.png",
                 root / "evaluation/eval_report.md",
                 root / "reports/summary.md",
                 tmp_path / "ATTRIBUTION.md"):
        assert path.exists(), str(path)
    for filename in ("acceptance_vs_soft_score.png",
                     "prevalence_accepted_rejected.png",
                     "selection_bias_by_finding.png",
                     "metadata_acceptance_rates.png"):
        assert (root / "audit/figures" / filename).stat().st_size > 2000
    # The exact disclaimer belongs to each narrative report and report table;
    # exact-schema downstream CSVs are exempt (their 'reason' must stay short).
    for path in (root / "audit/audit_report.md", root / "evaluation/eval_report.md",
                 tmp_path / "reports/summary.md", root / "reports/summary.md",
                 root / "audit/bias_table.csv", root / "audit/fairness_table.csv",
                 root / "evaluation/metrics_by_strategy.csv",
                 root / "evaluation/tier_metrics.csv",
                 root / "evaluation/finding_risk_checks.csv",
                 tmp_path / "ATTRIBUTION.md"):
        assert DISCLAIMER in path.read_text(encoding="utf-8"), str(path)
    notice = (release_dir / "NOTICE.txt").read_text(encoding="utf-8").strip()
    assert notice in (tmp_path / "ATTRIBUTION.md").read_text(encoding="utf-8")
    manifest = json.loads((root / "data_manifest.json").read_text())
    assert len(manifest["files"]) == 6
    assert manifest == json.loads((tmp_path / "data/manifest.json").read_text())
    nested = json.loads((root / "evaluation/proxy_training_diagnostics.json").read_text())
    assert set(nested["training_studies"]).isdisjoint(nested["expert_studies"])
    for f in nested["nested_propensity_folds"]:
        assert set(f["base_fit_studies"]).isdisjoint(f["test_studies"])
        assert set(f["platt_studies"]).isdisjoint(f["test_studies"])
        assert set(f["base_fit_studies"]).isdisjoint(f["platt_studies"])
    loo = pd.read_csv(root / "loo_conformal_scores.csv")
    assert len(loo) == 638 and loo.own_study_excluded.all()
    finding_checks = pd.read_csv(root / "evaluation/finding_risk_checks.csv")
    assert len(finding_checks) == 11
    assert finding_checks.empirical_bounds_met.all()
    assert set(finding_checks.finding) == set(weights.finding)
    metrics = pd.read_csv(root / "evaluation/metrics_by_strategy.csv").set_index("strategy")
    baseline = metrics.loc["released_accepted60"]
    assert baseline.n_evaluated == 638 and baseline.n_accepted == 383
    assert baseline.n_errors_accepted == 11
    assert abs(baseline.accuracy_at_coverage - 372 / 383) < 1e-5
    assert metrics.loc["D_IPCW_weighted", "selected_recall"] > metrics.loc[
        "A_naive_accepted_binary", "selected_recall"]
    folds = pd.read_csv(root / "evaluation/fold_assignments.csv")
    assert folds.study_id.nunique() == 58 and folds.outer_fold.nunique() == 5
    original = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (weights_path, masks_path, root / "evaluation/metrics_by_strategy.csv")}
    again = _run(command, tmp_path)
    assert again.returncode == 0, again.stderr[-5000:]
    for path, digest in original.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    # A report must not silently reuse metrics for a different/changed release.
    import shutil
    copied = tmp_path / "different-release"
    shutil.copytree(release_dir, copied)
    with (copied / "NOTICE.txt").open("a", encoding="utf-8") as notice_file:
        notice_file.write("Changed source notice; previous manifest must fail.\n")
    stale = _run(["report", "--release", str(copied),
                  "--config", str(config_path), "--out", "reports/summary.md"], tmp_path)
    assert stale.returncode != 0 and "Release inputs changed" in stale.stderr


def test_every_help_contains_disclaimer(tmp_path: Path) -> None:
    for command in ([], ["schema"], ["audit"], ["weights"], ["masks"],
                    ["evaluate"], ["report"], ["train"], ["all"]):
        result = _run([*command, "--help"], tmp_path)
        assert result.returncode == 0
        assert DISCLAIMER in result.stdout


def test_script_runs_pytest_before_pipeline() -> None:
    script = (Path(__file__).parent.parent / "scripts/run_all.sh").read_text()
    assert "set -euo pipefail" in script
    assert "export PYTHONHASHSEED" in script
    assert script.index("python -m pytest -q") < script.index("python -m safemask_knee all")
