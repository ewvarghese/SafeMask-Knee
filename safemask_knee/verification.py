"""Offline, read-only local integrity and optional numeric-reproduction checks.

The result is a deliberately allowlisted aggregate statement. Hash agreement
proves that bytes match a reference, not that weak labels are medically true.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .constants import DISCLAIMER, FINDINGS
from .doctor import KNOWN_ARCHIVE_SHA256
from .io import ReleaseData, expert_fold_map, group_keys, sha256_file
from .provenance import _git_revision, _requirements, config_hash
from .quality import quality_record
from .review import _aligned_weights
from .weights import make_weight_table


def _archive_check(archive: Path, release: Path) -> dict[str, Any]:
    """Rehash every member and cross-check extracted inputs; never extract."""
    archive = Path(archive)
    if not archive.is_file():
        raise ValueError("Bundled archive is missing")
    digest = sha256_file(archive)
    if digest != KNOWN_ARCHIVE_SHA256:
        raise ValueError("Bundled archive differs from the pinned SHA-256")
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or "FILES.json" not in names or any(
                Path(name).name != name or "\\" in name or name in ("", ".", "..")
                for name in names):
            raise ValueError("Archive has missing, duplicate or unsafe member names")
        listed = json.loads(source.read("FILES.json")).get("files")
        if not isinstance(listed, dict) or set(listed) != set(names) - {"FILES.json"}:
            raise ValueError("Archive member list differs from its file manifest")
        for name, expected in listed.items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("Archive manifest contains an invalid SHA-256")
            digest_member = hashlib.sha256()
            with source.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest_member.update(chunk)
            if digest_member.hexdigest() != expected:
                raise ValueError("An archive member differs from its file manifest")
        matched = 0
        for name in names:
            target = release / name
            if target.is_file():
                expected = (hashlib.sha256(source.read("FILES.json")).hexdigest()
                            if name == "FILES.json" else listed[name])
                if sha256_file(target) != expected:
                    raise ValueError("An extracted input differs from the pinned archive")
                matched += 1
        if matched == 0:
            raise ValueError("No extracted input files matched the pinned archive")
    return {"archive_sha256": digest, "members_checked": len(listed),
            "extracted_inputs_checked": matched}


def _source_manifest(data: ReleaseData, root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
        raise ValueError("Saved source manifest has an invalid file list")
    previous = {item["path"]: (item["sha256"], item["size_bytes"]) for item in entries}
    if len(previous) != len(entries):
        raise ValueError("Saved source manifest contains duplicate paths")
    current = {str(path.relative_to(data.release_dir)): (sha256_file(path), path.stat().st_size)
               for path in set(data.files.values()) | set(data.notices)}
    if previous != current:
        raise ValueError("Local source bytes differ from the saved run manifest")
    # Hash the *inventory*, not its platform-dependent JSON whitespace/CRLF.
    canonical = [{"path": name, "sha256": digest, "size_bytes": size}
                 for name, (digest, size) in sorted(current.items())]
    fingerprint = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"manifest_sha256": fingerprint,
            "source_files_checked": len(current), "manifest": manifest}


def _saved_schema(data: ReleaseData, root: Path) -> None:
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    frame = data.frame
    expected = {"findings": list(FINDINGS),
                "n_studies": int(frame.study_id.nunique()), "n_pairs": len(frame),
                "n_expert_studies": int(frame.loc[frame.expert_label.notna(),
                                                   "study_id"].nunique()),
                "n_evaluated_pairs": int(frame.expert_label.notna().sum()),
                "expert_truth_available": bool(frame.expert_label.notna().any()),
                "source_column_mapping": data.mapping}
    if any(schema.get(name) != value for name, value in expected.items()):
        raise ValueError("Saved schema does not match validated current release")


def _provenance(data: ReleaseData, root: Path, cfg: dict[str, Any],
                manifest: dict[str, Any] | None) -> dict[str, Any]:
    record = json.loads((root / "run_provenance.json").read_text(encoding="utf-8"))
    requirements = _requirements()
    if not requirements or any(spec["pinned"] != spec["installed"]
                               for spec in requirements.values()):
        raise ValueError("Pinned direct dependencies are not installed")
    if (record.get("config_sha256_canonical") != config_hash(cfg)
            or record.get("random_seed") != cfg["seed"]
            or record.get("expert_truth_available") != bool(data.frame.expert_label.notna().any())
            or record.get("direct_dependencies") != requirements
            or manifest is None or record.get("input_files") != manifest["files"]):
        raise ValueError("Run provenance differs from current config, inputs or environment")
    saved_revision, current_revision = record.get("git_revision"), _git_revision()
    if saved_revision is not None and (not isinstance(saved_revision, str) or
                                       not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
                                                        saved_revision)):
        raise ValueError("Saved code revision is not a Git digest")
    return {"saved_code_revision": saved_revision,
            "code_revision_unchanged": bool(saved_revision and current_revision and
                                            saved_revision == current_revision),
            "dependencies_checked": len(requirements)}


def _weights_check(data: ReleaseData, root: Path, cfg: dict[str, Any]
                   ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check values, 11-finding grouping, bounds and saved summary against inputs."""
    weights = _aligned_weights(data, root / "debiased_training_weights.csv")
    accepted = weights.accepted60.to_numpy(dtype=bool)
    numeric: dict[str, np.ndarray] = {}
    for name in ("propensity_accepted", "weight_raw", "weight_clipped", "weight_final"):
        numeric[name] = pd.to_numeric(weights[name], errors="coerce").to_numpy(float)
        if not np.isfinite(numeric[name]).all():
            raise ValueError("Saved weights have missing, infinite or nonnumeric values")
    raw, clipped, final = (numeric[name] for name in (
        "weight_raw", "weight_clipped", "weight_final"))
    probability = numeric["propensity_accepted"]
    lo, hi, floor = (cfg["weight_min"], cfg["weight_max"], cfg["propensity_floor"])
    if ((probability < floor - 1e-8) | (probability > 1 - floor + 1e-8)).any():
        raise ValueError("Saved OOF propensities exceed the configured floor/ceiling")
    if ((raw[~accepted] != 0).any() or (clipped[~accepted] != 0).any()
            or (raw[accepted] <= 0).any()
            or (clipped[accepted] < lo - 1e-6).any()
            or (clipped[accepted] > hi + 1e-6).any()
            or (final[accepted] < lo - 1e-6).any()
            or (final[accepted] > hi + 1e-6).any()):
        raise ValueError("Saved accepted/rejected weights violate finite bounds")
    if not np.allclose(clipped[accepted], np.clip(raw[accepted], lo, hi),
                       rtol=1e-7, atol=1e-7):
        raise ValueError("Saved clipped weights differ from clipped raw weights")
    if not (weights.loc[accepted, "notes"].eq("accepted_oof_ipcw").all() and
            weights.loc[~accepted, "notes"].eq("rejected_no_verified_label").all()):
        raise ValueError("Saved weight row roles do not match accepted/rejected status")
    expert = data.frame[["study_id", "finding", "expert_label"]].copy()
    expert["finding"] = expert.finding.astype(str)
    truth = weights[["study_id", "finding", "expert_label", "fold"]].merge(
        expert, on=["study_id", "finding"], validate="one_to_one",
        suffixes=("_saved", "_source"))
    saved_truth = pd.to_numeric(truth.expert_label_saved, errors="coerce")
    source_truth = pd.to_numeric(truth.expert_label_source, errors="coerce")
    if ((saved_truth.isna() != truth.expert_label_saved.isna()).any() or
            not saved_truth.isna().equals(source_truth.isna()) or not
            saved_truth.fillna(-1).eq(source_truth.fillna(-1)).all()):
        raise ValueError("Saved expert-label presence/values differ from the source")
    saved_folds = pd.to_numeric(truth.fold, errors="coerce")
    if (saved_folds.isna() != truth.fold.isna()).any():
        raise ValueError("Saved fold values are not integers")
    if source_truth.notna().any():
        if (saved_folds[source_truth.notna()].isna().any() or
                saved_folds[source_truth.isna()].notna().any() or
                not saved_folds.dropna().between(0, cfg["outer_folds"] - 1).all() or
                not np.equal(saved_folds.dropna() % 1, 0).all()):
            raise ValueError("Saved folds do not match expert-study membership")
        groups = pd.DataFrame({"study_id": data.frame.study_id.to_numpy(),
                               "group": group_keys(data.frame).to_numpy()}).drop_duplicates("study_id")
        grouped = truth.merge(groups, on="study_id", validate="many_to_one")
        grouped["saved_fold"] = saved_folds.to_numpy()
        if grouped.groupby("group").saved_fold.nunique().gt(1).any():
            raise ValueError("A patient/report group is split across saved expert folds")
    elif saved_folds.notna().any():
        raise ValueError("Saved folds exist without any expert truth")
    by_finding = []
    for name in FINDINGS:
        subset = weights.loc[weights.finding.eq(name)]
        selected = pd.to_numeric(subset.loc[subset.accepted60.eq(1), "weight_final"],
                                 errors="coerce").to_numpy(float)
        if len(subset) != data.frame.study_id.nunique() or not len(selected):
            raise ValueError("Weights lack a complete finding or accepted pseudo-labels")
        if abs(float(selected.mean()) - 1) > 5e-6:
            raise ValueError("Accepted weights are not normalized within finding")
        by_finding.append({"finding": name, "accepted_pairs": len(selected),
                           "acceptance_rate": len(selected) / len(subset),
                           "effective_sample_size": float(selected.sum() ** 2 /
                                                          np.square(selected).sum())})
    selected = final[accepted]
    stats = {"finding_checks": by_finding, "n_pairs": len(weights),
             "n_accepted": int(accepted.sum()),
             "effective_sample_size": float(selected.sum() ** 2 /
                                            np.square(selected).sum()),
             "propensity_lower_clip_fraction": float(np.mean(probability <= floor + 1e-8)),
             "propensity_upper_clip_fraction": float(np.mean(probability >= 1 - floor - 1e-8)),
             "weights_sha256": sha256_file(root / "debiased_training_weights.csv")}
    return weights, stats


def _diagnostics(root: Path, stats: dict[str, Any]) -> None:
    saved = json.loads((root / "propensity_diagnostics.json").read_text(
        encoding="utf-8"))["summary"]
    for name, value in (("n_rows", stats["n_pairs"]),
                        ("n_accepted", stats["n_accepted"]),
                        ("effective_sample_size_accepted", stats["effective_sample_size"]),
                        ("propensity_lower_clip_fraction",
                         stats["propensity_lower_clip_fraction"]),
                        ("propensity_upper_clip_fraction",
                         stats["propensity_upper_clip_fraction"])):
        if name not in saved or not np.isclose(float(saved[name]), float(value),
                                              rtol=1e-6, atol=2e-6):
            raise ValueError("Saved propensity diagnostics do not agree with weights")
    quality = json.loads((root / "quality/data_quality.json").read_text(encoding="utf-8"))
    if quality["n_pairs"] != stats["n_pairs"]:
        raise ValueError("Saved data quality counts do not agree with weights")


def _recompute(data: ReleaseData, cfg: dict[str, Any], saved: pd.DataFrame
               ) -> dict[str, float]:
    folds = (expert_fold_map(data.frame, cfg["outer_folds"], cfg["seed"])
             if data.frame.expert_label.notna().any() else {})
    reference, _ = make_weight_table(data, cfg, folds=folds)
    reference = reference.set_index(["study_id", "finding"])
    actual = saved.set_index(["study_id", "finding"]).loc[reference.index]
    errors: dict[str, float] = {}
    for name in ("propensity_accepted", "weight_raw", "weight_clipped", "weight_final"):
        left = pd.to_numeric(actual[name], errors="coerce").to_numpy(float)
        right = reference[name].to_numpy(float)
        difference = np.abs(left - right)
        if not np.allclose(left, right, rtol=1e-3, atol=1e-5):
            raise ValueError("Independent grouped refit disagrees with saved weights")
        errors[name] = float(difference.max())
    return errors


def _check(record: dict[str, Any], name: str, task: Callable[[], Any],
           success: str, failure: str) -> Any | None:
    """Keep report messages static, never include row-level exception text."""
    try:
        result = task()
    except (OSError, ValueError, KeyError, TypeError, AssertionError,
            zipfile.BadZipFile, json.JSONDecodeError):
        record["checks"].append({"check": name, "result": "fail", "detail": failure})
        return None
    record["checks"].append({"check": name, "result": "pass", "detail": success})
    return result


def verify_local(data: ReleaseData, cfg: dict[str, Any], root: Path,
                 archive: Path | None = None, *, recompute: bool = False
                 ) -> dict[str, Any]:
    """Check saved results without writing to the release or retraining artifacts."""
    root = Path(root)
    record: dict[str, Any] = {
        "disclaimer": DISCLAIMER, "checks": [], "n_studies": int(data.frame.study_id.nunique()),
        "n_pairs": len(data.frame), "n_findings": len(FINDINGS),
        "expert_truth_available": bool(data.frame.expert_label.notna().any()),
        "clinical_validation": "not_established", "outcome_accuracy": None,
        "warning": "Local byte/schema/weight consistency is not independent truth "
                   "or patient-level clinical validation. Predictions are not expert labels.",
    }
    record["checks"].append({
        "check": "source_schema", "result": "pass",
        "detail": "Strict release schema validated; all 11 findings stay together."})
    stale = not record["expert_truth_available"] and any(path.exists() for path in (
        root / "clinical_risk_masks.csv", root / "evaluation/metrics_by_strategy.csv"))
    record["checks"].append({
        "check": "no_stale_expert_artifacts",
        "result": "fail" if stale else "pass",
        "detail": ("Expert-dependent artifacts found without official expert truth."
                   if stale else "No incompatible expert-dependent artifacts found.")})
    manifest = _check(record, "source_manifest", lambda: _source_manifest(data, root),
                      "All local input bytes/sizes match the saved source manifest.",
                      "Inputs, saved hashes or manifest format differ; rerun training.")
    if manifest is not None:
        record["source_manifest_sha256"] = manifest["manifest_sha256"]
        record["source_files_checked"] = manifest["source_files_checked"]
    if archive is not None:
        pinned = _check(record, "pinned_archive",
                        lambda: _archive_check(archive, data.release_dir),
                        "Pinned bundle, every listed member, and extracted inputs match.",
                        "Bundle hash, member hashes or extracted input bytes differ.")
        if pinned is not None:
            record.update(pinned)
    else:
        record["checks"].append({
            "check": "pinned_archive", "result": "not_checked",
            "detail": "No pinned archive supplied; self-recorded hashes do not prove provenance."})
    _check(record, "saved_schema", lambda: _saved_schema(data, root),
           "Saved column mapping and study/finding counts match current data.",
           "Saved schema/column mapping differs from current release.")
    provenance = _check(record, "run_provenance",
                        lambda: _provenance(data, root, cfg,
                                            manifest["manifest"] if manifest else None),
                        "Saved inputs, config, seed and pinned dependencies agree.",
                        "Saved provenance, config, input hashes or pinned environment differ.")
    if provenance is not None:
        record["saved_code_revision"] = provenance["saved_code_revision"]
        record["code_revision_unchanged"] = provenance["code_revision_unchanged"]
        record["dependencies_checked"] = provenance["dependencies_checked"]
    record["checks"].append({
        "check": "code_revision",
        "result": "pass" if provenance and provenance["code_revision_unchanged"]
                  else "not_checked",
        "detail": ("Current Git revision matches saved training revision." if provenance and
                   provenance["code_revision_unchanged"] else
                   "Saved/current revision unavailable or changed; numeric refit "
                   "is checked separately." if recompute else
                   "Saved/current revision unavailable or changed; use --recompute "
                   "to check numeric reproducibility.")})
    quality = _check(record, "saved_quality", lambda: _check_quality(data, root),
                     "Stored aggregate quality/duplicate diagnostics match current inputs.",
                     "Stored aggregate data quality differs from the current release.")
    if quality is not None:
        record["quality_summary"] = {
            "cv_clusters": quality["cv_clusters"],
            "normalized_report_duplicate_groups": quality["normalized_report_duplicate_groups"],
            "patient_id_missing_studies": quality["metadata_missing_studies"]["patient_id"],
            "warnings": quality["warnings"]}
    checked = _check(record, "saved_weights", lambda: _weights_check(data, root, cfg),
                     "Key alignment, 11 findings, accepted/rejected weights and bounds agree.",
                     "Saved weights do not match inputs, group folds or bounded IPCW rules.")
    if checked is not None:
        weights, stats = checked
        record.update(stats)
        _check(record, "saved_diagnostics", lambda: _diagnostics(root, stats),
               "Stored aggregate propensity diagnostics agree with weights.",
               "Stored propensity/quality diagnostics disagree with weights.")
        plan = root / "review_plan"
        if plan.exists():
            _check(record, "review_plan_source",
                   lambda: _check_review_plan(plan, stats["weights_sha256"]),
                   "Existing review plan refers to the same saved weights.",
                   "Existing review plan refers to other weights or is malformed.")
        else:
            record["checks"].append({"check": "review_plan_source", "result": "not_checked",
                                     "detail": "No optional review plan found."})
        if recompute:
            errors = _check(record, "numeric_recomputation",
                            lambda: _recompute(data, cfg, weights),
                            "Independent seeded grouped refit matches within numeric tolerance.",
                            "Independent grouped refit differs from saved weights.")
            if errors is not None:
                record["max_absolute_recompute_difference"] = errors
    else:
        for name in ("saved_diagnostics", "review_plan_source", "numeric_recomputation"):
            record["checks"].append({"check": name, "result": "not_checked",
                                     "detail": "Requires valid saved weights."})
    if checked is not None and not recompute:
        record["checks"].append({"check": "numeric_recomputation", "result": "not_checked",
                                 "detail": "Pass --recompute for a slower grouped numeric refit."})
    record["checks"].append({"check": "clinical_outcomes", "result": "not_evaluable",
                             "detail": "Input integrity does not establish expert outcome accuracy, "
                                       "independent patient validation or a clinical mask."})
    record["status"] = ("integrity_checks_failed" if any(
        check["result"] == "fail" for check in record["checks"]) else
        "local_consistency_only_not_clinical_validation" if archive is None else
        "bundled_integrity_only_not_clinical_validation")
    return record


def _check_quality(data: ReleaseData, root: Path) -> dict[str, Any]:
    saved = json.loads((root / "quality/data_quality.json").read_text(encoding="utf-8"))
    actual = quality_record(data)
    if saved != actual:
        raise ValueError("Saved quality diagnostics differ from recomputation")
    return actual


def _check_review_plan(root: Path, weights_sha256: str) -> None:
    plan = json.loads((root / "plan_summary.json").read_text(encoding="utf-8"))
    if plan.get("source_weights_sha256") != weights_sha256:
        raise ValueError("Review plan is based on different weights")


def write_verification(record: dict[str, Any], root: Path) -> dict[str, Any]:
    """Write a repeatable, ID-free report; never copy source or per-study rows."""
    out = root / "verification"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(
        record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Local release and artifact verification", "", DISCLAIMER, "",
             f"**{record['status']}**. {record['n_studies']} studies × "
             f"{record['n_findings']} findings = {record['n_pairs']} pairs. "
             "This is an integrity/reproducibility audit, NOT accuracy validation.", "",
             "| Check | Result | Meaning |", "|---|---|---|"]
    lines += [f"| {item['check']} | {item['result']} | {item['detail']} |"
              for item in record["checks"]]
    if "quality_summary" in record:
        quality = record["quality_summary"]
        lines += ["", "## Aggregate data-quality cautions", "",
                  f"Connected known patient/report/study groups: {quality['cv_clusters']}; "
                  f"repeated normalized-report groups: "
                  f"{quality['normalized_report_duplicate_groups']}; "
                  f"studies missing patient IDs: {quality['patient_id_missing_studies']}.", ""]
        lines += [f"- {message}" for message in quality["warnings"]]
    lines += ["", "## Shareable aggregate fingerprints", "",
              f"Canonical input inventory SHA-256: "
              f"`{record.get('source_manifest_sha256', 'not verified')}`.  ",
              f"Saved weight CSV SHA-256: `{record.get('weights_sha256', 'not verified')}`.  ",
              f"Pinned archive SHA-256: `{record.get('archive_sha256', 'not checked')}`.  ",
              "Input inventory digests ignore JSON layout/line endings; "
              "weight CSV byte hashes may differ across OS/BLAS despite a "
              "successful tolerant numeric refit. A digest does not establish "
              "who created inputs, legal permission to share them, or correctness "
              "of labels.", ""]
    if "finding_checks" in record:
        lines += ["## Per-finding acceptance and weight stability (not accuracy)", "",
                  "| Finding | Accepted pairs | Acceptance | Weight ESS |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {row['finding']} | {row['accepted_pairs']} | "
                  f"{row['acceptance_rate']:.1%} | "
                  f"{row['effective_sample_size']:.1f} |"
                  for row in record["finding_checks"]]
    if "max_absolute_recompute_difference" in record:
        lines += ["", "Seeded grouped refit agreed within `atol=1e-5, rtol=1e-3` "
                  "(not a byte-for-byte cross-platform guarantee). Maximum "
                  "absolute differences: " + "; ".join(
                      f"{name}: {value:.6g}" for name, value in
                      record["max_absolute_recompute_difference"].items()) + "."]
    lines += ["", "**Limitations:** no official expert truth in the bundled archive; "
              "the 58 development studies, even if separately supplied, are "
              "not independent validation. Missing patient IDs and upstream "
              "teacher exposure prevent proof of patient-independent outcomes. "
              "No clinical mask, accepted-label accuracy, 1% per-finding risk "
              "certification or guaranteed coverage follows from these checks.", "",
              "Run this command on a separately obtained, appropriately licensed "
              "copy of the inputs before comparing fingerprints. The JSON/Markdown "
              "report contains only fixed check names, hashes, aggregate counts "
              "and finding names; never publish the release, private review "
              "queue, expert answers, weights CSV or patient IDs.", "", DISCLAIMER, ""]
    (out / "verification.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": record["status"], "disclaimer": DISCLAIMER,
            "checks_passed": sum(item["result"] == "pass" for item in record["checks"]),
            "checks_failed": sum(item["result"] == "fail" for item in record["checks"]),
            "report": str(out / "verification.md"),
            "record": str(out / "verification.json")}
