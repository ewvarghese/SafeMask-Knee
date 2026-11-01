"""Preflight check for the CPU runtime, archive, schema, licences and outputs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .constants import DISCLAIMER
from .io import ReleaseData, SchemaError, load_release, sha256_file
from .provenance import _requirements, config_hash

# The known bundled archive is pinned in scripts/bootstrap_release.py too.
KNOWN_ARCHIVE_SHA256 = "7898b2da2ebdefbbebbb15ae20f4ad06ffc69c9fe3ec6b1fdaf29f9e52785d62"


def check_doctor(release: Path, cfg: dict[str, Any], root: Path,
                 archive: Path | None = None, *, require_expert: bool = False
                 ) -> tuple[ReleaseData, dict[str, Any]]:
    """Check inputs without fitting anything; expose only aggregate counts."""
    requirements = _requirements()
    wrong = [f"{name}: expected {spec['pinned']}, installed {spec['installed']}"
             for name, spec in requirements.items()
             if spec["installed"] != spec["pinned"]]
    if wrong:
        raise SchemaError("Pinned dependencies are missing/mismatched: " + "; ".join(wrong))
    warnings: list[str] = []
    archive_digest = None
    if archive is not None:
        archive = Path(archive).expanduser()
        if archive.exists():
            archive_digest = sha256_file(archive)
            if archive_digest != KNOWN_ARCHIVE_SHA256:
                raise SchemaError("Bundled archive SHA-256 mismatch; run "
                                  "scripts/bootstrap_release.py with the known source")
        else:
            warnings.append("Archive not found here; a separately licensed local "
                            "release can still be checked")
    try:
        data = load_release(release, cfg, print_mapping=True,
                            require_expert=require_expert)
    except SchemaError as exc:
        if require_expert and "No evaluation CSV/JSON found" in str(exc):
            raise SchemaError("Expert truth is required for full analysis; "
                              "no official evaluation file was found") from exc
        raise
    if not data.notices:
        warnings.append("No source licence/notice file found; verify input terms")
    n_expert_studies = int(data.frame.loc[data.frame.expert_label.notna(),
                                          "study_id"].nunique())
    n_expert_pairs = int(data.frame.expert_label.notna().sum())
    if n_expert_studies == 0:
        warnings.append("Official expert truth unavailable: `train` can fit "
                        "acceptance/IPCW only; `all`/`masks`/`evaluate` will refuse")
    elif n_expert_studies != 58 or n_expert_pairs != 638:
        warnings.append("Expert panel is incomplete for `all`: exactly 58 studies "
                        "and 638 finding pairs are required")
    if "patient_id" not in data.frame or data.frame.patient_id.isna().any():
        warnings.append("Patient independence not verifiable for all studies")
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".write-check-", dir=root) as scratch:
            scratch.write(b"SafeMask-Knee write test")
    except OSError as exc:
        raise SchemaError(f"Output root not writable: {root}: {exc}") from exc
    stale = False
    manifest = root / "data_manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        previous = {item["path"]: item["sha256"] for item in saved["files"]}
        current = {str(path.relative_to(data.release_dir)): sha256_file(path)
                   for path in set(data.files.values()) | set(data.notices)}
        stale = previous != current
        if stale:
            warnings.append("Previous artifacts use different source hashes; rerun "
                            "the pipeline before reporting")
    provenance_path = root / "run_provenance.json"
    if provenance_path.is_file():
        previous_config = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous_config.get("config_sha256_canonical") != config_hash(cfg):
            stale = True
            warnings.append("Previous artifacts use a different configuration; "
                            "rerun before reporting")
    record = {
        "status": ("ready_label_free_only" if n_expert_studies == 0 else
                   "ready_full" if n_expert_studies == 58 and n_expert_pairs == 638
                   else "expert_panel_incomplete"),
        "disclaimer": DISCLAIMER,
        "n_studies": int(data.frame.study_id.nunique()),
        "n_pairs": len(data.frame),
        "expert_truth_studies": n_expert_studies,
        "expert_truth_pairs": n_expert_pairs,
        "archive_sha256": archive_digest,
        "archive_verified": archive_digest == KNOWN_ARCHIVE_SHA256,
        "dependencies_pinned_and_installed": True if requirements else None,
        "requirements_checked": len(requirements),
        "release_schema_valid": True,
        "release_notices_found": [item.name for item in data.notices],
        "output_writable": True,
        "previous_artifacts_stale": stale,
        "warnings": warnings,
    }
    return data, record


def write_doctor(record: dict[str, Any], root: Path) -> dict[str, Any]:
    """Write concise research-only preflight artifacts without study IDs."""
    path = root / "doctor.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# SafeMask-Knee preflight", "", DISCLAIMER, "",
             f"**{record['status']}**: {record['n_studies']} studies / "
             f"{record['n_pairs']} finding pairs; "
             f"{record['expert_truth_studies']} expert-truth studies.", "",
             f"Archive verified: {record['archive_verified']}. "
             f"Source notices: {', '.join(record['release_notices_found']) or 'none'}. "
             f"Pinned packages checked: {record['requirements_checked']}. "
             f"Output writable: {record['output_writable']}.", "",
             "## Warnings", ""]
    lines += [f"- {warning}" for warning in record["warnings"]] or ["- None."]
    lines += ["", "Preflight checks input shape and environment, NOT clinical "
              "accuracy or patient independence.", "", DISCLAIMER, ""]
    (root / "doctor.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": record["status"], "path": str(path),
            "n_studies": record["n_studies"], "warnings": record["warnings"],
            "disclaimer": DISCLAIMER}
