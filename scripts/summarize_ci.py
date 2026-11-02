#!/usr/bin/env python3
"""Export only aggregate, ID-free research status for GitHub Actions artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from safemask_knee.constants import DISCLAIMER


def public_summary(root: Path) -> dict[str, Any]:
    root = Path(root)
    quality = json.loads((root / "quality/data_quality.json").read_text(encoding="utf-8"))
    propensity = json.loads((root / "propensity_diagnostics.json").read_text(
        encoding="utf-8"))["summary"]
    sensitivity = json.loads((root / "sensitivity/sensitivity_summary.json").read_text(
        encoding="utf-8"))
    provenance = json.loads((root / "run_provenance.json").read_text(
        encoding="utf-8"))
    if propensity["expert_truth_available"] or (root / "clinical_risk_masks.csv").exists():
        raise ValueError("Public CI export is limited to label-free runs without masks")
    revision = provenance.get("git_revision")
    digest = provenance.get("config_sha256_canonical")
    if (revision is not None and (not isinstance(revision, str) or not re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision))) or (
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("Cannot export malformed code/config fingerprints")
    n_studies, n_pairs, n_accepted = (quality.get("n_studies"), quality.get("n_pairs"),
                                      propensity.get("n_accepted"))
    ess = propensity.get("effective_sample_size_accepted")
    fractions = (propensity.get("propensity_lower_clip_fraction"),
                 propensity.get("propensity_upper_clip_fraction"))
    if (any(type(x) is not int or x < 0 for x in (n_studies, n_pairs, n_accepted))
            or not n_studies or n_pairs != n_studies * 11 or not 0 < n_accepted <= n_pairs
            or not isinstance(ess, (int, float)) or not math.isfinite(ess)
            or not 0 < ess <= n_accepted + 1e-6 or any(
                not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1
                for x in fractions)
            or type(sensitivity.get("n_variants")) is not int or sensitivity["n_variants"] < 1):
        raise ValueError("Cannot export malformed aggregate research metrics")
    verification_status = "not_run"
    verification_hashes: dict[str, str] = {}
    recomputed = False
    verification_path = root / "verification/verification.json"
    if verification_path.is_file():
        verified = json.loads(verification_path.read_text(encoding="utf-8"))
        verification_status = verified.get("status")
        if verification_status not in ("bundled_integrity_only_not_clinical_validation",
                                       "local_consistency_only_not_clinical_validation"):
            raise ValueError("Cannot export a failed/malformed verification result")
        results = {item["check"]: item["result"] for item in verified["checks"]}
        if ("fail" in results.values() or results.get("source_manifest") != "pass" or
                results.get("saved_weights") != "pass"):
            raise ValueError("Cannot export unverifiable source/weight artifacts")
        recomputed = results.get("numeric_recomputation") == "pass"
        for field in ("source_manifest_sha256", "weights_sha256"):
            digest = verified.get(field)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Cannot export invalid aggregate fingerprint")
            verification_hashes[field] = digest
    # Strict allowlist: never copy individual study IDs, rows, source files or
    # the (locally licensed) full weight CSV into GitHub artifacts.
    return {"status": "aggregate_research_run_only",
            "disclaimer": DISCLAIMER,
            "code_revision": provenance["git_revision"],
            "config_sha256": provenance["config_sha256_canonical"],
            "n_studies": quality["n_studies"], "n_finding_pairs": quality["n_pairs"],
            "n_accepted_pairs": propensity["n_accepted"],
            "expert_truth_available": propensity["expert_truth_available"],
            "accepted_weight_ess": propensity["effective_sample_size_accepted"],
            "propensity_lower_clip_fraction": propensity["propensity_lower_clip_fraction"],
            "propensity_upper_clip_fraction": propensity["propensity_upper_clip_fraction"],
            "sensitivity_variants": sensitivity["n_variants"],
            "verification_status": verification_status,
            "recomputation_agrees": recomputed,
            "verification_fingerprints": verification_hashes,
            "clinical_mask_generated": False,
            "warning": "Aggregate statistics only. The bundled archive has no official "
                       "expert answers. No accepted accuracy, clinical mask, or "
                       "patient-independent risk guarantee was established."}


def write_public_summary(root: Path, target: Path) -> dict[str, Any]:
    target = Path(target)
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Safe CI summary must be written inside ignored outputs/")
    record = public_summary(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    markdown = ["# SafeMask-Knee aggregate research run", "", DISCLAIMER, "",
                f"Input: {record['n_studies']} studies / "
                f"{record['n_finding_pairs']} finding pairs. "
                f"Accepted pairs: {record['n_accepted_pairs']}. "
                f"IPCW effective sample size: {record['accepted_weight_ess']:.1f}.", "",
                f"Positivity warning: {record['propensity_lower_clip_fraction']:.1%} "
                "of OOF propensities at the floor and "
                f"{record['propensity_upper_clip_fraction']:.1%} at the ceiling.", "",
                f"Local integrity status: {record['verification_status']}; "
                f"seeded numeric refit agreed: {record['recomputation_agrees']}.", "",
                "No official expert truth, independently validated accuracy, "
                "MRI image model, or conformal risk mask is included. This CI "
                "artifact is deliberately aggregate-only; source/derived data "
                "remain subject to source licences, including CC BY-NC 4.0 "
                "for curated predictions.", "", DISCLAIMER, ""]
    (target.parent / "job_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ + "\n" + DISCLAIMER)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/ci/public_summary.json"))
    args = parser.parse_args()
    print(json.dumps(write_public_summary(args.root, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
