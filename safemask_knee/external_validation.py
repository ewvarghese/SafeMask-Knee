"""Additional held-out outcome audit on separately scored/labeled studies.

The inputs must contain external scores produced *without* consulting the
external answer key. This workflow can verify disjoint IDs and known patient/
report groups, but it cannot independently verify the upstream model's
training history or score provenance. No clinical certification is issued.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .conformal import fit_conformal
from .constants import DISCLAIMER, FINDINGS
from .eval import _metrics, study_bootstrap_ci
from .io import ReleaseData, SchemaError, _read_expert, sha256_file


def _finite_or_none(value: float | int) -> float | int | None:
    number = float(value)
    return value if np.isfinite(number) else None


def evaluate_external(source: ReleaseData, external: ReleaseData,
                      external_labels: Path, root: Path, out: Path,
                      cfg: dict[str, Any]) -> dict[str, Any]:
    """Evaluate frozen source-panel cutoffs, never fit to external outcomes.

    `external` has all three release-style score/acceptance CSVs for *new*
    studies. Its separate truth CSV must cover every study and finding exactly.
    The official development panel must already have been evaluated with `all`.
    Only aggregate metrics and hashes are written; no external IDs/labels.
    """
    if not source.frame.expert_label.notna().any():
        raise SchemaError("External outcome audit requires official development "
                          "expert truth and a completed `all` run")
    if external.frame.expert_label.notna().any():
        raise SchemaError("Keep external truth separate from external score files "
                          "to prevent fitting on the evaluation answer key")
    source_ids = set(source.frame.study_id.astype(str))
    ext_ids = set(external.frame.study_id.astype(str))
    if not ext_ids or source_ids & ext_ids:
        raise SchemaError("External score study IDs must be nonempty and disjoint "
                          "from ALL source release study IDs")
    for key in ("patient_id", "normalized_report_group"):
        if key in source.frame and key in external.frame:
            dev_values = set(source.frame.loc[source.frame.expert_label.notna(),
                                             key].dropna().astype(str))
            ext_values = set(external.frame[key].dropna().astype(str))
            if dev_values & ext_values:
                raise SchemaError(f"External and development studies share a known "
                                  f"{key} group; independent evaluation refused")
    labels_path = Path(external_labels).expanduser().resolve()
    if labels_path.name.lower() == "expert_oof.csv" or labels_path in set(
            source.files.values()) | set(external.files.values()):
        raise SchemaError("Prediction-only expert_oof.csv or source scores cannot "
                          "substitute for independent expert outcomes")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing separately licensed external truth: {labels_path}")
    truth, _map, _prediction_source = _read_expert(labels_path, cfg)
    if set(truth.study_id.astype(str)) != ext_ids or len(truth) != len(external.frame):
        raise SchemaError("External expert results must have all 11 labels for "
                          "every external score study, with no missing/extra IDs")
    # Form *all* conformal decisions before merging any external outcome.
    panel = source.frame.loc[source.frame.expert_label.notna()]
    model = fit_conformal(panel, cfg, heldout_studies=ext_ids)
    saved_path = root / "conformal_diagnostics.json"
    if not saved_path.is_file():
        raise FileNotFoundError("Run `all` with official development truth first; "
                                "frozen conformal diagnostics are missing")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    if (model.diagnostics() != saved["full_panel_cutoffs_for_nonexpert_only"] or
            set(saved["full_panel_calibration_studies"]) !=
            set(panel.study_id.astype(str))):
        raise SchemaError("Saved expert-panel conformal cutoffs do not match "
                          "the current config/source; rerun `all` before external audit")
    frame = external.frame.copy()
    decisions = model.select(frame)
    if not decisions.index.equals(frame.index):
        raise AssertionError("External mask decisions lost study/finding alignment")
    frame = frame.merge(truth[["study_id", "finding", "expert_label"]],
                        on=["study_id", "finding"], how="left", validate="one_to_one",
                        suffixes=("", "_external"))
    if frame.expert_label_external.isna().any():
        raise SchemaError("Missing external expert truth after joining by study/finding")
    frame["expert_label"] = frame.expert_label_external.astype(int)
    frame = frame.drop(columns=["expert_label_external"])
    predictions = frame.soft_score.ge(.5).to_numpy(dtype=int)
    mask = decisions.accepted_clinical.to_numpy(dtype=bool)
    strategies = {"source_accepted60": frame.accepted60.to_numpy(dtype=bool),
                  "frozen_expert_panel_mask": mask}
    rows: list[dict[str, Any]] = []
    overall: dict[str, Any] = {}
    for index, (strategy, selected) in enumerate(strategies.items()):
        metric = _metrics(frame.expert_label.to_numpy(dtype=int), predictions,
                          frame.soft_score.to_numpy(dtype=float), selected)
        ci = study_bootstrap_ci(frame, selected, predictions,
                                repetitions=cfg["bootstrap_repetitions"],
                                seed=cfg["seed"] + 1000 + index)
        overall[strategy] = {key: _finite_or_none(value)
                             for key, value in {**metric, **ci}.items()}
        for finding in FINDINGS:
            idx = frame.finding.eq(finding).to_numpy(dtype=bool)
            per = _metrics(frame.loc[idx, "expert_label"].to_numpy(dtype=int),
                           predictions[idx], frame.loc[idx, "soft_score"].to_numpy(dtype=float),
                           selected[idx])
            rows.append({"strategy": strategy, "finding": finding,
                         **per, "disclaimer": DISCLAIMER})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_finding_metrics.csv", index=False,
                              float_format="%.8g")
    warnings = ["User-supplied external score and answer provenance was not "
                "independently verified; no clinical/generalization guarantee"]
    for key in ("patient_id", "normalized_report_group"):
        if key not in source.frame or key not in external.frame:
            warnings.append(f"Cannot verify cross-cohort {key} independence: "
                            "metadata absent from one or both inputs")
        elif (source.frame[key].isna().any() or external.frame[key].isna().any()):
            warnings.append(f"Cannot verify cross-cohort {key} for all studies: "
                            "some metadata values are missing")
    if "full_mri_ensemble_weak_target_training_exposure" in external.frame:
        exposure = pd.to_numeric(
            external.frame.drop_duplicates("study_id")[
                "full_mri_ensemble_weak_target_training_exposure"], errors="coerce")
        if exposure.gt(0).any():
            warnings.append("External metadata report upstream MRI teacher "
                            "training exposure; independence is not established")
    if len(ext_ids) < 30:
        warnings.append("Small external cohort; per-finding CIs may be undefined "
                        "or highly unstable")
    hashes = {"external_labels.csv_sha256": sha256_file(labels_path)}
    for name, path in external.files.items():
        if name in ("soft_labels", "confidence", "accepted60", "metadata"):
            hashes[f"external_{name}_sha256"] = sha256_file(path)
    summary = {"status": "user_supplied_external_outcome_audit_not_certified",
               "disclaimer": DISCLAIMER,
               "n_external_studies": len(ext_ids), "n_external_pairs": len(frame),
               "development_studies_excluded": int(panel.study_id.nunique()),
               "external_labels_never_used_in_cutoffs": True,
               "known_id_and_group_overlap": False,
               "source_model_independence_verified": False,
               "outcomes": overall, "input_sha256": hashes, "warnings": warnings}
    (out / "external_summary.json").write_text(json.dumps(
        summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Additional externally scored/labeled cohort — descriptive audit", "",
             DISCLAIMER, "",
             f"**{len(ext_ids)} new studies / {len(frame)} finding pairs** "
             "not in the original release. No external answer was used to "
             "fit score processing, thresholds, or conformal ranks in this "
             "toolkit. The same frozen development-panel cutoffs saved by `all` "
             "were applied to externally supplied scores before outcome join.", "",
             "| Strategy | Coverage | Accepted accuracy | Accepted errors | "
             "Selected recall | Marginal FNR | 95% cluster-bootstrap accuracy CI |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for strategy, metric in overall.items():
        def percent(value: float | int | None) -> str:
            return f"{float(value):.2%}" if value is not None else "unavailable"
        ci_low = metric["accuracy_at_coverage_ci_low"]
        ci_high = metric["accuracy_at_coverage_ci_high"]
        lines.append(f"| {strategy} | {percent(metric['coverage'])} | "
                     f"{percent(metric['accuracy_at_coverage'])} | "
                     f"{metric['n_errors_accepted']} | {percent(metric['selected_recall'])} | "
                     f"{percent(metric['marginal_fnr'])} | "
                     f"{percent(ci_low)}–{percent(ci_high)} |")
    lines += ["", "Study/patient/report clusters (when provided) were resampled "
              "with all 11 findings together. `per_finding_metrics.csv` includes "
              "the complete finding-level outcomes; undefined metrics are blank, "
              "not zero. Input file hashes are in `external_summary.json`; raw "
              "external labels and identifiers are **not** copied to this report.", "",
              "## Provenance and safety limitations", ""]
    lines += [f"- {warning}" for warning in warnings]
    lines += ["", "No per-finding 1% risk certificate, simultaneous familywise "
              "guarantee, or independent clinical validation can be inferred "
              "from this evaluation alone. Separately licensed scores/labels "
              "remain local and Git-ignored; verify source licensing and "
              "upstream training independence separately.", "", DISCLAIMER, ""]
    (out / "external_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": summary["status"], "path": str(out / "external_report.md"),
            "n_external_studies": len(ext_ids), "input_sha256": hashes,
            "source_model_independence_verified": False,
            "disclaimer": DISCLAIMER}
