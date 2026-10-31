"""Phase 3b: outer-fold OOF expert decisions and full-panel non-expert masks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .conformal import fit_conformal, loo_conformal_scores
from .constants import DISCLAIMER, MASK_COLUMNS
from .io import ReleaseData, expert_fold_map

LOG = logging.getLogger(__name__)


def make_mask_table(data: ReleaseData, cfg: dict[str, Any],
                    folds: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calibrate each expert decision on *other outer-fold studies* only.

    Non-expert rows receive a deployment-oriented score cutoff calibrated on
    all expert studies. Those rows have no expert outcomes and must not be
    included in empirical expert-panel metrics. Reuse of the development panel
    is a material threat to any theoretical future-study guarantee.
    """
    frame = data.frame
    panel = frame.loc[frame.expert_label.notna()]
    fold_map = folds if folds is not None else expert_fold_map(
        frame, cfg["outer_folds"], cfg["seed"])
    decisions: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {"disclaimer": DISCLAIMER,
                                   "calibration_method": "fixed-score Mondrian conformal "
                                   "with study-level outer CV and LOO diagnostics; "
                                   "NOT model-refit Jackknife+", "folds": []}
    for fold in range(cfg["outer_folds"]):
        test_studies = {study for study, value in fold_map.items() if value == fold}
        calibration = panel.loc[~panel.study_id.isin(test_studies)]
        fit = fit_conformal(calibration, cfg, heldout_studies=test_studies)
        heldout_rows = panel.loc[panel.study_id.isin(test_studies)]
        decisions.append(fit.select(heldout_rows))
        LOG.debug("Conformal fold %d per-finding cutoffs: %s", fold, fit.diagnostics())
        diagnostics["folds"].append({"fold": fold,
                                     "heldout_studies": sorted(test_studies),
                                     "calibration_studies": sorted(fit.calibration_studies),
                                     "cutoffs": fit.diagnostics()})
        LOG.info("Mask fold %d: %d calibration studies, %d held-out studies",
                 fold, len(fit.calibration_studies), len(test_studies))
    final_fit = fit_conformal(panel, cfg)
    nonexpert = frame.loc[frame.expert_label.isna()]
    if not nonexpert.empty:
        decisions.append(final_fit.select(nonexpert))
    diagnostics["full_panel_cutoffs_for_nonexpert_only"] = final_fit.diagnostics()
    diagnostics["full_panel_calibration_studies"] = sorted(final_fit.calibration_studies)
    if not decisions:
        raise AssertionError("No rows evaluated by conformal calibration")
    selected = pd.concat(decisions).sort_index()
    if not selected.index.equals(frame.index) or selected.index.duplicated().any():
        raise AssertionError("Each study/finding must receive exactly one mask decision")
    table = frame[["study_id", "finding", "soft_score", "confidence", "accepted60"]].copy()
    table["finding"] = table.finding.astype(str)
    for col in ("tier", "risk_bound", "conformal_threshold", "accepted_clinical", "reason"):
        table[col] = selected[col].to_numpy()
    table["fold"] = table.study_id.map(fold_map).astype("Int64")
    table = table.loc[:, list(MASK_COLUMNS)]
    return table, diagnostics


def write_masks(data: ReleaseData, out: Path, cfg: dict[str, Any],
                folds: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write exactly eleven fields per finding and study (no ground truth needed)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    table, diagnostics = make_mask_table(data, cfg, folds)
    table.to_csv(out, index=False, float_format="%.9g")
    panel = data.frame.loc[data.frame.expert_label.notna()]
    loo = loo_conformal_scores(panel, cfg)
    if not loo.own_study_excluded.all():
        raise AssertionError("LOO conformal score calibrated on its own study")
    loo.to_csv(out.parent / "loo_conformal_scores.csv", index=False, float_format="%.9g")
    diagnostics["loo_diagnostic"] = {"n_pairs": len(loo),
                                     "all_own_studies_excluded": True,
                                     "min_calibration_studies": int(
                                         loo.calibration_study_n.min())}
    # This diagnostic includes only calibration IDs/counts/cutoffs, not source data.
    (out.parent / "conformal_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    panel = table.loc[table.fold.notna()]
    summary = {"path": str(out), "n_rows": len(table),
               "coverage_full_release_mixed_oof_and_full_panel": float(
                   table.accepted_clinical.mean()),
               "coverage_expert_outer_oof": float(panel.accepted_clinical.mean()),
               "accepted_expert_oof": int(panel.accepted_clinical.sum()),
               "method": "study-grouped outer CV for expert rows; full expert panel "
                         "only for non-expert rows; fixed-score class-conditional ranks"}
    LOG.info("Masks: %s/%s held-out expert pairs accepted (%.1f%% coverage)",
             summary["accepted_expert_oof"], len(panel),
             100 * summary["coverage_expert_outer_oof"])
    return table, summary
