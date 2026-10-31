"""Phase 2b: stabilized, bounded and finding-normalized training weights."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DISCLAIMER, FINDINGS, WEIGHT_COLUMNS
from .io import ReleaseData, expert_fold_map
from .propensity import PropensityResult, crossfit_propensity

LOG = logging.getLogger(__name__)


def _bounded_mean_one(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale and clip simultaneously: E[clip(scale * w, lo, hi)] == 1."""
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Accepted weights must be finite and positive")
    left, right = 0.0, max(1.0, hi / np.min(values) * 2)
    for _ in range(100):
        mid = (left + right) / 2
        if np.clip(values * mid, lo, hi).mean() < 1:
            left = mid
        else:
            right = mid
    result = np.clip(values * (left + right) / 2, lo, hi)
    if abs(result.mean() - 1) > 1e-8:
        raise AssertionError("Clipped accepted weights did not normalize to mean one")
    return result


def make_weight_table(data: ReleaseData, cfg: dict[str, Any],
                      propensity: PropensityResult | None = None,
                      folds: dict[str, int] | None = None) -> tuple[pd.DataFrame, PropensityResult]:
    """Compute stabilized IPCW (accepted rows only; rejected rows have weight 0).

    ``raw = P(A_f=1 in training folds) / e_OOF(X)``. The probability is
    out-of-fold and floored to handle measured positivity; weights are then
    clipped and jointly rescaled *inside* the bounds within each finding.
    Inverse weighting cannot correct selection on unmeasured expert outcome.
    """
    frame = data.frame
    prop = propensity if propensity is not None else crossfit_propensity(frame, cfg)
    if len(prop.probability) != len(frame):
        raise ValueError("Propensities and release rows must align")
    fold_map = folds if folds is not None else (
        expert_fold_map(frame, cfg["outer_folds"], cfg["seed"])
        if frame.expert_label.notna().any() else {})
    accepted = frame.accepted60.to_numpy(dtype=bool)
    raw = np.where(accepted, prop.finding_prior / prop.probability, 0.0)
    clipped = np.where(accepted, np.clip(raw, cfg["weight_min"], cfg["weight_max"]), 0.0)
    final = np.zeros(len(frame), dtype=float)
    for finding in FINDINGS:
        idx = (frame.finding.astype(str).to_numpy() == finding) & accepted
        if not idx.any():
            raise ValueError(f"Finding {finding} has no accepted labels; IPCW undefined")
        final[idx] = _bounded_mean_one(clipped[idx], cfg["weight_min"], cfg["weight_max"])
    table = frame[["study_id", "finding", "soft_score", "confidence", "accepted60",
                   "expert_label"]].copy()
    table["finding"] = table.finding.astype(str)
    table["propensity_accepted"] = prop.probability
    table["weight_raw"] = raw
    table["weight_clipped"] = clipped
    table["weight_final"] = final
    table["fold"] = table.study_id.map(fold_map).astype("Int64")
    table["notes"] = np.where(accepted, "accepted_oof_ipcw", "rejected_no_verified_label")
    table = table.loc[:, list(WEIGHT_COLUMNS)]
    if not np.isfinite(table[["propensity_accepted", "weight_raw", "weight_clipped",
                              "weight_final"]].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite propensity/weight")
    return table, prop


def write_weights(data: ReleaseData, out: Path, cfg: dict[str, Any],
                  propensity: PropensityResult | None = None,
                  folds: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write the exact twelve-column downstream CSV and propensity diagnostics."""
    out.parent.mkdir(parents=True, exist_ok=True)
    table, prop = make_weight_table(data, cfg, propensity, folds)
    table.to_csv(out, index=False, float_format="%.9g")
    accepted = table.loc[table.accepted60.eq(1)]
    summary = {"path": str(out), "n_rows": len(table),
               "n_accepted": len(accepted), "weight_mean_accepted": float(
                   accepted.weight_final.mean()),
               "weight_min_accepted": float(accepted.weight_final.min()),
               "weight_max_accepted": float(accepted.weight_final.max()),
               "effective_sample_size_accepted": float(
                   accepted.weight_final.sum() ** 2 / (accepted.weight_final ** 2).sum()),
               "propensity_floor": float(cfg["propensity_floor"]),
               "propensity_lower_clip_fraction": float(np.mean(
                   prop.probability <= cfg["propensity_floor"] + 1e-12)),
               "propensity_upper_clip_fraction": float(np.mean(
                   prop.probability >= 1 - cfg["propensity_floor"] - 1e-12)),
               "weight_at_max_fraction_accepted": float(np.mean(
                   accepted.weight_final >= cfg["weight_max"] - 1e-9)),
               "expert_truth_available": bool(data.frame.expert_label.notna().any()),
               "propensity_heldout": True}
    diagnostics = {"disclaimer": DISCLAIMER, "summary": summary,
                   "folds": prop.diagnostics["folds"],
                   "propensity_min": prop.diagnostics["propensity_min"],
                   "propensity_max": prop.diagnostics["propensity_max"]}
    (out.parent / "propensity_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    LOG.info("Weights: %s accepted records; finite clipped normalized weights [%.3f, %.3f]",
             len(accepted), summary["weight_min_accepted"], summary["weight_max_accepted"])
    return table, summary
