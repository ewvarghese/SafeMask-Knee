"""Phase 3a: leave-one-study-out, class-conditional conformal rank calibration.

This is LOO/Mondrian conformal with a *fixed* released soft score, not a genuine
Jackknife+ procedure: Jackknife+ would require refitting an underlying MRI model
for each held-out study, and images/models are not supplied. Guarantees are
nominal and require scores fixed independently of calibration outcomes and
exchangeability. Development-panel reuse may break those conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .constants import FINDINGS
from .io import SchemaError, group_keys


@dataclass(frozen=True)
class ConformalFit:
    """Per-finding positive/negative nonconformity ranks from calibration studies."""

    positive_scores: dict[str, np.ndarray]
    negative_scores: dict[str, np.ndarray]
    low_cutoff: dict[str, float | None]
    high_cutoff: dict[str, float | None]
    alpha_positive: dict[str, float]
    alpha_negative: dict[str, float]
    tier: dict[str, int]
    calibration_studies: frozenset[str]

    def select(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return OOF-safe acceptance decisions, p-values, and score cutoffs.

        A wrong accepted singleton excludes the true class. Therefore, for a
        fixed score and exchangeable studies, false-negative acceptance among
        all true positives is bounded by alpha_positive; the analogous false-
        positive event is bounded by alpha_negative. Neither bound applies to
        the conditional error rate *among accepted records*.
        """
        parts: list[pd.DataFrame] = []
        for finding in FINDINGS:
            subset = frame.loc[frame.finding == finding]
            if subset.empty:
                continue
            score = subset.soft_score.to_numpy(dtype=float)
            if ((score < 0) | (score > 1)).any():
                raise SchemaError("Conformal soft scores must be in [0,1]")
            pos = self.positive_scores[finding]
            neg = self.negative_scores[finding]
            p_pos = (1 + np.searchsorted(pos, score, side="right")) / (len(pos) + 1)
            p_neg = ((1 + len(neg) - np.searchsorted(neg, score, side="left")) /
                     (len(neg) + 1))
            include_pos = p_pos > self.alpha_positive[finding]
            include_neg = p_neg > self.alpha_negative[finding]
            predicted_pos = score >= 0.5
            pos_singleton = include_pos & ~include_neg
            neg_singleton = include_neg & ~include_pos
            accept = (predicted_pos & pos_singleton) | (~predicted_pos & neg_singleton)
            threshold = np.where(predicted_pos, self.high_cutoff[finding]
                                 if self.high_cutoff[finding] is not None else np.nan,
                                 self.low_cutoff[finding]
                                 if self.low_cutoff[finding] is not None else np.nan)
            reason = np.full(len(subset), "ambiguous_conformal_set", dtype=object)
            reason[~include_pos & ~include_neg] = "empty_conformal_set"
            reason[(pos_singleton & ~predicted_pos) |
                   (neg_singleton & predicted_pos)] = "score_sign_disagrees"
            reason[accept] = "singleton_conformal"
            reason[(~predicted_pos) & (self.low_cutoff[finding] is None)] = (
                "insufficient_positive_calibration")
            reason[predicted_pos & (self.high_cutoff[finding] is None)] = (
                "insufficient_negative_calibration")
            parts.append(pd.DataFrame({
                "accepted_clinical": accept.astype(np.int8),
                "conformal_threshold": threshold.astype(float),
                "reason": reason, "p_positive": p_pos,
                "p_negative": p_neg, "tier": self.tier[finding],
                "risk_bound": self.alpha_positive[finding],
            }, index=subset.index))
        return pd.concat(parts).sort_index()

    def diagnostics(self) -> dict[str, Any]:
        """Serialize calibration group counts and both score cutoffs."""
        return {finding: {"tier": self.tier[finding],
                          "n_positive": len(self.positive_scores[finding]),
                          "n_negative": len(self.negative_scores[finding]),
                          "fnr_bound": self.alpha_positive[finding],
                          "fpr_bound": self.alpha_negative[finding],
                          "negative_score_ceiling": self.low_cutoff[finding],
                          "positive_score_floor": self.high_cutoff[finding]}
                for finding in FINDINGS}


def _cutoffs(pos: np.ndarray, neg: np.ndarray,
             alpha_pos: float, alpha_neg: float) -> tuple[float | None, float | None]:
    """Find most permissive single-sided cutoffs allowed by fixed rank alphas."""
    # For negative singleton, exclude candidate positive: p_positive <= alpha_pos.
    k_pos = int(np.floor(alpha_pos * (len(pos) + 1) - 1 + 1e-12))
    low = float(np.nextafter(pos[k_pos], -np.inf)) if 0 <= k_pos < len(pos) else None
    # For positive singleton, exclude candidate negative: p_negative <= alpha_neg.
    k_neg = int(np.floor(alpha_neg * (len(neg) + 1) - 1 + 1e-12))
    descending = neg[::-1]
    high = float(np.nextafter(descending[k_neg], np.inf)) if 0 <= k_neg < len(neg) else None
    return low, high


def fit_conformal(panel: pd.DataFrame, cfg: dict[str, Any],
                  *, heldout_studies: set[str] | None = None) -> ConformalFit:
    """Fit per-finding ranks without looking at specified held-out study IDs.

    Parameters
    ----------
    panel : DataFrame
        Expert-calibrated *training* studies only. Never include a tested study.
    cfg : dict
        Validated tier-specific nominal risk bounds.
    heldout_studies : set of str, optional
        Defensive assertion preventing in-fold calibration.
    """
    heldout = heldout_studies or set()
    studies = set(panel.study_id.astype(str))
    if studies & heldout:
        raise AssertionError("Expert study appears in its own conformal calibration panel")
    if not len(panel) or panel.expert_label.isna().any():
        raise SchemaError("Conformal calibration requires observed expert outcomes only")
    if panel.groupby("study_id", observed=True).finding.nunique().ne(len(FINDINGS)).any():
        raise SchemaError("Each calibration study requires all 11 findings")
    pos: dict[str, np.ndarray] = {}
    neg: dict[str, np.ndarray] = {}
    low: dict[str, float | None] = {}
    high: dict[str, float | None] = {}
    alpha_pos: dict[str, float] = {}
    alpha_neg: dict[str, float] = {}
    tier_lookup: dict[str, int] = {}
    for tier, spec in cfg["tiers"].items():
        for finding in spec["findings"]:
            tier_lookup[finding] = int(tier)
            alpha_pos[finding] = float(spec["risk_bound"])
            alpha_neg[finding] = float(spec["false_positive_bound"])
    for finding in FINDINGS:
        rows = panel.loc[panel.finding == finding]
        pos[finding] = np.sort(rows.loc[rows.expert_label.eq(1), "soft_score"].to_numpy(
            dtype=float))
        neg[finding] = np.sort(rows.loc[rows.expert_label.eq(0), "soft_score"].to_numpy(
            dtype=float))
        low[finding], high[finding] = _cutoffs(pos[finding], neg[finding],
                                               alpha_pos[finding], alpha_neg[finding])
    return ConformalFit(pos, neg, low, high, alpha_pos, alpha_neg, tier_lookup,
                        frozenset(studies))


def loo_conformal_scores(panel: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute true-class conformal p-values leaving out the *entire study*.

    A repeated patient is also removed via patient grouping. This explicit
    leave-one-group-out diagnostic is not a refitted-model Jackknife+ interval.
    Results may be inspected, but cannot be used to report final independent
    accuracy on the same development panel.
    """
    panel = panel.loc[panel.expert_label.notna()].copy()
    panel["_group"] = group_keys(panel).to_numpy()
    parts: list[dict[str, Any]] = []
    for study, rows in panel.groupby("study_id", observed=True):
        group = str(rows._group.iloc[0])
        calibration = panel.loc[panel._group != group].drop(columns="_group")
        fit = fit_conformal(calibration, cfg, heldout_studies={str(study)})
        decision = fit.select(rows)
        for idx, row in rows.iterrows():
            finding = str(row.finding)
            true_pos = int(row.expert_label) == 1
            parts.append({"study_id": str(study), "finding": finding,
                          "true_class_p_loo": float(decision.loc[idx, "p_positive"
                                                              if true_pos else "p_negative"]),
                          "calibration_same_class_n": len(fit.positive_scores[finding]
                                                          if true_pos else
                                                          fit.negative_scores[finding]),
                          "calibration_study_n": len(fit.calibration_studies),
                          "own_study_excluded": str(study) not in fit.calibration_studies})
    return pd.DataFrame(parts)
