"""Phase 2a: grouped out-of-fold, held-out-Platt acceptance propensity."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import FINDINGS, normalize
from .io import SchemaError, grouped_study_folds

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PropensityResult:
    """Cross-fitted probabilities and training-fold marginal acceptance rates."""

    probability: np.ndarray
    finding_prior: np.ndarray
    fold: np.ndarray
    diagnostics: dict[str, Any]


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make tabular X without expert outcomes, IDs, or the row's own accepted flag.

    Parameters
    ----------
    frame : DataFrame
        Validated long table with all 11 rows per study.

    Returns
    -------
    DataFrame
        Identical feature columns across every split. Each accepted indicator
        for the *row's own finding* is replaced with zero to prevent leakage
        of the propensity target. All other findings are allowed covariates.
    """
    needed = {"study_id", "finding", "soft_score", "confidence", "accepted60"}
    if not needed.issubset(frame.columns):
        raise SchemaError(f"Missing features: {sorted(needed - set(frame.columns))}")
    score_wide = frame.pivot(index="study_id", columns="finding", values="soft_score")
    accepted_wide = frame.pivot(index="study_id", columns="finding", values="accepted60")
    score_matrix = score_wide.reindex(index=frame.study_id, columns=FINDINGS).to_numpy(dtype=float)
    accepted_matrix = accepted_wide.reindex(index=frame.study_id, columns=FINDINGS).to_numpy(dtype=float)
    if not np.isfinite(score_matrix).all() or not np.isfinite(accepted_matrix).all():
        raise SchemaError("Every study must have all 11 soft and accepted findings")
    own_score = frame.soft_score.to_numpy(dtype=float)
    own_accepted = frame.accepted60.to_numpy(dtype=float)
    names = frame.finding.astype(str).to_numpy()
    features = pd.DataFrame({
        "soft_score": own_score,
        "confidence": frame.confidence.to_numpy(dtype=float),
        "distance_from_half": np.abs(own_score - 0.5),
        "peer_soft_mean": (score_matrix.sum(axis=1) - own_score) / (len(FINDINGS) - 1),
        "peer_accepted_rate": (accepted_matrix.sum(axis=1) - own_accepted) /
                              (len(FINDINGS) - 1),
        "finding": names,
    }, index=frame.index)
    for i, finding in enumerate(FINDINGS):
        suffix = normalize(finding)
        self_row = names == finding
        features[f"other_soft_{suffix}"] = np.where(self_row, 0, score_matrix[:, i])
        features[f"other_accepted_{suffix}"] = np.where(self_row, 0, accepted_matrix[:, i])
    if "age" in frame:
        features["age"] = pd.to_numeric(frame.age, errors="coerce").to_numpy()
    for col in ("sex", "site", "scanner"):
        if col in frame:
            features[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    if "accepted60" in features or "expert_label" in features or "patient_id" in features:
        raise AssertionError("Label/ID leakage into features")
    return features


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Fit imputation/scaling only on the caller's training studies."""
    categorical = [c for c in ("finding", "sex", "site", "scanner") if c in features]
    numeric = [c for c in features if c not in categorical]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(
            strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(
                                      handle_unknown="infrequent_if_exist",
                                      min_frequency=0.01))]), categorical),
    ], remainder="drop", sparse_threshold=0.3)


def _base_classifier(features: pd.DataFrame, seed: int) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(features)),
                     ("logistic", LogisticRegression(
                         max_iter=600, solver="lbfgs", random_state=seed))])


def crossfit_propensity(frame: pd.DataFrame, cfg: dict[str, Any]) -> PropensityResult:
    """Cross-fit P(A=1|X) and Platt-calibrate on disjoint inner study groups.

    All 11 findings for a study (and all studies for a known patient) stay in
    the same outer fold. For each fold the base classifier is fit on 80% of the
    other studies, Platt is fit on the remaining studies, and the resulting
    predictor is used *only* on the outer holdout. No expert label is used.
    """
    frame = frame.reset_index(drop=True)
    features = feature_frame(frame)
    probability = np.full(len(frame), np.nan)
    prior = np.full(len(frame), np.nan)
    folds = np.full(len(frame), -1, dtype=int)
    diagnostics: dict[str, Any] = {"procedure": "study-grouped outer OOF + disjoint-study "
                                 "Platt holdout; no expert outcomes used", "folds": []}
    for fold, (train_studies, test_studies) in enumerate(grouped_study_folds(
            frame, cfg["propensity_folds"], cfg["seed"] + 1)):
        train = frame.study_id.isin(train_studies).to_numpy()
        test = frame.study_id.isin(test_studies).to_numpy()
        if not test.any() or (folds[test] >= 0).any():
            raise AssertionError("Invalid/overlapping propensity test fold")
        train_frame = frame.loc[train]
        # A second study-grouped split sets aside disjoint calibration studies.
        n_groups = train_frame.study_id.nunique()
        n_calib = max(1, round(n_groups * cfg["platt_calibration_fraction"]))
        n_calib = min(n_calib, n_groups // 2)
        if n_calib < 1 or n_groups < 4:
            raise SchemaError("At least four studies in each propensity training fold "
                              "are needed for disjoint Platt calibration")
        # Use deterministic shuffled unique group keys rather than row splitting.
        from .io import group_keys  # local import keeps the module dependency simple
        unique = train_frame[["study_id"]].copy()
        unique["group"] = group_keys(train_frame).to_numpy()
        groups = np.sort(unique.group.unique().astype(str))
        rng = np.random.default_rng(cfg["seed"] + 100 + fold)
        rng.shuffle(groups)
        n_calib = max(1, min(len(groups) - 2, round(len(groups) *
                                                    cfg["platt_calibration_fraction"])))
        calib_studies = set(unique.loc[unique.group.isin(groups[:n_calib]), "study_id"])
        fit = train & ~frame.study_id.isin(calib_studies).to_numpy()
        cal = train & frame.study_id.isin(calib_studies).to_numpy()
        y = frame.accepted60.to_numpy(dtype=int)
        if len(np.unique(y[fit])) < 2 or len(np.unique(y[cal])) < 2:
            raise SchemaError(f"Propensity fold {fold} lacks both acceptance classes "
                              "in base-fit or Platt calibration studies")
        base = _base_classifier(features, cfg["seed"] + fold)
        base.fit(features.loc[fit], y[fit])
        calibrated_base = np.clip(base.predict_proba(features.loc[cal])[:, 1], 1e-6, 1 - 1e-6)
        platt = LogisticRegression(max_iter=300, solver="lbfgs",
                                   random_state=cfg["seed"] + fold)
        platt.fit(logit(calibrated_base).reshape(-1, 1), y[cal])
        test_base = np.clip(base.predict_proba(features.loc[test])[:, 1], 1e-6, 1 - 1e-6)
        predicted = platt.predict_proba(logit(test_base).reshape(-1, 1))[:, 1]
        probability[test] = np.clip(predicted, cfg["propensity_floor"],
                                    1 - cfg["propensity_floor"])
        LOG.debug("Platt fold %d: coef=%s intercept=%s, held-out propensity range=[%.4g, %.4g]",
                  fold, platt.coef_.tolist(), platt.intercept_.tolist(),
                  probability[test].min(), probability[test].max())
        finding_marginal = train_frame.groupby("finding", observed=True)["accepted60"].mean()
        prior[test] = frame.loc[test, "finding"].astype(str).map(finding_marginal).to_numpy()
        folds[test] = fold
        diagnostics["folds"].append({
            "fold": fold, "test_studies": sorted(test_studies),
            "base_fit_studies": sorted(set(frame.loc[fit, "study_id"])),
            "platt_studies": sorted(calib_studies),
            "n_test_pairs": int(test.sum()),
        })
        LOG.info("Propensity fold %d: %d fit, %d calibration, %d held-out pairs",
                 fold, fit.sum(), cal.sum(), test.sum())
    if (folds < 0).any() or not np.isfinite(probability).all() or not np.isfinite(prior).all():
        raise AssertionError("Some rows lack valid OOF acceptance propensity/prior")
    diagnostics["propensity_min"] = float(probability.min())
    diagnostics["propensity_max"] = float(probability.max())
    diagnostics["propensity_mean"] = float(probability.mean())
    return PropensityResult(probability, prior, folds, diagnostics)
