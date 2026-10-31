"""Study-grouped evaluation, model-agnostic proxy comparisons, and CIs.

Four tabular distillation proxies compare training targets/weights; they are NOT
MRI image models and do not demonstrate that downstream networks inherit these
metrics. Their pseudo labels come from the release, never expert outcomes.
Expert-derived clinical-mask cutoffs are always outer-fold held-out here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .audit import bias_table
from .constants import DISCLAIMER, FINDINGS
from .io import ReleaseData, expert_fold_map, group_keys, grouped_study_folds
from .propensity import feature_frame, make_preprocessor

LOG = logging.getLogger(__name__)

STRATEGIES: dict[str, str] = {
    "A_naive_accepted_binary": "Naive: rejected pseudo-labels treated as negative",
    "B_soft_labels": "Soft targets on every release pair",
    "C_confidence_weighted": "Soft targets with confidence-weighted loss",
    "D_IPCW_weighted": "Binary accepted pseudo-labels with OOF stabilized IPCW",
}


def study_grouped_kfold(frame: pd.DataFrame, n_splits: int,
                        seed: int) -> Iterator[tuple[set[str], set[str]]]:
    """Public study/patient-grouped K-fold splitter; all 11 rows stay together."""
    return grouped_study_folds(frame, n_splits, seed)


@dataclass
class FractionalModel:
    """Scikit logistic model fitted to fractional binomial targets by duplication."""

    preprocess: Any
    logistic: LogisticRegression | None
    constant: float | None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict a positive-class probability for previously unseen studies."""
        if self.constant is not None:
            return np.full(len(features), self.constant)
        assert self.logistic is not None
        return self.logistic.predict_proba(self.preprocess.transform(features))[:, 1]


def _fractional_logistic(features: pd.DataFrame, target: np.ndarray,
                         weight: np.ndarray, seed: int) -> FractionalModel:
    if not len(features) or len(target) != len(features) or len(weight) != len(features):
        raise ValueError("Invalid proxy training lengths")
    if not np.isfinite(target).all() or not np.isfinite(weight).all() or (
            (target < 0) | (target > 1) | (weight < 0)).any():
        raise ValueError("Proxy targets must be [0,1] and weights nonnegative")
    mass = weight.sum()
    if mass == 0:
        raise ValueError("No weighted pseudo labels available for proxy training")
    positive_mass = float(np.sum(weight * target))
    negative_mass = float(np.sum(weight * (1 - target)))
    constant = np.clip(positive_mass / mass, 1e-6, 1 - 1e-6)
    if min(positive_mass, negative_mass) < 1e-6:
        return FractionalModel(None, None, float(constant))
    pre = make_preprocessor(features)
    matrix = pre.fit_transform(features)
    if sparse.issparse(matrix):
        x = sparse.vstack((matrix, matrix), format="csr")
    else:
        x = np.concatenate((matrix, matrix), axis=0)
    labels = np.concatenate((np.ones(len(target)), np.zeros(len(target))))
    weights = np.concatenate((target * weight, (1 - target) * weight))
    positive = weights > 0
    model = LogisticRegression(max_iter=600, solver="lbfgs", random_state=seed)
    model.fit(x[positive], labels[positive], sample_weight=weights[positive])
    return FractionalModel(pre, model, None)


def expected_calibration_error(y: np.ndarray, probability: np.ndarray,
                               n_bins: int = 10) -> float:
    """Return absolute-difference ECE using fixed-width bins on expert outcomes."""
    if not len(y):
        return float("nan")
    p = np.clip(np.asarray(probability, dtype=float), 0, 1)
    obs = np.asarray(y, dtype=int)
    ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        chosen = ids == bin_id
        if chosen.any():
            error += chosen.mean() * abs(p[chosen].mean() - obs[chosen].mean())
    return float(error)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _percent(value: float) -> str:
    return f"{value:.2%}" if np.isfinite(value) else "not estimable"


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray,
             selected: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    selected = np.asarray(selected, dtype=bool)
    prob = np.asarray(prob, dtype=float)
    correct = pred == y
    pos = y == 1
    neg = ~pos
    positive_pred = pred == 1
    n = len(y)
    if n == 0:
        raise ValueError("No expert outcomes for evaluation")
    return {
        "n_evaluated": n, "n_positive": int(pos.sum()), "n_negative": int(neg.sum()),
        "n_accepted": int(selected.sum()),
        "n_errors_accepted": int((selected & ~correct).sum()),
        "coverage": float(selected.mean()),
        "accuracy_at_coverage": _safe_div(int((selected & correct).sum()), int(selected.sum())),
        "conditional_error_accepted": _safe_div(int((selected & ~correct).sum()),
                                                   int(selected.sum())),
        "accepted_positive_precision": _safe_div(int((selected & positive_pred & pos).sum()),
                                                   int((selected & positive_pred).sum())),
        "selected_recall": _safe_div(int((selected & positive_pred & pos).sum()),
                                     int(pos.sum())),
        "score_recall": _safe_div(int((positive_pred & pos).sum()), int(pos.sum())),
        "specificity": _safe_div(int((~positive_pred & neg).sum()), int(neg.sum())),
        "marginal_fnr": _safe_div(int((selected & ~positive_pred & pos).sum()),
                                  int(pos.sum())),
        "marginal_fpr": _safe_div(int((selected & positive_pred & neg).sum()),
                                  int(neg.sum())),
        "marginal_error": float((selected & ~correct).mean()),
        "selection_bias_ratio": _safe_div(_safe_div(int((selected & pos).sum()),
                                                     int(pos.sum())),
                                           _safe_div(int((selected & neg).sum()),
                                                     int(neg.sum()))),
        "auprc": float(average_precision_score(y, prob))
        if np.unique(y).size == 2 else np.nan,
        "ece": expected_calibration_error(y, prob),
    }


def study_bootstrap_ci(panel: pd.DataFrame, selected: np.ndarray,
                       predicted: np.ndarray, *, repetitions: int,
                       seed: int) -> dict[str, float]:
    """95% percentile CIs by resampling studies (patient clusters if known).

    For a repeated patient, all its studies are sampled together. Every
    bootstrap draw repeats an entire cluster, not an individual finding row.
    Undefined replicates (zero selected or zero positives) are omitted.
    """
    if len(panel) != len(selected) or len(panel) != len(predicted):
        raise ValueError("Bootstrap panel and prediction lengths differ")
    clusters = group_keys(panel).to_numpy(dtype=str)
    _, integer = np.unique(clusters, return_inverse=True)
    g = int(integer.max()) + 1
    sel = np.asarray(selected, dtype=bool)
    pred = np.asarray(predicted, dtype=int)
    y = panel.expert_label.to_numpy(dtype=int)
    fields = {
        "n": np.ones(len(panel)), "selected": sel,
        "correct_selected": sel & (pred == y),
        "selected_positive_true": sel & (pred == 1) & (y == 1),
        "selected_pred_positive": sel & (pred == 1),
        "expert_positive": y == 1,
        "accepted_false_negative": sel & (pred == 0) & (y == 1),
    }
    sums = {key: np.bincount(integer, weights=np.asarray(value, dtype=float),
                             minlength=g) for key, value in fields.items()}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, g, size=(repetitions, g))
    taken = {key: vector[samples].sum(axis=1) for key, vector in sums.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        samples_by_metric = {
            "coverage": taken["selected"] / taken["n"],
            "accuracy_at_coverage": taken["correct_selected"] / taken["selected"],
            "selected_recall": taken["selected_positive_true"] / taken["expert_positive"],
            "accepted_positive_precision": taken["selected_positive_true"] /
                                           taken["selected_pred_positive"],
            "marginal_fnr": taken["accepted_false_negative"] / taken["expert_positive"],
        }
    ci: dict[str, float] = {}
    for key, values in samples_by_metric.items():
        valid = values[np.isfinite(values)]
        ci[f"{key}_ci_low"] = float(np.percentile(valid, 2.5)) if len(valid) else np.nan
        ci[f"{key}_ci_high"] = float(np.percentile(valid, 97.5)) if len(valid) else np.nan
    return ci


def _proxy_predictions(data: ReleaseData, weights: pd.DataFrame,
                       cfg: dict[str, Any], folds: dict[str, int]) -> pd.DataFrame:
    """Fit A-D ONLY on non-expert patient/study groups, evaluate all expert folds.

    D's propensity is nested inside this non-expert training pool. This avoids
    the otherwise subtle leakage of a held-out expert's acceptance/features
    through a globally cross-fitted weight or its normalization constant.
    The published all-study weight CSV remains a separate training artifact.
    """
    from dataclasses import replace

    from .propensity import crossfit_propensity
    from .weights import make_weight_table

    frame = data.frame
    if not frame[["study_id", "finding"]].astype(str).reset_index(drop=True).equals(
            weights[["study_id", "finding"]].astype(str).reset_index(drop=True)):
        raise ValueError("IPCW rows and source release must match exactly")
    expert = frame.loc[frame.expert_label.notna()]
    groups = group_keys(frame)
    expert_groups = set(groups.loc[expert.index])
    training_mask = ~groups.isin(expert_groups).to_numpy()
    training = frame.loc[training_mask].reset_index(drop=True)
    if training.study_id.nunique() < max(cfg["propensity_folds"] + 3, 8):
        raise ValueError("A-D proxy comparisons require at least eight non-expert "
                         "patient/study groups to fit nested training-only IPCW; "
                         "no held-out expert groups may enter training")
    assert training.expert_label.isna().all()
    if set(group_keys(training)) & expert_groups:
        raise AssertionError("Patient/study leakage into tabular proxy training")
    nested_propensity = crossfit_propensity(training, cfg)
    nested_weights, _ = make_weight_table(replace(data, frame=training), cfg,
                                          propensity=nested_propensity, folds={})
    features = feature_frame(frame)
    fit_features = features.loc[training_mask]
    expert_features = features.loc[expert.index]
    pseudo_positive = (training.soft_score.to_numpy() >= 0.5).astype(float)
    accepted = training.accepted60.to_numpy(dtype=float)
    score = training.soft_score.to_numpy(dtype=float)
    confidence = training.confidence.to_numpy(dtype=float)
    expert_confidence = expert.confidence.to_numpy(dtype=float)
    ipcw_train = nested_weights.weight_final.to_numpy(dtype=float)
    result = pd.DataFrame(index=expert.index)
    for idx, name in enumerate(STRATEGIES):
        if name.startswith("A_"):
            target = accepted * pseudo_positive
            sample = np.ones(len(training))
        elif name.startswith("B_"):
            target = score
            sample = np.ones(len(training))
        elif name.startswith("C_"):
            target = score
            sample = np.clip(confidence, 0.05, 1)
        else:
            target = pseudo_positive
            sample = ipcw_train
        model = _fractional_logistic(fit_features, target, sample,
                                     seed=cfg["seed"] + idx)
        probability = model.predict(expert_features)
        quality_train = 2 * np.abs(model.predict(fit_features) - 0.5)
        quality_expert = 2 * np.abs(probability - 0.5)
        if name.startswith("C_"):
            quality_train *= confidence
            quality_expert *= expert_confidence
        cutoff = float(np.quantile(quality_train, 1 - cfg["proxy_coverage"]))
        result[f"prob_{name}"] = probability
        result[f"selected_{name}"] = (quality_expert >= cutoff).astype(int)
    for fold in range(cfg["outer_folds"]):
        heldout = {study for study, num in folds.items() if num == fold}
        if set(training.study_id) & heldout or not set(expert.study_id) >= heldout:
            raise AssertionError("Invalid expert outer fold / proxy training overlap")
        LOG.info("Expert outer fold %d: frozen A-D models trained exclusively on "
                 "%d non-expert studies", fold, training.study_id.nunique())
    if result.isna().any().any() or len(result) != len(expert):
        raise AssertionError("Every expert prediction needs a frozen out-of-panel model")
    result.attrs["proxy_training_diagnostics"] = {
        "method": "A-D trained ONLY on non-expert groups; nested non-expert OOF IPCW "
                  "for D; 5 outer expert folds evaluated by frozen models",
        "training_studies": sorted(training.study_id.unique().astype(str)),
        "expert_studies": sorted(expert.study_id.unique().astype(str)),
        "nested_propensity_folds": nested_propensity.diagnostics["folds"],
    }
    return result


def _tier_metrics(panel: pd.DataFrame, clinical_selected: np.ndarray,
                  cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    prediction = (panel.soft_score.to_numpy() >= 0.5).astype(int)
    for tier, spec in cfg["tiers"].items():
        loc = panel.finding.isin(spec["findings"]).to_numpy()
        y = panel.loc[loc, "expert_label"].to_numpy(dtype=int)
        selected = clinical_selected[loc]
        pred = prediction[loc]
        metric = _metrics(y, pred, panel.loc[loc, "soft_score"].to_numpy(), selected)
        intervals = study_bootstrap_ci(panel.loc[loc], selected, pred,
                                       repetitions=cfg["bootstrap_repetitions"],
                                       seed=cfg["seed"] + 3000 + tier)
        baseline = float(panel.loc[loc, "accepted60"].mean())
        rows.append({"tier": tier, "findings": ", ".join(spec["findings"]),
                     **intervals,
                     "n_evaluated": int(loc.sum()), "n_positive": int(y.sum()),
                     "baseline_coverage": baseline,
                     "clinical_coverage_outer_oof": metric["coverage"],
                     "coverage_gain": metric["coverage"] - baseline,
                     "conditional_accuracy_accepted": metric["accuracy_at_coverage"],
                     "observed_marginal_fnr": metric["marginal_fnr"],
                     "observed_marginal_fpr": metric["marginal_fpr"],
                     "nominal_fnr_bound": spec["risk_bound"],
                     "nominal_fpr_bound": spec["false_positive_bound"],
                     "empirical_bounds_met": bool(metric["marginal_fnr"] <= spec["risk_bound"]
                                                  and metric["marginal_fpr"] <=
                                                  spec["false_positive_bound"]),
                     "disclaimer": DISCLAIMER})
    return pd.DataFrame(rows)


def _finding_risk_checks(per_finding: pd.DataFrame,
                         cfg: dict[str, Any]) -> pd.DataFrame:
    """Check each finding separately; a pooled tier rate cannot certify it."""
    clinical = per_finding.loc[per_finding.strategy.eq("clinical_risk_mask")].set_index("finding")
    released = per_finding.loc[per_finding.strategy.eq("released_accepted60")].set_index("finding")
    rows = []
    for tier, spec in cfg["tiers"].items():
        for finding in spec["findings"]:
            row, base = clinical.loc[finding], released.loc[finding]
            estimable = bool(row.n_positive > 0 and row.n_negative > 0)
            within = bool(estimable and row.marginal_fnr <= spec["risk_bound"] and
                          row.marginal_fpr <= spec["false_positive_bound"])
            rows.append({"finding": finding, "tier": tier,
                         "n_evaluated": int(row.n_evaluated),
                         "n_positive": int(row.n_positive),
                         "n_negative": int(row.n_negative),
                         "baseline_coverage": float(base.coverage),
                         "clinical_coverage_outer_oof": float(row.coverage),
                         "coverage_gain": float(row.coverage - base.coverage),
                         "clinical_errors_accepted": int(row.n_errors_accepted),
                         "accepted_accuracy_outer_oof": float(row.accuracy_at_coverage),
                         "marginal_fnr_outer_oof": float(row.marginal_fnr),
                         "nominal_fnr_bound": spec["risk_bound"],
                         "marginal_fpr_outer_oof": float(row.marginal_fpr),
                         "nominal_fpr_bound": spec["false_positive_bound"],
                         "empirical_bounds_met": within,
                         "note": "descriptive OOF, NOT an independent certification"
                         if estimable else "class unobserved; empirical risk not estimable",
                         "disclaimer": DISCLAIMER})
    return pd.DataFrame(rows)


def _reliability(ax: Any, y: np.ndarray, probability: np.ndarray,
                 label: str) -> None:
    bins = np.minimum((probability * 10).astype(int), 9)
    empirical = [y[bins == i].mean() if (bins == i).any() else np.nan for i in range(10)]
    avg_p = [probability[bins == i].mean() if (bins == i).any() else np.nan
             for i in range(10)]
    ax.plot(avg_p, empirical, marker="o", label=label)


def _plots(panel: pd.DataFrame, preds: pd.DataFrame,
           clinical: np.ndarray, metrics: pd.DataFrame, out: Path) -> None:
    y = panel.expert_label.to_numpy(dtype=int)
    p = panel.soft_score.to_numpy(dtype=float)
    prediction = (p >= 0.5).astype(int)
    q = 2 * np.abs(p - 0.5) * panel.confidence.to_numpy(dtype=float)
    points = []
    for cutoff in np.linspace(0, 1, 61):
        selected = q >= cutoff
        if selected.any():
            points.append((selected.mean(), 1 - (prediction[selected] == y[selected]).mean()))
    fig, ax = plt.subplots(figsize=(7, 5))
    if points:
        ax.plot([v[0] for v in points], [v[1] for v in points],
                label="Soft-score/confidence rank (descriptive)")
    error_values = [value[1] for value in points]
    for strategy, marker, color in (("released_accepted60", "o", "black"),
                                    ("clinical_risk_mask", "s", "tab:orange")):
        row = metrics.loc[metrics.strategy == strategy].iloc[0]
        if np.isfinite(row.accuracy_at_coverage):
            conditional_error = 1 - row.accuracy_at_coverage
            error_values.append(conditional_error)
            ax.scatter([row.coverage], [conditional_error], marker=marker,
                       s=85, c=color, label=strategy)
    ax.set(xlabel="Coverage over expert pairs", ylabel="Error among accepted (descriptive)",
           xlim=(0, 1), ylim=(0, min(1, max(.05, 1.1 * max(error_values, default=0)))),
           title="Risk/coverage on outer-fold expert decisions (not independent test data)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "risk_coverage_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    _reliability(ax, y, p, "released soft score")
    _reliability(ax, y, preds["prob_D_IPCW_weighted"].to_numpy(), "D tabular proxy (OOF)")
    ax.set(xlabel="Mean probability", ylabel="Expert-positive frequency",
           xlim=(0, 1), ylim=(0, 1), title="Reliability on expert development panel")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=150)
    plt.close(fig)


def run_evaluation(data: ReleaseData, weights: pd.DataFrame,
                   masks: pd.DataFrame, out: Path,
                   cfg: dict[str, Any], folds: dict[str, int] | None = None) -> dict[str, Any]:
    """Compare four identically assessed proxies plus release/mask baselines.

    Metrics for expert-dependent masks use only their outer-fold holdouts. All
    11 findings in a study share a fold; no final full-panel cutoff is tested
    on that panel. A-D use identical features/outer folds, but different
    pseudo-targets/weights; thresholding uses only non-held-out X, not truth.
    """
    out.mkdir(parents=True, exist_ok=True)
    frame = data.frame
    fold_map = folds if folds is not None else expert_fold_map(
        frame, cfg["outer_folds"], cfg["seed"])
    if not frame[["study_id", "finding"]].astype(str).reset_index(drop=True).equals(
            masks[["study_id", "finding"]].astype(str).reset_index(drop=True)):
        raise ValueError("Risk-mask rows and release rows must match exactly")
    panel = frame.loc[frame.expert_label.notna()].copy()
    pred = _proxy_predictions(data, weights, cfg, fold_map)
    (out / "proxy_training_diagnostics.json").write_text(
        json.dumps({"disclaimer": DISCLAIMER, **pred.attrs["proxy_training_diagnostics"]},
                   indent=2) + "\n", encoding="utf-8")
    clinical = masks.loc[panel.index, "accepted_clinical"].to_numpy(dtype=bool)
    y = panel.expert_label.to_numpy(dtype=int)
    baseline_pred = panel.released_prediction.to_numpy(dtype=int)
    score_pred = (panel.soft_score.to_numpy() >= 0.5).astype(int)
    eval_cases = {
        "released_accepted60": (panel.soft_score.to_numpy(dtype=float), baseline_pred,
                                panel.accepted60.to_numpy(dtype=bool)),
        **{name: (pred[f"prob_{name}"].to_numpy(dtype=float),
                  (pred[f"prob_{name}"].to_numpy(dtype=float) >= 0.5).astype(int),
                  pred[f"selected_{name}"].to_numpy(dtype=bool))
           for name in STRATEGIES},
        "clinical_risk_mask": (panel.soft_score.to_numpy(dtype=float), score_pred,
                               clinical),
    }
    results: list[dict[str, Any]] = []
    per_finding: list[dict[str, Any]] = []
    for i, (strategy, (prob, predicted, selected)) in enumerate(eval_cases.items()):
        metrics = _metrics(y, predicted, prob, selected)
        metrics.update(study_bootstrap_ci(panel, selected, predicted,
                                          repetitions=cfg["bootstrap_repetitions"],
                                          seed=cfg["seed"] + 1000 + i))
        metrics.update({"strategy": strategy, "outer_folds": cfg["outer_folds"],
                        "n_expert_studies": int(panel.study_id.nunique()),
                        "evidence": "outer study-grouped CV; baseline descriptive",
                        "disclaimer": DISCLAIMER})
        results.append(metrics)
        for j, finding in enumerate(FINDINGS):
            loc = panel.finding.eq(finding).to_numpy()
            one = _metrics(y[loc], predicted[loc], prob[loc], selected[loc])
            one.update(study_bootstrap_ci(panel.loc[loc], selected[loc], predicted[loc],
                                          repetitions=cfg["bootstrap_repetitions"],
                                          seed=cfg["seed"] + 2000 + i * 13 + j))
            one.update({"strategy": strategy, "finding": finding, "disclaimer": DISCLAIMER})
            per_finding.append(one)
    metrics_frame = pd.DataFrame(results)
    finding_frame = pd.DataFrame(per_finding)
    # This SBR uses accepted60, expert truth, and the actual OOF IPCW weights;
    # it is an estimate, not evidence of ignorability or causal debiasing.
    positive = y == 1
    negative = ~positive
    weight = weights.loc[panel.index, "weight_final"].to_numpy(dtype=float)
    a = panel.accepted60.to_numpy(dtype=bool)
    weighted_pos = _safe_div(float(weight[a & positive].sum()), int(positive.sum()))
    weighted_neg = _safe_div(float(weight[a & negative].sum()), int(negative.sum()))
    adjusted_sbr = _safe_div(weighted_pos, weighted_neg)
    metrics_frame["ipcw_adjusted_selection_ratio"] = np.nan
    metrics_frame.loc[metrics_frame.strategy.eq("D_IPCW_weighted"),
                      "ipcw_adjusted_selection_ratio"] = adjusted_sbr
    finding_frame["ipcw_adjusted_selection_ratio"] = np.nan
    for finding in FINDINGS:
        loc = panel.finding.eq(finding).to_numpy()
        this_y, this_a, this_w = y[loc], a[loc], weight[loc]
        pos_f, neg_f = this_y == 1, this_y == 0
        positive_rate = _safe_div(float(this_w[this_a & pos_f].sum()), int(pos_f.sum()))
        negative_rate = _safe_div(float(this_w[this_a & neg_f].sum()), int(neg_f.sum()))
        finding_frame.loc[finding_frame.strategy.eq("D_IPCW_weighted") &
                          finding_frame.finding.eq(finding),
                          "ipcw_adjusted_selection_ratio"] = _safe_div(
                              positive_rate, negative_rate)
    tier = _tier_metrics(panel, clinical, cfg)
    risk_checks = _finding_risk_checks(finding_frame, cfg)
    metrics_frame.to_csv(out / "metrics_by_strategy.csv", index=False, float_format="%.6g")
    finding_frame.to_csv(out / "per_finding_metrics.csv", index=False, float_format="%.6g")
    tier.to_csv(out / "tier_metrics.csv", index=False, float_format="%.6g")
    risk_checks.to_csv(out / "finding_risk_checks.csv", index=False, float_format="%.6g")
    pd.DataFrame({"study_id": sorted(fold_map),
                  "outer_fold": [fold_map[s] for s in sorted(fold_map)]}).to_csv(
        out / "fold_assignments.csv", index=False)
    _plots(panel, pred, clinical, metrics_frame, out)
    baseline = metrics_frame.loc[metrics_frame.strategy.eq("released_accepted60")].iloc[0]
    clinical_row = metrics_frame.loc[metrics_frame.strategy.eq("clinical_risk_mask")].iloc[0]
    naive = finding_frame.loc[finding_frame.strategy.eq("A_naive_accepted_binary")].set_index("finding")
    ours = finding_frame.loc[finding_frame.strategy.eq("D_IPCW_weighted")].set_index("finding")
    flagged = bias_table(frame).query('mnar_flag == "positive_underselected"').finding.tolist()
    improvements = {f: round(float(ours.loc[f, "selected_recall"] -
                                   naive.loc[f, "selected_recall"]), 4)
                    for f in flagged if np.isfinite(ours.loc[f, "selected_recall"])
                    and np.isfinite(naive.loc[f, "selected_recall"])}
    target_precision = cfg["accepted_accuracy_target"]
    d = metrics_frame.loc[metrics_frame.strategy.eq("D_IPCW_weighted")].iloc[0]
    precision_ok = bool(d.accepted_positive_precision >= target_precision)
    coverage_ok = bool(clinical_row.coverage >= max(baseline.coverage,
                                                    cfg["coverage_target"]))
    aspirational_coverage_ok = bool(clinical_row.coverage >= 0.70)
    observed_bound_ok = bool(tier.empirical_bounds_met.all() and
                             risk_checks.empirical_bounds_met.all())
    violations = risk_checks.loc[~risk_checks.empirical_bounds_met, "finding"].tolist()
    expected = (len(panel) == 638 and int(baseline.n_accepted) == 383 and
                int(baseline.n_errors_accepted) == 11)
    note = ("Observed 372/383 accepted correct on 638 expert pairs, matching the "
            "described release counts." if expected else
            "The described release counts (638 expert pairs, 383 accepted, 372 "
            "correct) are NOT reproduced. Inspect the source expert prediction "
            "column, 0.5 score-to-label convention, cohort/version, and mask "
            f"semantics. This input has {len(panel)} evaluated pairs, "
            f"{int(baseline.n_accepted)} accepted, "
            f"{int(baseline.n_accepted - baseline.n_errors_accepted)} correct. "
            f"Prediction source: {data.prediction_source}.")
    prediction_disagreements = int((baseline_pred != score_pred).sum())
    prediction_note = (
        f"The supplied released prediction disagrees with soft_score>=0.5 for "
        f"{prediction_disagreements} expert pairs. The baseline uses the released "
        "prediction; the new masks use >=0.5. These are different label semantics."
        if prediction_disagreements else
        "All supplied released expert predictions agree with soft_score>=0.5."
    )
    lines = ["# Evaluation: study-grouped expert-panel estimates", "", DISCLAIMER, "",
             "Source: user-supplied RSNA knee MRI release; see generated "
             "`ATTRIBUTION.md`. **All comparative expert estimates use an outer "
             f"{cfg['outer_folds']}-fold study/patient-grouped split.** All 11 findings "
             "of a study share a fold. Clinical masks on those studies are calibrated "
             "only on other folds; non-expert rows use the full 58-study panel. "
             "LOO diagnostics exclude the whole study. Released baseline is a "
             "descriptive sanity check, not an independent test; the expert panel "
             "was used during release development. A-D are trained exclusively "
             "on non-expert patient/study groups (D's propensity is nested there), "
             "then compared across the outer expert folds. These are tabular "
             "distillation proxies, **not image models**.", "",
             "## Acceptance-mask sanity check", "", note, "", prediction_note, "",
             f"Released baseline: **{baseline.accuracy_at_coverage:.2%}** accuracy "
             f"among accepted at **{baseline.coverage:.2%}** coverage "
             f"(95% study-bootstrap accuracy interval "
             f"{baseline.accuracy_at_coverage_ci_low:.2%}–"
             f"{baseline.accuracy_at_coverage_ci_high:.2%}; "
             "not an independent validation interval).", "",
             "| Strategy | Coverage | Accuracy among accepted | Selected recall | "
             "Accepted-positive precision | AUPRC | ECE |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics_frame.itertuples():
        lines.append(f"| {row.strategy} | {_percent(row.coverage)} | "
                     f"{_percent(row.accuracy_at_coverage)} | "
                     f"{_percent(row.selected_recall)} | "
                     f"{_percent(row.accepted_positive_precision)} | "
                     f"{row.auprc:.3f} | {row.ece:.3f} |")
    lines += ["", "**Interpretation:** selected recall = accepted true positives / "
              "all expert positives; score recall/specificity ignore acceptance. "
              "IPCW is a *training weight*, not a new class label. Proxy metrics "
              "are tabular diagnostics, not improvements promised for a future CNN.", "",
              "## Tiered risk on outer-fold expert decisions", "",
              "The nominal risk bounds concern **marginal class-conditional "
              "wrong-singleton events over all new studies**, not misclassification "
              "conditional on selection. Tier 1 is the accepted-negative FNR among "
              "all positives. Positive predictive value/97% accepted accuracy "
              "are empirical targets, NOT conformal guarantees. Small positive "
              "class samples at 1% make negative acceptance mathematically "
              "unavailable in many Tier 1 folds.", "",
              "| Tier | Released coverage | OOF mask coverage | OOF accepted accuracy | "
              "Observed marginal FNR (bound) | Observed marginal FPR (bound) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in tier.itertuples():
        lines.append(f"| {row.tier} | {row.baseline_coverage:.2%} | "
                     f"{row.clinical_coverage_outer_oof:.2%} | "
                     f"{row.conditional_accuracy_accepted:.2%} | "
                     f"{row.observed_marginal_fnr:.2%} ({row.nominal_fnr_bound:.1%}) | "
                     f"{row.observed_marginal_fpr:.2%} ({row.nominal_fpr_bound:.1%}) |")
    lines += ["", "Coverage goal vs released mask: " + ("met" if coverage_ok else
              "NOT MET; refusing to relax the rank bound merely to boost coverage") +
              f" ({clinical_row.coverage:.2%} vs {baseline.coverage:.2%}). "
              "Aspirational overall 70–85% coverage: " +
              ("at least 70% reached" if aspirational_coverage_ok else
               "NOT REACHED; Tier 1's 1% rule abstains on predicted negatives") + ". "
              "Observed outer-fold per-tier AND per-finding bounds: " +
              ("within configured values" if observed_bound_ok else
               "one or more EXCEEDED or unestimable; see finding_risk_checks.csv") +
              (". Finding flags: " + ", ".join(violations) if violations else "") + ". "
              "This finite sample cannot certify the operational release because "
              "the score/model development reused the expert panel.", "",
              "## Bias/weight comparison and accepted errors", "",
              "Subtle/under-selected findings (descriptive ratio <0.80): " +
              (", ".join(flagged) or "none flagged") + ". D−A change in selected "
              "recall on these findings: " + (", ".join(f"{f}: {v:+.2%}"
              for f, v in improvements.items()) or "not estimable") + ".",
              f"Accepted60 selection-bias ratio: {baseline.selection_bias_ratio:.3f}; "
              f"OOF-IPCW-adjusted expert ratio: {adjusted_sbr:.3f}. "
              "If the ratio does not move toward 1, observed X does not adequately "
              "explain selection; unmeasured MNAR is not repaired.",
              f"D accepted-positive precision is {d.accepted_positive_precision:.2%}: " +
              ("at/above the 97% empirical target, with no guarantee." if precision_ok else
               "below the 97% empirical target. The available tabular labels/weights "
               "do not establish preservation of released precision."),
              f"Released accepted mistakes: {int(baseline.n_errors_accepted)}; see "
              "`../audit/bias_table.csv` for each finding's false negatives/positives.", "",
              "![Risk/coverage](risk_coverage_curves.png)", "",
              "![Calibration](calibration.png)", "",
              "## Limitations", "",
              "1. The 58 expert studies were used to develop the release; outer-fold "
              "estimates avoid direct fitting/evaluation reuse here but not "
              "upstream data-dependent score selection. No independent test set.",
              "2. Conformal coverage assumes exchangeable new studies and a fixed "
              "score independent of calibration outcomes. This release may violate "
              "those assumptions. LOO fixed-score conformal is **not Jackknife+**.",
              "3. Small per-finding positive counts preclude meaningful 1% finite-"
              "sample risk certification with useful negative acceptance. "
              "Prevalence, site, scanner and calibration shifts break transfer.",
              "4. IPCW assumes selection exchangeable conditional on observed X, "
              "positivity and a correctly estimated propensity; genuine MNAR on "
              "unobserved expert truth cannot be removed by weighting.",
              "5. Neither pseudo-label proxy evaluation nor accepted-mask precision "
              "is evidence of medical safety. No DICOM, images or radiologist "
              "replacement are provided.", "", DISCLAIMER, ""]
    (out / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"n_expert_pairs": len(panel), "released_accuracy": float(
                   baseline.accuracy_at_coverage),
               "released_accuracy_ci_95": [float(baseline.accuracy_at_coverage_ci_low),
                                            float(baseline.accuracy_at_coverage_ci_high)],
               "released_coverage": float(baseline.coverage),
               "released_errors": int(baseline.n_errors_accepted),
               "mask_accuracy_outer_oof": _json_float(clinical_row.accuracy_at_coverage),
               "mask_coverage_outer_oof": float(clinical_row.coverage),
               "coverage_goal_met": coverage_ok,
               "aspirational_70pct_coverage_met": aspirational_coverage_ok,
               "empirical_tier_bounds_met": observed_bound_ok,
               "per_finding_risk_flags": violations,
               "ipcw_accepted_positive_precision": _json_float(
                   d.accepted_positive_precision),
               "ipcw_adjusted_sbr": _json_float(adjusted_sbr),
               "ipcw_subtle_recall_changes": improvements,
               "sanity_check_counts_match": expected}
    (out / "summary.json").write_text(json.dumps(
        {"disclaimer": DISCLAIMER, **summary}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    LOG.info("Expert OOF: baseline %.2f%% @ %.2f%%; risk mask %.2f%% @ %.2f%%",
             100 * baseline.accuracy_at_coverage, 100 * baseline.coverage,
             100 * clinical_row.accuracy_at_coverage, 100 * clinical_row.coverage)
    return summary
