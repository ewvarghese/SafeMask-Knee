"""Outer-heldout mask risk and independent IID conformal rank smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from safemask_knee.config import load_config
from safemask_knee.conformal import fit_conformal
from safemask_knee.constants import FINDINGS
from safemask_knee.io import load_release
from safemask_knee.masks import make_mask_table


def test_synthetic_outer_fold_risk_and_coverage(release_dir: Path,
                                               config_path: Path) -> None:
    cfg = load_config(config_path)
    data = load_release(release_dir, cfg, print_mapping=False)
    table, diagnostics = make_mask_table(data, cfg)
    assert table.accepted_clinical.isin([0, 1]).all()
    assert table.tier.isin([1, 2, 3]).all()
    panel = table.loc[table.fold.notna()].merge(
        data.frame.loc[data.frame.expert_label.notna(),
                       ["study_id", "finding", "expert_label"]].assign(
                           finding=lambda d: d.finding.astype(str)),
        on=["study_id", "finding"], validate="one_to_one")
    assert len(panel) == 638
    assert panel.accepted_clinical.mean() >= panel.accepted60.mean()
    assert panel.accepted_clinical.mean() > .6
    for spec in diagnostics["folds"]:
        for name in cfg["tiers"][1]["findings"]:
            assert spec["cutoffs"][name]["negative_score_ceiling"] is None
        assert set(spec["heldout_studies"]).isdisjoint(spec["calibration_studies"])
    for tier, spec in cfg["tiers"].items():
        sub = panel.loc[panel.tier.eq(tier)]
        pred = (sub.soft_score >= .5).to_numpy()
        y = sub.expert_label.to_numpy(dtype=int)
        accepted = sub.accepted_clinical.to_numpy(dtype=bool)
        fnr = ((accepted & ~pred & (y == 1)).sum() / (y == 1).sum())
        fpr = ((accepted & pred & (y == 0)).sum() / (y == 0).sum())
        assert fnr <= spec["risk_bound"]
        assert fpr <= spec["false_positive_bound"]


def test_independent_iid_classconditional_risks_approximately_bounded(
        config_path: Path) -> None:
    cfg = load_config(config_path)
    rng = np.random.default_rng(2026)

    def sample(n: int, offset: int) -> pd.DataFrame:
        y = rng.binomial(1, .35, size=(n, 11))
        score = np.where(y == 1, rng.beta(7, 3, size=(n, 11)),
                         rng.beta(3, 7, size=(n, 11)))
        return pd.DataFrame({"study_id": np.repeat(
                                [f"IID-{offset+i:05d}" for i in range(n)], 11),
                             "finding": np.tile(FINDINGS, n),
                             "soft_score": score.reshape(-1),
                             "expert_label": y.reshape(-1)})

    fit = fit_conformal(sample(400, 0), cfg)
    heldout = sample(3000, 10000)
    selected = fit.select(heldout)
    for finding in FINDINGS:
        loc = heldout.finding.eq(finding)
        sub = heldout.loc[loc]
        mask = selected.loc[loc].accepted_clinical.to_numpy(dtype=bool)
        y = sub.expert_label.to_numpy(dtype=int)
        predicted = (sub.soft_score >= .5).to_numpy()
        fnr = (mask & ~predicted & (y == 1)).sum() / (y == 1).sum()
        fpr = (mask & predicted & (y == 0)).sum() / (y == 0).sum()
        assert fnr <= fit.alpha_positive[finding] + 0.03
        assert fpr <= fit.alpha_negative[finding] + 0.03
    # The distribution-free 1% claim does NOT preclude negative acceptance
    # here: this independent calibration has ~140 positives/finding (>99).
    # A separate tiny (58-study) panel cannot exclude the positive class.
    small = fit_conformal(sample(58, 20000), cfg)
    assert all(small.low_cutoff[f] is None for f in cfg["tiers"][1]["findings"])
