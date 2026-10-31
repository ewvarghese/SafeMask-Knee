"""IPCW clipping/normalization and a known synthetic selection mechanism."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from safemask_knee.audit import bias_table
from safemask_knee.config import load_config
from safemask_knee.io import load_release
from safemask_knee.weights import make_weight_table


def test_ipcw_is_oof_bounded_positive_and_mean_one(release_dir: Path,
                                                   config_path: Path) -> None:
    cfg = load_config(config_path)
    data = load_release(release_dir, cfg, print_mapping=False)
    table, prop = make_weight_table(data, cfg)
    accepted = table.loc[table.accepted60.eq(1)]
    rejected = table.loc[table.accepted60.eq(0)]
    assert len(table) == 1760 and len(accepted) == 1056
    assert np.isfinite(table[["weight_raw", "weight_clipped", "weight_final",
                              "propensity_accepted"]].to_numpy()).all()
    assert (accepted[["weight_raw", "weight_clipped", "weight_final"]] > 0).all().all()
    assert (rejected[["weight_raw", "weight_clipped", "weight_final"]] == 0).all().all()
    assert accepted.weight_final.between(cfg["weight_min"], cfg["weight_max"]).all()
    assert accepted.weight_clipped.between(cfg["weight_min"], cfg["weight_max"]).all()
    assert table.propensity_accepted.between(cfg["propensity_floor"],
                                             1 - cfg["propensity_floor"]).all()
    for _, group in accepted.groupby("finding"):
        assert abs(group.weight_final.mean() - 1) < 1e-8
    assert len(prop.diagnostics["folds"]) == cfg["propensity_folds"]
    assert table.loc[table.expert_label.isna(), "fold"].isna().all()
    assert table.loc[table.expert_label.notna(), "fold"].notna().all()


def test_synthetic_rare_positives_upweighted_and_selection_ratio_improves(
        release_dir: Path, config_path: Path) -> None:
    cfg = load_config(config_path)
    data = load_release(release_dir, cfg, print_mapping=False)
    weights, _ = make_weight_table(data, cfg)
    panel = weights.loc[weights.expert_label.notna()]
    selected = panel.loc[panel.accepted60.eq(1)]
    pos = selected.loc[selected.expert_label.eq(1), "weight_final"]
    neg = selected.loc[selected.expert_label.eq(0), "weight_final"]
    assert len(pos) > 0
    assert pos.mean() > neg.mean() * 1.15
    y = panel.expert_label.to_numpy(dtype=int)
    a = panel.accepted60.to_numpy(dtype=int)
    w = panel.weight_final.to_numpy(dtype=float)
    baseline_sbr = (a[y == 1].mean() / a[y == 0].mean())
    weighted_sbr = ((w[y == 1].sum() / (y == 1).sum()) /
                    (w[y == 0].sum() / (y == 0).sum()))
    assert abs(1 - weighted_sbr) < abs(1 - baseline_sbr)
    assert bias_table(data.frame).accepted_errors.sum() == 11
