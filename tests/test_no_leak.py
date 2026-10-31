"""All splitting, calibration, and model fitting exclude held-out study groups."""

from __future__ import annotations

from pathlib import Path

from safemask_knee.config import load_config
from safemask_knee.conformal import fit_conformal, loo_conformal_scores
from safemask_knee.io import expert_fold_map, grouped_study_folds, load_release
from safemask_knee.masks import make_mask_table
from safemask_knee.propensity import crossfit_propensity, feature_frame


def test_all_eleven_and_patient_group_stay_in_one_fold(release_dir: Path,
                                                        config_path: Path) -> None:
    data = load_release(release_dir, load_config(config_path), print_mapping=False)
    frame = data.frame.copy()
    # Simulate two studies belonging to one patient; they must not cross folds.
    studies = sorted(frame.study_id.unique())
    first_patient = frame.loc[frame.study_id.eq(studies[0]), "patient_id"].iloc[0]
    frame.loc[frame.study_id.eq(studies[1]), "patient_id"] = first_patient
    folds = list(grouped_study_folds(frame, 5, 10))
    all_test: set[str] = set()
    for train, heldout in folds:
        assert not (train & heldout)
        assert (studies[0] in heldout) == (studies[1] in heldout)
        assert all_test.isdisjoint(heldout)
        all_test.update(heldout)
    assert all_test == set(frame.study_id)
    expert = expert_fold_map(data.frame, 5, 10)
    assert len(expert) == 58
    assert set(expert.values()) == set(range(5))
    assert all(frame.loc[frame.study_id.eq(s)].shape[0] == 11 for s in expert)


def test_propensity_inner_and_outer_calibration_disjoint(release_dir: Path,
                                                        config_path: Path) -> None:
    data = load_release(release_dir, load_config(config_path), print_mapping=False)
    x = feature_frame(data.frame)
    assert "accepted60" not in x and "expert_label" not in x and "patient_id" not in x
    acl = data.frame.finding.astype(str).eq("ACL")
    assert x.loc[acl, "other_accepted_acl"].eq(0).all()
    fitted = crossfit_propensity(data.frame, load_config(config_path))
    covered: set[str] = set()
    for spec in fitted.diagnostics["folds"]:
        fit = set(spec["base_fit_studies"])
        platt = set(spec["platt_studies"])
        test = set(spec["test_studies"])
        assert fit.isdisjoint(platt)
        assert fit.isdisjoint(test)
        assert platt.isdisjoint(test)
        assert covered.isdisjoint(test)
        covered |= test
    assert covered == set(data.frame.study_id)
    assert fitted.fold.min() == 0 and fitted.fold.max() == 4


def test_heldout_expert_truth_cannot_affect_own_mask(release_dir: Path,
                                                     config_path: Path) -> None:
    cfg = load_config(config_path)
    data = load_release(release_dir, cfg, print_mapping=False)
    folds = expert_fold_map(data.frame, cfg["outer_folds"], cfg["seed"])
    table, diagnostic = make_mask_table(data, cfg, folds)
    for item in diagnostic["folds"]:
        assert set(item["heldout_studies"]).isdisjoint(item["calibration_studies"])
    study = sorted(folds)[0]
    altered = data.frame.copy()
    loc = altered.study_id.eq(study)
    altered.loc[loc, "expert_label"] = 1 - altered.loc[loc, "expert_label"].astype(int)
    from dataclasses import replace
    perturbed = replace(data, frame=altered)
    changed, _ = make_mask_table(perturbed, cfg, folds)
    assert table.loc[loc, ["accepted_clinical", "conformal_threshold", "reason"]].equals(
        changed.loc[loc, ["accepted_clinical", "conformal_threshold", "reason"]])
    panel = data.frame.loc[data.frame.expert_label.notna()]
    loo = loo_conformal_scores(panel, cfg)
    assert len(loo) == 638
    assert loo.own_study_excluded.all()
    assert set(loo.calibration_study_n) == {57}
    assert loo.true_class_p_loo.between(0, 1).all()
    fold0_test = set(diagnostic["folds"][0]["heldout_studies"])
    train = panel.loc[~panel.study_id.isin(fold0_test)]
    fit_conformal(train, cfg, heldout_studies=fold0_test)
    import pytest
    with pytest.raises(AssertionError, match="own conformal"):
        fit_conformal(panel, cfg, heldout_studies=fold0_test)


def test_proxy_models_and_nested_ipcw_exclude_every_expert_patient(
        release_dir: Path, config_path: Path) -> None:
    """Even a non-expert study for an expert patient cannot train A-D or D's e(X)."""
    from dataclasses import replace

    from safemask_knee.eval import _proxy_predictions
    from safemask_knee.weights import make_weight_table

    cfg = load_config(config_path)
    data = load_release(release_dir, cfg, print_mapping=False)
    frame = data.frame.copy()
    expert_id = sorted(frame.loc[frame.expert_label.notna(), "study_id"].unique())[0]
    other_id = sorted(frame.loc[frame.expert_label.isna(), "study_id"].unique())[0]
    expert_patient = frame.loc[frame.study_id.eq(expert_id), "patient_id"].iloc[0]
    frame.loc[frame.study_id.eq(other_id), "patient_id"] = expert_patient
    modified = replace(data, frame=frame)
    folds = expert_fold_map(modified.frame, cfg["outer_folds"], cfg["seed"])
    weights, _ = make_weight_table(modified, cfg, folds=folds)
    preds = _proxy_predictions(modified, weights, cfg, folds)
    diagnostic = preds.attrs["proxy_training_diagnostics"]
    assert expert_id not in diagnostic["training_studies"]
    assert other_id not in diagnostic["training_studies"]
    for spec in diagnostic["nested_propensity_folds"]:
        assert expert_id not in spec["base_fit_studies"] + spec["platt_studies"]
        assert other_id not in spec["base_fit_studies"] + spec["platt_studies"]
    assert not preds.isna().any().any()
