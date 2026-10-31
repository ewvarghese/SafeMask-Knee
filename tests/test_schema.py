"""Hard-failing schema checks and explicit alias mapping."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from safemask_knee.audit import fairness_table
from safemask_knee.config import load_config
from safemask_knee.constants import FINDINGS
from safemask_knee.io import SchemaError, load_release, write_manifest, write_schema


def test_loads_wide_long_and_maps_ids(release_dir: Path, config_path: Path,
                                     tmp_path: Path) -> None:
    data = load_release(release_dir, load_config(config_path), print_mapping=False)
    assert len(data.frame) == 160 * len(FINDINGS)
    assert data.frame.study_id.nunique() == 160
    assert data.frame.expert_label.notna().sum() == 638
    assert data.frame.loc[data.frame.expert_label.notna(), "study_id"].nunique() == 58
    assert set(data.frame.finding.astype(str)) == set(FINDINGS)
    assert "Synovitis" not in set(data.frame.finding.astype(str))
    assert {"age", "sex", "site", "scanner", "patient_id"} <= set(data.metadata_fields)
    assert any(item["source_column"] == "exam_id" and
               item["canonical_field"] == "study_id" for item in data.mapping)
    assert any(item["canonical_field"] == "accepted60[ACL]" and
               item["source_column"] == "accepted60_ACL" for item in data.mapping)
    schema = write_schema(data, tmp_path / "schema.json")
    manifest = write_manifest(data, tmp_path / "outputs", tmp_path / "data")
    assert schema["n_evaluated_pairs"] == 638
    assert len(manifest["files"]) == 6  # 3 release + metadata + expert + NOTICE
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert "NOTICE.txt" in {entry["path"] for entry in manifest["files"]}
    assert not (tmp_path / "outputs/accuracy_soft_labels.csv").exists()


def test_bad_wide_finding_fails_loudly(release_dir: Path, config_path: Path,
                                      tmp_path: Path) -> None:
    broken = tmp_path / "release"
    shutil.copytree(release_dir, broken)
    path = broken / "accuracy_soft_labels.csv"
    source = pd.read_csv(path)
    source.drop(columns="ACL").to_csv(path, index=False)
    with pytest.raises(SchemaError, match="missing 11-finding wide schema.*ACL"):
        load_release(broken, load_config(config_path), print_mapping=False)


def test_duplicate_and_misaligned_pairs_fail(release_dir: Path, config_path: Path,
                                              tmp_path: Path) -> None:
    broken = tmp_path / "release"
    shutil.copytree(release_dir, broken)
    path = broken / "accuracy_soft_labels.csv"
    source = pd.read_csv(path)
    pd.concat([source, source.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="duplicate study_id"):
        load_release(broken, load_config(config_path), print_mapping=False)
    source.to_csv(path, index=False)
    accepted_path = broken / "accuracy_accepted60.csv"
    accepted = pd.read_csv(accepted_path)
    accepted.loc[0, "exam_id"] = "NOT-A-SOFT-STUDY"
    accepted.to_csv(accepted_path, index=False)
    with pytest.raises(SchemaError, match="keys do not exactly match"):
        load_release(broken, load_config(config_path), print_mapping=False)


def test_invalid_values_and_missing_metadata(release_dir: Path, config_path: Path,
                                             tmp_path: Path) -> None:
    broken = tmp_path / "release"
    shutil.copytree(release_dir, broken)
    path = broken / "accuracy_accepted60.csv"
    mask = pd.read_csv(path)
    mask["accepted60_ACL"] = mask["accepted60_ACL"].astype(float)
    mask.loc[0, "accepted60_ACL"] = 0.5
    mask.to_csv(path, index=False)
    with pytest.raises(SchemaError, match="binary 0/1"):
        load_release(broken, load_config(config_path), print_mapping=False)
    mask.loc[0, "accepted60_ACL"] = 0
    mask.to_csv(path, index=False)
    (broken / "study_metadata.csv").unlink()
    data = load_release(broken, load_config(config_path), print_mapping=False)
    assert any("metadata" in note.lower() for note in data.warnings)
    assert not {"sex", "site", "age_band"} & set(fairness_table(data.frame).attribute)


def test_config_rejects_unknown_keys_and_missing_tier(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("seed: 3\nsecret_internet_url: http://example.com\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(config)
    config.write_text("tiers:\n  1:\n    findings: [ACL]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="11 findings"):
        load_config(config)


def test_all_three_long_json_metadata_and_role_specific_override(
        release_dir: Path, config_path: Path, tmp_path: Path) -> None:
    """Changing physical layout must not change long-format join semantics."""
    import json
    import yaml

    alternate = tmp_path / "long-release"
    shutil.copytree(release_dir, alternate)
    raw = pd.read_csv(alternate / "accuracy_soft_labels.csv")
    raw = raw.melt(id_vars="exam_id", var_name="finding", value_name="soft_score")
    raw.to_csv(alternate / "accuracy_soft_labels.csv", index=False)
    conf = pd.read_csv(alternate / "accuracy_confidence.csv")
    conf.columns = ["study_id", *[c.removesuffix("_confidence") for c in conf.columns[1:]]]
    conf = conf.melt(id_vars="study_id", var_name="label", value_name="value")
    conf.to_csv(alternate / "accuracy_confidence.csv", index=False)
    accepted = pd.read_csv(alternate / "accuracy_accepted60.csv")
    accepted.columns = ["exam_id", *[c.removeprefix("accepted60_") for c in accepted.columns[1:]]]
    accepted = accepted.melt(id_vars="exam_id", var_name="finding", value_name="accepted_60")
    accepted.to_csv(alternate / "accuracy_accepted60.csv", index=False)
    meta = pd.read_csv(alternate / "study_metadata.csv")
    (alternate / "study_metadata.json").write_text(
        json.dumps({"studies": meta.to_dict("records")}), encoding="utf-8")
    (alternate / "study_metadata.csv").unlink()
    cfg = load_config(config_path)
    data = load_release(alternate, cfg, print_mapping=False)
    assert len(data.frame) == 1760
    assert data.frame.expert_label.notna().sum() == 638
    assert data.frame.groupby("study_id", observed=True).finding.nunique().eq(11).all()
    assert {"long"} == {r["shape"] for r in data.mapping if r["file"].startswith("accuracy_")}
    # On a wide input a role-specific override should affect only that file.
    cfg["columns"]["findings"] = {"ACL": {"soft_score": "ACL"}}
    assert len(load_release(release_dir, cfg, print_mapping=False).frame) == 1760
    cfg["columns"]["findings"] = {"ACL": {"accepted60": "accepted60_ACL"}}
    assert len(load_release(release_dir, cfg, print_mapping=False).frame) == 1760
    config = tmp_path / "overrides.yaml"
    cfg["columns"]["findings"] = {"ACL": {"garbage": "ACL"}}
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid role"):
        load_config(config)


def test_duplicate_raw_header_cannot_be_mangled_silently(release_dir: Path,
                                                       config_path: Path,
                                                       tmp_path: Path) -> None:
    broken = tmp_path / "release"
    shutil.copytree(release_dir, broken)
    path = broken / "accuracy_soft_labels.csv"
    content = path.read_text(encoding="utf-8").splitlines(keepends=True)
    content[0] = content[0].replace("ACL,MCL", "ACL,ACL")
    path.write_text("".join(content), encoding="utf-8")
    with pytest.raises(SchemaError, match="duplicate CSV header"):
        load_release(broken, load_config(config_path), print_mapping=False)


def test_image_tree_is_ignored_entirely(release_dir: Path, config_path: Path,
                                        tmp_path: Path) -> None:
    """An image directory is not needed, traversed for notices, or hashed."""
    broken = tmp_path / "release-with-unused-images"
    shutil.copytree(release_dir, broken)
    image_dir = broken / "images" / "many_studies"
    image_dir.mkdir(parents=True)
    (image_dir / "not_read.dcm").write_bytes(b"not even a valid DICOM file")
    data = load_release(broken, load_config(config_path), print_mapping=False)
    manifest = write_manifest(data, tmp_path / "artifacts", tmp_path / "hashes")
    assert len(data.frame) == 1760
    assert not any(".dcm" in entry["path"] for entry in manifest["files"])
