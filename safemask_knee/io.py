"""Local release loading, strict schema checks, provenance, and grouped splits.

Only tabular files and text notices are opened. No image or network access is
performed anywhere in the package.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .constants import DISCLAIMER, FINDINGS, canonical_finding, normalize

LOG = logging.getLogger(__name__)


class SchemaError(ValueError):
    """An input file is missing, ambiguous, duplicated, or malformed."""


@dataclass(frozen=True)
class ReleaseData:
    """Validated, complete long-format release with optional expert outcomes."""

    frame: pd.DataFrame
    mapping: list[dict[str, str]]
    files: dict[str, Path]
    notices: tuple[Path, ...]
    warnings: tuple[str, ...]
    metadata_fields: tuple[str, ...]
    release_dir: Path
    prediction_source: str


_ID_ALIASES = ("studyid", "examid", "studyinstanceuid", "examuid",
               "caseid", "study", "exam")
_FINDING_ALIASES = ("finding", "findingname", "diagnosis", "condition",
                    "labelname", "label")
_ROLE_ALIASES = {
    "soft_score": ("softscore", "softlabel", "probability", "prob", "score", "value"),
    "confidence": ("confidence", "conf", "confidencescore", "value"),
    "accepted60": ("accepted60", "accepted", "acceptance", "mask", "value"),
    "expert_label": ("expertlabel", "groundtruth", "truth", "expert",
                     "target", "label", "ytrue", "value"),
    "predicted_label": ("predictedlabel", "prediction", "predicted", "pred", "ypred"),
}
_METADATA_ALIASES = {
    "patient_id": ("patientid", "patientidentifier", "subjectid", "personid"),
    "age": ("age", "patientage", "ageyears"),
    "sex": ("sex", "gender", "patientsex"),
    "site": ("site", "siteid", "institution", "center", "centre", "hospital"),
    "scanner": ("scanner", "scannermodel", "scannermanufacturer", "manufacturer",
                "scannername", "machine"),
    # These fields are operational release metadata, NOT protected demographic
    # variables or patient identifiers. A normalized-report hash is a stronger
    # grouping key when identical reports appear under multiple study UIDs.
    "normalized_report_group": ("normalizedreportgroup",),
    "report_script_bucket": ("reportscriptbucket",),
    "report_length_bucket": ("reportlengthbucket",),
    "multi_report_context_flag": ("multireportcontextflag",),
    "verified_mri_teacher_available": ("verifiedmriteacheravailable",),
    "official_expert_evaluation_row": ("officialexpertevaluationrow",),
    "full_mri_ensemble_weak_target_training_exposure": (
        "fullmriensembleweaktargettrainingexposure",),
}


def _column(df: pd.DataFrame, aliases: tuple[str, ...], override: str | None,
            path: Path, label: str, *, optional: bool = False,
            exclude: set[str] | None = None) -> str | None:
    """Identify one source column, erroring rather than guessing ambiguous matches."""
    exclude = exclude or set()
    if override is not None:
        if override not in df.columns:
            raise SchemaError(f"{path.name}: override for {label}={override!r} is absent")
        if override in exclude:
            raise SchemaError(f"{path.name}: {label} override reuses an ID/finding column")
        return override
    candidates = [c for c in df.columns if normalize(c) in aliases and c not in exclude]
    if not candidates:
        if optional:
            return None
        raise SchemaError(f"{path.name}: missing {label}; looked for {list(aliases)}. "
                          "Set columns.<field> in YAML to override.")
    if len(candidates) > 1:
        # Prefer a role-specific name over the generic 'value'/'label'.
        specific = [c for c in candidates if normalize(c) not in ("value", "label")]
        if len(specific) == 1:
            return specific[0]
        raise SchemaError(f"{path.name}: ambiguous {label} columns {candidates}; "
                          "set an explicit YAML column override")
    return candidates[0]


def _study_column(df: pd.DataFrame, override: str | None, path: Path) -> str:
    return str(_column(df, _ID_ALIASES, override, path, "study_id"))


def _text_ids(series: pd.Series, path: Path, field: str) -> pd.Series:
    if series.isna().any():
        raise SchemaError(f"{path.name}: {field} contains missing IDs")
    values = series.astype(str).str.strip()
    if (values == "").any() or values.str.lower().isin(("nan", "none", "null")).any():
        raise SchemaError(f"{path.name}: {field} contains blank/null IDs")
    return values


def _values(series: pd.Series, path: Path, role: str) -> pd.Series:
    raw = series.copy()
    if role in ("accepted60", "expert_label", "predicted_label"):
        raw = raw.astype(str).str.strip().str.lower().replace(
            {"true": "1", "false": "0", "yes": "1", "no": "0"})
    try:
        numeric = pd.to_numeric(raw, errors="raise")
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"{path.name}: {role} must be numeric in [0,1]: {exc}") from exc
    arr = numeric.to_numpy(dtype=float)
    if not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
        raise SchemaError(f"{path.name}: {role} has missing, infinite, or out-of-range values; "
                          "expected [0,1]")
    if role in ("accepted60", "expert_label", "predicted_label"):
        if not np.isin(arr, [0, 1]).all():
            raise SchemaError(f"{path.name}: {role} must be exactly binary 0/1")
        return pd.Series(arr.astype(np.int8), index=series.index)
    return pd.Series(arr, index=series.index)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream))
            if len(header) != len(set(header)):
                raise SchemaError(f"{path.name}: duplicate CSV header names; "
                                  "pandas would silently mangle them")
            df = pd.read_csv(path, dtype=str, low_memory=False)
        except (pd.errors.ParserError, UnicodeError, StopIteration) as exc:
            raise SchemaError(f"Cannot parse {path}: {exc}") from exc
    elif path.suffix.lower() == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise SchemaError(f"Cannot parse {path}: {exc}") from exc
        if isinstance(obj, dict):
            if isinstance(obj.get("studies"), list):
                obj = obj["studies"]
            elif isinstance(obj.get("rows"), list):
                obj = obj["rows"]
            elif all(isinstance(v, dict) for v in obj.values()):
                obj = [dict(study_id=k, **v) for k, v in obj.items()]
        if not isinstance(obj, list) or not all(isinstance(item, dict) for item in obj):
            raise SchemaError(f"{path.name}: JSON must contain a list of study/row objects")
        df = pd.DataFrame(obj)
    else:
        raise SchemaError(f"Unsupported input type: {path.name} (CSV/JSON only)")
    if df.empty or df.columns.duplicated().any():
        raise SchemaError(f"{path.name}: empty table or duplicated header names")
    return df


def _mapping(path: Path, source: str, target: str, shape: str) -> dict[str, str]:
    return {"file": path.name, "source_column": source, "canonical_field": target,
            "shape": shape}


def _finding_column(df: pd.DataFrame, cfg: dict[str, Any], path: Path,
                    sid: str) -> str | None:
    override = cfg["columns"]["finding"]
    if override is not None:
        return _column(df, _FINDING_ALIASES, override, path, "finding", exclude={sid})
    candidates = [c for c in df.columns if normalize(c) in _FINDING_ALIASES and c != sid]
    for candidate in candidates:
        # The generic 'label' often means a binary label, not the finding name.
        if df[candidate].notna().all() and df[candidate].map(
                lambda v: canonical_finding(v) is not None).all():
            return candidate
    return None


def _wide_findings(df: pd.DataFrame, path: Path, cfg: dict[str, Any],
                   role: str, sid: str) -> dict[str, str]:
    overrides = cfg["columns"]["findings"]
    selected: dict[str, str] = {}
    for name, setting in overrides.items():
        col = setting.get(role) if isinstance(setting, dict) else setting
        if col is None:
            continue
        if col not in df:
            raise SchemaError(f"{path.name}: wide column override {name}: {col!r} not present")
        if col == sid:
            raise SchemaError(f"{path.name}: a finding override cannot equal the study ID")
        selected[name] = col
    overridden_names = set(selected)
    for col in df.columns:
        if col == sid or col in selected.values():
            continue
        name = canonical_finding(col, allow_affixes=True)
        if name is None or name in overridden_names:
            continue
        if name in selected:
            prev = selected[name]
            # An expert-wide file may carry both target and prediction columns.
            if role == "expert_label":
                markers = ("expert", "truth", "target", "label")
                prev_marker = any(x in normalize(prev) for x in markers)
                col_marker = any(x in normalize(col) for x in markers)
                if col_marker and not prev_marker:
                    selected[name] = col
                    continue
                if prev_marker and not col_marker:
                    continue
            raise SchemaError(f"{path.name}: ambiguous {name} wide columns "
                              f"{prev!r}, {col!r}; provide columns.findings override")
        # A predicted column must NOT be misread as expert truth.
        if role == "expert_label" and any(t in normalize(col) for t in ("pred", "prediction")):
            continue
        selected[name] = col
    missing = set(FINDINGS) - set(selected)
    if missing:
        raise SchemaError(f"{path.name}: missing 11-finding wide schema: {sorted(missing)}; "
                          "provide all canonical findings or use (study_id, finding, value) long format")
    if len(set(selected.values())) != len(FINDINGS):
        raise SchemaError(f"{path.name}: repeated wide source column in finding map")
    return selected


def _release_file(release: Path, requested: str | None, kind: str,
                  *, optional: bool = False) -> Path | None:
    if requested:
        path = (release / requested).resolve()
        if not path.is_relative_to(release):
            raise SchemaError(f"paths.{kind} must stay within the release directory")
        if not path.is_file():
            raise SchemaError(f"Missing required {kind} file: {path}")
        return path
    if kind == "metadata":
        names = ("metadata", "study_metadata", "studies", "study_info")
    elif kind == "evaluation":
        names = ("evaluation_results", "expert_results", "expert_labels",
                 "evaluation", "expert", "ground_truth")
    else:
        raise SchemaError(f"No configured file for {kind}")
    candidates = [p.resolve() for p in release.iterdir() if p.is_file()
                  and p.suffix.lower() in (".csv", ".json")
                  and p.stem.lower() in names]
    if not candidates and kind == "evaluation" and not optional:
        candidates = [p.resolve() for p in release.iterdir() if p.is_file()
                      and p.suffix.lower() in (".csv", ".json")
                      and ("expert" in p.stem.lower() or "eval" in p.stem.lower())]
    if len(candidates) > 1:
        raise SchemaError(f"Ambiguous {kind} files: {[p.name for p in candidates]}; "
                          f"specify paths.{kind} in YAML")
    if not candidates and not optional:
        raise SchemaError(f"No {kind} CSV/JSON found in {release}; specify paths.{kind}")
    return candidates[0] if candidates else None


def _read_findings(path: Path, cfg: dict[str, Any], role: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    df = _read_table(path)
    sid = _study_column(df, cfg["columns"]["study_id"], path)
    finding = _finding_column(df, cfg, path, sid)
    shape = "long" if finding is not None else "wide"
    mappings = [_mapping(path, sid, "study_id", shape)]
    if finding is not None:
        value = _column(df, _ROLE_ALIASES[role], cfg["columns"][role], path,
                        role, exclude={sid, finding})
        assert value is not None
        mapped = df[finding].map(canonical_finding)
        if mapped.isna().any():
            unknown = df.loc[mapped.isna(), finding].astype(str).unique()[:8].tolist()
            raise SchemaError(f"{path.name}: unknown findings {unknown}; expected exactly 11, "
                              "Synovitis is not included")
        data = pd.DataFrame({"study_id": _text_ids(df[sid], path, "study_id"),
                             "finding": mapped, role: _values(df[value], path, role)})
        mappings += [_mapping(path, finding, "finding", shape),
                     _mapping(path, value, role, shape)]
    else:
        selected = _wide_findings(df, path, cfg, role, sid)
        ids = _text_ids(df[sid], path, "study_id")
        if ids.duplicated().any():
            raise SchemaError(f"{path.name}: duplicate study_id in wide file")
        parts = []
        for name in FINDINGS:
            col = selected[name]
            parts.append(pd.DataFrame({"study_id": ids, "finding": name,
                                       role: _values(df[col], path, role)}))
            mappings.append(_mapping(path, col, f"{role}[{name}]", shape))
        data = pd.concat(parts, ignore_index=True)
    _validate_pairs(data, path, role)
    return data, mappings


def _validate_pairs(df: pd.DataFrame, path: Path, role: str) -> None:
    if df.duplicated(["study_id", "finding"]).any():
        first = df.loc[df.duplicated(["study_id", "finding"]),
                       ["study_id", "finding"]].head(2).to_dict("records")
        raise SchemaError(f"{path.name}: duplicate (study_id, finding) for {role}: {first}")
    counts = df.groupby("study_id", sort=False)["finding"].nunique()
    if not counts.eq(len(FINDINGS)).all():
        bad = counts[counts != len(FINDINGS)].head(3).to_dict()
        raise SchemaError(f"{path.name}: each study needs all 11 findings; "
                          f"observed counts {bad} (Synovitis must not be added)")


def _read_expert(path: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, str]], str]:
    expert, mappings = _read_findings(path, cfg, "expert_label")
    df = _read_table(path)
    sid = _study_column(df, cfg["columns"]["study_id"], path)
    finding = _finding_column(df, cfg, path, sid)
    if finding is not None:
        pred_col = _column(df, _ROLE_ALIASES["predicted_label"],
                           cfg["columns"]["predicted_label"], path,
                           "predicted_label", optional=True, exclude={sid, finding})
        if pred_col is not None:
            pred = pd.DataFrame({"study_id": _text_ids(df[sid], path, "study_id"),
                                 "finding": df[finding].map(canonical_finding),
                                 "released_prediction": _values(
                                     df[pred_col], path, "predicted_label")})
            _validate_pairs(pred, path, "predicted_label")
            expert = expert.merge(pred, on=["study_id", "finding"], validate="one_to_one")
            mappings.append(_mapping(path, pred_col, "released_prediction", "long"))
            return expert, mappings, f"{path.name}:{pred_col}"
    else:
        # Prediction columns must be explicitly marked, not plain wide label columns.
        pred_map: dict[str, str] = {}
        for col in df.columns:
            name = canonical_finding(col, allow_affixes=True)
            if name and any(t in normalize(col) for t in ("pred", "prediction")):
                if name in pred_map:
                    raise SchemaError(f"{path.name}: ambiguous predicted wide columns for {name}")
                pred_map[name] = col
        if pred_map:
            if set(pred_map) != set(FINDINGS):
                raise SchemaError(f"{path.name}: predicted-wide columns must cover all 11 findings")
            ids = _text_ids(df[sid], path, "study_id")
            pred = pd.concat([
                pd.DataFrame({"study_id": ids, "finding": name,
                              "released_prediction": _values(df[pred_map[name]], path,
                                                             "predicted_label")})
                for name in FINDINGS], ignore_index=True)
            expert = expert.merge(pred, on=["study_id", "finding"], validate="one_to_one")
            mappings.extend(_mapping(path, col, f"released_prediction[{name}]", "wide")
                            for name, col in pred_map.items())
            return expert, mappings, f"{path.name}:predicted-wide"
    expert["released_prediction"] = np.nan
    return expert, mappings, "soft_score>=0.5 (no expert prediction column supplied)"


def _read_metadata(path: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    df = _read_table(path)
    sid = _study_column(df, cfg["columns"]["study_id"], path)
    ids = _text_ids(df[sid], path, "study_id")
    if ids.duplicated().any():
        raise SchemaError(f"{path.name}: duplicate study_id in metadata")
    out = pd.DataFrame({"study_id": ids})
    mappings = [_mapping(path, sid, "study_id", "metadata")]
    for role, aliases in _METADATA_ALIASES.items():
        col = _column(df, aliases, None, path, role, optional=True, exclude={sid})
        if col is None:
            continue
        if role == "age":
            out[role] = pd.to_numeric(df[col], errors="coerce")
            if out[role].notna().any() and ((out[role].dropna() < 0) |
                                            (out[role].dropna() > 120)).any():
                raise SchemaError(f"{path.name}: age outside [0,120]")
        else:
            out[role] = df[col].astype("string").str.strip().replace("", pd.NA)
        mappings.append(_mapping(path, col, role, "metadata"))
    return out, mappings


def _notices(release: Path) -> tuple[Path, ...]:
    """Inspect text notices, never recursively traverse image/data directories."""
    keys = ("license", "licence", "notice", "attribution", "citation", "dataset_card")
    parents = [release]
    parents.extend(p for p in release.iterdir() if p.is_dir() and
                   p.name.lower() in ("notices", "licenses", "licences", "docs", "documentation"))
    found = [p for parent in parents for p in parent.iterdir()
             if p.is_file() and p.resolve().is_relative_to(release)
             and p.suffix.lower() in ("", ".txt", ".md")
             and any(k in p.stem.lower() for k in keys)]
    return tuple(sorted(set(found)))


def load_release(release_dir: str | Path, cfg: dict[str, Any],
                 *, print_mapping: bool = True,
                 require_expert: bool = True) -> ReleaseData:
    """Read, join, and validate the release without consulting images or network.

    Parameters
    ----------
    release_dir : str or Path
        User-supplied directory. IDs remain strings to preserve leading zeroes.
    cfg : dict
        Validated configuration.
    print_mapping : bool
        Print explicit source-to-canonical mapping table to stderr.
    require_expert : bool
        True for masks/evaluation; False for prediction-only releases where
        schema, label-free audit and OOF IPCW may still be trained. NEVER
        interpret ``expert_oof.csv`` predictions as expert ground truth.

    Returns
    -------
    ReleaseData
        One row for each of the 11 findings in each study.

    Raises
    ------
    SchemaError
        For missing/ambiguous columns, misaligned pairs, or invalid values.
    """
    release = Path(release_dir).expanduser().resolve()
    if not release.is_dir():
        raise SchemaError(f"Release directory does not exist: {release}")
    paths: dict[str, Path] = {}
    for role in ("soft_labels", "confidence", "accepted60"):
        chosen = _release_file(release, cfg["paths"][role], role)
        assert chosen is not None
        paths[role] = chosen
    for role in ("metadata", "evaluation"):
        chosen = _release_file(release, cfg["paths"][role], role,
                               optional=(role == "metadata" or not require_expert))
        if chosen is not None:
            paths[role] = chosen
    if require_expert and "evaluation" in paths and paths["evaluation"].name == "expert_oof.csv":
        raise SchemaError("expert_oof.csv contains predictions, NOT expert ground truth. "
                          "Supply the separately licensed official expert labels "
                          "as expert_results.csv (all 11 findings for 58 studies), "
                          "or run `python -m safemask_knee train` for label-free IPCW.")
    for name in ("expert_oof.csv", "benchmark_results.json", "FILES.json",
                 "source_provenance.json"):
        context = release / name
        if context.is_file() and context.resolve() not in paths.values():
            paths["context_" + name] = context.resolve()
    if len(set(paths.values())) != len(paths):
        raise SchemaError("Input paths must point to distinct files")
    soft, m1 = _read_findings(paths["soft_labels"], cfg, "soft_score")
    conf, m2 = _read_findings(paths["confidence"], cfg, "confidence")
    accepted, m3 = _read_findings(paths["accepted60"], cfg, "accepted60")
    mapping = m1 + m2 + m3
    ids = soft[["study_id", "finding"]]
    for role, table in (("confidence", conf), ("accepted60", accepted)):
        if set(map(tuple, table[["study_id", "finding"]].to_numpy())) != set(
                map(tuple, ids.to_numpy())):
            raise SchemaError(f"{role} (study_id, finding) keys do not exactly match "
                              "accuracy_soft_labels.csv")
    frame = soft.merge(conf, on=["study_id", "finding"], validate="one_to_one").merge(
        accepted, on=["study_id", "finding"], validate="one_to_one")
    warnings: list[str] = []
    metadata_fields: tuple[str, ...] = ()
    if "metadata" in paths:
        meta, mm = _read_metadata(paths["metadata"], cfg)
        mapping += mm
        metadata_fields = tuple(c for c in meta if c != "study_id")
        unavailable = [name for name in ("age", "sex", "site", "scanner")
                       if name not in meta or meta[name].notna().sum() == 0]
        if unavailable:
            warnings.append("No usable metadata for " + ", ".join(unavailable) +
                            "; those fairness strata will be skipped")
        missing_studies = len(set(frame.study_id) - set(meta.study_id))
        if missing_studies:
            warnings.append(f"Metadata missing for {missing_studies} study/studies; "
                            "their missing strata are skipped")
        frame = frame.merge(meta, on="study_id", how="left", validate="many_to_one")
    else:
        warnings.append("No metadata supplied: age/sex/site/scanner fairness strata skipped")
    soft_pred = (frame.soft_score >= 0.5).astype(np.int8)
    if "evaluation" in paths:
        expert, me, prediction_source = _read_expert(paths["evaluation"], cfg)
        mapping += me
        if not set(expert.study_id).issubset(set(frame.study_id)):
            raise SchemaError("Expert evaluation contains study IDs absent from the three release CSVs")
        frame = frame.merge(expert, on=["study_id", "finding"], how="left", validate="one_to_one")
        if len(expert) < 11 or expert.study_id.nunique() < 2:
            raise SchemaError("Expert evaluation needs at least two complete 11-finding studies")
        frame["expert_label"] = frame["expert_label"].astype("Int64")
        mismatches = int(((frame["released_prediction"].notna()) &
                          (frame["released_prediction"] != soft_pred)).sum())
        if mismatches:
            warnings.append(f"Expert released predictions differ from soft_score>=0.5 on "
                            f"{mismatches} pairs; baseline uses released predictions, "
                            "new masks use soft_score>=0.5")
        frame["released_prediction"] = frame["released_prediction"].fillna(soft_pred).astype(np.int8)
        if expert.study_id.nunique() != 58 or len(expert) != 638:
            if require_expert:
                raise SchemaError(f"Full analysis requires the 58 complete expert studies "
                                  f"(638 pairs), not {expert.study_id.nunique()} studies "
                                  f"and {len(expert)} pairs")
            warnings.append(f"Expert panel has {expert.study_id.nunique()} studies and "
                            f"{len(expert)} pairs, not the described 58/638; inspect source semantics")
        if "official_expert_evaluation_row" in frame:
            marked = set(frame.loc[frame.official_expert_evaluation_row.astype(str).eq("1"),
                                   "study_id"])
            if marked != set(expert.study_id):
                raise SchemaError("Expert results do not match metadata's 58 marked "
                                  "development study IDs; check official answer-key join")
        n_expert_studies, n_expert_pairs = expert.study_id.nunique(), len(expert)
    else:
        if require_expert:
            raise SchemaError("Expert truth is required for masks/evaluation. "
                              "Prediction-only files and aggregate benchmark accuracy "
                              "cannot substitute for expert labels.")
        frame["expert_label"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        frame["released_prediction"] = soft_pred.to_numpy()
        prediction_source = "soft_score>=0.5; expert truth NOT PROVIDED"
        mismatches = 0
        n_expert_studies, n_expert_pairs = 0, 0
        context_note = ("expert_oof.csv is predictions only; "
                        if "context_expert_oof.csv" in paths else "")
        warnings.append("EXPERT TRUTH ABSENT: no outcome-associated MNAR verdict, "
                        "accepted-label accuracy, conformal cutoffs, or risk guarantees "
                        "can be estimated. " + context_note +
                        "use separately licensed official expert labels for full run.")
    frame["finding"] = pd.Categorical(frame["finding"], categories=FINDINGS, ordered=True)
    frame = frame.sort_values(["study_id", "finding"], kind="stable").reset_index(drop=True)
    result = ReleaseData(frame, mapping, paths, _notices(release), tuple(warnings),
                         metadata_fields, release, prediction_source)
    LOG.debug("Validated column mapping: %s", mapping)
    LOG.debug("Expert prediction source: %s; %d score-sign mismatches",
              prediction_source, mismatches)
    if print_mapping:
        print_mapping_table(result)
    for warning in warnings:
        LOG.warning(warning)
    LOG.info("Validated %s studies, %s pairs, %s studies with expert truth (%s expert pairs)",
             frame.study_id.nunique(), len(frame), n_expert_studies, n_expert_pairs)
    return result


def print_mapping_table(data: ReleaseData) -> None:
    """Print explicit source-to-canonical mapping even when names already match."""
    print("COLUMN MAPPING (source file | source column -> canonical field | format)",
          file=sys.stderr)
    for item in data.mapping:
        print(f"  {item['file']} | {item['source_column']} -> "
              f"{item['canonical_field']} | {item['shape']}", file=sys.stderr)


def group_keys(frame: pd.DataFrame) -> pd.Series:
    """Cluster studies sharing a patient ID OR an identical normalized report.

    If neither metadata key exists, a study is its own group. The report hash
    is only a duplicate-report guard, not a substitute for unknown patient
    identity; no claim of independently sampled patients is made.
    """
    fields = [key for key in ("patient_id", "normalized_report_group") if key in frame]
    if not fields:
        return pd.Series("study:" + frame.study_id.astype(str), index=frame.index)
    for field in fields:
        if frame.groupby("study_id", observed=True)[field].nunique(dropna=True).gt(1).any():
            raise SchemaError(f"A study_id maps to multiple {field} values")
    unique = frame[["study_id", *fields]].drop_duplicates("study_id")
    ids = unique.study_id.astype(str).tolist()
    parent = {study: study for study in ids}

    def root(study: str) -> str:
        while parent[study] != study:
            parent[study] = parent[parent[study]]
            study = parent[study]
        return study

    for field in fields:
        first: dict[str, str] = {}
        for study, raw in unique[["study_id", field]].itertuples(index=False, name=None):
            if pd.isna(raw) or not str(raw).strip():
                continue
            value = str(raw).strip()
            if value not in first:
                first[value] = str(study)
            else:
                left, right = root(str(study)), root(first[value])
                if left != right:
                    parent[max(left, right)] = min(left, right)
    identity = {study: "cluster:" + root(study) for study in ids}
    return frame.study_id.astype(str).map(identity)


def grouped_study_folds(frame: pd.DataFrame, n_splits: int,
                        seed: int) -> Iterator[tuple[set[str], set[str]]]:
    """Yield study-level train/test IDs, keeping repeated patient IDs together."""
    unique = frame[["study_id"]].copy()
    unique["_group"] = group_keys(frame).to_numpy()
    unique = unique.drop_duplicates("study_id").sort_values("study_id")
    clusters = np.sort(unique._group.unique().astype(str))
    if len(clusters) < n_splits:
        raise SchemaError(f"Need >= {n_splits} distinct patient/study groups; "
                          f"have {len(clusters)}")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(clusters):
        train_groups, test_groups = set(clusters[train]), set(clusters[test])
        assert train_groups.isdisjoint(test_groups)
        train_studies = set(unique.loc[unique._group.isin(train_groups), "study_id"])
        test_studies = set(unique.loc[unique._group.isin(test_groups), "study_id"])
        assert train_studies.isdisjoint(test_studies)
        yield train_studies, test_studies


def expert_fold_map(frame: pd.DataFrame, n_splits: int, seed: int) -> dict[str, int]:
    """Assign each expert study to exactly one fixed outer study-grouped fold."""
    expert = frame.loc[frame.expert_label.notna()]
    mapping: dict[str, int] = {}
    for fold, (_train, test) in enumerate(grouped_study_folds(expert, n_splits, seed)):
        for study in test:
            if study in mapping:
                raise AssertionError(f"Study {study} in multiple outer folds")
            mapping[study] = fold
    if set(mapping) != set(expert.study_id):
        raise AssertionError("Not all expert studies received an outer fold")
    return mapping


def sha256_file(path: Path) -> str:
    """Hash an input without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(data: ReleaseData, outputs_root: Path,
                   data_dir: Path | None = None) -> dict[str, Any]:
    """Write identical hash-only input manifests in outputs/ and data/."""
    files = sorted(set(data.files.values()) | set(data.notices))
    entries = [{"path": str(path.relative_to(data.release_dir)),
                "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
               for path in files]
    manifest = {"description": "SHA-256 of local inputs; no source data copied",
                "disclaimer": DISCLAIMER, "files": entries}
    outputs_root.mkdir(parents=True, exist_ok=True)
    (outputs_root / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target = data_dir or (Path.cwd() / "data")
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_schema(data: ReleaseData, path: Path) -> dict[str, Any]:
    """Record auto-detection, the exact long table fields, and validation counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = data.frame
    result = {
        "disclaimer": DISCLAIMER, "source_column_mapping": data.mapping,
        "long_format_fields": ["study_id", "finding", "soft_score", "confidence",
                               "accepted60", *data.metadata_fields, "expert_label",
                               "released_prediction"],
        "findings": list(FINDINGS), "n_studies": int(frame.study_id.nunique()),
        "n_pairs": len(frame), "n_expert_studies": int(
            frame.loc[frame.expert_label.notna(), "study_id"].nunique()),
        "n_evaluated_pairs": int(frame.expert_label.notna().sum()),
        "expert_truth_available": bool(frame.expert_label.notna().any()),
        "n_source_development_studies": int(frame.loc[
            frame.official_expert_evaluation_row.astype(str).eq("1"), "study_id"].nunique())
        if "official_expert_evaluation_row" in frame else None,
        "prediction_only_expert_oof_present": "context_expert_oof.csv" in data.files,
        "prediction_source": data.prediction_source,
        "metadata_fields": list(data.metadata_fields), "warnings": list(data.warnings),
    }
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def write_attribution(data: ReleaseData, path: Path) -> None:
    """Copy supplied licence/attribution notices verbatim into a local document."""
    lines = ["# Attribution for this run", "",
             "Derived from the user-supplied RSNA knee MRI competition label release. "
             "The competition data and third-party labels remain under their own terms. "
             "The MIT license applies ONLY to SafeMask-Knee code; no images or "
             "original release tables are redistributed.", "", DISCLAIMER, "",
             "## Notices copied from the release", ""]
    if not data.notices:
        lines += ["No licence/attribution notices found in the supplied release. "
                  "Check the source terms before using any release data.", ""]
    for notice in data.notices:
        content = notice.read_text(encoding="utf-8")
        longest = max((len(match.group()) for match in re.finditer(r"`+", content)),
                      default=0)
        fence = "`" * max(3, longest + 1)
        lines += [f"### `{notice.relative_to(data.release_dir)}`", "",
                  f"{fence}text", content, fence, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
