#!/usr/bin/env python3
"""Offline adapter for a separately licensed official train.csv (no downloads).

Writes ONLY the 58 user-authorized expert studies to an ignored local release
file. The source competition CSV remains under its own terms and is never
copied or downloaded by SafeMask-Knee. Do NOT run this on expert_oof.csv:
that file has predictions, not ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safemask_knee.constants import DISCLAIMER, FINDINGS, canonical_finding, normalize


def prepare_expert(official_train: Path, release: Path, out: Path) -> dict[str, object]:
    """Filter actual expert labels to the source-marked 58-study development panel.

    Parameters
    ----------
    official_train : Path
        Locally held, separately licensed official competition training CSV.
    release : Path
        Directory containing `study_metadata.csv` and accuracy soft scores.
    out : Path
        Local, Git-ignored expert_results.csv inside the release directory.

    Raises
    ------
    ValueError
        If truth is incomplete/ambiguous, source IDs mismatch, or predictions
        are substituted for official labels.
    """
    official_train, release, out = (Path(official_train).expanduser().resolve(),
                                    Path(release).expanduser().resolve(),
                                    Path(out).expanduser().resolve())
    if not official_train.is_file():
        raise FileNotFoundError(f"Official licensed train.csv not found: {official_train}")
    if not out.is_relative_to(release):
        raise ValueError("The expert-output path must remain inside the ignored release directory")
    if official_train.name in ("expert_oof.csv", "accuracy_soft_labels.csv"):
        raise ValueError("Predictions are not an official expert answer key")
    meta = pd.read_csv(release / "study_metadata.csv", dtype=str)
    for field in ("StudyInstanceUID", "official_expert_evaluation_row"):
        if field not in meta:
            raise ValueError(f"Study metadata lacks {field}")
    panel = meta.loc[meta.official_expert_evaluation_row.eq("1"), "StudyInstanceUID"]
    if len(panel) != 58 or panel.duplicated().any():
        raise ValueError("Expected exactly 58 distinct source-marked expert studies")
    official = pd.read_csv(official_train, dtype=str)
    candidates = [c for c in official if normalize(c) in ("studyinstanceuid", "studyid", "examid")]
    if len(candidates) != 1:
        raise ValueError("Official CSV needs one unambiguous study-ID column")
    sid = candidates[0]
    if official[sid].duplicated().any():
        raise ValueError("Official CSV has duplicate study IDs")
    columns: dict[str, str] = {}
    for col in official:
        if col == sid:
            continue
        name = canonical_finding(col)
        if name:
            if name in columns:
                raise ValueError(f"Official CSV has ambiguous columns for {name}")
            columns[name] = col
    if set(columns) != set(FINDINGS):
        raise ValueError("Official CSV must provide exactly all 11 requested findings "
                         f"(Synovitis is excluded); missing: {sorted(set(FINDINGS)-set(columns))}")
    subset = official.loc[official[sid].isin(panel)].copy()
    if set(subset[sid]) != set(panel):
        raise ValueError("Official competition CSV does not contain all 58 source expert IDs")
    soft = pd.read_csv(release / "accuracy_soft_labels.csv", dtype=str)
    if "StudyInstanceUID" not in soft:
        raise ValueError("Release accuracy scores lack StudyInstanceUID")
    soft = soft.set_index("StudyInstanceUID")
    result: list[pd.DataFrame] = []
    for finding in FINDINGS:
        label = pd.to_numeric(subset[columns[finding]], errors="coerce")
        if label.isna().any() or not label.isin([0, 1]).all():
            raise ValueError(f"Official expert label for {finding} must be complete binary 0/1")
        source_col = next((col for col in soft if canonical_finding(col) == finding), None)
        if source_col is None:
            raise ValueError(f"Missing release soft score for {finding}")
        score = pd.to_numeric(soft.loc[subset[sid].tolist(), source_col], errors="raise")
        if not np.isfinite(score.to_numpy()).all():
            raise ValueError("Non-finite release soft scores")
        result.append(pd.DataFrame({"StudyInstanceUID": subset[sid].to_numpy(),
                                    "finding": finding,
                                    "expert_label": label.to_numpy(dtype=int),
                                    "predicted_label": (score.to_numpy(dtype=float) >= .5).astype(int)}))
    table = pd.concat(result, ignore_index=True).sort_values(
        ["StudyInstanceUID", "finding"], kind="stable")
    if len(table) != 638 or table.duplicated(["StudyInstanceUID", "finding"]).any():
        raise ValueError("Incomplete or repeated expert study/finding rows")
    accepted = pd.read_csv(release / "accuracy_accepted60.csv", dtype=str).set_index(
        "StudyInstanceUID")
    accepted_pairs = 0
    correct = 0
    for row in table.itertuples(index=False):
        col = next(c for c in accepted if canonical_finding(c) == row.finding)
        if float(accepted.at[row.StudyInstanceUID, col]) == 1.0:
            accepted_pairs += 1
            correct += int(row.expert_label == row.predicted_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return {"path": str(out), "expert_studies": 58, "expert_pairs": 638,
            "accepted_pairs": accepted_pairs, "correct_accepted": correct,
            "source_count_sanity_match": bool(accepted_pairs == 383 and correct == 372),
            "disclaimer": DISCLAIMER}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ + "\n" + DISCLAIMER)
    parser.add_argument("--official-train", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_expert(args.official_train, args.release, args.out),
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
