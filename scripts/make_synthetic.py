#!/usr/bin/env python3
"""Deterministic *invented* RSNA-shaped fixture; contains no RSNA patient data.

With defaults, the 58-study evaluation panel has exactly 638 findings, 383
accepted pairs, and 11 deliberately injected accepted errors (372/383 correct).
A latent positive-dependent selection step creates known outcome-associated
missingness so the audit and weighting code have a nontrivial test target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from safemask_knee.constants import FINDINGS

_PREVALENCE = np.array([.29, .23, .35, .31, .34, .29, .28, .36, .28, .26, .23])
_ERASURES = {"ACL", "Medial Meniscus", "Lateral Meniscus", "Effusion",
             "Contusion", "PF OA"}


def generate_release(out: Path, studies: int = 160,
                     expert_studies: int = 58,
                     seed: int = 317) -> dict[str, int]:
    """Write a tiny complete 11-finding release and simulated expert results.

    Parameters
    ----------
    out : Path
        Dedicated *synthetic* output directory, never the real release.
    studies : int
        Number of completely synthetic studies, at least expert_studies + 5.
    expert_studies : int
        Simulated expert subset; default exactly 58.
    seed : int
        NumPy PCG64 seed.

    Returns
    -------
    dict
        Counts of generated studies/pairs, expert selections and mistakes.
    """
    if studies < expert_studies + 5 or expert_studies < 10:
        raise ValueError("Need >=10 expert studies and >=5 non-expert studies")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = np.array([f"SYN-{i:05d}" for i in range(studies)])
    age = np.clip(np.rint(rng.normal(51, 15, studies)), 18, 86).astype(int)
    sex = rng.choice(["F", "M"], studies, p=[.51, .49])
    site = rng.choice(["North", "South", "UndersampledSite"], studies,
                      p=[.42, .42, .16])
    scanner = rng.choice(["1.5T", "3T"], studies, p=[.5, .5])
    meta = pd.DataFrame({"exam_id": ids, "patient_id": [f"PX-{i:05d}"
                                                          for i in range(studies)],
                         "age": age, "sex": sex, "site": site, "scanner": scanner})
    rows = studies * len(FINDINGS)
    latent = rng.normal(0, 1, studies)
    score = np.zeros((studies, len(FINDINGS)))
    confidence = np.zeros_like(score)
    truth = np.zeros_like(score, dtype=int)
    subtle = np.zeros_like(score, dtype=bool)
    for i in range(studies):
        for j, finding in enumerate(FINDINGS):
            prior = _PREVALENCE[j]
            if "OA" in finding:
                prior = np.clip(prior + (age[i] - 50) / 250, .10, .65)
            chance = np.clip(prior + .08 * latent[i], .05, .75)
            truth[i, j] = int(rng.random() < chance)
            subtle[i, j] = bool(truth[i, j] == 1 and rng.random() < .17)
            if truth[i, j]:
                score[i, j] = (rng.uniform(.52, .70) if subtle[i, j] else
                               rng.uniform(.73, .98))
            else:
                score[i, j] = (rng.uniform(.32, .48) if rng.random() < .08 else
                               rng.uniform(.02, .28))
            confidence[i, j] = np.clip(.57 + .36 * (2 * abs(score[i, j] - .5)) +
                                       rng.normal(0, .07), .12, .99)
    flatten = score.reshape(-1)
    conf_flat = confidence.reshape(-1)
    latent_y = truth.reshape(-1)
    site_flat = np.repeat(site, len(FINDINGS))
    finding_flat = np.tile(np.array(FINDINGS, dtype=object), studies)
    # Source selection is known to be outcome-associated in this fixture;
    # the real package never sees non-expert true outcomes.
    accept_rank = (.72 * conf_flat + .15 * (1 - flatten) +
                   .11 * (latent_y == 0) -
                   .24 * (np.isin(finding_flat, list(_ERASURES)) & (latent_y == 1)) -
                   .11 * (site_flat == "UndersampledSite") +
                   rng.normal(0, .17, rows))
    chosen = rng.choice(studies, size=expert_studies, replace=False)
    panel_rows = np.isin(np.repeat(np.arange(studies), len(FINDINGS)), chosen)
    accepted = np.zeros(rows, dtype=int)
    n_panel = int(panel_rows.sum())
    accept_expert = 383 if expert_studies == 58 else round(.60 * n_panel)
    for selector, n_select in ((panel_rows, accept_expert),
                               (~panel_rows, round(.60 * (rows - n_panel)))):
        indices = np.flatnonzero(selector)
        ordered = indices[np.argsort(-accept_rank[indices], kind="stable")]
        accepted[ordered[:n_select]] = 1
    # Define an expert simulation aligned with score sign, then inject exactly
    # 11 mistakes among accepted. The audit's hard-coded counts are *not* used
    # anywhere in fitting or evaluation.
    prediction = (flatten >= .5).astype(int)
    expert_truth = prediction.copy()
    injected: list[int] = []
    if expert_studies == 58:
        fn_targets = ["ACL", "ACL", "Medial Meniscus", "Medial Meniscus",
                      "Lateral Meniscus", "Effusion", "Contusion", "PF OA"]
        fp_targets = ["MCL", "Lateral OA", "Baker's cyst"]
        for target, predicted in ([(name, 0) for name in fn_targets] +
                                  [(name, 1) for name in fp_targets]):
            candidates = np.flatnonzero(panel_rows & (accepted == 1) &
                                        (finding_flat == target) &
                                        (prediction == predicted))
            candidates = [idx for idx in candidates if idx not in injected]
            if not candidates:
                raise RuntimeError(f"Synthetic seed lacks accepted {target} candidate; "
                                   "select a different seed")
            pick = min(candidates, key=lambda idx: abs(flatten[idx] - .5))
            expert_truth[pick] = 1 - prediction[pick]
            injected.append(pick)
    for stem, array, mode in (
            ("accuracy_soft_labels.csv", score, "soft"),
            ("accuracy_confidence.csv", confidence, "conf"),
            ("accuracy_accepted60.csv", accepted.reshape(score.shape), "accepted")):
        header = "study_id" if mode == "conf" else "exam_id"
        df = pd.DataFrame({header: ids})
        for j, finding in enumerate(FINDINGS):
            name = (f"{finding}_confidence" if mode == "conf" else
                    f"accepted60_{finding}" if mode == "accepted" else finding)
            df[name] = array[:, j]
        if mode == "conf":
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        df.to_csv(out / stem, index=False)
    meta.to_csv(out / "study_metadata.csv", index=False)
    expert_indices = np.flatnonzero(panel_rows)
    evaluation = pd.DataFrame({"exam_id": np.repeat(ids, len(FINDINGS))[expert_indices],
                               "finding": finding_flat[expert_indices],
                               "expert_label": expert_truth[expert_indices],
                               "predicted_label": prediction[expert_indices]})
    evaluation.sample(frac=1, random_state=seed + 1).to_csv(
        out / "expert_results.csv", index=False)
    (out / "NOTICE.txt").write_text(
        "Synthetic, invented knee-label fixture, CC0. This test input is NOT "
        "RSNA competition data. No actual patient, MRI image, expert result "
        "or third-party release is bundled.\n", encoding="utf-8")
    return {"studies": studies, "pairs": rows, "expert_studies": expert_studies,
            "expert_pairs": n_panel, "expert_accepted": accept_expert,
            "accepted_errors": len(injected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--studies", default=160, type=int)
    parser.add_argument("--expert-studies", default=58, type=int)
    parser.add_argument("--seed", default=317, type=int)
    args = parser.parse_args()
    print(generate_release(args.out, args.studies, args.expert_studies, args.seed))


if __name__ == "__main__":
    main()
