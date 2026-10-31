"""Immutable labels, output schemas, and safety language."""

from __future__ import annotations

import re
from typing import Final

FINDINGS: Final[tuple[str, ...]] = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Baker's cyst", "Contusion",
    "Fracture",
)

TIERS: Final[dict[int, tuple[str, ...]]] = {
    1: ("ACL", "MCL", "Contusion", "Fracture"),
    2: ("Medial Meniscus", "Lateral Meniscus", "Effusion"),
    3: ("Medial OA", "Lateral OA", "PF OA", "Baker's cyst"),
}

DISCLAIMER: Final[str] = (
    "For research only; not for clinical use; requires expert radiologist "
    "oversight; statistical guarantees are not clinical guarantees."
)

WEIGHT_COLUMNS: Final[tuple[str, ...]] = (
    "study_id", "finding", "soft_score", "confidence", "accepted60",
    "expert_label", "propensity_accepted", "weight_raw", "weight_clipped",
    "weight_final", "fold", "notes",
)
MASK_COLUMNS: Final[tuple[str, ...]] = (
    "study_id", "finding", "tier", "risk_bound", "conformal_threshold",
    "soft_score", "confidence", "accepted60", "accepted_clinical",
    "reason", "fold",
)


def normalize(text: object) -> str:
    """Return an alphanumeric, lower-case token used only for schema lookup."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "ACL": ("acl", "anteriorcruciateligament"),
    "MCL": ("mcl", "medialcollateralligament"),
    "Medial Meniscus": ("medialmeniscus", "medmeniscus", "mm"),
    "Lateral Meniscus": ("lateralmeniscus", "latmeniscus", "lm"),
    "Medial OA": ("medialoa", "medialosteoarthritis", "medoa"),
    "Lateral OA": ("lateraloa", "lateralosteoarthritis", "latoa"),
    "PF OA": ("pfoa", "patellofemoraloa", "patellofemoralosteoarthritis"),
    "Effusion": ("effusion", "jointeffusion"),
    "Baker's cyst": ("bakerscyst", "bakercyst", "poplitealcyst", "bakers"),
    "Contusion": ("contusion", "bonecontusion", "bonebruise"),
    "Fracture": ("fracture", "bonefracture"),
}

_FINDING_LOOKUP: Final[dict[str, str]] = {
    alias: finding for finding, aliases in _ALIASES.items() for alias in aliases
}


def canonical_finding(text: object, *, allow_affixes: bool = False) -> str | None:
    """Map a recognized release finding to a canonical name, never guessing labels."""
    token = normalize(text)
    if token in _FINDING_LOOKUP:
        return _FINDING_LOOKUP[token]
    if allow_affixes:
        # Only unambiguous, documented technical prefixes/suffixes are removed.
        affixes = ("accuracy", "softlabels", "softscore", "soft", "score",
                   "confidence", "conf", "accepted60", "accepted", "mask",
                   "expertlabel", "groundtruth", "truth", "target",
                   "prediction", "predictedlabel", "predicted", "pred", "label")
        for affix in affixes:
            for candidate in (token.removeprefix(affix), token.removesuffix(affix)):
                if candidate != token and candidate in _FINDING_LOOKUP:
                    return _FINDING_LOOKUP[candidate]
    return None
