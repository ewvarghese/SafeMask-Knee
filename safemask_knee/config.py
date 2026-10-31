"""Configuration parsing and validation (no implicit network or remote defaults)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .constants import FINDINGS, TIERS

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {"soft_labels": "accuracy_soft_labels.csv",
              "confidence": "accuracy_confidence.csv",
              "accepted60": "accuracy_accepted60.csv",
              "metadata": None, "evaluation": None, "output_root": "outputs"},
    "columns": {"study_id": None, "finding": None, "soft_score": None,
                "confidence": None, "accepted60": None, "expert_label": None,
                "predicted_label": None, "findings": {}},
    "seed": 20261101, "outer_folds": 5, "propensity_folds": 5,
    "platt_calibration_fraction": 0.2, "propensity_floor": 0.02,
    "weight_min": 0.1, "weight_max": 10.0, "bootstrap_repetitions": 1000,
    "proxy_coverage": 0.6, "coverage_target": 0.6,
    "accepted_accuracy_target": 0.97,
    "tiers": {1: {"findings": list(TIERS[1]), "risk_bound": 0.01,
                  "false_positive_bound": 0.10},
              2: {"findings": list(TIERS[2]), "risk_bound": 0.12,
                  "false_positive_bound": 0.12},
              3: {"findings": list(TIERS[3]), "risk_bound": 0.20,
                  "false_positive_bound": 0.20}},
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and strictly validate a config.

    Parameters
    ----------
    path : str or Path, optional
        Local YAML file. With no file, use immutable built-in defaults.

    Returns
    -------
    dict
        Independent, validated config; global defaults are never mutated.
    """
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            custom = yaml.safe_load(handle)
        if not isinstance(custom, dict):
            raise ValueError("Config must be a YAML mapping")
        _merge(cfg, custom, "config")
    for key in ("seed", "outer_folds", "propensity_folds", "bootstrap_repetitions"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int):
            raise ValueError(f"{key} must be an integer")
    if cfg["outer_folds"] < 2 or cfg["propensity_folds"] < 2:
        raise ValueError("outer_folds and propensity_folds must be >= 2")
    if cfg["bootstrap_repetitions"] < 20:
        raise ValueError("bootstrap_repetitions must be >= 20")
    for key in ("platt_calibration_fraction", "propensity_floor", "weight_min",
                "weight_max", "proxy_coverage", "coverage_target",
                "accepted_accuracy_target"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be numeric")
        cfg[key] = float(value)
    if not 0 < cfg["platt_calibration_fraction"] < 0.5:
        raise ValueError("platt_calibration_fraction must be in (0, 0.5)")
    if not 0 < cfg["propensity_floor"] < 0.5:
        raise ValueError("propensity_floor must be in (0, 0.5)")
    if not 0 < cfg["weight_min"] <= 1 <= cfg["weight_max"]:
        raise ValueError("weight_min <= 1 <= weight_max and both > 0 are required")
    for key in ("proxy_coverage", "coverage_target", "accepted_accuracy_target"):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if not isinstance(cfg["columns"]["findings"], dict):
        raise ValueError("columns.findings must map canonical finding names to source names")
    if set(cfg["columns"]["findings"]) - set(FINDINGS):
        raise ValueError("columns.findings has unknown canonical finding(s)")
    for name, override in cfg["columns"]["findings"].items():
        if isinstance(override, dict):
            if not override or set(override) - {"soft_score", "confidence", "accepted60",
                                                "expert_label"}:
                raise ValueError(f"columns.findings.{name} has invalid role(s)")
            if not all(isinstance(value, str) and value for value in override.values()):
                raise ValueError(f"columns.findings.{name} must specify source column names")
        elif not isinstance(override, str) or not override:
            raise ValueError(f"columns.findings.{name} must be a name or role->name map")
    for key, value in cfg["paths"].items():
        if value is not None and not isinstance(value, str):
            raise ValueError(f"paths.{key} must be a path string or null")
    for key, value in cfg["columns"].items():
        if key != "findings" and value is not None and not isinstance(value, str):
            raise ValueError(f"columns.{key} must be a column name string or null")
    if set(cfg["tiers"]) != {1, 2, 3}:
        raise ValueError("tiers must contain exactly integer tiers 1, 2, 3")
    listed = []
    for tier, spec in cfg["tiers"].items():
        if set(spec) != {"findings", "risk_bound", "false_positive_bound"}:
            raise ValueError(f"tiers.{tier} requires findings, risk_bound, false_positive_bound")
        if not isinstance(spec["findings"], list):
            raise ValueError(f"tiers.{tier}.findings must be a list")
        listed.extend(spec["findings"])
        for risk_key in ("risk_bound", "false_positive_bound"):
            value = spec[risk_key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 < value < 1:
                raise ValueError(f"tiers.{tier}.{risk_key} must be in (0,1)")
            spec[risk_key] = float(value)
    if sorted(listed) != sorted(FINDINGS):
        raise ValueError("Each of the 11 findings (no Synovitis) must occur exactly once in tiers")
    return cfg


def _merge(base: dict[str, Any], custom: dict[str, Any], location: str) -> None:
    for key, value in custom.items():
        if key not in base:
            raise ValueError(f"Unknown {location} key: {key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            # The findings override map is intentionally open to canonical names.
            if location == "config.columns" and key == "findings":
                base[key].update(value)
            else:
                _merge(base[key], value, f"{location}.{key}")
        else:
            base[key] = value
