"""Stress-test OOF IPCW floors and caps without refitting or inventing truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .constants import DISCLAIMER, FINDINGS
from .weights import _bounded_mean_one


def _grid(cfg: dict[str, Any], floors: Iterable[float] | None,
          caps: Iterable[float] | None) -> tuple[list[float], list[float]]:
    base_floor = float(cfg["propensity_floor"])
    floor_values = ([base_floor] + [x for x in (.05, .10) if x >= base_floor]
                    if floors is None else [base_floor, *floors])
    cap_values = ([3., 5., float(cfg["weight_max"])] if caps is None else
                  [float(cfg["weight_max"]), *caps])
    floor_values = sorted(set(float(x) for x in floor_values))
    cap_values = sorted(set(float(x) for x in cap_values))
    if not floor_values or any(not np.isfinite(x) or x < base_floor or x >= .5
                               for x in floor_values):
        raise ValueError("Sensitivity floors must be finite and >= the fitted "
                         "propensity floor and < 0.5; lower values cannot undo clipping")
    if not cap_values or any(not np.isfinite(x) or x < 1 or x <= cfg["weight_min"]
                             for x in cap_values):
        raise ValueError("Sensitivity caps must be finite and >=1 and > weight_min")
    if len(floor_values) * len(cap_values) > 100:
        raise ValueError("Sensitivity grid exceeds 100 variants")
    return floor_values, cap_values


def sensitivity_tables(weights: pd.DataFrame, cfg: dict[str, Any], *,
                       floors: Iterable[float] | None = None,
                       caps: Iterable[float] | None = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reweight accepted pairs using stored grouped-OOF propensities only.

    Reconstruct the outer-training-fold acceptance prior as
    weight_raw * (clipped OOF propensity). The stored OOF probability was
    already clipped at the configured floor; smaller hypothetical floors
    cannot be recovered and are explicitly rejected.
    """
    required = {"finding", "accepted60", "weight_raw", "propensity_accepted"}
    if not required.issubset(weights):
        raise ValueError(f"Weights missing: {sorted(required - set(weights))}")
    accepted = weights.loc[weights.accepted60.eq(1)]
    if accepted.empty or set(accepted.finding) != set(FINDINGS):
        raise ValueError("Sensitivity needs accepted rows for all 11 findings")
    e = accepted.propensity_accepted.to_numpy(dtype=float)
    raw = accepted.weight_raw.to_numpy(dtype=float)
    if not np.isfinite(e).all() or not np.isfinite(raw).all() or (
            (e <= 0) | (e >= 1) | (raw <= 0)).any():
        raise ValueError("Invalid OOF propensity or raw weights")
    prior = raw * e
    floor_values, cap_values = _grid(cfg, floors, caps)
    finding = accepted.finding.astype(str).to_numpy()
    by_finding: list[dict[str, Any]] = []
    overview: list[dict[str, Any]] = []
    base = float(cfg["propensity_floor"])
    original_cap = float(cfg["weight_max"])
    for floor in floor_values:
        propensity = np.clip(e, floor, 1 - floor)
        for cap in cap_values:
            all_final = np.empty(len(accepted), dtype=float)
            for name in FINDINGS:
                idx = finding == name
                candidates = np.clip(prior[idx] / propensity[idx],
                                     cfg["weight_min"], cap)
                final = _bounded_mean_one(candidates, cfg["weight_min"], cap)
                all_final[idx] = final
                by_finding.append({
                    "propensity_floor": floor, "weight_cap": cap,
                    "finding": name, "n_accepted": int(idx.sum()),
                    "effective_sample_size": float(final.sum() ** 2 / (final ** 2).sum()),
                    "weight_p95": float(np.quantile(final, .95)),
                    "weight_max": float(final.max()),
                    "fraction_at_cap": float(np.mean(final >= cap - 1e-9)),
                    "fraction_propensity_at_floor": float(np.mean(e[idx] <= floor + 1e-12)),
                    "mean_weight": float(final.mean()),
                    "is_configured_baseline": bool(abs(floor - base) < 1e-12 and
                                                   abs(cap - original_cap) < 1e-12),
                    "disclaimer": DISCLAIMER,
                })
            overview.append({
                "propensity_floor": floor, "weight_cap": cap,
                "n_accepted": len(accepted),
                "effective_sample_size": float(all_final.sum() ** 2 /
                                               (all_final ** 2).sum()),
                "weight_p99": float(np.quantile(all_final, .99)),
                "weight_max": float(all_final.max()),
                "fraction_at_cap": float(np.mean(all_final >= cap - 1e-9)),
                "is_configured_baseline": bool(abs(floor - base) < 1e-12 and
                                               abs(cap - original_cap) < 1e-12),
                "disclaimer": DISCLAIMER,
            })
    return pd.DataFrame(by_finding), pd.DataFrame(overview)


def write_sensitivity(weights: pd.DataFrame, out: Path, cfg: dict[str, Any], *,
                      floors: Iterable[float] | None = None,
                      caps: Iterable[float] | None = None) -> dict[str, Any]:
    """Write sensitivity tables and a label-free interpretation, not a risk claim."""
    table, overview = sensitivity_tables(weights, cfg, floors=floors, caps=caps)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "sensitivity_by_finding.csv", index=False, float_format="%.9g")
    overview.to_csv(out / "sensitivity_overview.csv", index=False, float_format="%.9g")
    baseline = overview.loc[overview.is_configured_baseline]
    if len(baseline) != 1:
        raise ValueError("Sensitivity grid must include the configured baseline "
                         "propensity floor and weight cap")
    base_ess = float(baseline.effective_sample_size.iloc[0])
    record = {"status": "label_free_weight_stability_only",
              "disclaimer": DISCLAIMER, "n_variants": len(overview),
              "accepted_pairs": int(baseline.n_accepted.iloc[0]),
              "configured_baseline_ess": base_ess,
              "ess_min": float(overview.effective_sample_size.min()),
              "ess_max": float(overview.effective_sample_size.max()),
              "warning": "Deterministic selection / lack of overlap cannot be fixed "
                         "by choosing a floor or cap; no expert outcome metrics here."}
    (out / "sensitivity_summary.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    lines = ["# IPCW sensitivity to propensity floors and weight caps", "", DISCLAIMER, "",
             "**Label-free weight stability only.** All variants reuse the same "
             "study-grouped out-of-fold propensities and training-fold "
             "acceptance priors; no expert truth, model refit, or independent "
             "evaluation is involved. A lower floor than the original cannot "
             "be reconstructed after clipping. Each variant retains only "
             "accepted pseudo-labels, bounded and mean-normalized within "
             "each finding; rejected pairs remain unknown.", "",
             f"Configured baseline: ESS {base_ess:.1f} across "
             f"{int(baseline.n_accepted.iloc[0])} accepted pairs. Other caps/floors "
             f"yield ESS {record['ess_min']:.1f}–{record['ess_max']:.1f}. "
             "This is **not** evidence of less outcome bias.", "",
             "| Propensity floor | Weight cap | Effective sample size | "
             "99th percentile | Fraction at cap |",
             "|---:|---:|---:|---:|---:|"]
    for row in overview.itertuples():
        lines.append(f"| {row.propensity_floor:.3f} | {row.weight_cap:g} | "
                     f"{row.effective_sample_size:.1f} | {row.weight_p99:.3f} | "
                     f"{row.fraction_at_cap:.2%} |")
    lines += ["", "See `sensitivity_by_finding.csv` for all eleven findings. "
              "**Severe overlap/positivity violations and unmeasured MNAR "
              "remain possible under every variant.** This comparison is "
              "about numerical stability, never a clinical guarantee.", "", DISCLAIMER, ""]
    (out / "sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": record["status"], "path": str(out / "sensitivity_report.md"),
            "n_variants": record["n_variants"], "configured_baseline_ess": base_ess,
            "ess_min": record["ess_min"], "ess_max": record["ess_max"]}
