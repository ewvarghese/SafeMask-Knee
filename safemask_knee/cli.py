"""Local, CPU-only command-line workflow with JSON stdout summaries."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import run_audit
from .config import load_config
from .constants import DISCLAIMER
from .dashboard import serve_dashboard
from .doctor import check_doctor, write_doctor
from .external_validation import evaluate_external
from .eval import run_evaluation
from .io import (ReleaseData, SchemaError, expert_fold_map, load_release,
                 sha256_file, write_attribution, write_manifest, write_schema)
from .masks import write_masks
from .provenance import config_hash, write_provenance
from .quality import write_quality
from .report import write_summary, write_training_summary
from .sensitivity import write_sensitivity
from .weights import write_weights

LOG = logging.getLogger(__name__)


def _logging(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs/pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for old in root.handlers[:]:
        root.removeHandler(old)
        old.close()
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    file = logging.FileHandler(log_path, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)
    root.addHandler(file)


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m safemask_knee",
        description="SafeMask-Knee: grouped audit, IPCW labels and conservative "
                    "selective risk masks (CPU/tabular only). " + DISCLAIMER,
        epilog=DISCLAIMER, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    help_text = {
        "doctor": "Preflight environment, archive, schema, licence and output access.",
        "schema": "Validate inputs; print column map; hash source files.",
        "quality": "Aggregate data-quality, duplicate and leakage diagnostics.",
        "audit": "Expert outcome selection and metadata fairness diagnostics.",
        "weights": "Out-of-fold Platt acceptance propensity and bounded IPCW.",
        "sensitivity": "Stress-test IPCW floors and caps without expert outcomes.",
        "masks": "Class-conditional LOO conformal masks, expert outer-CV.",
        "evaluate": "Outer study-grouped comparison of A-D and release/mask.",
        "external-evaluate": "New disjoint score/label cohort, frozen expert cutoffs.",
        "report": "Summarize existing artifacts, including partial truth-free training.",
        "train": "Fit grouped acceptance propensity and IPCW without expert truth; "
                 "abstain from conformal masks.",
        "all": "Run schema -> audit -> weights -> masks -> evaluate -> report "
               "(expert truth required).",
        "dashboard": "Serve a read-only aggregate research dashboard.",
    }
    defaults = {
        "doctor": "outputs/doctor.json", "schema": "outputs/schema.json",
        "quality": "outputs/quality", "audit": "outputs/audit",
        "weights": "outputs/debiased_training_weights.csv",
        "sensitivity": "outputs/sensitivity",
        "masks": "outputs/clinical_risk_masks.csv",
        "evaluate": "outputs/evaluation",
        "external-evaluate": "outputs/external_evaluation",
        "report": "reports/summary.md",
    }
    for command, description in help_text.items():
        part = sub.add_parser(command, help=description, description=description + " " +
                              DISCLAIMER, epilog=DISCLAIMER,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
        if command == "dashboard":
            part.add_argument("--config", default=None, metavar="YAML")
            part.add_argument("--output-root", default=None, metavar="DIR",
                              help="Previously generated output root (default from config)")
            part.add_argument("--host", default="127.0.0.1",
                              help="Bind address; use 0.0.0.0 only for a private preview")
            part.add_argument("--port", type=int, default=8765)
            continue
        part.add_argument("--release", required=True, metavar="RELEASE_DIR",
                          help="Local tabular release directory; no images or network")
        part.add_argument("--config", default=None, metavar="YAML",
                          help="Optional local YAML; built-in defaults match configs/default.yaml")
        if command == "doctor":
            part.add_argument("--archive", type=Path, default=Path("archive.zip"),
                              help="Optional bundled archive to verify if present")
            part.add_argument("--require-expert", action="store_true",
                              help="Fail preflight if official expert truth is missing")
        if command == "external-evaluate":
            part.add_argument("--external-release", required=True, type=Path,
                              help="Separate frozen-score release for unseen studies")
            part.add_argument("--external-labels", required=True, type=Path,
                              help="Separately licensed 11-finding expert truth CSV/JSON")
        if command not in ("all", "train", "doctor"):
            part.add_argument("--out", default=None, metavar="PATH",
                              help=f"Output path (default {defaults[command]})")
    return parser


def _verify_saved_manifest(data: ReleaseData, root: Path,
                           cfg: dict[str, Any] | None = None) -> None:
    path = root / "data_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run `train` or `all` first; no source hash manifest: {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    current = {str(p.relative_to(data.release_dir)): sha256_file(p)
               for p in set(data.files.values()) | set(data.notices)}
    previous = {item["path"]: item["sha256"] for item in saved["files"]}
    if current != previous:
        raise ValueError("Release inputs changed since the artifacts were generated; "
                         "rerun `train` or `all` before reporting")
    provenance = root / "run_provenance.json"
    if cfg is not None and provenance.is_file():
        saved_config = json.loads(provenance.read_text(encoding="utf-8"))[
            "config_sha256_canonical"]
        if saved_config != config_hash(cfg):
            raise ValueError("Config changed since artifact generation; rerun "
                             "`train` or `all` before using saved results")


def _prepare(args: argparse.Namespace, cfg: dict[str, Any], root: Path,
             *, verify_only: bool = False) -> ReleaseData:
    data = load_release(args.release, cfg,
                        require_expert=args.command in (
                            "masks", "evaluate", "external-evaluate", "all"))
    if verify_only:
        _verify_saved_manifest(data, root, cfg)
        return data
    # Inspector/partial subcommands must not rehash a new source over old
    # downstream weights/masks and make stale metrics look current.
    if args.command not in ("train", "all") and (
            root / "debiased_training_weights.csv").exists():
        _verify_saved_manifest(data, root, cfg)
    if not data.frame.expert_label.notna().any() and args.command in (
            "train", "quality", "audit", "weights") and (
            (root / "clinical_risk_masks.csv").exists() or
            (root / "evaluation/metrics_by_strategy.csv").exists()):
        raise ValueError("Refusing to leave stale expert-dependent masks/evaluation "
                         "alongside truth-free artifacts. Use a fresh output_root.")
    schema_path = Path(args.out) if args.command == "schema" and args.out else root / "schema.json"
    write_schema(data, schema_path)
    write_manifest(data, root)
    write_attribution(data, Path.cwd() / "ATTRIBUTION.md")
    if args.command in ("train", "all", "weights", "masks", "evaluate") or not (
            root / "debiased_training_weights.csv").exists():
        write_provenance(data, root, cfg, args.command,
                         Path(args.config) if args.config else None)
    return data


def _target(args: argparse.Namespace, root: Path, suffix: str) -> Path:
    requested = getattr(args, "out", None)
    return Path(requested) if requested else root / suffix


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    root = Path(args.output_root if args.command == "dashboard" and args.output_root
                else cfg["paths"]["output_root"]).expanduser()
    _logging(root)
    if args.command == "dashboard":
        serve_dashboard(root, args.host, args.port)
        return {"command": "dashboard", "status": "stopped", "disclaimer": DISCLAIMER}
    if args.command == "doctor":
        _data, diagnostics = check_doctor(args.release, cfg, root, args.archive,
                                          require_expert=args.require_expert)
        return {"command": "doctor", **write_doctor(diagnostics, root)}
    if args.command == "report":
        data = _prepare(args, cfg, root, verify_only=True)
        target = _target(args, root, "reports/summary.md") if args.out else Path(
            "reports/summary.md")
        writer = write_summary if data.frame.expert_label.notna().any() else write_training_summary
        return {"command": "report", **writer(data, root, target)}
    if args.command in ("sensitivity", "external-evaluate"):
        data = _prepare(args, cfg, root, verify_only=True)
        if args.command == "sensitivity":
            weights_path = root / "debiased_training_weights.csv"
            if not weights_path.is_file():
                raise FileNotFoundError("Run `train` or `all` first; no weights CSV")
            result = write_sensitivity(pd.read_csv(weights_path),
                                       _target(args, root, "sensitivity"), cfg)
            return {"command": "sensitivity", **result}
        external = load_release(args.external_release, cfg, require_expert=False)
        result = evaluate_external(data, external, args.external_labels, root,
                                   _target(args, root, "external_evaluation"), cfg)
        return {"command": "external-evaluate", **result}
    data = _prepare(args, cfg, root)
    if args.command == "schema":
        return {"command": "schema", "path": str(_target(args, root, "schema.json")),
                "n_studies": int(data.frame.study_id.nunique()),
                "n_expert_pairs": int(data.frame.expert_label.notna().sum()),
                "expert_truth_available": bool(data.frame.expert_label.notna().any()),
                "n_mapped_columns": len(data.mapping),
                "manifest": str(root / "data_manifest.json")}
    if args.command == "quality":
        return {"command": "quality", **write_quality(
            data, _target(args, root, "quality"))}
    has_expert = bool(data.frame.expert_label.notna().any())
    if args.command == "train":
        if has_expert:
            raise ValueError("Expert truth is supplied; use `all` for the full outer-CV analysis")
        quality = write_quality(data, root / "quality")
        audit = run_audit(data, root / "audit", cfg)
        weight_table, weight_summary = write_weights(
            data, root / "debiased_training_weights.csv", cfg, folds={})
        sensitivity = write_sensitivity(weight_table, root / "sensitivity", cfg)
        report = write_training_summary(data, root, Path("reports/summary.md"))
        return {"command": "train", "status": "incomplete_missing_expert_truth",
                "disclaimer": DISCLAIMER, "schema": str(root / "schema.json"),
                "manifest": str(root / "data_manifest.json"),
                "provenance": str(root / "run_provenance.json"),
                "quality": quality, "audit": audit, "weights": weight_summary,
                "sensitivity": sensitivity, "masks": None,
                "evaluation": None, "report": report}
    folds = expert_fold_map(data.frame, cfg["outer_folds"], cfg["seed"]) if has_expert else {}
    if args.command == "all":
        quality = write_quality(data, root / "quality")
    if args.command in ("audit", "all"):
        audit = run_audit(data, _target(args, root, "audit"), cfg)
        if args.command == "audit":
            return {"command": "audit", **audit}
    if args.command in ("weights", "evaluate", "all"):
        weight_table, weight_summary = write_weights(
            data, _target(args, root, "debiased_training_weights.csv")
            if args.command == "weights" else root / "debiased_training_weights.csv",
            cfg, folds=folds)
        if args.command == "weights":
            return {"command": "weights", **weight_summary}
        if args.command == "all":
            sensitivity = write_sensitivity(weight_table, root / "sensitivity", cfg)
    if args.command in ("masks", "evaluate", "all"):
        mask_table, mask_summary = write_masks(
            data, _target(args, root, "clinical_risk_masks.csv")
            if args.command == "masks" else root / "clinical_risk_masks.csv",
            cfg, folds=folds)
        if args.command == "masks":
            return {"command": "masks", **mask_summary}
    if args.command in ("evaluate", "all"):
        evaluation = run_evaluation(data, weight_table, mask_table,
                                    _target(args, root, "evaluation"), cfg, folds=folds)
        if args.command == "evaluate":
            return {"command": "evaluate", **evaluation}
    if args.command == "all":
        report = write_summary(data, root, Path("reports/summary.md"))
        return {"command": "all", "schema": str(root / "schema.json"),
                "manifest": str(root / "data_manifest.json"),
                "provenance": str(root / "run_provenance.json"),
                "quality": quality, "audit": audit, "weights": weight_summary,
                "sensitivity": sensitivity, "masks": mask_summary,
                "evaluation": evaluation, "report": report}
    raise AssertionError(f"Unrecognized CLI command: {args.command}")


def main() -> None:
    """Parse CLI arguments; put only a compact JSON object on stdout."""
    parser = _parse()
    args = parser.parse_args()
    try:
        summary = _execute(args)
    except (SchemaError, ValueError, FileNotFoundError, AssertionError) as exc:
        LOG.exception("SafeMask-Knee failed: %s", exc)
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(summary, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
