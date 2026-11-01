"""Reproducible, ID-free records of code, inputs, configuration and runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import DISCLAIMER
from .io import ReleaseData


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requirements() -> dict[str, dict[str, str | None]]:
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    if not requirements.is_file():
        return {}
    packages: dict[str, dict[str, str | None]] = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        name, expected = text.split("==", 1)
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        packages[name] = {"pinned": expected, "installed": installed}
    return packages


def _git_revision() -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]),
                                 "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True, timeout=3)
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable hash of validated run settings, independent of YAML formatting."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode(
        "utf-8")).hexdigest()


def provenance_record(data: ReleaseData, root: Path, cfg: dict[str, Any],
                      command: str, config_path: Path | None = None) -> dict[str, Any]:
    """Describe provenance without serializing patients, labels or source rows."""
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    config_digest = config_hash(cfg)
    config_file = Path(config_path).expanduser().resolve() if config_path else None
    return {
        "disclaimer": DISCLAIMER,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "package_version": _installed_version(),
        "direct_dependencies": _requirements(),
        "config_sha256_canonical": config_digest,
        "config_file_sha256": _sha256(config_file) if config_file and
        config_file.is_file() else None,
        "random_seed": int(cfg["seed"]),
        "input_files": manifest["files"],  # names/sizes/hashes only; no data rows
        "expert_truth_available": bool(data.frame.expert_label.notna().any()),
        "license_notice_files": [p.name for p in data.notices],
        "data_policy": "source and derived data local/Git-ignored; do not upload "
                       "restricted labels or study IDs to public CI artifacts",
    }


def _installed_version() -> str | None:
    try:
        return importlib.metadata.version("safemask-knee")
    except importlib.metadata.PackageNotFoundError:
        return None


def write_provenance(data: ReleaseData, root: Path, cfg: dict[str, Any],
                     command: str, config_path: Path | None = None) -> dict[str, Any]:
    """Write output-root provenance after the source-hash manifest exists."""
    record = provenance_record(data, root, cfg, command, config_path)
    path = root / "run_provenance.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "git_revision": record["git_revision"],
            "config_sha256_canonical": record["config_sha256_canonical"]}
