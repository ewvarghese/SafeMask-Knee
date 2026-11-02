"""A fresh checkout can verify/extract the bundled archive without source downloads."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts.bootstrap_release import RELEASE_FILES, prepare_release


def _archive(path: Path, *, tamper_member: str | None = None,
             extra: str = "unneeded.txt") -> str:
    # An invented fixture, not a competition file. FILES.json is self-excluded.
    payload = {name: f"fixture {name}\n".encode() for name in RELEASE_FILES
               if name != "FILES.json"}
    payload[extra] = b"An extra member must be checked, not extracted.\n"
    expected = {name: hashlib.sha256(content).hexdigest()
                for name, content in payload.items()}
    if tamper_member:
        expected[tamper_member] = "0" * 64
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("FILES.json", json.dumps({"files": expected}))
        for name, content in payload.items():
            z.writestr(name, content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_all_members_then_extract_only_toolkit_files(tmp_path: Path) -> None:
    archive = tmp_path / "synthetic.zip"
    sha = _archive(archive)
    out = tmp_path / "local" / "release"
    with pytest.raises(ValueError, match="Archive SHA-256 mismatch"):
        prepare_release(archive, out)
    assert not out.exists()
    result = prepare_release(archive, out, expected_archive_sha256=sha)
    assert result["selected_files"] == len(RELEASE_FILES)
    assert result["verified_members"] == len(RELEASE_FILES)
    assert result["extracted_files"] == len(RELEASE_FILES)
    assert {p.name for p in out.iterdir()} == set(RELEASE_FILES)
    assert not (out / "unneeded.txt").exists()
    # Local expert answers (separately licensed) must never be clobbered.
    expert = out / "expert_results.csv"
    expert.write_text("locally licensed answer key", encoding="utf-8")
    again = prepare_release(archive, out, expected_archive_sha256=sha)
    assert again["extracted_files"] == 0
    assert expert.read_text() == "locally licensed answer key"
    (out / "accuracy_soft_labels.csv").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        prepare_release(archive, out, expected_archive_sha256=sha)
    assert (out / "accuracy_soft_labels.csv").read_text() == "modified"


def test_unselected_tampering_and_zip_slip_fail_before_extraction(tmp_path: Path) -> None:
    out = tmp_path / "release"
    archive = tmp_path / "tampered.zip"
    sha = _archive(archive, tamper_member="unneeded.txt")
    with pytest.raises(ValueError, match="ZIP member failed SHA-256 verification"):
        prepare_release(archive, out, expected_archive_sha256=sha)
    assert not out.exists()
    sha = _archive(archive, extra="../outside.txt")
    with pytest.raises(ValueError, match="unsafe or nested"):
        prepare_release(archive, out, expected_archive_sha256=sha)
    assert not out.exists() and not (tmp_path / "outside.txt").exists()


def test_runner_has_strict_order_and_offline_runtime() -> None:
    runner = (Path(__file__).resolve().parents[1] / "scripts/run_release.sh").read_text()
    assert "set -euo pipefail" in runner
    assert runner.index("bootstrap_release.py") < runner.index("-m safemask_knee train")
    assert "OPENBLAS_NUM_THREADS=1" in runner
    assert "outputs/training_summary.json" in runner


def test_runner_explains_missing_ensurepip_and_partial_venv(tmp_path: Path) -> None:
    """Bash from PowerShell uses WSL: fail with actionable distro-package help."""
    project = tmp_path / "invented-project"
    (project / "scripts").mkdir(parents=True)
    (project / ".venv/bin").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts/run_release.sh"
    shutil.copyfile(source, project / "scripts/run_release.sh")
    # Represents the half-created Python executable left by a failed venv.
    partial_python = project / ".venv/bin/python"
    partial_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    partial_python.chmod(0o755)
    fake = project / "fake-bin"
    fake.mkdir()
    python3 = fake / "python3"
    python3.write_text("#!/bin/sh\n"
                       "if [ \"$1\" = -c ]; then\n"
                       "  case \"$2\" in\n"
                       "    *sys.exit*) [ \"$FAKE_MINOR\" != 3.14 ]; exit $?;;\n"
                       "    *) echo \"$FAKE_MINOR\"; exit 0;;\n"
                       "  esac\n"
                       "fi\n"
                       "echo 'ensurepip is not available' >&2\nexit 1\n", encoding="utf-8")
    python3.chmod(0o755)
    for name in ("python3.11", "python3.12", "python3.13", "python3.10"):
        (fake / name).symlink_to(python3)
    env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ.get("PATH", ""),
               FAKE_MINOR="3.11")
    outcome = subprocess.run(["bash", "scripts/run_release.sh"], cwd=project,
                             env=env, capture_output=True, text=True, timeout=20)
    assert outcome.returncode == 2
    assert "python3.11-venv" in outcome.stderr
    assert "PowerShell" in outcome.stderr
    assert "Then rerun: bash scripts/run_release.sh" in outcome.stderr
    env["FAKE_MINOR"] = "3.14"
    unsupported = subprocess.run(["bash", "scripts/run_release.sh"], cwd=project,
                                 env=env, capture_output=True, text=True, timeout=20)
    assert unsupported.returncode == 2
    assert "Python 3.14 is incompatible" in unsupported.stderr
    assert "python3.14-venv" not in unsupported.stderr
