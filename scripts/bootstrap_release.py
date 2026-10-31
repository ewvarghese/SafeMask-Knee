#!/usr/bin/env python3
"""Verify the bundled archive and extract only CPU-toolkit inputs locally.

No competition labels, images, or source CSVs are added to Git. This step uses
only the local archive; the official expert answer key is NOT inside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

ARCHIVE_SHA256 = "7898b2da2ebdefbbebbb15ae20f4ad06ffc69c9fe3ec6b1fdaf29f9e52785d62"
DISCLAIMER = ("For research only; not for clinical use; requires expert radiologist "
              "oversight; statistical guarantees are not clinical guarantees.")
RELEASE_FILES = (
    "DATASET_CARD.txt",
    "THIRD_PARTY_NOTICES.txt",
    "FILES.json",
    "source_provenance.json",
    "benchmark_results.json",
    "expert_oof.csv",  # predictions ONLY; the toolkit never treats this as truth
    "study_metadata.csv",
    "accuracy_soft_labels.csv",
    "accuracy_confidence.csv",
    "accuracy_accepted60.csv",
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_release(archive: Path, out: Path, *,
                    expected_archive_sha256: str = ARCHIVE_SHA256) -> dict[str, object]:
    """Check every listed member's SHA-256 before writing any selected file.

    Refuses an unfamiliar archive, extra/unlisted files, unsafe paths, missing
    members, or existing local release files that no longer match the archive.
    Does not overwrite a separately licensed local `expert_results.csv`.
    """
    archive = Path(archive).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Missing local archive: {archive}")
    digest = _hash_file(archive)
    if digest != expected_archive_sha256:
        raise ValueError(f"Archive SHA-256 mismatch: {digest}; expected "
                         f"{expected_archive_sha256}. Refusing unknown inputs.")
    selected: dict[str, bytes] = {}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or "FILES.json" not in names:
            raise ValueError("ZIP has duplicate names or lacks FILES.json")
        if any(Path(name).name != name or "\\" in name or name in ("", ".", "..")
               for name in names):
            raise ValueError("ZIP has unsafe or nested file paths")
        manifest = json.loads(z.read("FILES.json"))
        expected = manifest.get("files")
        if not isinstance(expected, dict) or set(expected) != set(names) - {"FILES.json"}:
            raise ValueError("ZIP members and FILES.json do not match exactly")
        if not set(RELEASE_FILES).issubset(set(names)):
            raise ValueError(f"Archive lacks toolkit inputs: "
                             f"{sorted(set(RELEASE_FILES) - set(names))}")
        for name, checksum in expected.items():
            if not isinstance(checksum, str) or len(checksum) != 64 or any(
                    digit not in "0123456789abcdef" for digit in checksum):
                raise ValueError(f"Invalid SHA-256 in FILES.json for {name}")
            hash_member = hashlib.sha256()
            chunks: list[bytes] = []
            with z.open(name) as stream:
                while chunk := stream.read(1024 * 1024):
                    hash_member.update(chunk)
                    if name in RELEASE_FILES:
                        chunks.append(chunk)
            if hash_member.hexdigest() != checksum:
                raise ValueError(f"ZIP member failed SHA-256 verification: {name}")
            if name in RELEASE_FILES:
                selected[name] = b"".join(chunks)
        selected["FILES.json"] = z.read("FILES.json")

    # Check every existing destination before writing any new files. In
    # particular, never clobber a user-provided expert answer key.
    for name, content in selected.items():
        target = out / name
        if target.exists() and (not target.is_file() or
                                _hash_file(target) != hashlib.sha256(content).hexdigest()):
            raise ValueError(f"Existing release input differs from the verified "
                             f"archive; refusing to overwrite {target}")
    out.mkdir(parents=True, exist_ok=True)
    extracted = 0
    for name, content in selected.items():
        target = out / name
        if target.is_file():
            continue
        with tempfile.NamedTemporaryFile(dir=out, prefix=".release-", suffix=".tmp",
                                         delete=False) as tmp:
            temp_path = Path(tmp.name)
            try:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise
        os.replace(temp_path, target)
        extracted += 1
    return {"archive": str(archive), "sha256": digest,
            "verified_members": len(expected), "release_dir": str(out),
            "selected_files": len(selected), "extracted_files": extracted,
            "expert_truth_in_archive": False, "disclaimer": DISCLAIMER}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ + "\n" + DISCLAIMER,
                                     epilog=DISCLAIMER)
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    parser.add_argument("--out", type=Path, default=Path("data/release"))
    args = parser.parse_args()
    print(json.dumps(prepare_release(args.archive, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
