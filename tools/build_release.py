#!/usr/bin/env python3
"""Audit the source tree and build a deterministic, audio-free release ZIP."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"
ARCHIVE = ROOT / "dist" / f"fxrepbench-v{VERSION}.zip"
FORBIDDEN_SUFFIXES = {".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
FORBIDDEN_TEXT = (
    "/data/" + "workspace",
    "/" + "data2/",
    "fayelliu" + "@",
    "AK" + "IA",
    "BEGIN OPENSSH" + " PRIVATE KEY",
)
EXCLUDED_PARTS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", "dist"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def release_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"audio file must not enter the release: {relative}")
        if path.name.endswith((".pyc", ".pyo")):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def audit_text(files: list[Path]) -> None:
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in FORBIDDEN_TEXT:
            if marker in text:
                raise RuntimeError(f"private marker {marker!r} found in {path.relative_to(ROOT)}")


def validate_manifests() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from fxrepbench.manifests import load_tracks, validate_track

    reports = [validate_track(track) for track in load_tracks(ROOT).values()]
    errors = {report["track"]: report["errors"] for report in reports if report["errors"]}
    if errors:
        raise RuntimeError(json.dumps(errors, indent=2, sort_keys=True))


def build(files: list[Path]) -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    checksum_lines = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        checksum_lines.append(f"{sha256_bytes(path.read_bytes())}  {relative}")
    checksum_data = ("\n".join(checksum_lines) + "\n").encode("utf-8")

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = f"fxrepbench-v{VERSION}/{path.relative_to(ROOT).as_posix()}"
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        info = zipfile.ZipInfo(
            f"fxrepbench-v{VERSION}/MANIFEST.sha256",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, checksum_data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> None:
    validate_manifests()
    files = release_files()
    audit_text(files)
    build(files)
    print(f"archive: {ARCHIVE}")
    print(f"files: {len(files) + 1}")
    print(f"bytes: {ARCHIVE.stat().st_size}")
    print(f"sha256: {sha256_bytes(ARCHIVE.read_bytes())}")


if __name__ == "__main__":
    main()
