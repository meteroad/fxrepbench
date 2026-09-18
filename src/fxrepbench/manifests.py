from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EFFECT_IDS = {
    "highpass_filter",
    "lowpass_filter",
    "peak_filter",
    "compressor",
    "distortion",
    "chorus",
    "delay",
    "reverb",
}


@dataclass(frozen=True)
class TrackFiles:
    key: str
    role: str
    expected_items: int
    benchmark_version: str
    audio_manifest: Path
    target_manifest: Path
    summary: Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_tracks(root: Path | None = None) -> dict[str, TrackFiles]:
    root = root or repository_root()
    index = json.loads(
        (root / "benchmarks/counterfx/manifests/index.json").read_text(
            encoding="utf-8"
        )
    )
    tracks: dict[str, TrackFiles] = {}
    for key, item in index["tracks"].items():
        tracks[key] = TrackFiles(
            key=key,
            role=str(item["role"]),
            expected_items=int(item["expected_items"]),
            benchmark_version=str(item["legacy_benchmark_version"]),
            audio_manifest=root / item["audio_manifest"],
            target_manifest=root / item["target_manifest"],
            summary=root / item["summary"],
        )
    return tracks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def validate_track(track: TrackFiles) -> dict[str, Any]:
    audio_rows = load_csv(track.audio_manifest)
    target_rows = load_jsonl(track.target_manifest)
    summary = json.loads(track.summary.read_text(encoding="utf-8"))
    errors: list[str] = []

    if len(audio_rows) != track.expected_items:
        errors.append(f"audio rows: expected {track.expected_items}, found {len(audio_rows)}")
    if len(target_rows) != track.expected_items:
        errors.append(f"target rows: expected {track.expected_items}, found {len(target_rows)}")
    if int(summary.get("items", -1)) != track.expected_items:
        errors.append(f"summary items: expected {track.expected_items}, found {summary.get('items')}")

    audio_ids = [row.get("item_id", "") for row in audio_rows]
    target_ids = [str(row.get("item_id", "")) for row in target_rows]
    if _duplicates(audio_ids):
        errors.append(f"duplicate audio IDs: {_duplicates(audio_ids)[:5]}")
    if _duplicates(target_ids):
        errors.append(f"duplicate target IDs: {_duplicates(target_ids)[:5]}")
    if set(audio_ids) != set(target_ids):
        errors.append("audio and target item IDs do not match")

    for row in audio_rows:
        item_id = row.get("item_id", "<missing>")
        if row.get("benchmark_version") != track.benchmark_version:
            errors.append(f"{item_id}: unexpected benchmark version")
        if not row.get("license") or not row.get("license_url"):
            errors.append(f"{item_id}: missing source license metadata")
        digest = row.get("source_sha256", "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            errors.append(f"{item_id}: invalid source SHA256")
        audio_relpath = row.get("audio_relpath", "")
        if Path(audio_relpath).is_absolute() or ".." in Path(audio_relpath).parts:
            errors.append(f"{item_id}: unsafe audio path")
        try:
            adjacent = float(row["start_b_sec"]) == float(row["start_a_sec"]) + float(row["duration_sec"])
        except (KeyError, ValueError):
            adjacent = False
        if not adjacent:
            errors.append(f"{item_id}: windows are not adjacent")

    for row in target_rows:
        item_id = str(row.get("item_id", "<missing>"))
        if row.get("benchmark_version") != track.benchmark_version:
            errors.append(f"{item_id}: unexpected target benchmark version")
        order = row.get("chain_order")
        active = row.get("active_effects")
        if not isinstance(order, list) or not 1 <= len(order) <= 4:
            errors.append(f"{item_id}: invalid chain order")
            continue
        if not isinstance(active, list) or set(active) != set(order):
            errors.append(f"{item_id}: active effects do not match chain order")
        if not set(order) <= EFFECT_IDS:
            errors.append(f"{item_id}: unknown effect ID")
        params = row.get("continuous_params_physical")
        if not isinstance(params, dict) or set(params) != set(order):
            errors.append(f"{item_id}: physical parameters do not match chain")
        if row.get("renderer_version") != "0.9.23":
            errors.append(f"{item_id}: unexpected renderer version")
        if float(row.get("source_target_ld_min", -1.0)) < 0.8:
            errors.append(f"{item_id}: target is below the frozen salience threshold")

    return {
        "track": track.key,
        "role": track.role,
        "audio_items": len(audio_rows),
        "target_items": len(target_rows),
        "errors": errors,
    }


def get_item(track: TrackFiles, item_id: str) -> tuple[dict[str, str], dict[str, Any]]:
    audio = {row["item_id"]: row for row in load_csv(track.audio_manifest)}
    targets = {str(row["item_id"]): row for row in load_jsonl(track.target_manifest)}
    if item_id not in audio or item_id not in targets:
        raise KeyError(f"unknown item {item_id!r} in track {track.key!r}")
    return audio[item_id], targets[item_id]
