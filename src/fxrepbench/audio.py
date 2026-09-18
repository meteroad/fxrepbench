from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .manifests import TrackFiles, load_csv, load_jsonl, sha256_file
from .render import render_chain


def resolve_audio_path(data_root: Path, audio_relpath: str) -> Path:
    relative = Path(audio_relpath)
    candidates = [data_root / relative]
    parts = relative.parts
    if parts[:2] == ("fma", "fma_small"):
        candidates.append(data_root / Path(*parts[2:]))
        candidates.append(data_root / "fma_small" / Path(*parts[2:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"source audio not found; tried: {tried}")


def _normalize_lufs(audio: Any, sample_rate: int, target_lufs: float) -> Any:
    import numpy as np
    import pyloudnorm as pyln

    loudness = float(
        pyln.Meter(sample_rate).integrated_loudness(
            audio.transpose(0, 1).cpu().numpy().astype(np.float64, copy=False)
        )
    )
    if not math.isfinite(loudness) or loudness < -70.0:
        raise ValueError(f"invalid integrated loudness: {loudness}")
    gain = 10.0 ** ((target_lufs - loudness) / 20.0)
    return audio * gain


def reconstruct_item(
    audio_row: dict[str, str],
    target_row: dict[str, Any],
    data_root: Path,
    *,
    verify_hash: bool = True,
) -> tuple[Any, Any, Any, int]:
    import numpy as np
    import torchaudio

    path = resolve_audio_path(data_root, audio_row["audio_relpath"])
    if verify_hash:
        actual = sha256_file(path)
        expected = audio_row["source_sha256"].lower()
        if actual != expected:
            raise ValueError(f"source SHA256 mismatch for {path}: {actual} != {expected}")

    decoded, source_rate = torchaudio.load(str(path))
    if decoded.shape[0] == 1:
        decoded = decoded.repeat(2, 1)
    elif decoded.shape[0] != 2:
        raise ValueError(f"expected mono or stereo source, found {decoded.shape[0]} channels")

    sample_rate = int(audio_row["sample_rate"])
    if source_rate != sample_rate:
        decoded = torchaudio.functional.resample(decoded, source_rate, sample_rate)
    start_a = int(round(float(audio_row["start_a_sec"]) * sample_rate))
    start_b = int(round(float(audio_row["start_b_sec"]) * sample_rate))
    length = int(round(float(audio_row["duration_sec"]) * sample_rate))
    if decoded.shape[-1] < start_b + length:
        raise ValueError("decoded source is too short for the frozen windows")

    source = _normalize_lufs(
        decoded[:, start_a : start_a + length],
        sample_rate,
        float(audio_row["target_lufs"]),
    )
    reference_base = _normalize_lufs(
        decoded[:, start_b : start_b + length],
        sample_rate,
        float(audio_row["target_lufs"]),
    )
    chain_order = [str(value) for value in target_row["chain_order"]]
    physical_params = target_row["continuous_params_physical"]
    source_np = source.cpu().numpy().astype(np.float32, copy=False)
    reference_base_np = reference_base.cpu().numpy().astype(np.float32, copy=False)
    reference = render_chain(reference_base_np, chain_order, physical_params, sample_rate)
    target = render_chain(source_np, chain_order, physical_params, sample_rate)
    return source_np, reference, target, sample_rate


def write_item(output_dir: Path, source: Any, reference: Any, target: Any, sample_rate: int) -> None:
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "source.wav", source.T, sample_rate, subtype="FLOAT")
    sf.write(output_dir / "reference.wav", reference.T, sample_rate, subtype="FLOAT")
    sf.write(output_dir / "target.wav", target.T, sample_rate, subtype="FLOAT")


def prepare_track(
    track: TrackFiles,
    data_root: Path,
    output_dir: Path,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    import json

    audio_rows = {row["item_id"]: row for row in load_csv(track.audio_manifest)}
    target_rows = {str(row["item_id"]): row for row in load_jsonl(track.target_manifest)}
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    for index, item_id in enumerate(sorted(audio_rows), start=1):
        source, reference, target, sample_rate = reconstruct_item(
            audio_rows[item_id],
            target_rows[item_id],
            data_root,
            verify_hash=verify_hash,
        )
        item_dir = output_dir / item_id
        write_item(item_dir, source, reference, target, sample_rate)
        index_rows.append(
            {
                "item_id": item_id,
                "source": f"{item_id}/source.wav",
                "reference": f"{item_id}/reference.wav",
                "target": f"{item_id}/target.wav",
                "sample_rate": sample_rate,
                "chain_order": target_rows[item_id]["chain_order"],
            }
        )
        print(f"[{index}/{len(audio_rows)}] prepared {item_id}")
    index_path = output_dir / "items.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": "1.0",
        "track": track.key,
        "items": len(index_rows),
        "audio_included_in_repository": False,
        "format": "float32 WAV",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
