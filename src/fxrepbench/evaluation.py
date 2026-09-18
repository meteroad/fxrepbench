from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from .audio import reconstruct_item
from .config import metric_config
from .manifests import EFFECT_IDS, TrackFiles, load_csv, load_jsonl
from .metrics import MultiResolutionStftDistance, fx_set_f1


@dataclass(frozen=True)
class SubmissionRow:
    item_id: str
    candidate_id: str
    audio_path: Path
    score: float | None
    chain_order: tuple[str, ...] | None


def _parse_chain(value: Any) -> tuple[str, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            value = json.loads(stripped)
        else:
            value = [part.strip() for part in stripped.split(",") if part.strip()]
    if not isinstance(value, list) or not all(isinstance(part, str) for part in value):
        raise ValueError("chain_order must be a JSON list or comma-separated effect IDs")
    if not set(value) <= EFFECT_IDS:
        unknown = sorted(set(value) - EFFECT_IDS)
        raise ValueError(f"unknown effects in chain_order: {unknown}")
    return tuple(value)


def load_submission(path: Path) -> list[SubmissionRow]:
    if path.suffix.lower() == ".jsonl":
        raw_rows = load_jsonl(path)
    elif path.suffix.lower() == ".csv":
        raw_rows = load_csv(path)
    else:
        raise ValueError("submission must be a .jsonl or .csv file")

    rows: list[SubmissionRow] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_rows):
        item_id = str(raw.get("item_id", "")).strip()
        audio_value = str(raw.get("audio_path", "")).strip()
        if not item_id or not audio_value:
            raise ValueError(f"submission row {index + 1} requires item_id and audio_path")
        candidate_id = str(raw.get("candidate_id", index)).strip()
        key = (item_id, candidate_id)
        if key in seen:
            raise ValueError(f"duplicate candidate_id {candidate_id!r} for {item_id!r}")
        seen.add(key)

        score_value = raw.get("score")
        score = None if score_value is None or score_value == "" else float(score_value)
        if score is not None and not math.isfinite(score):
            raise ValueError(f"non-finite score for {item_id}/{candidate_id}")
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        rows.append(
            SubmissionRow(
                item_id=item_id,
                candidate_id=candidate_id,
                audio_path=audio_path,
                score=score,
                chain_order=_parse_chain(raw.get("chain_order")),
            )
        )
    if not rows:
        raise ValueError("submission is empty")
    return rows


def _load_audio(path: Path, sample_rate: int, channels: int, samples: int) -> Any:
    import torch
    import torchaudio

    if not path.is_file():
        raise FileNotFoundError(f"candidate audio not found: {path}")
    audio, actual_rate = torchaudio.load(str(path))
    if actual_rate != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz, found {actual_rate} Hz")
    if audio.shape[0] != channels:
        raise ValueError(f"{path}: expected {channels} channels, found {audio.shape[0]}")
    if audio.shape[-1] != samples:
        raise ValueError(f"{path}: expected {samples} samples, found {audio.shape[-1]}")
    if not torch.isfinite(audio).all():
        raise ValueError(f"{path}: audio contains non-finite samples")
    return audio


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "p90": _percentile(values, 0.9),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_submission(
    track: TrackFiles,
    submission_path: Path,
    data_root: Path,
    output_dir: Path,
    *,
    device: str = "cpu",
    batch_size: int = 8,
    allow_partial: bool = False,
    verify_source_hash: bool = True,
) -> dict[str, Any]:
    import torch

    submission = load_submission(submission_path)
    grouped: dict[str, list[SubmissionRow]] = defaultdict(list)
    for row in submission:
        grouped[row.item_id].append(row)
    for item_id, rows in grouped.items():
        if len(rows) > 1 and any(row.score is None for row in rows):
            raise ValueError(f"{item_id}: score is required when multiple candidates are submitted")

    audio_rows = {row["item_id"]: row for row in load_csv(track.audio_manifest)}
    target_rows = {str(row["item_id"]): row for row in load_jsonl(track.target_manifest)}
    unknown = sorted(set(grouped) - set(audio_rows))
    if unknown:
        raise ValueError(f"submission contains unknown item IDs: {unknown[:5]}")
    missing = sorted(set(audio_rows) - set(grouped))
    if missing and not allow_partial:
        raise ValueError(f"submission is missing {len(missing)} items; first: {missing[:5]}")

    metric_spec = metric_config()
    metric = MultiResolutionStftDistance(
        fft_sizes=metric_spec["fft_sizes"],
        hop_sizes=metric_spec["hop_sizes"],
        win_lengths=metric_spec["win_lengths"],
        epsilon=metric_spec["epsilon"],
    )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not visible in this environment")

    candidate_results: list[dict[str, Any]] = []
    item_results: list[dict[str, Any]] = []
    for item_id in sorted(grouped):
        audio_row = audio_rows[item_id]
        target_row = target_rows[item_id]
        _, _, target, sample_rate = reconstruct_item(
            audio_row,
            target_row,
            data_root,
            verify_hash=verify_source_hash,
        )
        target_tensor = torch.as_tensor(target)
        candidates = [
            _load_audio(
                row.audio_path,
                sample_rate,
                int(target_tensor.shape[0]),
                int(target_tensor.shape[-1]),
            )
            for row in grouped[item_id]
        ]
        losses = metric(
            torch.stack(candidates),
            target_tensor,
            batch_size=batch_size,
            device=device,
        ).tolist()

        rows = grouped[item_id]
        selected_index = 0 if len(rows) == 1 else max(
            range(len(rows)), key=lambda index: float(rows[index].score)
        )
        oracle_index = min(range(len(rows)), key=lambda index: losses[index])
        selected = rows[selected_index]
        selected_f1 = (
            None
            if selected.chain_order is None
            else fx_set_f1(selected.chain_order, target_row["chain_order"])
        )
        for index, (row, loss) in enumerate(zip(rows, losses)):
            candidate_results.append(
                {
                    "item_id": item_id,
                    "candidate_id": row.candidate_id,
                    "score": "" if row.score is None else row.score,
                    "ld": loss,
                    "selected": int(index == selected_index),
                    "oracle": int(index == oracle_index),
                    "chain_order": "" if row.chain_order is None else json.dumps(row.chain_order),
                }
            )
        item_results.append(
            {
                "item_id": item_id,
                "candidates": len(rows),
                "selected_candidate_id": selected.candidate_id,
                "selected_score": "" if selected.score is None else selected.score,
                "selected_ld": losses[selected_index],
                "oracle_candidate_id": rows[oracle_index].candidate_id,
                "oracle_ld": losses[oracle_index],
                "selected_oracle_gap": losses[selected_index] - losses[oracle_index],
                "selected_fx_set_f1": "" if selected_f1 is None else selected_f1,
                "target_chain": json.dumps(target_row["chain_order"]),
            }
        )
        print(
            f"[{len(item_results)}/{len(grouped)}] {item_id}: "
            f"selected={losses[selected_index]:.4f} oracle={losses[oracle_index]:.4f}"
        )

    selected_values = [float(row["selected_ld"]) for row in item_results]
    oracle_values = [float(row["oracle_ld"]) for row in item_results]
    gap_values = [float(row["selected_oracle_gap"]) for row in item_results]
    f1_values = [
        float(row["selected_fx_set_f1"])
        for row in item_results
        if row["selected_fx_set_f1"] != ""
    ]
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "track": track.key,
        "items_evaluated": len(item_results),
        "items_expected": track.expected_items,
        "complete": len(item_results) == track.expected_items,
        "candidate_rows": len(candidate_results),
        "device": str(device),
        "metric": metric_spec,
        "selected_ld": _metric_summary(selected_values),
        "oracle_ld": _metric_summary(oracle_values),
        "selected_oracle_gap": _metric_summary(gap_values),
        "fx_set_f1": None
        if not f1_values
        else {"items": len(f1_values), **_metric_summary(f1_values)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "candidates.csv", candidate_results)
    _write_csv(output_dir / "items.csv", item_results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
