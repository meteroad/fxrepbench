from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import reconstruct_item
from .config import chain_config, effects_config, load_config, sample_parameter
from .manifests import TrackFiles, load_csv, load_jsonl
from .render import render_chain


@dataclass(frozen=True)
class CandidateSpec:
    chain_order: tuple[str, ...]
    physical_params: dict[str, dict[str, float]]


def stable_seed(seed: int, *parts: object) -> int:
    text = ":".join((str(seed), *(str(part) for part in parts)))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _sample_params(
    chain_order: tuple[str, ...],
    effect_spec: dict[str, Any],
    rng: Any,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for effect_id in chain_order:
        output[effect_id] = {
            name: sample_parameter(parameter, rng)[0]
            for name, parameter in effect_spec["effects"][effect_id]["parameters"].items()
        }
    return output


def sample_known_candidate(
    target_row: dict[str, Any],
    effect_spec: dict[str, Any],
    rng: Any,
) -> CandidateSpec:
    chain_order = tuple(str(value) for value in target_row["chain_order"])
    return CandidateSpec(chain_order, _sample_params(chain_order, effect_spec, rng))


def _sample_active_effects(
    effect_order: tuple[str, ...],
    prior: dict[str, Any],
    rng: Any,
) -> set[str]:
    import numpy as np

    probabilities = prior["activation_probabilities"]
    selected = {
        effect_id
        for effect_id in effect_order
        if float(rng.random()) < float(probabilities[effect_id])
    }
    minimum = max(1, min(int(prior["minimum_effects"]), len(effect_order)))
    maximum = max(minimum, min(int(prior["maximum_effects"]), len(effect_order)))
    if len(selected) < minimum:
        candidates = [effect_id for effect_id in effect_order if effect_id not in selected]
        weights = np.asarray([probabilities[effect_id] for effect_id in candidates], dtype=np.float64)
        weights /= weights.sum()
        selected.update(
            str(value)
            for value in rng.choice(
                candidates,
                size=minimum - len(selected),
                replace=False,
                p=weights,
            )
        )
    if len(selected) > maximum:
        candidates = sorted(selected, key=effect_order.index)
        weights = np.asarray([probabilities[effect_id] for effect_id in candidates], dtype=np.float64)
        weights /= weights.sum()
        selected = {
            str(value)
            for value in rng.choice(
                candidates,
                size=maximum,
                replace=False,
                p=weights,
            )
        }
    return selected


def sample_hidden_candidate(
    effect_spec: dict[str, Any],
    prior: dict[str, Any],
    rng: Any,
) -> CandidateSpec:
    effect_order = tuple(str(value) for value in effect_spec["effect_order"])
    selected = _sample_active_effects(effect_order, prior, rng)
    templates = prior["order_templates"]
    template = templates[int(rng.integers(0, len(templates)))]
    chain_order = tuple(str(effect_id) for effect_id in template if effect_id in selected)
    return CandidateSpec(chain_order, _sample_params(chain_order, effect_spec, rng))


def generate_random_pool(
    track: TrackFiles,
    data_root: Path,
    output_path: Path,
    *,
    topology: str,
    budget: int,
    item_id: str | None = None,
    audio_dir: Path | None = None,
    seed: int | None = None,
    verify_source_hash: bool = True,
    maximum_attempt_factor: int = 20,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    if topology not in {"known", "hidden"}:
        raise ValueError("topology must be 'known' or 'hidden'")
    if budget <= 0:
        raise ValueError("budget must be positive")
    effect_spec = effects_config()
    prior = chain_config()
    benchmark = load_config("counterfx.json")
    candidate_seed = int(benchmark["candidate_seed"] if seed is None else seed)
    clipping_peak = float(benchmark["clipping_peak"])

    audio_rows = {row["item_id"]: row for row in load_csv(track.audio_manifest)}
    target_rows = {str(row["item_id"]): row for row in load_jsonl(track.target_manifest)}
    item_ids = sorted(audio_rows)
    if item_id is not None:
        if item_id not in audio_rows:
            raise KeyError(f"unknown item {item_id!r} in track {track.key!r}")
        item_ids = [item_id]

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing pool: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    item_summaries: list[dict[str, Any]] = []
    for item_index, current_id in enumerate(item_ids, start=1):
        source, _, _, sample_rate = reconstruct_item(
            audio_rows[current_id],
            target_rows[current_id],
            data_root,
            verify_hash=verify_source_hash,
        )
        item_seed = stable_seed(candidate_seed, current_id, topology)
        rng = np.random.default_rng(item_seed)
        accepted = 0
        attempts = 0
        rejected = 0
        while accepted < budget:
            attempts += 1
            if attempts > maximum_attempt_factor * budget:
                raise RuntimeError(
                    f"candidate acceptance exhausted for {current_id}: "
                    f"{accepted}/{budget} after {attempts - 1} attempts"
                )
            if topology == "known":
                candidate = sample_known_candidate(target_rows[current_id], effect_spec, rng)
            else:
                candidate = sample_hidden_candidate(effect_spec, prior, rng)
            rendered = render_chain(
                source,
                list(candidate.chain_order),
                candidate.physical_params,
                sample_rate,
            )
            peak = float(np.abs(rendered).max())
            if not np.isfinite(rendered).all() or not peak < clipping_peak:
                rejected += 1
                continue

            candidate_id = f"{accepted:04d}"
            record: dict[str, Any] = {
                "schema_version": "1.0",
                "track": track.key,
                "item_id": current_id,
                "candidate_id": candidate_id,
                "topology": topology,
                "candidate_seed": candidate_seed,
                "item_seed": item_seed,
                "draw_index": attempts - 1,
                "chain_order": list(candidate.chain_order),
                "physical_params": candidate.physical_params,
                "peak": peak,
            }
            if audio_dir is not None:
                candidate_path = audio_dir / current_id / f"{candidate_id}.wav"
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(candidate_path, rendered.T, sample_rate, subtype="FLOAT")
                record["audio_path"] = os.path.relpath(candidate_path, output_path.parent)
            records.append(record)
            accepted += 1

        item_summaries.append(
            {
                "item_id": current_id,
                "valid_candidates": accepted,
                "renderer_calls": attempts,
                "rejected_candidates": rejected,
            }
        )
        print(
            f"[{item_index}/{len(item_ids)}] {current_id}: "
            f"valid={accepted} calls={attempts} rejected={rejected}"
        )

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(output_path)
    total_calls = sum(row["renderer_calls"] for row in item_summaries)
    summary = {
        "schema_version": "1.0",
        "track": track.key,
        "topology": topology,
        "candidate_seed": candidate_seed,
        "budget_per_item": budget,
        "items": len(item_ids),
        "valid_candidates": len(records),
        "renderer_calls": total_calls,
        "rejected_candidates": sum(row["rejected_candidates"] for row in item_summaries),
        "renderer": benchmark["renderer"],
        "items_detail": item_summaries,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
