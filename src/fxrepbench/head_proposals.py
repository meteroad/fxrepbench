from __future__ import annotations

import math
from typing import Any, Sequence

from .candidates import CandidateSpec
from .config import chain_config, effects_config, load_config, unit_to_parameter


def parameter_order(effect_spec: dict[str, Any] | None = None) -> tuple[tuple[str, str], ...]:
    effect_spec = effect_spec or effects_config()
    return tuple(
        (effect_id, parameter_name)
        for effect_id in effect_spec["effect_order"]
        for parameter_name, parameter in effect_spec["effects"][effect_id]["parameters"].items()
        if parameter.get("fixed") is None
    )


def logistic_normal_perturb(
    center: Any,
    rng: Any,
    scale_min: float = 0.08,
    scale_max: float = 2.5,
) -> tuple[Any, float]:
    import numpy as np

    center = np.asarray(center, dtype=np.float64)
    if center.ndim != 1:
        raise ValueError("proposal center must be one-dimensional")
    if not 0.0 < scale_min <= scale_max:
        raise ValueError("proposal scales must satisfy 0 < minimum <= maximum")
    scale = float(
        math.exp(rng.uniform(math.log(scale_min), math.log(scale_max)))
        if scale_min < scale_max
        else scale_min
    )
    clipped = np.clip(center, 1e-5, 1.0 - 1e-5)
    logits = np.log(clipped) - np.log1p(-clipped)
    perturbed = 1.0 / (1.0 + np.exp(-(logits + scale * rng.standard_normal(center.shape))))
    return perturbed.astype(np.float32), scale


def decode_hidden_topology(
    usage_probabilities: dict[str, float],
    order_probabilities: Sequence[Sequence[float]],
    *,
    rng: Any | None,
    effect_spec: dict[str, Any] | None = None,
    prior: dict[str, Any] | None = None,
    floor: float = 0.10,
    head_weight: float = 0.62,
    prior_weight: float = 0.28,
) -> tuple[str, ...]:
    import numpy as np

    effect_spec = effect_spec or effects_config()
    prior = prior or chain_config()
    effects = tuple(effect_spec["effect_order"])
    minimum = max(1, min(int(prior["minimum_effects"]), len(effects)))
    maximum = max(minimum, min(int(prior["maximum_effects"]), len(effects)))
    order = np.asarray(order_probabilities, dtype=np.float64)
    if order.shape != (maximum, len(effects) + 1):
        raise ValueError(
            f"order probability shape {order.shape} != {(maximum, len(effects) + 1)}"
        )
    usage = np.asarray([usage_probabilities[effect_id] for effect_id in effects], dtype=np.float64)
    library_prior = np.asarray(
        [prior["activation_probabilities"][effect_id] for effect_id in effects],
        dtype=np.float64,
    )
    support = floor + head_weight * np.sqrt(usage) + prior_weight * library_prior
    support = np.clip(support, 1e-8, None)

    selected: list[str] = []
    available = np.ones(len(effects), dtype=bool)
    for slot in range(maximum):
        slot_probability = np.clip(order[slot], 0.0, None)
        effect_mass = float(slot_probability[1:].sum())
        effect_scores = slot_probability[1:] * support
        effect_scores[~available] = 0.0
        if float(effect_scores.sum()) > 0.0:
            effect_scores *= effect_mass / effect_scores.sum()
        stop_score = float(slot_probability[0]) if len(selected) >= minimum else 0.0
        if len(selected) + int(available.sum()) <= minimum:
            stop_score = 0.0
        scores = np.concatenate(([stop_score], effect_scores))
        if not np.isfinite(scores).all() or float(scores.sum()) <= 0.0:
            scores = np.concatenate(([0.0], support * available))
        choice = int(np.argmax(scores)) if rng is None else int(rng.choice(len(effects) + 1, p=scores / scores.sum()))
        if choice == 0:
            break
        effect_index = choice - 1
        selected.append(effects[effect_index])
        available[effect_index] = False

    if len(selected) < minimum:
        remaining = [index for index in range(len(effects)) if available[index]]
        remaining.sort(key=lambda index: support[index], reverse=True)
        selected.extend(effects[index] for index in remaining[: minimum - len(selected)])
    return tuple(selected[:maximum])


def candidate_from_center(
    chain_order: Sequence[str],
    center: Sequence[float],
    effect_spec: dict[str, Any] | None = None,
) -> CandidateSpec:
    import numpy as np

    effect_spec = effect_spec or effects_config()
    center = np.asarray(center, dtype=np.float32)
    order = parameter_order(effect_spec)
    if center.shape != (len(order),):
        raise ValueError(f"parameter center shape {center.shape} != {(len(order),)}")
    active = set(chain_order)
    physical: dict[str, dict[str, float]] = {}
    cursor = 0
    for effect_id in effect_spec["effect_order"]:
        effect_values: dict[str, float] = {}
        for parameter_name, parameter in effect_spec["effects"][effect_id]["parameters"].items():
            if parameter.get("fixed") is None:
                unit = float(center[cursor])
                cursor += 1
                if effect_id in active:
                    effect_values[parameter_name] = unit_to_parameter(parameter, unit)
            elif effect_id in active:
                effect_values[parameter_name] = float(parameter["fixed"])
        if effect_id in active:
            physical[effect_id] = effect_values
    return CandidateSpec(tuple(chain_order), physical)


def sample_head_candidates(
    count: int,
    center: Sequence[float],
    *,
    topology: str,
    rng: Any,
    target_chain: Sequence[str] | None = None,
    usage_probabilities: dict[str, float] | None = None,
    order_probabilities: Sequence[Sequence[float]] | None = None,
) -> list[CandidateSpec]:
    if count <= 0:
        raise ValueError("count must be positive")
    settings = load_config("counterfx.json")["head_proposal"]
    candidates: list[CandidateSpec] = []
    for index in range(count):
        if topology == "known":
            if target_chain is None:
                raise ValueError("target_chain is required for known topology")
            chain = tuple(target_chain)
        elif topology == "hidden":
            if usage_probabilities is None or order_probabilities is None:
                raise ValueError("usage and order probabilities are required for hidden topology")
            chain = decode_hidden_topology(
                usage_probabilities,
                order_probabilities,
                rng=None if index == 0 else rng,
                floor=float(settings["topology_floor"]),
                head_weight=float(settings["topology_head_weight"]),
                prior_weight=float(settings["topology_prior_weight"]),
            )
        else:
            raise ValueError("topology must be 'known' or 'hidden'")
        parameters = center
        if index > 0:
            parameters, _ = logistic_normal_perturb(
                center,
                rng,
                float(settings["scale_min"]),
                float(settings["scale_max"]),
            )
        candidates.append(candidate_from_center(chain, parameters))
    return candidates
