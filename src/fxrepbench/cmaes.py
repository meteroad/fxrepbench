from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .candidates import CandidateSpec
from .config import chain_config, effects_config
from .head_proposals import candidate_from_center, parameter_order


@dataclass(frozen=True)
class SearchLayout:
    topology: str
    dimension: int
    active_parameter_indices: tuple[int, ...]
    effect_count: int
    parameter_count: int


def make_layout(topology: str, target_chain: Sequence[str] | None = None) -> SearchLayout:
    effect_spec = effects_config()
    effects = tuple(effect_spec["effect_order"])
    parameters = parameter_order(effect_spec)
    if topology == "known":
        if target_chain is None:
            raise ValueError("target_chain is required for known topology")
        active = set(target_chain)
        indices = tuple(
            index for index, (effect_id, _) in enumerate(parameters) if effect_id in active
        )
        if not indices:
            raise ValueError("known target chain has no trainable controls")
        return SearchLayout(topology, len(indices), indices, len(effects), len(parameters))
    if topology != "hidden":
        raise ValueError("topology must be 'known' or 'hidden'")
    return SearchLayout(
        topology,
        2 * len(effects) + len(parameters),
        tuple(range(len(parameters))),
        len(effects),
        len(parameters),
    )


def initial_solution(layout: SearchLayout) -> Any:
    import numpy as np

    if layout.topology == "known":
        return np.full(layout.dimension, 0.5, dtype=np.float64)
    effect_spec = effects_config()
    prior = chain_config()
    effects = tuple(effect_spec["effect_order"])
    activation = np.asarray(
        [prior["activation_probabilities"][effect_id] for effect_id in effects],
        dtype=np.float64,
    )
    template = tuple(prior["order_templates"][0])
    denominator = max(1, len(template) - 1)
    order = np.asarray(
        [template.index(effect_id) / denominator for effect_id in effects],
        dtype=np.float64,
    )
    parameters = np.full(layout.parameter_count, 0.5, dtype=np.float64)
    return np.concatenate((activation, order, parameters))


def decode_hidden_chain(
    solution: Sequence[float],
    *,
    activation_threshold: float = 0.5,
) -> tuple[str, ...]:
    import numpy as np

    effect_spec = effects_config()
    prior = chain_config()
    effects = tuple(effect_spec["effect_order"])
    values = np.asarray(solution, dtype=np.float64)
    activation = values[: len(effects)]
    order_keys = values[len(effects) : 2 * len(effects)]
    minimum = max(1, min(int(prior["minimum_effects"]), len(effects)))
    maximum = max(minimum, min(int(prior["maximum_effects"]), len(effects)))
    selected = [
        index for index, value in enumerate(activation) if value >= activation_threshold
    ]
    ranked = sorted(range(len(effects)), key=lambda index: (-activation[index], index))
    if len(selected) < minimum:
        selected_set = set(selected)
        for index in ranked:
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
            if len(selected) >= minimum:
                break
    if len(selected) > maximum:
        selected = sorted(selected, key=lambda index: (-activation[index], index))[:maximum]
    selected.sort(key=lambda index: (order_keys[index], index))
    return tuple(effects[index] for index in selected)


def decode_solution(
    solution: Sequence[float],
    layout: SearchLayout,
    *,
    target_chain: Sequence[str] | None = None,
    activation_threshold: float = 0.5,
) -> CandidateSpec:
    import numpy as np

    values = np.clip(np.asarray(solution, dtype=np.float64), 0.0, 1.0)
    if values.shape != (layout.dimension,):
        raise ValueError(f"solution shape {values.shape} != {(layout.dimension,)}")
    if layout.topology == "known":
        if target_chain is None:
            raise ValueError("target_chain is required for known topology")
        full = np.full(layout.parameter_count, 0.5, dtype=np.float32)
        full[np.asarray(layout.active_parameter_indices, dtype=np.int64)] = values
        chain = tuple(target_chain)
    else:
        chain = decode_hidden_chain(values, activation_threshold=activation_threshold)
        full = values[2 * layout.effect_count :].astype(np.float32)
    return candidate_from_center(chain, full)


class CmaEsAskTell:
    """Renderer-agnostic CMA-ES loop matching the paper's search encoding."""

    def __init__(
        self,
        topology: str,
        *,
        target_chain: Sequence[str] | None = None,
        population_size: int = 128,
        sigma0: float = 0.33,
        seed: int = 1,
        activation_threshold: float = 0.5,
    ) -> None:
        import cma

        self.layout = make_layout(topology, target_chain)
        self.target_chain = None if target_chain is None else tuple(target_chain)
        self.activation_threshold = float(activation_threshold)
        options: dict[str, Any] = {
            "bounds": [0.0, 1.0],
            "popsize": int(population_size),
            "seed": int(seed),
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
            "verb_filenameprefix": "",
        }
        if self.layout.dimension == 1:
            options["maxstd"] = float("inf")
        self.strategy = cma.CMAEvolutionStrategy(
            initial_solution(self.layout).tolist(),
            float(sigma0),
            options,
        )

    def ask(self) -> tuple[list[list[float]], list[CandidateSpec]]:
        solutions = self.strategy.ask()
        candidates = [
            decode_solution(
                solution,
                self.layout,
                target_chain=self.target_chain,
                activation_threshold=self.activation_threshold,
            )
            for solution in solutions
        ]
        return solutions, candidates

    def tell(self, solutions: Sequence[Sequence[float]], objectives: Sequence[float]) -> None:
        if len(solutions) != len(objectives):
            raise ValueError("solutions and objectives must have equal lengths")
        self.strategy.tell(solutions, [float(value) for value in objectives])
