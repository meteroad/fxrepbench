from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .manifests import repository_root


def load_config(name: str, root: Path | None = None) -> dict[str, Any]:
    base = root or repository_root()
    return json.loads(
        (base / "benchmarks" / "counterfx" / "configs" / name).read_text(
            encoding="utf-8"
        )
    )


def effects_config(root: Path | None = None) -> dict[str, Any]:
    return load_config("effects.json", root)


def chain_config(root: Path | None = None) -> dict[str, Any]:
    return load_config("chain_prior.json", root)


def metric_config(root: Path | None = None) -> dict[str, Any]:
    return load_config("metric.json", root)


def parameter_to_unit(spec: dict[str, Any], value: float) -> float:
    fixed = spec.get("fixed")
    if fixed is not None:
        return 0.0
    minimum = float(spec["prior_min"])
    maximum = float(spec["prior_max"])
    if spec["distribution"] == "log":
        numerator = math.log(max(float(value), minimum)) - math.log(minimum)
        denominator = math.log(maximum) - math.log(minimum)
    else:
        numerator = float(value) - minimum
        denominator = maximum - minimum
    return min(1.0, max(0.0, numerator / max(denominator, 1e-12)))


def unit_to_parameter(spec: dict[str, Any], unit: float) -> float:
    fixed = spec.get("fixed")
    if fixed is not None:
        return float(fixed)
    unit = min(1.0, max(0.0, float(unit)))
    minimum = float(spec["prior_min"])
    maximum = float(spec["prior_max"])
    if spec["distribution"] == "log":
        return float(math.exp(math.log(minimum) + unit * (math.log(maximum) - math.log(minimum))))
    return float(minimum + unit * (maximum - minimum))


def sample_parameter(spec: dict[str, Any], rng: Any) -> tuple[float, float]:
    if spec.get("fixed") is not None:
        return float(spec["fixed"]), 0.0
    unit = float(rng.random())
    return unit_to_parameter(spec, unit), unit
