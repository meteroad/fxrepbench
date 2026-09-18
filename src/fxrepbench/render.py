from __future__ import annotations

from typing import Any


def make_plugin(effect_id: str, params: dict[str, float]) -> Any:
    from pedalboard import (
        Chorus,
        Compressor,
        Delay,
        Distortion,
        HighpassFilter,
        LowpassFilter,
        PeakFilter,
        Reverb,
    )

    constructors = {
        "highpass_filter": lambda: HighpassFilter(
            cutoff_frequency_hz=float(params["cutoff_frequency_hz"])
        ),
        "lowpass_filter": lambda: LowpassFilter(
            cutoff_frequency_hz=float(params["cutoff_frequency_hz"])
        ),
        "peak_filter": lambda: PeakFilter(
            cutoff_frequency_hz=float(params["cutoff_frequency_hz"]),
            gain_db=float(params["gain_db"]),
            q=float(params["q"]),
        ),
        "compressor": lambda: Compressor(
            threshold_db=float(params["threshold_db"]),
            ratio=float(params["ratio"]),
            attack_ms=float(params["attack_ms"]),
            release_ms=float(params["release_ms"]),
        ),
        "distortion": lambda: Distortion(drive_db=float(params["drive_db"])),
        "chorus": lambda: Chorus(
            rate_hz=float(params["rate_hz"]),
            depth=float(params["depth"]),
            centre_delay_ms=float(params["centre_delay_ms"]),
            feedback=float(params["feedback"]),
            mix=float(params["mix"]),
        ),
        "delay": lambda: Delay(
            delay_seconds=float(params["delay_seconds"]),
            feedback=float(params["feedback"]),
            mix=float(params["mix"]),
        ),
        "reverb": lambda: Reverb(
            room_size=float(params["room_size"]),
            damping=float(params["damping"]),
            wet_level=float(params["wet_level"]),
            dry_level=float(params["dry_level"]),
            width=float(params["width"]),
            freeze_mode=float(params["freeze_mode"]),
        ),
    }
    try:
        return constructors[effect_id]()
    except KeyError as exc:
        raise KeyError(f"unsupported effect {effect_id!r}") from exc


def render_chain(
    audio: "Any",
    chain_order: list[str],
    physical_params: dict[str, dict[str, float]],
    sample_rate: int,
    *,
    buffer_size: int = 8192,
) -> "Any":
    import numpy as np
    from pedalboard import Pedalboard

    plugins = [make_plugin(effect_id, physical_params[effect_id]) for effect_id in chain_order]
    board = Pedalboard(plugins)
    contiguous = np.ascontiguousarray(audio.astype(np.float32, copy=False))
    rendered = board.process(
        contiguous,
        float(sample_rate),
        buffer_size=buffer_size,
        reset=True,
    )
    return np.asarray(rendered, dtype=np.float32)

