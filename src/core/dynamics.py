from __future__ import annotations

import numpy as np
from .utils import in_window


def _meal_profile(dt: float, peak: float, width: float) -> float:
    if dt < 0 or dt > width:
        return 0.0

    x = dt / width
    p = peak / width

    sigma = 0.20
    gaussian = np.exp(-0.5 * ((x - p) / sigma) ** 2)
    window = max(0.0, np.sin(np.pi * x))

    return float(gaussian * window)


def meal_effect_mgdl_per_step(
    cho_g: float,
    t_min: int,
    meal_time_min: int,
    width_min: int = 180,
    peak_min: int = 60,
    gain: float = 1.95,
) -> float:
    dt = t_min - meal_time_min
    shape = _meal_profile(dt, peak=float(peak_min), width=float(width_min))

    absorption_factor = np.random.normal(1.0, 0.12)
    absorption_factor = float(np.clip(absorption_factor, 0.75, 1.35))

    if np.random.rand() < 0.06:
        absorption_factor *= np.random.uniform(1.10, 1.30)

    return float(gain * cho_g * shape * absorption_factor / 100.0)


def stress_effect_mgdl_per_step(stress_level: float) -> float:
    variability = np.random.uniform(0.85, 1.25)
    return float(0.40 * stress_level * variability)


def activity_effect_mgdl_per_step(
    activity_level: float,
    t_min: int,
    start_hhmm: str | None,
    end_hhmm: str | None,
) -> float:
    if start_hhmm is None or end_hhmm is None:
        return 0.0

    if in_window(t_min, start_hhmm, end_hhmm):
        variability = np.random.uniform(0.85, 1.25)
        return float(-1.55 * activity_level * variability)

    return 0.0