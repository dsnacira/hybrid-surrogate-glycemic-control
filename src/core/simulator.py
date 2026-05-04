from __future__ import annotations

import numpy as np

from .metrics import tir_tbr_tar, compute_fitness
from .utils import clamp
from ..safety.rules import apply_fitness_penalties, clamp_actions
from .dynamics import (
    meal_effect_mgdl_per_step,
    stress_effect_mgdl_per_step,
    activity_effect_mgdl_per_step,
)


def simulate_day(
    dose_basal: float,
    insulin_duration_h: float,
    scenario,
    dt_minutes: int = 5,
    horizon_hours: int = 24,
    g0: float = 110.0,
    gmin: float = 40.0,
    gmax: float = 400.0,
    kI: float = 0.025,
    kB: float = 0.002,
    safety_action_level: bool = False,
    safety_fitness_level: bool = True,
) -> dict:

    n_steps = int(horizon_hours * 60 / dt_minutes)

    G = np.zeros(n_steps + 1, dtype=float)
    G[0] = float(g0)

    insulin_duration_h = max(float(insulin_duration_h), 1e-6)

    beta = np.exp(-dt_minutes / (60.0 * insulin_duration_h))
    iob = 0.0

    for k in range(n_steps):
        t_min = k * dt_minutes
        inp = scenario.inputs_at(t_min)

        ub = float(dose_basal)
        ubol = 0.0

        if safety_action_level:
            ub, ubol = clamp_actions(
                ub,
                ubol,
                G[k],
                iob,
                scenario.name,
            )

        insulin_variability = np.random.normal(1.0, 0.03)
        insulin_variability = float(np.clip(insulin_variability, 0.90, 1.10))

        iob = beta * iob + (ub + ubol) * insulin_variability

        dG_meal = 0.0
        for m in inp["meals"]:
            dG_meal += meal_effect_mgdl_per_step(
                cho_g=float(m["cho_g"]),
                t_min=t_min,
                meal_time_min=int(m["time_min"]),
                width_min=180,
                peak_min=60,
                gain=1.95,
            )

        dG_stress = stress_effect_mgdl_per_step(float(inp["stress"]))

        dG_activity = activity_effect_mgdl_per_step(
            float(inp["activity"]),
            t_min,
            inp["activity_start"],
            inp["activity_end"],
        )

        eps = np.random.normal(0.0, float(inp["noise_sigma"]) * 1.25)

        # Rare unannounced physiological disturbances.
        # These events prevent overly smooth trajectories and make the benchmark
        # more discriminative while keeping the simulation numerically stable.
        rare_event = 0.0

        # Occasional upward spike: missed snack, stress pulse, sensor disturbance.
        if np.random.rand() < 0.02:
            rare_event += np.random.uniform(15.0, 40.0)

        # Less frequent downward fluctuation: activity burst or sensitivity increase.
        if np.random.rand() < 0.01:
            rare_event -= np.random.uniform(8.0, 18.0)

        g_next = (
            G[k]
            + dG_meal
            - kI * iob
            + dG_stress
            + dG_activity
            + eps
            + rare_event
        )

        # Weak homeostatic pull toward baseline
        g_next += kB * (g0 - G[k])

        G[k + 1] = clamp(g_next, gmin, gmax)

    mets = tir_tbr_tar(G)
    fitness = compute_fitness(mets, G)

    if safety_fitness_level:
        fitness = apply_fitness_penalties(fitness, G, scenario.name)

    return {
        "G": G,
        "metrics": mets,
        "fitness": float(fitness),
    }