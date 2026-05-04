from __future__ import annotations

import numpy as np


def apply_fitness_penalties(
    fitness: float,
    glucose,
    scenario_name: str,
    lambda_extreme: float = 600.0,
    lambda_severe: float = 300.0,
    lambda_hypo_duration: float = 3.0,
    lambda_hyper_duration: float = 0.5,
) -> float:
    """
    Multi-level safety penalties applied after simulation.

    Penalizes:
    - severe hypoglycemia: min(G) < 54 mg/dL
    - extreme numerical hypoglycemia: min(G) <= 40 mg/dL
    - prolonged time below range: G < 70 mg/dL
    - prolonged time above range: G > 180 mg/dL
    """

    g = np.asarray(glucose, dtype=float)

    if g.size == 0:
        return float(fitness)

    n = len(g)
    g_min = float(np.min(g))
    g_max = float(np.max(g))

    tbr70 = float(np.sum(g < 70.0) / n)
    tbr54 = float(np.sum(g < 54.0) / n)
    tar180 = float(np.sum(g > 180.0) / n)

    penalized_fitness = float(fitness)

    # Severe hypoglycemia
    if g_min < 54.0:
        penalized_fitness -= lambda_severe

    # Extreme numerical / physiological lower bound
    if g_min <= 40.0:
        penalized_fitness -= lambda_extreme

    # Duration-based hypoglycemia penalty
    penalized_fitness -= lambda_hypo_duration * tbr70 * 100.0

    # Extra penalty for severe hypo duration
    penalized_fitness -= 1.5 * tbr54 * 100.0

    # Mild hyperglycemia duration penalty
    penalized_fitness -= lambda_hyper_duration * tar180 * 100.0

    # Optional scenario-specific conservatism
    name = str(scenario_name).lower()

    if ("exercise" in name or "lowcarb" in name) and g_min < 70.0:
        penalized_fitness -= 150.0

    if ("ramadan" in name or "fasting" in name) and g_min < 70.0:
        penalized_fitness -= 150.0

    if ("stress" in name or "highcarb" in name) and g_max > 220.0:
        penalized_fitness -= 100.0

    return float(penalized_fitness)


def clamp_actions(
    ub: float,
    ubol: float,
    glucose_now: float,
    iob: float,
    scenario_name: str,
) -> tuple[float, float]:
    """
    Action-level rule-based safety supervision.

    This function is called before glucose update. It prevents obviously unsafe
    insulin delivery configurations, especially under low glucose, fasting,
    exercise, or high insulin-on-board conditions.

    Parameters
    ----------
    ub:
        Basal insulin action.
    ubol:
        Bolus insulin action.
    glucose_now:
        Current glucose value.
    iob:
        Insulin-on-board proxy.
    scenario_name:
        Scenario identifier.

    Returns
    -------
    tuple[float, float]
        Safety-adjusted basal and bolus actions.
    """

    ub = float(ub)
    ubol = float(ubol)
    glucose_now = float(glucose_now)
    iob = float(iob)
    name = str(scenario_name).lower()

    # Never allow negative insulin actions
    ub = max(0.0, ub)
    ubol = max(0.0, ubol)

    # Severe low-glucose protection
    if glucose_now < 70.0:
        ub = 0.0
        ubol = 0.0
        return ub, ubol

    # Moderate low-glucose protection
    if glucose_now < 80.0:
        ub *= 0.3
        ubol = 0.0

    # Conservative action under near-low glucose
    elif glucose_now < 90.0:
        ub *= 0.5
        ubol = 0.0

    # Anti-stacking rule
    if iob > 5.0 and glucose_now < 100.0:
        ubol = 0.0
        ub *= 0.5

    # Exercise and low-carbohydrate scenarios: higher hypo risk
    if "exercise" in name or "lowcarb" in name:
        if glucose_now < 110.0:
            ub *= 0.5
            ubol = 0.0

    # Ramadan / fasting scenarios: conservative daytime insulin
    if "ramadan" in name or "fasting" in name:
        if glucose_now < 110.0:
            ub *= 0.5
            ubol = 0.0

    # Final numerical clamp
    ub = max(0.0, ub)
    ubol = max(0.0, ubol)

    return ub, ubol