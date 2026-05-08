from __future__ import annotations

import math
import warnings
from datetime import datetime
from typing import Any

import numpy as np


# =========================================================
# Robust simglucose / UVA-Padova wrapper
# Public API:
#     simulate_day_simglucose(...)
#
# Decision variables:
#     x = [basal_rate, insulin_carb_ratio, correction_factor]
#
# IMPORTANT UNIT CONVENTION:
#     basal_rate is optimized in U/hour
#     simglucose Action.basal expects U/min
#     therefore basal_u_min = basal_rate / 60
# =========================================================


# =========================================================
# Utility functions
# =========================================================

def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _scenario_get(scenario: Any, key: str, default: Any = None) -> Any:
    if isinstance(scenario, dict):
        return scenario.get(key, default)
    return getattr(scenario, key, default)


def _time_to_minutes(t: Any) -> int:
    if t is None:
        return 0

    if isinstance(t, (int, float, np.integer, np.floating)):
        return int(t)

    if isinstance(t, str):
        s = t.strip()
        if ":" in s:
            h, m = s.split(":", 1)
            return int(h) * 60 + int(m)
        return int(float(s))

    return 0


def _normalize_meals(scenario: Any) -> list[dict[str, float]]:
    meals = _scenario_get(scenario, "meals", []) or []
    out: list[dict[str, float]] = []

    for meal in meals:
        if isinstance(meal, dict):
            t = meal.get("time", meal.get("t", meal.get("minute", meal.get("minutes", 0))))
            cho = meal.get("cho_g", meal.get("carbs", meal.get("CHO", meal.get("amount", 0))))
        elif isinstance(meal, (list, tuple)) and len(meal) >= 2:
            t, cho = meal[0], meal[1]
        else:
            continue

        cho = _safe_float(cho, 0.0)
        if cho <= 0.0:
            continue

        out.append({
            "time_min": float(_time_to_minutes(t)),
            "cho_g": float(cho),
        })

    return out


# =========================================================
# Robust observation parsing
# =========================================================

def _extract_glucose(obj: Any) -> float | None:
    """
    Extract CGM/BG from multiple possible simglucose/Gym formats.
    """

    if obj is None:
        return None

    if isinstance(obj, tuple) and len(obj) > 0:
        for item in obj:
            val = _extract_glucose(item)
            if val is not None:
                return val
        return None

    if isinstance(obj, list):
        for item in obj:
            val = _extract_glucose(item)
            if val is not None:
                return val
        return None

    if isinstance(obj, dict):
        for key in ("CGM", "cgm", "BG", "bg", "glucose", "Glucose"):
            if key in obj:
                try:
                    v = float(obj[key])
                    if math.isfinite(v):
                        return v
                except Exception:
                    pass

        for value in obj.values():
            val = _extract_glucose(value)
            if val is not None:
                return val

        return None

    for attr in ("CGM", "cgm", "BG", "bg", "glucose", "Glucose"):
        if hasattr(obj, attr):
            try:
                v = float(getattr(obj, attr))
                if math.isfinite(v):
                    return v
            except Exception:
                pass

    fields = getattr(obj, "_fields", None)
    if fields:
        for f in fields:
            if str(f).lower() in ("cgm", "bg", "glucose"):
                try:
                    v = float(getattr(obj, f))
                    if math.isfinite(v):
                        return v
                except Exception:
                    pass

    try:
        for key in ("CGM", "cgm", "BG", "bg", "glucose", "Glucose"):
            v = float(obj[key])
            if math.isfinite(v):
                return v
    except Exception:
        pass

    return None


def _unwrap_reset(reset_out: Any) -> Any:
    """
    Compatible with:
      old Gym: obs
      new Gym/Gymnasium: (obs, info)
    """
    if isinstance(reset_out, tuple) and len(reset_out) >= 1:
        return reset_out[0]
    return reset_out


def _unwrap_step(step_out: Any) -> tuple[Any, float, bool, dict]:
    """
    Compatible with:
      old Gym: obs, reward, done, info
      new Gym/Gymnasium: obs, reward, terminated, truncated, info
    """
    if not isinstance(step_out, tuple):
        raise RuntimeError("env.step(action) did not return a tuple.")

    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, float(reward), bool(done), dict(info or {})

    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
        return obs, float(reward), done, dict(info or {})

    raise RuntimeError(f"Unexpected env.step() output length: {len(step_out)}")


# =========================================================
# Metrics and fitness
# =========================================================

def _compute_metrics(
    G: np.ndarray,
    gmin: float = 70.0,
    gmax: float = 180.0,
) -> dict[str, float]:
    G = np.asarray(G, dtype=float).ravel()
    G = G[np.isfinite(G)]

    if G.size == 0:
        raise RuntimeError("Empty glucose trajectory.")

    tir = 100.0 * np.mean((G >= gmin) & (G <= gmax))
    tbr = 100.0 * np.mean(G < gmin)
    tar = 100.0 * np.mean(G > gmax)

    mean_g = float(np.mean(G))
    std_g = float(np.std(G))
    cv = float(100.0 * std_g / mean_g) if mean_g > 1e-9 else float("nan")

    return {
        "TIR": float(tir),
        "TBR": float(tbr),
        "TAR": float(tar),
        "mean_glucose": mean_g,
        "std_glucose": std_g,
        "CV": cv,
        "Gmin": float(np.min(G)),
        "Gmax": float(np.max(G)),
    }


def _fitness_from_metrics(
    metrics: dict[str, float],
    G: np.ndarray,
    severe_hypo: float = 54.0,
    severe_hyper: float = 250.0,
) -> float:
    """
    Maximization fitness.
    Higher is better.
    """

    fitness = float(metrics["TIR"])
    fitness -= 8.0 * float(metrics["TBR"])
    fitness -= 2.0 * float(metrics["TAR"])
    fitness -= 0.25 * float(metrics["CV"])

    if np.any(G < severe_hypo):
        fitness -= 250.0

    if np.any(G > severe_hyper):
        fitness -= 80.0

    return float(fitness)


def _format_output(
    G: Any,
    metrics: dict[str, float] | None = None,
    fitness: float | None = None,
    backend: str = "unknown",
    **extra: Any,
) -> dict[str, Any]:
    G_arr = np.asarray(G, dtype=float).ravel()
    G_arr = G_arr[np.isfinite(G_arr)]

    if G_arr.size == 0:
        raise RuntimeError("Empty glucose trajectory after formatting.")

    if metrics is None:
        metrics = _compute_metrics(G_arr)

    if fitness is None:
        fitness = _fitness_from_metrics(metrics, G_arr)

    out: dict[str, Any] = {
        "G": G_arr,
        "metrics": metrics,
        "fitness": float(fitness),
        "backend": backend,
    }
    out.update(extra)
    return out


# =========================================================
# Internal fallback simulator
# =========================================================

def _fallback_simulation(
    basal_rate: float,
    insulin_carb_ratio: float,
    correction_factor: float,
    scenario: Any,
    dt_minutes: int = 5,
    horizon_hours: int = 24,
    g0: float = 140.0,
    gmin: float = 70.0,
    gmax: float = 180.0,
    seed: int | None = None,
    safety_action_level: bool = False,
    safety_fitness_level: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """
    Robust internal fallback used only if simglucose fails.
    Do NOT use fallback-generated results as final publication results.
    """

    rng = np.random.default_rng(seed)

    dt_minutes = int(max(1, dt_minutes))
    dt_h = float(dt_minutes) / 60.0
    n_steps = max(2, int(float(horizon_hours) * 60.0 / float(dt_minutes)))

    G = np.zeros(n_steps, dtype=float)
    G[0] = float(g0)

    meals = _normalize_meals(scenario)
    meal_signal = np.zeros(n_steps, dtype=float)

    for meal in meals:
        idx = int(meal["time_min"] / float(dt_minutes))
        if 0 <= idx < n_steps:
            cho = float(meal["cho_g"])
            duration_steps = int(4 * 60 / float(dt_minutes))
            for k in range(idx, min(n_steps, idx + duration_steps)):
                tau = (k - idx) * dt_h
                meal_signal[k] += cho * 2.8 * tau * math.exp(-tau / 1.2)

    stress = _safe_float(_scenario_get(scenario, "stress_level", 0.0), 0.0)
    activity = _safe_float(_scenario_get(scenario, "activity_level", 0.0), 0.0)
    noise_sigma = _safe_float(_scenario_get(scenario, "noise_sigma", 2.0), 2.0)

    basal_rate = max(0.01, float(basal_rate))
    insulin_carb_ratio = max(1.0, float(insulin_carb_ratio))
    correction_factor = max(1.0, float(correction_factor))

    insulin_effect = 1.0 * basal_rate
    carb_sensitivity = 45.0 / insulin_carb_ratio
    correction_effect = 80.0 / correction_factor

    for k in range(1, n_steps):
        drift = -0.015 * (G[k - 1] - 115.0)
        meal_effect = carb_sensitivity * meal_signal[k] / 100.0
        insulin = -insulin_effect * dt_h
        correction = -correction_effect * max(G[k - 1] - 160.0, 0.0) * dt_h
        stress_effect = 0.08 * stress
        activity_effect = -0.10 * activity
        noise = rng.normal(0.0, noise_sigma * math.sqrt(dt_h))

        G[k] = G[k - 1] + drift + meal_effect + insulin + correction
        G[k] += stress_effect + activity_effect + noise

        if safety_action_level and G[k] < 75.0:
            G[k] += 4.0

        G[k] = float(np.clip(G[k], 35.0, 450.0))

    metrics = _compute_metrics(G, gmin=gmin, gmax=gmax)
    fitness = _fitness_from_metrics(metrics, G)

    if safety_fitness_level:
        if metrics["Gmin"] < 54.0:
            fitness -= 200.0
        if metrics["Gmax"] > 250.0:
            fitness -= 50.0

    return _format_output(
        G,
        metrics=metrics,
        fitness=fitness,
        backend="fallback_internal",
    )


# =========================================================
# simglucose backend using env.reset / env.step
# =========================================================

def _try_simglucose_simulation(
    basal_rate: float,
    insulin_carb_ratio: float,
    correction_factor: float,
    scenario: Any,
    dt_minutes: int = 5,
    horizon_hours: int = 24,
    g0: float = 140.0,
    gmin: float = 70.0,
    gmax: float = 180.0,
    seed: int | None = None,
    patient_name: str | None = None,
    safety_action_level: bool = False,
    safety_fitness_level: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    UVA/Padova via simglucose.

    Uses env.reset()/env.step(), not user_interface.simulate(), because
    simulate() signatures differ across simglucose versions.
    """

    from simglucose.actuator.pump import InsulinPump
    from simglucose.controller.base import Action
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.scenario import CustomScenario

    patient_name = (
        patient_name
        or _scenario_get(scenario, "patient_name", None)
        or "adolescent#001"
    )

    meals = _normalize_meals(scenario)
    scenario_events = [
        (float(m["time_min"]), float(m["cho_g"]))
        for m in meals
    ]

    start_time = datetime(2020, 1, 1, 0, 0, 0)

    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    pump = InsulinPump.withName("Insulet")
    meal_scenario = CustomScenario(
        start_time=start_time,
        scenario=scenario_events,
    )

    env = T1DSimEnv(
        patient=patient,
        sensor=sensor,
        pump=pump,
        scenario=meal_scenario,
    )

    basal_rate_u_hour = max(0.0, float(basal_rate))
    basal_rate_u_min = basal_rate_u_hour / 60.0

    insulin_carb_ratio = max(1.0, float(insulin_carb_ratio))
    correction_factor = max(1.0, float(correction_factor))

    target = 110.0
    correction_threshold = 150.0

    dt_minutes = int(max(1, dt_minutes))
    n_steps = max(2, int(float(horizon_hours) * 60.0 / float(dt_minutes)))

    meal_by_step: dict[int, float] = {}
    for m in meals:
        idx = int(round(float(m["time_min"]) / float(dt_minutes)))
        if 0 <= idx < n_steps:
            meal_by_step[idx] = meal_by_step.get(idx, 0.0) + float(m["cho_g"])

    glucose_values: list[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        obs = _unwrap_reset(env.reset())
        done = False

        cgm0 = _extract_glucose(obs)
        if cgm0 is not None and math.isfinite(cgm0):
            glucose_values.append(float(cgm0))

        for k in range(n_steps):
            cgm = _extract_glucose(obs)
            if cgm is None or not math.isfinite(cgm):
                cgm = target

            meal_g = float(meal_by_step.get(k, 0.0))

            bolus = 0.0
            if meal_g > 0.0:
                bolus += meal_g / insulin_carb_ratio

            if cgm > correction_threshold:
                bolus += (cgm - target) / correction_factor

            if safety_action_level and cgm < 80.0:
                action = Action(basal=0.0, bolus=0.0)
            else:
                action = Action(
                    basal=float(max(0.0, basal_rate_u_min)),
                    bolus=float(max(0.0, bolus)),
                )

            step_out = env.step(action)
            obs, _reward, done, info = _unwrap_step(step_out)

            cgm_next = _extract_glucose(obs)
            if cgm_next is None:
                cgm_next = _extract_glucose(info)

            if cgm_next is not None and math.isfinite(cgm_next):
                glucose_values.append(float(cgm_next))

            if done:
                break

    G = np.asarray(glucose_values, dtype=float).ravel()
    G = G[np.isfinite(G)]

    if G.size == 0:
        raise RuntimeError(
            "simglucose returned an empty glucose trajectory. "
            "The environment ran, but no CGM/BG value could be extracted."
        )

    metrics = _compute_metrics(G, gmin=gmin, gmax=gmax)
    fitness = _fitness_from_metrics(metrics, G)

    if safety_fitness_level:
        if metrics["Gmin"] < 54.0:
            fitness -= 200.0
        if metrics["Gmax"] > 250.0:
            fitness -= 50.0

    return _format_output(
        G,
        metrics=metrics,
        fitness=fitness,
        backend="simglucose",
        patient_name=patient_name,
        basal_rate_u_hour=basal_rate_u_hour,
        basal_rate_u_min=basal_rate_u_min,
    )


# =========================================================
# Public API used by GA / surrogate / hybrid code
# =========================================================

def simulate_day_simglucose(
    basal_rate: float | None = None,
    insulin_carb_ratio: float | None = None,
    correction_factor: float | None = None,
    scenario: Any | None = None,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Robust public function.

    Supports:
      simulate_day_simglucose(
          basal_rate=...,
          insulin_carb_ratio=...,
          correction_factor=...,
          scenario=...
      )

      simulate_day_simglucose(x0, x1, x2, scenario, ...)

      simulate_day_simglucose(x0, x1, scenario, ...)
      for legacy compatibility.
    """

    remaining = list(args)

    if basal_rate is None and remaining:
        basal_rate = remaining.pop(0)

    if insulin_carb_ratio is None and remaining:
        insulin_carb_ratio = remaining.pop(0)

    if correction_factor is None and remaining:
        third = remaining.pop(0)
        if isinstance(third, (int, float, np.integer, np.floating)):
            correction_factor = third
        else:
            scenario = third
            correction_factor = kwargs.pop("correction_factor", 50.0)

    if scenario is None and remaining:
        scenario = remaining.pop(0)

    if scenario is None:
        scenario = kwargs.pop("scenario", None)

    if scenario is None:
        raise TypeError("scenario is required in simulate_day_simglucose().")

    basal_rate = _safe_float(basal_rate, 0.8)
    insulin_carb_ratio = _safe_float(insulin_carb_ratio, 12.0)
    correction_factor = _safe_float(correction_factor, 50.0)

    try:
        return _try_simglucose_simulation(
            basal_rate=basal_rate,
            insulin_carb_ratio=insulin_carb_ratio,
            correction_factor=correction_factor,
            scenario=scenario,
            **kwargs,
        )

    except Exception as e:
        print("[WARNING] simglucose backend failed; using internal fallback.")
        print(e)

        return _fallback_simulation(
            basal_rate=basal_rate,
            insulin_carb_ratio=insulin_carb_ratio,
            correction_factor=correction_factor,
            scenario=scenario,
            **kwargs,
        )

