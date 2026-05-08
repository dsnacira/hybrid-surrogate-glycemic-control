from __future__ import annotations

import numpy as np

from ..optimization.ga import init_population, evolve_population
from ..core.simulator_interface import simulate_day
from ..surrogate.model import SurrogateModel


def _evaluate_candidate(x, scenario, sim_kwargs=None):
    """Evaluate one GA candidate with robust support for 2D legacy or 3D simglucose chromosomes."""
    sim_kwargs = dict(sim_kwargs or {})
    x = np.asarray(x, dtype=float)

    if x.size >= 3:
        return simulate_day(
            basal_rate=float(x[0]),
            insulin_carb_ratio=float(x[1]),
            correction_factor=float(x[2]),
            scenario=scenario,
            **sim_kwargs,
        )

    return simulate_day(float(x[0]), float(x[1]), scenario, **sim_kwargs)


def run_ga_surrogate(
    scenario,
    bounds_lo,
    bounds_hi,
    n_pop,
    n_gen,
    pc,
    pm,
    elitism,
    tournament_k,
    surrogate_type: str = "random_forest",
    warmup_generations: int = 2,
    retrain_period: int = 5,
    rho: float = 0.30,
    sim_kwargs=None,
) -> dict:
    """
    GA + surrogate baseline.

    Robust points:
      - supports 3D simglucose chromosome
      - warm-up uses true simulations
      - surrogate ranks candidates after warm-up
      - only top rho candidates are truly simulated
      - best solution is tracked only from true simulations
      - returns pred/true pairs for surrogate accuracy plots
    """
    sim_kwargs = dict(sim_kwargs or {})
    bounds_lo = np.asarray(bounds_lo, dtype=float)
    bounds_hi = np.asarray(bounds_hi, dtype=float)

    n_pop = int(n_pop)
    n_gen = int(n_gen)
    warmup_generations = max(1, int(warmup_generations))
    retrain_period = int(retrain_period)
    rho = float(np.clip(rho, 0.0, 1.0))

    pop = init_population(n_pop, bounds_lo, bounds_hi)
    surrogate = SurrogateModel(model_type=surrogate_type)

    X_data: list[np.ndarray] = []
    y_data: list[float] = []
    pred_true_pairs: list[tuple[float, float]] = []

    sim_calls = 0
    best_hist_sel: list[float] = []
    best_hist_true: list[float] = []

    best_solution_true = None
    best_true_fitness = -1e18
    best_true_out = None

    for g in range(n_gen):
        fitness_sel = np.full(n_pop, -1e18, dtype=float)
        fitness_true = np.full(n_pop, np.nan, dtype=float)
        outs = [None] * n_pop

        model_ready = getattr(surrogate, "model", None) is not None

        if g < warmup_generations or not model_ready:
            selected_idx = np.arange(n_pop)
            pred = None
        else:
            pred = np.asarray(surrogate.predict(pop), dtype=float)
            fitness_sel[:] = pred
            n_sim = int(np.ceil(rho * n_pop))
            n_sim = max(1, min(n_pop, n_sim))
            selected_idx = np.argsort(pred, kind="mergesort")[-n_sim:]

        for i in selected_idx:
            x = pop[int(i)]
            out = _evaluate_candidate(x, scenario, sim_kwargs=sim_kwargs)
            sim_calls += 1

            f = float(out.get("fitness", -1e18))
            fitness_true[int(i)] = f
            fitness_sel[int(i)] = f
            outs[int(i)] = out

            X_data.append(x.copy())
            y_data.append(f)

            if pred is not None:
                pred_true_pairs.append((float(pred[int(i)]), f))

            if f > best_true_fitness:
                best_true_fitness = f
                best_solution_true = x.copy()
                best_true_out = out

        if len(y_data) >= max(5, bounds_lo.size + 2):
            if g < warmup_generations or retrain_period <= 0 or (g % retrain_period == 0):
                surrogate.fit(np.asarray(X_data, dtype=float), np.asarray(y_data, dtype=float))

        best_hist_sel.append(float(np.max(fitness_sel)))

        if np.any(np.isfinite(fitness_true)):
            gen_best_true = float(np.nanmax(fitness_true))
        else:
            gen_best_true = best_true_fitness

        prev = best_hist_true[-1] if best_hist_true else -1e18
        best_hist_true.append(max(prev, gen_best_true))

        pop = evolve_population(pop, fitness_sel, bounds_lo, bounds_hi, pc, pm, elitism, tournament_k)

    best_true_history = np.maximum.accumulate(np.asarray(best_hist_true, dtype=float))
    best_history = np.asarray(best_hist_sel, dtype=float)

    pairs = np.asarray(pred_true_pairs, dtype=float)
    if pairs.size == 0:
        pairs = pairs.reshape(0, 2)
        surrogate_pred = np.asarray([], dtype=float)
        surrogate_true = np.asarray([], dtype=float)
    else:
        surrogate_pred = pairs[:, 0].copy()
        surrogate_true = pairs[:, 1].copy()

    return {
        "best_solution": best_solution_true,
        "best_fitness": float(best_true_fitness),
        "best_history": best_history,
        "best_solution_true": best_solution_true,
        "best_true_solution": best_solution_true,
        "best_true_fitness": float(best_true_fitness),
        "best_true_history": best_true_history,
        "best_out": best_true_out,
        "sim_calls": int(sim_calls),
        "dataset_size": int(len(y_data)),
        "pred_true_pairs": pairs,
        "surrogate_pred": surrogate_pred,
        "surrogate_true": surrogate_true,
    }

