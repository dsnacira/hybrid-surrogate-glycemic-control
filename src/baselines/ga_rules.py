from __future__ import annotations

import numpy as np

from ..optimization.ga import init_population, evolve_population
from ..core.simulator_interface import simulate_day


def _evaluate_candidate(x, scenario, sim_kwargs=None):
    """Evaluate one candidate with safety-aware rules enabled."""
    sim_kwargs = dict(sim_kwargs or {})
    x = np.asarray(x, dtype=float)

    if x.size >= 3:
        return simulate_day(
            basal_rate=float(x[0]),
            insulin_carb_ratio=float(x[1]),
            correction_factor=float(x[2]),
            scenario=scenario,
            safety_action_level=True,
            safety_fitness_level=True,
            **sim_kwargs,
        )

    return simulate_day(
        float(x[0]),
        float(x[1]),
        scenario,
        safety_action_level=True,
        safety_fitness_level=True,
        **sim_kwargs,
    )


def run_ga_rules(
    scenario,
    bounds_lo,
    bounds_hi,
    n_pop,
    n_gen,
    pc,
    pm,
    elitism,
    tournament_k,
    sim_kwargs=None,
) -> dict:
    """
    GA + safety/rule constraints baseline.

    Robust for simglucose migration:
      - supports 3D chromosome
      - enables safety_action_level and safety_fitness_level
      - stores best_out without re-simulation
      - returns monotone best_true_history
    """
    sim_kwargs = dict(sim_kwargs or {})
    bounds_lo = np.asarray(bounds_lo, dtype=float)
    bounds_hi = np.asarray(bounds_hi, dtype=float)

    n_pop = int(n_pop)
    n_gen = int(n_gen)

    pop = init_population(n_pop, bounds_lo, bounds_hi)

    best_hist_gen: list[float] = []
    best_solution = None
    best_fitness = -1e18
    best_out = None
    sim_calls = 0

    for _g in range(n_gen):
        fitness = np.full(n_pop, -1e18, dtype=float)
        outs = [None] * n_pop

        for i in range(n_pop):
            x = pop[i]
            out = _evaluate_candidate(x, scenario, sim_kwargs=sim_kwargs)
            sim_calls += 1

            f = float(out.get("fitness", -1e18))
            fitness[i] = f
            outs[i] = out

        gen_best_idx = int(np.argmax(fitness))
        gen_best_fit = float(fitness[gen_best_idx])
        best_hist_gen.append(gen_best_fit)

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_solution = pop[gen_best_idx].copy()
            best_out = outs[gen_best_idx]

        pop = evolve_population(
            pop, fitness, bounds_lo, bounds_hi, pc, pm, elitism, tournament_k
        )

    best_history = np.asarray(best_hist_gen, dtype=float)
    best_true_history = np.maximum.accumulate(best_history) if best_history.size else best_history

    return {
        "best_solution": best_solution,
        "best_fitness": float(best_fitness),
        "best_history": best_history,
        "best_solution_true": best_solution,
        "best_true_solution": best_solution,
        "best_true_fitness": float(best_fitness),
        "best_true_history": best_true_history,
        "best_out": best_out,
        "sim_calls": int(sim_calls),
        "dataset_size": int(sim_calls),
    }

