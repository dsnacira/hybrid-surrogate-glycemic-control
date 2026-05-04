from __future__ import annotations
import numpy as np

from .ga import init_population, evolve_population
from ..core.simulator import simulate_day
from ..surrogate.model import SurrogateModel


def run_hybrid(
    scenario,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    n_pop: int,
    n_gen: int,
    pc: float,
    pm: float,
    elitism: int,
    tournament_k: int,
    alpha: float = 0.2,
    rho: float = 0.7,
    warmup_generations: int = 6,
    retrain_period: int = 2,
    surrogate_type: str = "random_forest",
    sim_kwargs: dict | None = None,
) -> dict:
    """
    Robust surrogate-assisted GA.

    Key design:
    - Warm-up: full simulation.
    - Hybrid phase: surrogate ranks candidates only.
    - Only top-rho candidates are simulated.
    - GA selection uses true simulated fitness only.
    - Non-simulated candidates receive a very low selection score.
    - Best solution is tracked only from true simulations.
    """

    sim_kwargs = dict(sim_kwargs or {})

    n_pop = int(n_pop)
    n_gen = int(n_gen)
    rho = float(np.clip(rho, 0.0, 1.0))
    warmup_generations = int(warmup_generations)
    retrain_period = int(retrain_period)

    pop = init_population(n_pop, bounds_lo, bounds_hi)

    D_X = []
    D_y = []

    sim_calls = 0
    surrogate = SurrogateModel(model_type=surrogate_type)

    best_true_fitness = -1e18
    best_true_solution = None
    best_out = None

    best_true_history = []
    best_history = []

    pred_true_pairs = []

    for g in range(n_gen):
        fitness_sel = np.full(n_pop, -1e18, dtype=float)
        true_fits_this_gen = []

        # --------------------------------------------------
        # Warm-up phase: simulate full population
        # --------------------------------------------------
        if g < warmup_generations or getattr(surrogate, "model", None) is None:
            for i in range(n_pop):
                x = pop[i]

                out = simulate_day(
                    float(x[0]),
                    float(x[1]),
                    scenario,
                    **sim_kwargs,
                )

                sim_calls += 1
                f_true = float(out["fitness"])

                fitness_sel[i] = f_true
                true_fits_this_gen.append(f_true)

                D_X.append(x.copy())
                D_y.append(f_true)

                if f_true > best_true_fitness:
                    best_true_fitness = f_true
                    best_true_solution = x.copy()
                    best_out = out

            if len(D_y) >= 5:
                surrogate.fit(
                    np.vstack(D_X),
                    np.asarray(D_y, dtype=float),
                )

        # --------------------------------------------------
        # Hybrid phase: surrogate ranks, simulation validates
        # --------------------------------------------------
        else:
            pred = np.asarray(surrogate.predict(pop), dtype=float)

            n_top = int(np.ceil(rho * n_pop))
            n_top = max(1, min(n_pop, n_top))

            top_idx = np.argsort(pred, kind="mergesort")[-n_top:]

            for i in top_idx:
                x = pop[i]

                out = simulate_day(
                    float(x[0]),
                    float(x[1]),
                    scenario,
                    **sim_kwargs,
                )

                sim_calls += 1
                f_true = float(out["fitness"])

                # Only true fitness is used for GA selection
                fitness_sel[i] = f_true
                true_fits_this_gen.append(f_true)

                D_X.append(x.copy())
                D_y.append(f_true)

                pred_true_pairs.append((float(pred[i]), f_true))

                if f_true > best_true_fitness:
                    best_true_fitness = f_true
                    best_true_solution = x.copy()
                    best_out = out

            if retrain_period > 0 and (g % retrain_period == 0):
                if len(D_y) >= 5:
                    surrogate.fit(
                        np.vstack(D_X),
                        np.asarray(D_y, dtype=float),
                    )

        # --------------------------------------------------
        # Histories
        # --------------------------------------------------
        if true_fits_this_gen:
            gen_best_true = float(np.max(true_fits_this_gen))
        else:
            gen_best_true = best_true_fitness

        if len(best_true_history) == 0:
            best_true_history.append(gen_best_true)
        else:
            best_true_history.append(
                max(best_true_history[-1], gen_best_true)
            )

        best_history.append(best_true_history[-1])

        # --------------------------------------------------
        # Evolve population using true simulated candidates only
        # --------------------------------------------------
        pop = evolve_population(
            pop,
            fitness_sel,
            bounds_lo,
            bounds_hi,
            pc,
            pm,
            elitism,
            tournament_k,
        )

    pred_true_pairs = np.asarray(pred_true_pairs, dtype=float)

    if pred_true_pairs.size == 0:
        pred_true_pairs = pred_true_pairs.reshape(0, 2)
        surrogate_pred = np.asarray([], dtype=float)
        surrogate_true = np.asarray([], dtype=float)
    else:
        surrogate_pred = pred_true_pairs[:, 0]
        surrogate_true = pred_true_pairs[:, 1]

    return {
        "best_solution": best_true_solution,
        "best_fitness": float(best_true_fitness),
        "best_history": np.asarray(best_history, dtype=float),

        "best_true_solution": best_true_solution,
        "best_true_fitness": float(best_true_fitness),
        "best_true_history": np.asarray(best_true_history, dtype=float),
        "best_out": best_out,

        "dataset_size": int(len(D_y)),
        "sim_calls": int(sim_calls),

        "pred_true_pairs": pred_true_pairs,
        "surrogate_pred": surrogate_pred,
        "surrogate_true": surrogate_true,
    }