from __future__ import annotations

import numpy as np

from .ga import init_population, evolve_population
from ..core.simulator_interface import simulate_day
from ..surrogate.model import SurrogateModel


def _evaluate_candidate(
    x,
    scenario,
    sim_kwargs=None,
    safety_action_level: bool = False,
    safety_fitness_level: bool = False,
):
    sim_kwargs = dict(sim_kwargs or {})
    x = np.asarray(x, dtype=float)

    if x.size >= 3:
        return simulate_day(
            basal_rate=float(x[0]),
            insulin_carb_ratio=float(x[1]),
            correction_factor=float(x[2]),
            scenario=scenario,
            safety_action_level=safety_action_level,
            safety_fitness_level=safety_fitness_level,
            **sim_kwargs,
        )

    return simulate_day(
        float(x[0]),
        float(x[1]),
        scenario,
        safety_action_level=safety_action_level,
        safety_fitness_level=safety_fitness_level,
        **sim_kwargs,
    )


def _safe_int(value, default: int, min_value: int = 0) -> int:
    try:
        v = int(value)
    except Exception:
        v = int(default)
    return max(int(min_value), v)


def _select_real_evaluation_indices(
    pred: np.ndarray,
    n_pop: int,
    top_k_real_eval: int | None = None,
    random_real_eval: int = 0,
    rho: float = 0.5,
    elite_indices: list[int] | None = None,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).ravel()
    if pred.size != n_pop:
        raise ValueError("Prediction size does not match population size.")

    selected: set[int] = set()

    if elite_indices is not None:
        for idx in elite_indices:
            if 0 <= int(idx) < n_pop:
                selected.add(int(idx))

    if top_k_real_eval is None:
        rho = float(np.clip(rho, 0.0, 1.0))
        n_top = int(np.ceil(rho * n_pop))
    else:
        n_top = int(top_k_real_eval)

    n_top = max(1, min(n_pop, n_top))

    ranked = np.argsort(pred, kind="mergesort")
    top_idx = ranked[-n_top:]

    for idx in top_idx:
        selected.add(int(idx))

    remaining = [i for i in range(n_pop) if i not in selected]

    random_real_eval = max(0, int(random_real_eval))
    if random_real_eval > 0 and remaining:
        n_rand = min(random_real_eval, len(remaining))
        rand_idx = np.random.choice(remaining, size=n_rand, replace=False)
        for idx in rand_idx:
            selected.add(int(idx))

    return np.asarray(sorted(selected), dtype=int)


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
    rho: float = 0.5,
    warmup_generations: int = 3,
    retrain_period: int = 1,
    surrogate_type: str = "random_forest",
    top_k_real_eval: int | None = None,
    random_real_eval: int | None = None,
    sim_kwargs: dict | None = None,
) -> dict:
    """
    Robust hybrid surrogate-assisted GA.

    Warm-up:
      evaluate the full population using the high-fidelity simulator.

    Hybrid phase:
      surrogate predicts all candidates, then only selected candidates are
      evaluated by the high-fidelity simulator:
        - top_k_real_eval best predicted candidates
        - random_real_eval exploratory candidates
        - previous elite candidates when possible

    Best solution is tracked only from true simulations.
    """

    sim_kwargs = dict(sim_kwargs or {})

    bounds_lo = np.asarray(bounds_lo, dtype=float)
    bounds_hi = np.asarray(bounds_hi, dtype=float)

    n_pop = int(n_pop)
    n_gen = int(n_gen)
    pc = float(pc)
    pm = float(pm)
    elitism = int(elitism)
    tournament_k = int(tournament_k)

    rho = float(np.clip(rho, 0.0, 1.0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    warmup_generations = max(1, int(warmup_generations))
    retrain_period = int(retrain_period)

    if top_k_real_eval is not None:
        top_k_real_eval = _safe_int(
            top_k_real_eval,
            default=max(1, int(np.ceil(rho * n_pop))),
            min_value=1,
        )
        top_k_real_eval = min(top_k_real_eval, n_pop)

    if random_real_eval is not None:
        random_real_eval = _safe_int(random_real_eval, default=0, min_value=0)

    pop = init_population(n_pop, bounds_lo, bounds_hi)
    surrogate = SurrogateModel(model_type=surrogate_type)

    D_X: list[np.ndarray] = []
    D_y: list[float] = []
    pred_true_pairs: list[tuple[float, float]] = []

    sim_calls = 0

    best_true_fitness = -1e18
    best_true_solution = None
    best_out = None

    best_true_history: list[float] = []
    best_history: list[float] = []

    last_elite_solutions: list[np.ndarray] = []

    for g in range(n_gen):
        fitness_sel = np.full(n_pop, -1e18, dtype=float)
        true_fits_this_gen: list[float] = []

        model_ready = getattr(surrogate, "model", None) is not None

        if g < warmup_generations or not model_ready:
            selected_idx = np.arange(n_pop, dtype=int)
            pred = None
        else:
            pred = np.asarray(surrogate.predict(pop), dtype=float)

            elite_indices: list[int] = []
            if last_elite_solutions:
                for elite_x in last_elite_solutions:
                    distances = np.linalg.norm(pop - elite_x.reshape(1, -1), axis=1)
                    elite_indices.append(int(np.argmin(distances)))

            if random_real_eval is None:
                if top_k_real_eval is None:
                    n_top_tmp = max(1, min(n_pop, int(np.ceil(rho * n_pop))))
                else:
                    n_top_tmp = int(top_k_real_eval)
                remaining_count = max(0, n_pop - n_top_tmp)
                random_eval_this_gen = int(np.ceil(alpha * remaining_count))
            else:
                random_eval_this_gen = int(random_real_eval)

            selected_idx = _select_real_evaluation_indices(
                pred=pred,
                n_pop=n_pop,
                top_k_real_eval=top_k_real_eval,
                random_real_eval=random_eval_this_gen,
                rho=rho,
                elite_indices=elite_indices,
            )

        for idx in selected_idx:
            i = int(idx)
            x = pop[i]

            out = _evaluate_candidate(
                x,
                scenario,
                sim_kwargs=sim_kwargs,
                safety_action_level=True,
                safety_fitness_level=True,
            )

            sim_calls += 1

            f_true = float(out.get("fitness", -1e18))
            if not np.isfinite(f_true):
                f_true = -1e18

            fitness_sel[i] = f_true
            true_fits_this_gen.append(f_true)

            D_X.append(x.copy())
            D_y.append(f_true)

            if pred is not None:
                pred_true_pairs.append((float(pred[i]), f_true))

            if f_true > best_true_fitness:
                best_true_fitness = f_true
                best_true_solution = x.copy()
                best_out = out

        min_train_size = max(5, bounds_lo.size + 2)

        should_train = False
        if len(D_y) >= min_train_size:
            if g < warmup_generations:
                should_train = True
            elif retrain_period > 0 and (g % retrain_period == 0):
                should_train = True

        if should_train:
            surrogate.fit(
                np.asarray(D_X, dtype=float),
                np.asarray(D_y, dtype=float),
            )

        if true_fits_this_gen:
            gen_best_true = float(np.max(true_fits_this_gen))
        else:
            gen_best_true = best_true_fitness

        prev_best = best_true_history[-1] if best_true_history else -1e18
        best_true_history.append(max(prev_best, gen_best_true))
        best_history.append(best_true_history[-1])

        evaluated_indices = np.where(np.isfinite(fitness_sel) & (fitness_sel > -1e17))[0]

        last_elite_solutions = []
        if evaluated_indices.size > 0:
            n_elite_store = max(1, min(int(elitism), evaluated_indices.size))
            local_order = evaluated_indices[
                np.argsort(fitness_sel[evaluated_indices], kind="mergesort")
            ]
            elite_idx = local_order[-n_elite_store:]
            last_elite_solutions = [pop[int(i)].copy() for i in elite_idx]

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

    pred_true_pairs_arr = np.asarray(pred_true_pairs, dtype=float)

    if pred_true_pairs_arr.size == 0:
        pred_true_pairs_arr = pred_true_pairs_arr.reshape(0, 2)
        surrogate_pred = np.asarray([], dtype=float)
        surrogate_true = np.asarray([], dtype=float)
    else:
        surrogate_pred = pred_true_pairs_arr[:, 0].copy()
        surrogate_true = pred_true_pairs_arr[:, 1].copy()

    best_true_history_arr = np.asarray(best_true_history, dtype=float)
    if best_true_history_arr.size > 0:
        best_true_history_arr = np.maximum.accumulate(best_true_history_arr)

    return {
        "best_solution": best_true_solution,
        "best_fitness": float(best_true_fitness),
        "best_history": np.asarray(best_history, dtype=float),

        "best_solution_true": best_true_solution,
        "best_true_solution": best_true_solution,
        "best_true_fitness": float(best_true_fitness),
        "best_true_history": best_true_history_arr,

        "best_out": best_out,
        "dataset_size": int(len(D_y)),
        "sim_calls": int(sim_calls),

        "pred_true_pairs": pred_true_pairs_arr,
        "surrogate_pred": surrogate_pred,
        "surrogate_true": surrogate_true,

        "top_k_real_eval": int(top_k_real_eval) if top_k_real_eval is not None else -1,
        "random_real_eval": int(random_real_eval) if random_real_eval is not None else -1,
        "rho": float(rho),
        "alpha": float(alpha),
        "warmup_generations": int(warmup_generations),
        "retrain_period": int(retrain_period),
    }

