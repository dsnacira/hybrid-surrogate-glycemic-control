from __future__ import annotations

import os
import sys
import argparse
import yaml
import inspect
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.utils import set_seed
from src.core.scenarios import Scenario

from src.baselines.ga_only import run_ga_only
from src.baselines.ga_rules import run_ga_rules
from src.baselines.ga_surrogate import run_ga_surrogate
from src.optimization.hybrid_loop import run_hybrid

from src.analysis.convergence_auc import auc_sum, auc_mean, auc_regret

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METHODS = ["ga_only", "ga_rules", "ga_surrogate", "hybrid"]


# =========================================================
# Basic utilities
# =========================================================

def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] Missing config file: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pretty_metrics(m: dict) -> str:
    return (
        f"TIR={float(m.get('TIR', np.nan)):5.1f}% "
        f"TBR={float(m.get('TBR', np.nan)):5.1f}% "
        f"TAR={float(m.get('TAR', np.nan)):5.1f}%"
    )


def stable_seed(base_seed: int, scenario_name: str, method: str) -> int:
    s = f"{scenario_name}::{method}"
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 2_147_483_647
    return int((int(base_seed) + h) % 2_147_483_647)


def _as_hhmm(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        m = int(x)
        hh, mm = m // 60, m % 60
        return f"{hh:02d}:{mm:02d}"
    return None


def normalize_window(w):
    if w is None:
        return None
    if isinstance(w, dict):
        return {"start": _as_hhmm(w.get("start")), "end": _as_hhmm(w.get("end"))}
    if isinstance(w, (list, tuple)) and len(w) >= 2:
        return {"start": _as_hhmm(w[0]), "end": _as_hhmm(w[1])}
    return None


def ensure_true_history(res: dict) -> dict:
    if "best_true_history" in res and "best_true_fitness" in res:
        h = np.asarray(res["best_true_history"], dtype=float)
        res["best_true_history"] = np.maximum.accumulate(h) if h.size else h
        return res

    if "best_history" in res:
        h = np.asarray(res["best_history"], dtype=float)
        h = np.maximum.accumulate(h) if h.size else h
        res["best_true_history"] = h
        res["best_true_fitness"] = float(np.max(h)) if h.size else float("nan")
        return res

    raise RuntimeError("Result dict missing best_true_history / best_history.")


def get_cfg_param(cfg: dict, *names, default=None):
    for name in names:
        if name in cfg:
            return cfg[name]
    if default is not None:
        return default
    raise KeyError(f"Missing config key. Tried: {names}")


def get_ga_param(cfg_ga: dict, *names, default=None):
    return get_cfg_param(cfg_ga, *names, default=default)


# =========================================================
# Configuration
# =========================================================

def merge_ml_config(cfg_ga: dict, cfg_ml: dict) -> dict:
    """
    Robust ML config merge.

    Priority:
      1. config/ml.yaml
      2. config/ga.yaml sections: surrogate, hybrid
      3. defaults below
    """

    merged = {}

    if isinstance(cfg_ga.get("surrogate"), dict):
        merged.update(cfg_ga.get("surrogate", {}))

    if isinstance(cfg_ga.get("hybrid"), dict):
        merged.update(cfg_ga.get("hybrid", {}))

    if isinstance(cfg_ml, dict):
        merged.update(cfg_ml)

    merged.setdefault("model_type", "random_forest")
    merged.setdefault("alpha", 0.2)
    merged.setdefault("rho", 0.7)
    merged.setdefault("warmup_generations", 6)
    merged.setdefault("retrain_period", 2)

    return merged


def get_bounds(cfg_ga: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Prefer simglucose/UVA-Padova 3D decision vector:
        x = [basal_rate, insulin_carb_ratio, correction_factor]

    Fallback to legacy 2D vector:
        x = [dose_basal, insulin_duration]
    """

    required_3d = [
        "basal_rate_min", "basal_rate_max",
        "carb_ratio_min", "carb_ratio_max",
        "correction_factor_min", "correction_factor_max",
    ]

    if all(k in cfg_ga for k in required_3d):
        lo = np.array([
            cfg_ga["basal_rate_min"],
            cfg_ga["carb_ratio_min"],
            cfg_ga["correction_factor_min"],
        ], dtype=float)

        hi = np.array([
            cfg_ga["basal_rate_max"],
            cfg_ga["carb_ratio_max"],
            cfg_ga["correction_factor_max"],
        ], dtype=float)

        print("[INFO] Using 3D simglucose chromosome: [basal_rate, carb_ratio, correction_factor]")
        return lo, hi

    lo = np.array([
        cfg_ga["dose_basal_min"],
        cfg_ga["insulin_duration_min"],
    ], dtype=float)

    hi = np.array([
        cfg_ga["dose_basal_max"],
        cfg_ga["insulin_duration_max"],
    ], dtype=float)

    print("[WARN] Using legacy 2D chromosome: [dose_basal, insulin_duration]")
    return lo, hi


def make_sim_kwargs(cfg_default: dict) -> dict:
    """
    Build simulator kwargs, compatible with both legacy simulator and
    simglucose wrapper.
    """

    out = {}

    if "dt_minutes" in cfg_default:
        out["dt_minutes"] = int(cfg_default["dt_minutes"])

    if "horizon_hours" in cfg_default:
        out["horizon_hours"] = int(cfg_default["horizon_hours"])
    else:
        out["horizon_hours"] = 24

    if "g0_mgdl" in cfg_default:
        out["g0"] = float(cfg_default["g0_mgdl"])
    if "gmin_mgdl" in cfg_default:
        out["gmin"] = float(cfg_default["gmin_mgdl"])
    if "gmax_mgdl" in cfg_default:
        out["gmax"] = float(cfg_default["gmax_mgdl"])

    for key in [
        "patient_name",
        "patient_group",
        "seed",
        "sample_time",
        "animate",
        "parallel",
    ]:
        if key in cfg_default:
            out[key] = cfg_default[key]

    return out


def build_scenario(scenario_name: str, sc: dict) -> Scenario:
    return Scenario(
        name=scenario_name,
        meals=sc.get("meals", []),
        stress_level=float(sc.get("stress_level", 0.0)),
        activity_level=float(sc.get("activity_level", 0.0)),
        noise_sigma=float(sc.get("noise_sigma", 2.0)),
        fasting=bool(sc.get("fasting", False)),
        fasting_window=normalize_window(sc.get("fasting_window")),
        activity_window=normalize_window(sc.get("activity_window")),
    )


# =========================================================
# Method dispatcher
# =========================================================

def _run_hybrid_compatible(
    scenario,
    lo,
    hi,
    n_pop,
    n_gen,
    pc,
    pm,
    elitism,
    tournament_k,
    cfg_ml,
    sim_kwargs,
):
    """
    Calls run_hybrid robustly.

    If your current hybrid_loop.py already supports:
        top_k_real_eval
        random_real_eval
    they are passed.

    If your old hybrid_loop.py does not support them, they are ignored
    instead of causing a TypeError.
    """

    kwargs = dict(
        alpha=float(cfg_ml.get("alpha", cfg_ml.get("random_real_eval", 0.2))),
        rho=float(cfg_ml.get("rho", cfg_ml.get("surrogate_screening_ratio", 0.7))),
        warmup_generations=int(cfg_ml.get("warmup_generations", cfg_ml.get("warmup_real_evals", 6))),
        retrain_period=int(cfg_ml.get("retrain_period", cfg_ml.get("retrain_frequency", 2))),
        surrogate_type=str(cfg_ml.get("model_type", "random_forest")),
        sim_kwargs=sim_kwargs,
    )

    sig = inspect.signature(run_hybrid)
    params = set(sig.parameters.keys())

    if "top_k_real_eval" in params:
        kwargs["top_k_real_eval"] = cfg_ml.get("top_k_real_eval", None)

    if "random_real_eval" in params:
        kwargs["random_real_eval"] = cfg_ml.get("random_real_eval", None)

    return run_hybrid(
        scenario,
        lo,
        hi,
        n_pop,
        n_gen,
        pc,
        pm,
        int(elitism),
        tournament_k,
        **kwargs,
    )


def run_method(method, scenario, lo, hi, cfg_ga, cfg_ml, sim_kwargs) -> dict:
    n_pop = int(get_ga_param(cfg_ga, "n_pop", "population_size"))
    n_gen = int(get_ga_param(cfg_ga, "n_gen", "n_generations"))

    pc = float(get_ga_param(cfg_ga, "pc", "crossover_rate", default=0.9))
    pm = float(get_ga_param(cfg_ga, "pm", "mutation_rate", default=0.15))

    elitism = get_ga_param(cfg_ga, "elitism", default=None)
    if elitism is None:
        elite_fraction = float(get_ga_param(cfg_ga, "elite_fraction", default=0.1))
        elitism = max(1, int(round(elite_fraction * n_pop)))

    tournament_k = int(get_ga_param(cfg_ga, "tournament_k", "tournament_size", default=3))

    if method == "ga_only":
        return run_ga_only(
            scenario,
            lo,
            hi,
            n_pop,
            n_gen,
            pc,
            pm,
            int(elitism),
            tournament_k,
            sim_kwargs=sim_kwargs,
        )

    if method == "ga_rules":
        return run_ga_rules(
            scenario,
            lo,
            hi,
            n_pop,
            n_gen,
            pc,
            pm,
            int(elitism),
            tournament_k,
            sim_kwargs=sim_kwargs,
        )

    if method == "ga_surrogate":
        return run_ga_surrogate(
            scenario,
            lo,
            hi,
            n_pop,
            n_gen,
            pc,
            pm,
            int(elitism),
            tournament_k,
            warmup_generations=int(cfg_ml.get("warmup_generations", cfg_ml.get("warmup_real_evals", 2))),
            retrain_period=int(cfg_ml.get("retrain_period", cfg_ml.get("retrain_frequency", 2))),
            rho=float(cfg_ml.get("rho", cfg_ml.get("surrogate_screening_ratio", 0.7))),
            surrogate_type=str(cfg_ml.get("model_type", "random_forest")),
            sim_kwargs=sim_kwargs,
        )

    if method == "hybrid":
        return _run_hybrid_compatible(
            scenario,
            lo,
            hi,
            n_pop,
            n_gen,
            pc,
            pm,
            int(elitism),
            tournament_k,
            cfg_ml,
            sim_kwargs,
        )

    raise ValueError(f"Unknown method: {method}")



# =========================================================
# Saving
# =========================================================

def save_method_arrays(out_results: str, scenario_name: str, method: str, res: dict) -> None:
    """
    Save arrays with unambiguous names:
      - G_*.npy and glucose_*.npy: glucose trajectory
      - history_*.npy: best true fitness convergence history
      - best_history_*.npy: legacy history if available
      - surrogate_pred/surrogate_true if available
    """
    best_out = res.get("best_out", None)

    if best_out is not None and "G" in best_out:
        G = np.asarray(best_out["G"], dtype=float)
        if G.size:
            np.save(os.path.join(out_results, f"G_{scenario_name}_{method}.npy"), G)
            np.save(os.path.join(out_results, f"glucose_{scenario_name}_{method}.npy"), G)

    if "best_true_history" in res:
        hist = np.asarray(res["best_true_history"], dtype=float)
        if hist.size:
            np.save(os.path.join(out_results, f"history_{scenario_name}_{method}.npy"), hist)

    if "best_history" in res:
        hist_legacy = np.asarray(res["best_history"], dtype=float)
        if hist_legacy.size:
            np.save(os.path.join(out_results, f"best_history_{scenario_name}_{method}.npy"), hist_legacy)

    if "surrogate_pred" in res and "surrogate_true" in res:
        pred = np.asarray(res.get("surrogate_pred", []), dtype=float)
        true = np.asarray(res.get("surrogate_true", []), dtype=float)
        if pred.size and true.size:
            np.save(os.path.join(out_results, f"surrogate_pred_{scenario_name}_{method}.npy"), pred)
            np.save(os.path.join(out_results, f"surrogate_true_{scenario_name}_{method}.npy"), true)


# =========================================================
# Plotting
# =========================================================

def plot_convergence(scenario_name: str, all_res: dict, fig_dir: str) -> None:
    labels = {
        "ga_only": "GA-only",
        "ga_rules": "GA+Rules",
        "ga_surrogate": "GA+Surrogate",
        "hybrid": "Hybrid",
    }

    plt.figure(figsize=(7, 4))

    for method, res in all_res.items():
        h = np.asarray(res["best_true_history"], dtype=float)
        plt.plot(
            np.arange(1, len(h) + 1),
            h,
            label=labels.get(method, method),
            linewidth=2,
        )

    plt.xlabel("Generation")
    plt.ylabel("Best true fitness")
    plt.title(f"Convergence - {scenario_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"convergence_{scenario_name}.pdf"), bbox_inches="tight")
    plt.close()


def plot_glucose_trajectories(scenario_name: str, all_res: dict, fig_dir: str) -> None:
    labels = {
        "ga_only": "GA-only",
        "ga_rules": "GA+Rules",
        "ga_surrogate": "GA+Surrogate",
        "hybrid": "Hybrid",
    }

    plt.figure(figsize=(8, 4))

    for method, res in all_res.items():
        best_out = res.get("best_out")
        if not best_out:
            continue

        G = np.asarray(best_out.get("G", []), dtype=float)
        if G.size:
            plt.plot(G, label=labels.get(method, method), linewidth=1.8)

    plt.axhline(70, linestyle="--", linewidth=1)
    plt.axhline(180, linestyle="--", linewidth=1)
    plt.xlabel("Time step")
    plt.ylabel("Glucose (mg/dL)")
    plt.title(f"Glucose Trajectories - {scenario_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"glucose_{scenario_name}.pdf"), bbox_inches="tight")
    plt.close()


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="ALL")
    parser.add_argument("--methods", type=str, default="ALL")
    parser.add_argument("--save_figs", action="store_true")
    parser.add_argument("--fig_dir", type=str, default="figures")
    args = parser.parse_args()

    base = PROJECT_ROOT

    cfg_default = load_yaml(os.path.join(base, "config", "default.yaml"))
    cfg_ga = load_yaml(os.path.join(base, "config", "ga.yaml"))
    cfg_ml_raw = load_yaml(os.path.join(base, "config", "ml.yaml"))
    cfg_sc = load_yaml(os.path.join(base, "config", "scenarios.yaml"))

    cfg_ml = merge_ml_config(cfg_ga, cfg_ml_raw)

    base_seed = int(cfg_default.get("seed", cfg_ga.get("seed", 0)))

    lo, hi = get_bounds(cfg_ga)
    sim_kwargs_base = make_sim_kwargs(cfg_default)

    out_results = os.path.join(base, "outputs", "results")
    fig_dir = os.path.join(base, args.fig_dir)

    ensure_dir(out_results)
    ensure_dir(fig_dir)

    if args.methods.strip().upper() == "ALL":
        methods = DEFAULT_METHODS
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.scenario.strip().upper() == "ALL":
        scenario_names = list(cfg_sc.keys())
    else:
        scenario_names = [args.scenario.strip()]

    all_summary_rows = []
    all_res_by_scenario = {}

    for scenario_name in scenario_names:
        if scenario_name not in cfg_sc:
            raise KeyError(f"Scenario '{scenario_name}' not found in config/scenarios.yaml")

        sc = cfg_sc[scenario_name]
        scenario = build_scenario(scenario_name, sc)

        print(f"\n=== Running scenario: {scenario.name} ===")

        all_res = {}
        best_final_list = []

        sim_kwargs = dict(sim_kwargs_base)
        for key in [
            "patient_name",
            "patient_group",
            "seed",
            "horizon_hours",
            "sample_time",
        ]:
            if key in sc:
                sim_kwargs[key] = sc[key]

        for method in methods:
            method_seed = stable_seed(base_seed, scenario.name, method)

            set_seed(method_seed)

            method_sim_kwargs = dict(sim_kwargs)
            method_sim_kwargs["seed"] = method_seed

            res = run_method(
                method,
                scenario,
                lo,
                hi,
                cfg_ga,
                cfg_ml,
                method_sim_kwargs,
            )

            res = ensure_true_history(res)

            if res.get("best_out") is None:
                raise RuntimeError(f"{method}: best_out is missing. Check simulator wrapper output.")

            all_res[method] = res
            best_final_list.append(float(res.get("best_true_fitness", -1e18)))

        all_res_by_scenario[scenario.name] = all_res
        f_star = float(np.max(best_final_list)) if best_final_list else float("nan")

        summary_rows = []

        for method in methods:
            res = all_res[method]
            best_out = res["best_out"]

            G = np.asarray(best_out.get("G", []), dtype=float)
            mets = best_out.get("metrics", {})

            fit_out = float(best_out.get("fitness", res.get("best_true_fitness", np.nan)))
            best_final = float(res.get("best_true_fitness", fit_out))
            hist = np.asarray(res["best_true_history"], dtype=float)

            AUCsum = float(auc_sum(hist))
            AUCmean = float(auc_mean(hist))
            AUCreg = float(auc_regret(hist, f_star=f_star))

            minG = float(np.min(G)) if G.size else float("nan")
            maxG = float(np.max(G)) if G.size else float("nan")
            meanG = float(np.mean(G)) if G.size else float("nan")
            stdG = float(np.std(G)) if G.size else float("nan")
            cvG = float(100.0 * stdG / meanG) if np.isfinite(meanG) and meanG > 1e-9 else float("nan")

            severe_hypo = int(np.isfinite(minG) and minG < 54.0)

            sim_calls = int(res.get("sim_calls", -1))
            dataset = res.get("dataset_size", None)

            # Save glucose trajectory and convergence history.
            save_method_arrays(out_results, scenario.name, method, res)

            row = {
                "Scenario": scenario.name,
                "Method": method,
                "Fitness": fit_out,
                "BestTrueFitness": best_final,
                "TIR": float(mets.get("TIR", np.nan)),
                "TBR": float(mets.get("TBR", np.nan)),
                "TAR": float(mets.get("TAR", np.nan)),
                "Gmin": minG,
                "Gmax": maxG,
                "MeanG": meanG,
                "StdG": stdG,
                "CV": cvG,
                "SevereHypo": severe_hypo,
                "SimCalls": sim_calls,
                "DatasetSize": int(dataset) if dataset is not None else -1,
                "AUCsum": AUCsum,
                "AUCmean": AUCmean,
                "AUCregret": AUCreg,
                "Backend": best_out.get("backend", "unknown"),
                "PatientName": best_out.get("patient_name", ""),
                "TopKRealEval": int(res.get("top_k_real_eval", -1)),
                "RandomRealEval": int(res.get("random_real_eval", -1)),
                "Rho": float(res.get("rho", cfg_ml.get("rho", np.nan))),
                "Alpha": float(res.get("alpha", cfg_ml.get("alpha", np.nan))),
            }

            summary_rows.append(row)
            all_summary_rows.append(row)

            extra = ""
            if dataset is not None:
                extra += f" | dataset={int(dataset)}"
            if sim_calls >= 0:
                extra += f" | sim_calls={sim_calls}"
            if row["Backend"]:
                extra += f" | backend={row['Backend']}"

            print(
                f"{method:<12} | best={best_final:7.2f} | fit_out={fit_out:7.2f} "
                f"| AUCsum={AUCsum:8.1f} | AUCmean={AUCmean:7.2f} "
                f"| AUCregret={AUCreg:8.1f} | {pretty_metrics(mets)} "
                f"| minG={minG:6.1f} | maxG={maxG:6.1f}{extra}"
            )

        try:
            import pandas as pd
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_results, f"summary_{scenario.name}.csv"),
                index=False,
            )
        except Exception as e:
            print(f"[WARN] Could not save scenario summary: {e}")

    try:
        import pandas as pd

        df_global = pd.DataFrame(all_summary_rows)

        # Add reduction percentage vs GA-only baseline.
        if not df_global.empty and "ga_only" in set(df_global["Method"]):
            baseline = df_global[df_global["Method"] == "ga_only"][["Scenario", "SimCalls"]]
            baseline = baseline.rename(columns={"SimCalls": "GAOnlyCalls"})
            df_global = df_global.merge(baseline, on="Scenario", how="left")
            df_global["ReductionPct"] = 100.0 * (1.0 - df_global["SimCalls"] / df_global["GAOnlyCalls"])
        else:
            df_global["GAOnlyCalls"] = np.nan
            df_global["ReductionPct"] = np.nan

        summary_path = os.path.join(out_results, "summary_all.csv")
        summary_reduction_path = os.path.join(out_results, "summary_all_with_reduction.csv")

        df_global.to_csv(summary_path, index=False)
        df_global.to_csv(summary_reduction_path, index=False)

        print("\n[DEBUG] summary_all.csv preview:")
        print(df_global.head())

        print("\n[DEBUG] summary_all.csv columns:")
        print(df_global.columns.tolist())

        print(f"\n[OK] Global summary saved in: {summary_path}")
        print(f"[OK] Global summary with reduction saved in: {summary_reduction_path}")

    except Exception as e:
        print(f"[WARN] Could not save global summary: {e}")

    if args.save_figs:
        for scenario_name, all_res in all_res_by_scenario.items():
            plot_convergence(scenario_name, all_res, fig_dir)
            plot_glucose_trajectories(scenario_name, all_res, fig_dir)

        print(f"\n[OK] Figures saved in: {fig_dir}")

    print(f"\n[OK] Results saved in: {out_results}")


if __name__ == "__main__":
    main()


