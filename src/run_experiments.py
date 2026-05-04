from __future__ import annotations

import os
import sys
import argparse
import yaml
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


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pretty_metrics(m: dict) -> str:
    return f"TIR={m['TIR']:5.1f}% TBR={m['TBR']:5.1f}% TAR={m['TAR']:5.1f}%"


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
        return {
            "start": _as_hhmm(w.get("start")),
            "end": _as_hhmm(w.get("end")),
        }

    if isinstance(w, (list, tuple)) and len(w) >= 2:
        return {
            "start": _as_hhmm(w[0]),
            "end": _as_hhmm(w[1]),
        }

    return None


def ensure_true_history(res: dict) -> dict:
    if "best_true_history" in res and "best_true_fitness" in res:
        return res

    if "best_history" in res:
        h = np.asarray(res["best_history"], dtype=float)
        h = np.maximum.accumulate(h)

        res["best_true_history"] = h
        res["best_true_fitness"] = float(np.max(h)) if h.size else float("nan")
        return res

    raise RuntimeError("Result dict missing best_true_history / best_history.")


def run_method(
    method: str,
    scenario: Scenario,
    lo: np.ndarray,
    hi: np.ndarray,
    cfg_ga: dict,
    cfg_ml: dict,
    sim_kwargs: dict,
) -> dict:

    if method == "ga_only":
        return run_ga_only(
            scenario,
            lo,
            hi,
            cfg_ga["n_pop"],
            cfg_ga["n_gen"],
            cfg_ga["pc"],
            cfg_ga["pm"],
            cfg_ga["elitism"],
            cfg_ga["tournament_k"],
            sim_kwargs=sim_kwargs,
        )

    if method == "ga_rules":
        return run_ga_rules(
            scenario,
            lo,
            hi,
            cfg_ga["n_pop"],
            cfg_ga["n_gen"],
            cfg_ga["pc"],
            cfg_ga["pm"],
            cfg_ga["elitism"],
            cfg_ga["tournament_k"],
            sim_kwargs=sim_kwargs,
        )

    if method == "ga_surrogate":
        return run_ga_surrogate(
            scenario,
            lo,
            hi,
            cfg_ga["n_pop"],
            cfg_ga["n_gen"],
            cfg_ga["pc"],
            cfg_ga["pm"],
            cfg_ga["elitism"],
            cfg_ga["tournament_k"],
            warmup_generations=cfg_ml.get("warmup_generations", 6),
            retrain_period=cfg_ml.get("retrain_period", 2),
            rho=cfg_ml.get("rho", 0.7),
            surrogate_type=cfg_ml.get("model_type", "random_forest"),
            sim_kwargs=sim_kwargs,
        )

    if method == "hybrid":
        return run_hybrid(
            scenario,
            lo,
            hi,
            cfg_ga["n_pop"],
            cfg_ga["n_gen"],
            cfg_ga["pc"],
            cfg_ga["pm"],
            cfg_ga["elitism"],
            cfg_ga["tournament_k"],
            alpha=cfg_ml.get("alpha", 0.2),
            rho=cfg_ml.get("rho", 0.7),
            warmup_generations=cfg_ml.get("warmup_generations", 6),
            retrain_period=cfg_ml.get("retrain_period", 2),
            surrogate_type=cfg_ml.get("model_type", "random_forest"),
            sim_kwargs=sim_kwargs,
        )

    raise ValueError(f"Unknown method: {method}")


def plot_convergence(scenario_name: str, all_res: dict, fig_dir: str) -> None:
    labels = {
        "ga_only": "GA-only",
        "ga_rules": "GA+Rules",
        "ga_surrogate": "GA+Surrogate",
        "hybrid": "Hybrid",
    }

    colors = {
        "ga_only": "#1f77b4",
        "ga_rules": "#d62728",
        "ga_surrogate": "#ff7f0e",
        "hybrid": "#2ca02c",
    }

    plt.figure(figsize=(7, 4))

    for method, res in all_res.items():
        h = np.asarray(res["best_true_history"], dtype=float)

        plt.plot(
            np.arange(1, len(h) + 1),
            h,
            label=labels.get(method, method),
            color=colors.get(method, None),
            linewidth=2,
        )

    plt.xlabel("Generation")
    plt.ylabel("Best true fitness")
    plt.title(f"Convergence - {scenario_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(fig_dir, f"convergence_{scenario_name}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_glucose_trajectories(scenario_name: str, all_res: dict, fig_dir: str) -> None:
    labels = {
        "ga_only": "GA-only",
        "ga_rules": "GA+Rules",
        "ga_surrogate": "GA+Surrogate",
        "hybrid": "Hybrid",
    }

    colors = {
        "ga_only": "#1f77b4",
        "ga_rules": "#d62728",
        "ga_surrogate": "#ff7f0e",
        "hybrid": "#2ca02c",
    }

    plt.figure(figsize=(8, 4))

    for method, res in all_res.items():
        best_out = res.get("best_out")
        if best_out is None:
            continue

        G = np.asarray(best_out["G"], dtype=float)

        plt.plot(
            G,
            label=labels.get(method, method),
            color=colors.get(method, None),
            linewidth=1.8,
        )

    plt.axhline(70, linestyle="--", color="black", linewidth=1)
    plt.axhline(180, linestyle="--", color="black", linewidth=1)

    plt.xlabel("Time step")
    plt.ylabel("Glucose (mg/dL)")
    plt.title(f"Glucose Trajectories - {scenario_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(fig_dir, f"glucose_{scenario_name}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()


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
    cfg_ml = load_yaml(os.path.join(base, "config", "ml.yaml"))
    cfg_sc = load_yaml(os.path.join(base, "config", "scenarios.yaml"))

    cfg_ml.setdefault("model_type", "random_forest")
    cfg_ml.setdefault("alpha", 0.2)
    cfg_ml.setdefault("rho", 0.7)
    cfg_ml.setdefault("warmup_generations", 6)
    cfg_ml.setdefault("retrain_period", 2)

    base_seed = int(cfg_default.get("seed", 0))

    lo = np.array(
        [cfg_ga["dose_basal_min"], cfg_ga["insulin_duration_min"]],
        dtype=float,
    )

    hi = np.array(
        [cfg_ga["dose_basal_max"], cfg_ga["insulin_duration_max"]],
        dtype=float,
    )

    sim_kwargs = {
        "dt_minutes": int(cfg_default["dt_minutes"]),
        "horizon_hours": int(cfg_default["horizon_hours"]),
        "g0": float(cfg_default["g0_mgdl"]),
        "gmin": float(cfg_default["gmin_mgdl"]),
        "gmax": float(cfg_default["gmax_mgdl"]),
    }

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
        sc = cfg_sc[scenario_name]

        scenario = Scenario(
            name=scenario_name,
            meals=sc.get("meals", []),
            stress_level=float(sc.get("stress_level", 0.0)),
            activity_level=float(sc.get("activity_level", 0.0)),
            noise_sigma=float(sc.get("noise_sigma", 2.0)),
            fasting=bool(sc.get("fasting", False)),
            fasting_window=normalize_window(sc.get("fasting_window")),
            activity_window=normalize_window(sc.get("activity_window")),
        )

        print(f"\n=== Running scenario: {scenario.name} ===")

        all_res = {}
        best_final_list = []

        for method in methods:
            set_seed(stable_seed(base_seed, scenario.name, method))

            res = run_method(
                method,
                scenario,
                lo,
                hi,
                cfg_ga,
                cfg_ml,
                sim_kwargs,
            )

            res = ensure_true_history(res)

            if res.get("best_out") is None:
                raise RuntimeError(f"{method}: best_out is missing.")

            best_out = res["best_out"]
            best_final = float(res.get("best_true_fitness", best_out["fitness"]))

            all_res[method] = res
            best_final_list.append(best_final)

        all_res_by_scenario[scenario.name] = all_res
        f_star = float(np.max(best_final_list))

        summary_rows = []

        for method in methods:
            res = all_res[method]
            best_out = res["best_out"]

            G = np.asarray(best_out["G"], dtype=float)
            mets = best_out["metrics"]

            fit_out = float(best_out["fitness"])
            best_final = float(res.get("best_true_fitness", fit_out))

            hist = np.asarray(res["best_true_history"], dtype=float)

            AUCsum = float(auc_sum(hist))
            AUCmean = float(auc_mean(hist))
            AUCreg = float(auc_regret(hist, f_star=f_star))

            minG = float(np.min(G))
            maxG = float(np.max(G))
            meanG = float(np.mean(G))
            stdG = float(np.std(G))
            cvG = float(100.0 * stdG / meanG) if meanG > 1e-9 else float("nan")

            severe_hypo = int(minG < 54.0)

            sim_calls = int(res.get("sim_calls", -1))
            dataset = res.get("dataset_size", None)

            np.save(
                os.path.join(out_results, f"G_{scenario.name}_{method}.npy"),
                G,
            )

            row = {
                "Scenario": scenario.name,
                "Method": method,
                "Fitness": fit_out,
                "BestTrueFitness": best_final,
                "TIR": float(mets["TIR"]),
                "TBR": float(mets["TBR"]),
                "TAR": float(mets["TAR"]),
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
            }

            summary_rows.append(row)
            all_summary_rows.append(row)

            extra = ""
            if dataset is not None:
                extra += f" | dataset={int(dataset)}"
            if sim_calls >= 0:
                extra += f" | sim_calls={sim_calls}"

            print(
                f"{method:<12} | best={best_final:7.2f} | fit_out={fit_out:7.2f} "
                f"| AUCsum={AUCsum:8.1f} | AUCmean={AUCmean:7.2f} "
                f"| AUCregret={AUCreg:8.1f} | {pretty_metrics(mets)} "
                f"| minG={minG:6.1f} | maxG={maxG:6.1f}{extra}"
            )

        try:
            import pandas as pd

            df_scenario = pd.DataFrame(summary_rows)
            df_scenario.to_csv(
                os.path.join(out_results, f"summary_{scenario.name}.csv"),
                index=False,
            )

        except Exception as e:
            print(f"[WARN] Could not save scenario summary: {e}")

    try:
        import pandas as pd

        df_global = pd.DataFrame(all_summary_rows)

        summary_path = os.path.join(out_results, "summary_all.csv")
        df_global.to_csv(summary_path, index=False)

        print("\n[DEBUG] summary_all.csv preview:")
        print(df_global.head())

        print("\n[DEBUG] summary_all.csv columns:")
        print(df_global.columns.tolist())

        print(f"\n[OK] Global summary saved in: {summary_path}")

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