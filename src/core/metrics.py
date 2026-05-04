from __future__ import annotations

import numpy as np


def tir_tbr_tar(glucose: np.ndarray) -> dict:
    g = np.asarray(glucose, dtype=float)
    n = len(g)

    tir = 100.0 * np.sum((g >= 70.0) & (g <= 180.0)) / n
    tbr = 100.0 * np.sum(g < 70.0) / n
    tar = 100.0 * np.sum(g > 180.0) / n

    return {
        "TIR": float(tir),
        "TBR": float(tbr),
        "TAR": float(tar),
    }


def glucose_stats(glucose: np.ndarray) -> dict:
    g = np.asarray(glucose, dtype=float)

    mean = float(np.mean(g))
    std = float(np.std(g))
    cv = float(100.0 * std / mean) if mean > 1e-9 else float("nan")

    return {
        "mean": mean,
        "std": std,
        "CV": cv,
        "min": float(np.min(g)),
        "max": float(np.max(g)),
    }


def compute_fitness(
    metrics: dict,
    glucose: np.ndarray,
    target: float = 110.0,
) -> float:
    g = np.asarray(glucose, dtype=float)

    tir = float(metrics["TIR"])
    tbr = float(metrics["TBR"])
    tar = float(metrics["TAR"])

    mean_g = float(np.mean(g))
    std_g = float(np.std(g))
    cv_g = float(100.0 * std_g / mean_g) if mean_g > 1e-9 else 0.0
    gmax = float(np.max(g))
    gmin = float(np.min(g))

    fitness = tir - 2.0 * tbr - 0.5 * tar

    fitness -= 0.08 * abs(mean_g - target)
    fitness -= 0.06 * std_g
    fitness -= 0.01 * cv_g

    if gmax > 180.0:
        fitness -= 0.15 * (gmax - 180.0)

    if gmin < 70.0:
        fitness -= 0.30 * (70.0 - gmin)

    return float(fitness)