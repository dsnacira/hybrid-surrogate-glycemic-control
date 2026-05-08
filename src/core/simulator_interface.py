from __future__ import annotations

try:
    from .simglucose_wrapper import simulate_day_simglucose as _simulate_simglucose
    SIMULATOR_BACKEND = "simglucose"
    _HAS_SIMGLUCOSE = True

except Exception as e:
    print("[WARNING] simglucose unavailable.")
    print(e)

    from .simulator import simulate_day as _simulate_legacy
    SIMULATOR_BACKEND = "legacy"
    _HAS_SIMGLUCOSE = False


def simulate_day(*args, **kwargs):
    """
    Unified simulator interface.

    Supports new 3D chromosome:
        [basal_rate, insulin_carb_ratio, correction_factor]

    If simglucose is unavailable, falls back to legacy simulator using:
        dose_basal = basal_rate
        insulin_duration_h = insulin_carb_ratio
    """

    if _HAS_SIMGLUCOSE:
        return _simulate_simglucose(*args, **kwargs)

    # -----------------------------
    # Legacy fallback compatibility
    # -----------------------------
    basal_rate = kwargs.pop("basal_rate", None)
    insulin_carb_ratio = kwargs.pop("insulin_carb_ratio", None)
    correction_factor = kwargs.pop("correction_factor", None)
    scenario = kwargs.pop("scenario", None)

    if len(args) >= 3:
        basal_rate = args[0]
        insulin_carb_ratio = args[1]
        scenario = args[2]

    elif len(args) == 2:
        basal_rate = args[0]
        insulin_carb_ratio = args[1]

    elif len(args) == 1:
        basal_rate = args[0]

    if scenario is None:
        raise TypeError("scenario is required in simulator_interface.simulate_day()")

    # legacy simulator expects:
    # simulate_day(dose_basal, insulin_duration_h, scenario, ...)
    return _simulate_legacy(
        float(basal_rate),
        float(insulin_carb_ratio),
        scenario,
        **kwargs,
    )