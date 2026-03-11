"""Arrhenius analysis for reaction kinetics."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from src.fitting import run_fit
from src.models import ArrheniusResult, FitResult

R_GAS = 8.314  # J/(mol·K)


def run_arrhenius(
    temp_groups: dict,
    reaction_type: str,
    k_index: int = 1,
    per_temp_fits: Optional[list] = None,
) -> Optional[ArrheniusResult]:
    """
    Fit Arrhenius equation ln(k) = ln(A) - Ea/R * 1/T from multiple temperatures.

    Parameters
    ----------
    temp_groups    : {T_celsius: sub_df}
    reaction_type  : "simple" | "sequential" | "parallel"
    k_index        : 1 for k (or k1), 2 for k2
    per_temp_fits  : pre-computed list of (T_celsius, FitResult). When provided,
                     k values are taken from these fits directly (avoids re-fitting).

    Skips:
    - Failed fits (success=False)
    - Low R² (R² < 0.5)
    """
    valid_temps: list[float] = []
    valid_ks:    list[float] = []

    if per_temp_fits is not None:
        # Use pre-computed fit results — avoids redundant ODE fitting
        for T_c, fit in per_temp_fits:
            if not fit.success:
                continue
            if fit.r2 < 0.5:
                continue
            k_val = fit.k if k_index == 1 else fit.k2
            if k_val is None or not np.isfinite(k_val) or k_val <= 0:
                continue
            valid_temps.append(T_c + 273.15)
            valid_ks.append(float(k_val))
    else:
        # Fallback: re-fit each temperature group independently
        for T_c, sub_df in temp_groups.items():
            mask_A = sub_df["concentration"].notna()
            if mask_A.sum() < 3:
                continue
            try:
                fit: FitResult = run_fit(sub_df, reaction_type)
                if not fit.success:
                    continue
                if fit.r2 < 0.5:
                    continue
                k_val = fit.k if k_index == 1 else fit.k2
                if k_val is None or not np.isfinite(k_val) or k_val <= 0:
                    continue
                valid_temps.append(T_c + 273.15)
                valid_ks.append(float(k_val))
            except Exception:
                continue

    if len(valid_temps) < 2:
        return None

    T_K   = np.array(valid_temps)
    k_arr = np.array(valid_ks)
    inv_T = 1.0 / T_K
    ln_k  = np.log(k_arr)

    res = stats.linregress(inv_T, ln_k)
    slope: float     = float(res.slope)
    intercept: float = float(res.intercept)
    r2: float        = float(res.rvalue ** 2)

    Ea       = float(-slope * R_GAS)
    A_factor = float(np.exp(intercept))

    return ArrheniusResult(
        temps_celsius=[t - 273.15 for t in T_K.tolist()],
        k_values=k_arr.tolist(),
        ln_k=ln_k.tolist(),
        inv_T=inv_T.tolist(),
        Ea=Ea,
        Ea_kJmol=Ea / 1000.0,
        A=A_factor,
        r2=r2,
        slope=slope,
        intercept=intercept,
        k_method="RK4+LSQ",
    )
