"""Reaction kinetics analysis: integral, LSQ (analytical), RK4+LSQ, and Arrhenius methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

R_GAS = 8.314  # J/(mol·K)

# Minimum valid data points required for RK4 / LSQ
_MIN_POINTS_RK4 = 4
_MIN_POINTS_LSQ = 3


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    order: int | float
    k: float
    r2: float
    slope: float
    intercept: float
    stderr: float
    aic: float
    k_ci_lower: float
    k_ci_upper: float
    n_points: int
    transform_y: np.ndarray
    transform_x: np.ndarray
    residuals: np.ndarray


@dataclass
class LsqSimpleResult:
    """Result of analytical-solution least-squares fit for A→products."""
    n: float        # estimated reaction order (free parameter)
    k: float
    C0: float       # fitted initial concentration
    r2: float
    rmse: float
    success: bool
    message: str
    t_pred: np.ndarray   # dense time grid for plotting
    c_pred: np.ndarray   # predicted concentrations on t_pred
    residuals: np.ndarray


@dataclass
class RK4LsqResult:
    reaction_type: str          # "simple" | "sequential" | "parallel"
    order: float                # estimated reaction order (simple only)
    k: float
    k2: float | None
    r2: float
    rmse: float
    success: bool
    message: str
    t_pred: np.ndarray
    c_pred: dict[str, np.ndarray]
    residuals: dict[str, np.ndarray]


@dataclass
class DifferentialResult:
    """Result of differential (log-log) method: -dC/dt = k * C^n."""
    n: float        # estimated reaction order
    k: float        # rate constant
    r2: float       # R² of log-log regression
    log_c: np.ndarray
    log_rate: np.ndarray


@dataclass
class ArrheniusResult:
    temperatures_K: np.ndarray
    k_values: np.ndarray
    k_method: str               # "integral" | "lsq" | "rk4"
    k_label: str                # "k" | "k1" | "k2"
    Ea: float                   # J/mol
    A: float                    # pre-exponential factor
    r2: float
    Ea_ci_lower: float
    Ea_ci_upper: float
    n_temperatures: int


@dataclass
class PerTempResult:
    temperature_C: float
    temperature_K: float
    n: float          # 推算反応次数
    k: float          # 速度定数
    r2: float         # 当てはまりの良さ
    method: str       # "integral" | "lsq" | "rk4"
    success: bool
    message: str = ""


@dataclass
class FullAnalysisResult:
    integral: dict[int, RegressionResult]
    lsq: LsqSimpleResult | None
    best_order: int
    best_result: RegressionResult
    warnings: list[str] = field(default_factory=list)
    rk4lsq: RK4LsqResult | None = None
    differential: DifferentialResult | None = None
    arrhenius: ArrheniusResult | None = None
    arrhenius_k2: ArrheniusResult | None = None
    detected_reaction_type: str = "simple"
    detected_reaction_reason: str = ""
    per_temp_results: list[PerTempResult] = field(default_factory=list)
    optimal_order_multi_temp: float | None = None
    optimal_order_explanation: str = ""


# ---------------------------------------------------------------------------
# Reaction type auto-detection
# ---------------------------------------------------------------------------

def auto_detect_reaction_type(df: pd.DataFrame) -> tuple[str, str]:
    """
    Suggest reaction type from data patterns.

    Returns
    -------
    (suggested_type, reason_message)
    suggested_type: "simple" | "sequential" | "parallel"
    """
    has_B = "concentration_B" in df.columns and df["concentration_B"].notna().any()
    has_C = "concentration_C" in df.columns and df["concentration_C"].notna().any()

    if not has_B:
        return "simple", "濃度Bデータがないため単純反応 A→products として解析します。"

    B_valid = df["concentration_B"].dropna().values
    if len(B_valid) < 3:
        return "sequential", "濃度Bのデータ点数が少ないため逐次反応を仮定します。"

    # Check for peak in B (sequential indicator)
    peak_idx = int(np.argmax(B_valid))
    if 0 < peak_idx < len(B_valid) - 1:
        return "sequential", (
            f"濃度Bが時刻インデックス {peak_idx} でピークを持つため、"
            "逐次反応 A→B→C を推奨します。"
        )

    # B monotonically increases -> parallel
    if has_C:
        return "parallel", (
            "濃度Bが単調増加かつ濃度Cデータがあるため、"
            "並列反応 A→B + A→C を推奨します。"
        )

    return "parallel", (
        "濃度Bが単調増加するため並列反応 A→B + A→C を推奨します。"
        "（濃度CがなければAのみの解析も可能です）"
    )


# ---------------------------------------------------------------------------
# AIC helper
# ---------------------------------------------------------------------------

def _aic(n: int, sse: float, k_params: int = 2) -> float:
    if sse <= 0 or n <= k_params:
        return np.inf
    sigma2 = sse / n
    return n * np.log(sigma2) + 2 * k_params


# ---------------------------------------------------------------------------
# Integral method
# ---------------------------------------------------------------------------

def _integral_order(
    time: np.ndarray,
    conc: np.ndarray,
    order: int,
) -> RegressionResult:
    """Fit one reaction order via integral method using only finite-valued data."""
    finite_mask = np.isfinite(conc)
    time = time[finite_mask]
    conc = conc[finite_mask]

    pos_mask = conc > 0

    if order == 0:
        y = conc.copy()
    elif order == 1:
        if not pos_mask.all():
            time = time[pos_mask]
            conc = conc[pos_mask]
        y = np.log(conc)
    elif order == 2:
        if not pos_mask.all():
            time = time[pos_mask]
            conc = conc[pos_mask]
        y = 1.0 / conc
    else:
        raise ValueError(f"Unsupported order: {order}")

    n = len(time)
    if n < 2:
        return RegressionResult(
            order=order, k=0.0, r2=0.0, slope=0.0, intercept=float(np.mean(y)) if len(y) else 0.0,
            stderr=np.inf, aic=np.inf, k_ci_lower=0.0, k_ci_upper=0.0,
            n_points=n, transform_y=y, transform_x=time, residuals=np.zeros_like(y),
        )

    result = stats.linregress(time, y)
    slope: float = result.slope
    intercept: float = result.intercept
    r2: float = result.rvalue ** 2
    stderr: float = result.stderr

    residuals = y - (slope * time + intercept)
    sse = float(np.sum(residuals ** 2))
    aic_val = _aic(n, sse)

    if order in (0, 1):
        k = float(-slope)
    else:
        k = float(slope)

    t_crit = stats.t.ppf(0.975, df=n - 2) if n > 2 else 1.96
    ci_half = t_crit * stderr
    if order in (0, 1):
        k_lower = float(-(slope + ci_half))
        k_upper = float(-(slope - ci_half))
    else:
        k_lower = float(slope - ci_half)
        k_upper = float(slope + ci_half)

    return RegressionResult(
        order=order, k=k, r2=r2, slope=slope, intercept=intercept,
        stderr=stderr, aic=aic_val, k_ci_lower=k_lower, k_ci_upper=k_upper,
        n_points=n, transform_y=y, transform_x=time, residuals=residuals,
    )


def run_integral_analysis(df: pd.DataFrame) -> dict[int, RegressionResult]:
    """Run integral method for orders 0, 1, 2 using valid A rows only."""
    mask = df["concentration"].notna()
    time = df.loc[mask, "time"].to_numpy(dtype=float)
    conc = df.loc[mask, "concentration"].to_numpy(dtype=float)
    return {order: _integral_order(time.copy(), conc.copy(), order) for order in (0, 1, 2)}


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_best_order(integral: dict[int, RegressionResult]) -> int:
    best_r2 = max(r.r2 for r in integral.values())
    candidates = [o for o, r in integral.items() if np.isclose(r.r2, best_r2, atol=1e-4)]
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda o: integral[o].aic)


# ---------------------------------------------------------------------------
# Analytical solution helper
# ---------------------------------------------------------------------------

def _analytical_conc(t: np.ndarray, k: float, n: float, c0: float) -> np.ndarray:
    """
    Analytical solution of dC/dt = -k*C^n.

    n=1 : C(t) = C0 * exp(-k*t)
    n≠1 : C(t) = (C0^(1-n) - (1-n)*k*t)^(1/(1-n))
    """
    if abs(n - 1.0) < 1e-6:
        return c0 * np.exp(-k * t)
    inner = c0 ** (1.0 - n) - (1.0 - n) * k * t
    inner = np.maximum(inner, 1e-30)   # avoid 0^negative
    return inner ** (1.0 / (1.0 - n))


# ---------------------------------------------------------------------------
# LSQ: analytical-solution least squares (simple reaction, no ODE)
# ---------------------------------------------------------------------------

def run_lsq_simple(df: pd.DataFrame) -> LsqSimpleResult:
    """
    Fit A→products using the analytical solution of dC/dt = -k*C^n
    and scipy.optimize.least_squares (no ODE integration).

    Free parameters: k, n (reaction order), C0 (initial concentration).
    """
    mask_A = df["concentration"].notna()
    n_valid = int(mask_A.sum())

    _fail = lambda msg: LsqSimpleResult(
        n=1.0, k=0.0, C0=0.0, r2=0.0, rmse=np.nan,
        success=False, message=msg,
        t_pred=np.array([]), c_pred=np.array([]), residuals=np.array([]),
    )

    if n_valid < _MIN_POINTS_LSQ:
        return _fail(f"有効データが {n_valid} 点のみです（最低 {_MIN_POINTS_LSQ} 点必要）。")

    time = df.loc[mask_A, "time"].to_numpy(dtype=float)
    conc = df.loc[mask_A, "concentration"].to_numpy(dtype=float)

    # Initial estimates from integral method
    integral = run_integral_analysis(df)
    best_order = select_best_order(integral)
    k_init = max(integral[best_order].k, 1e-6)
    n_init = float(best_order)
    c0_init = max(float(conc[0]), 1e-9)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        k, n, c0 = params
        if k <= 0 or c0 <= 0:
            return np.full(len(time), 1e6)
        try:
            c_pred = _analytical_conc(time, k, n, c0)
            if not np.all(np.isfinite(c_pred)):
                return np.full(len(time), 1e6)
            return c_pred - conc
        except Exception:
            return np.full(len(time), 1e6)

    try:
        res = least_squares(
            residual_fn,
            [k_init, n_init, c0_init],
            bounds=([1e-9, 0.0, 1e-9], [np.inf, 5.0, np.inf]),
            method="trf",
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            max_nfev=500,
        )
        success = bool(res.success or res.cost < 1e-6)
        msg = res.message
        k_fit, n_fit, c0_fit = float(res.x[0]), float(res.x[1]), float(res.x[2])
    except Exception as e:
        success = False
        msg = str(e)
        k_fit, n_fit, c0_fit = k_init, n_init, c0_init

    t_pred = np.linspace(time[0], time[-1], 300)
    c_pred_dense = _analytical_conc(t_pred, k_fit, n_fit, c0_fit)

    c_at_obs = _analytical_conc(time, k_fit, n_fit, c0_fit)
    if np.all(np.isfinite(c_at_obs)):
        r2   = _safe_r2(conc, c_at_obs)
        rmse = float(np.sqrt(np.nanmean((conc - c_at_obs) ** 2)))
        resid = conc - c_at_obs
    else:
        r2, rmse = 0.0, np.nan
        resid = np.full(len(time), np.nan)

    return LsqSimpleResult(
        n=n_fit, k=k_fit, C0=c0_fit,
        r2=r2, rmse=rmse, success=success, message=msg,
        t_pred=t_pred, c_pred=c_pred_dense, residuals=resid,
    )


# ---------------------------------------------------------------------------
# ODE definitions
# ---------------------------------------------------------------------------

def _ode_simple(t, y, k, n):
    A = max(y[0], 0.0)
    return [-k * A ** n]


def _ode_sequential(t, y, k1, k2):
    A = max(y[0], 0.0)
    B = max(y[1], 0.0)
    return [-k1 * A, k1 * A - k2 * B, k2 * B]


def _ode_parallel(t, y, k1, k2):
    A = max(y[0], 0.0)
    return [-(k1 + k2) * A, k1 * A, k2 * A]


# ---------------------------------------------------------------------------
# Helper: R² and prediction utilities
# ---------------------------------------------------------------------------

def _r2_score(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = obs[np.isfinite(obs) & np.isfinite(pred)]
    pred = pred[np.isfinite(obs) & np.isfinite(pred)] if obs.shape == pred.shape else pred
    valid = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[valid], pred[valid]
    if len(o) == 0:
        return 0.0
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _safe_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return 0.0
    return _r2_score(obs[valid], pred[valid])


def _solve_and_predict(
    ode_fn,
    t_span: tuple[float, float],
    y0: list[float],
    t_eval: np.ndarray,
    args: tuple,
) -> np.ndarray | None:
    """Run solve_ivp; returns sol.y or None on failure."""
    try:
        sol = solve_ivp(
            ode_fn, t_span, y0,
            method="RK45", t_eval=t_eval, args=args,
            rtol=1e-6, atol=1e-9, dense_output=False,
        )
        if sol.success and sol.y.shape[1] == len(t_eval):
            return sol.y
    except Exception:
        pass
    return None


def _initial_k_from_A(df: pd.DataFrame) -> float:
    """Quick k estimate from 1st-order integral fit of A data."""
    try:
        integral = run_integral_analysis(df)
        return max(integral[1].k, 1e-6)
    except Exception:
        return 1e-3


# ---------------------------------------------------------------------------
# RK4 + Least Squares: simple reaction
# ---------------------------------------------------------------------------

def run_rk4lsq_simple(
    df: pd.DataFrame,
    order_hint: float | None = None,
) -> RK4LsqResult:
    """
    Fit A→products via RK45 + least_squares (ODE-based).
    Works with NaN in concentration_B/C (ignored).
    """
    mask_A = df["concentration"].notna()
    n_valid = mask_A.sum()

    if n_valid < _MIN_POINTS_RK4:
        return RK4LsqResult(
            reaction_type="simple", order=1.0, k=0.0, k2=None,
            r2=0.0, rmse=np.nan, success=False,
            message=f"有効データが {n_valid} 点のみです（最低 {_MIN_POINTS_RK4} 点必要）。",
            t_pred=np.array([]), c_pred={"A": np.array([])},
            residuals={"A": np.array([])},
        )

    time = df.loc[mask_A, "time"].to_numpy(dtype=float)
    conc = df.loc[mask_A, "concentration"].to_numpy(dtype=float)

    integral = run_integral_analysis(df)
    best_order = select_best_order(integral)
    k_init = max(integral[best_order].k, 1e-6)
    n_init = order_hint if order_hint is not None else float(best_order)
    c0_init = max(float(conc[0]), 1e-9)

    def residual_fn(params):
        k, n, c0 = params
        if k <= 0 or c0 <= 0:
            return np.full(len(time), 1e6)
        sol_y = _solve_and_predict(
            _ode_simple, (time[0], time[-1]), [c0], time, (k, max(n, 0.0))
        )
        if sol_y is None:
            return np.full(len(time), 1e6)
        return sol_y[0] - conc

    try:
        res = least_squares(
            residual_fn,
            [k_init, n_init, c0_init],
            bounds=([1e-9, 0.0, 1e-9], [np.inf, 5.0, np.inf]),
            method="trf",
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            max_nfev=1000,
        )
        success = bool(res.success or res.cost < 1e-6)
        msg = res.message
        k_fit, n_fit, c0_fit = float(res.x[0]), float(res.x[1]), float(res.x[2])
    except Exception as e:
        success = False
        msg = str(e)
        k_fit, n_fit, c0_fit = k_init, n_init, c0_init

    t_pred = np.linspace(time[0], time[-1], 300)
    sol_pred = _solve_and_predict(_ode_simple, (time[0], time[-1]), [c0_fit], t_pred, (k_fit, n_fit))
    c_pred_A = sol_pred[0] if sol_pred is not None else np.full(300, np.nan)

    sol_obs = _solve_and_predict(_ode_simple, (time[0], time[-1]), [c0_fit], time, (k_fit, n_fit))
    if sol_obs is not None:
        c_at_obs = sol_obs[0]
        r2   = _safe_r2(conc, c_at_obs)
        rmse = float(np.sqrt(np.nanmean((conc - c_at_obs) ** 2)))
        resid_A = conc - c_at_obs
    else:
        r2, rmse = 0.0, np.nan
        resid_A = np.full(len(time), np.nan)

    return RK4LsqResult(
        reaction_type="simple", order=n_fit, k=k_fit, k2=None,
        r2=r2, rmse=rmse, success=success, message=msg,
        t_pred=t_pred, c_pred={"A": c_pred_A}, residuals={"A": resid_A},
    )


# ---------------------------------------------------------------------------
# RK4 + Least Squares: sequential reaction A→B→C
# ---------------------------------------------------------------------------

def run_rk4lsq_sequential(df: pd.DataFrame) -> RK4LsqResult:
    """
    Fit A→B→C via RK45 + least_squares.
    Handles NaN in any species column (different time points OK).
    Requires concentration_B; concentration_C is optional.
    """
    if "concentration_B" not in df.columns:
        raise ValueError("逐次反応解析には concentration_B 列が必要です。")

    time    = df["time"].to_numpy(dtype=float)
    concA   = df["concentration"].to_numpy(dtype=float)
    concB   = df["concentration_B"].to_numpy(dtype=float)
    has_C   = "concentration_C" in df.columns
    concC   = df["concentration_C"].to_numpy(dtype=float) if has_C else np.full(len(time), np.nan)

    maskA = np.isfinite(concA)
    maskB = np.isfinite(concB)
    maskC = np.isfinite(concC)

    if maskA.sum() < 2:
        raise ValueError(f"有効な濃度A データが {maskA.sum()} 点のみです（最低2点必要）。")
    if maskB.sum() < 2:
        raise ValueError(f"有効な濃度B データが {maskB.sum()} 点のみです（最低2点必要）。")

    c0A = float(concA[maskA][0])
    c0B = float(concB[maskB][0]) if maskB.any() else 0.0
    c0C = float(concC[maskC][0]) if maskC.any() else 0.0
    t_start, t_end = float(time[0]), float(time[-1])

    n_res = int(maskA.sum() + maskB.sum() + (maskC.sum() if has_C else 0))

    k_init  = _initial_k_from_A(df)
    k2_init = _estimate_k2_sequential(time, concB, maskB, k_init)

    def residual_fn(params):
        k1, k2 = params
        if k1 <= 0 or k2 <= 0:
            return np.full(n_res, 1e6)
        sol_y = _solve_and_predict(
            _ode_sequential, (t_start, t_end), [c0A, c0B, c0C], time, (k1, k2)
        )
        if sol_y is None:
            return np.full(n_res, 1e6)
        rA = sol_y[0][maskA] - concA[maskA]
        rB = sol_y[1][maskB] - concB[maskB]
        parts = [rA, rB]
        if has_C and maskC.any():
            parts.append(sol_y[2][maskC] - concC[maskC])
        return np.concatenate(parts)

    try:
        res = least_squares(
            residual_fn,
            [k_init, k2_init],
            bounds=([1e-9, 1e-9], [np.inf, np.inf]),
            method="trf",
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            max_nfev=1000,
        )
        success = bool(res.success or res.cost < 1e-6)
        msg = res.message
        k1_fit, k2_fit = float(res.x[0]), float(res.x[1])
    except Exception as e:
        success = False
        msg = str(e)
        k1_fit, k2_fit = k_init, k2_init

    t_pred = np.linspace(t_start, t_end, 300)
    sol_pred = _solve_and_predict(_ode_sequential, (t_start, t_end), [c0A, c0B, c0C], t_pred, (k1_fit, k2_fit))
    if sol_pred is not None:
        cA_pred, cB_pred, cC_pred = sol_pred[0], sol_pred[1], sol_pred[2]
    else:
        cA_pred = cB_pred = cC_pred = np.full(300, np.nan)

    sol_obs = _solve_and_predict(_ode_sequential, (t_start, t_end), [c0A, c0B, c0C], time, (k1_fit, k2_fit))
    if sol_obs is not None:
        r2_vals = [_safe_r2(concA[maskA], sol_obs[0][maskA]),
                   _safe_r2(concB[maskB], sol_obs[1][maskB])]
        resid_parts = {
            "A": concA - sol_obs[0],
            "B": concB - sol_obs[1],
            "C": (concC - sol_obs[2]) if has_C else np.zeros(len(time)),
        }
        if has_C and maskC.any():
            r2_vals.append(_safe_r2(concC[maskC], sol_obs[2][maskC]))
        r2   = float(np.mean(r2_vals))
        rmse = float(np.sqrt(np.nanmean((concA[maskA] - sol_obs[0][maskA]) ** 2)))
    else:
        r2, rmse = 0.0, np.nan
        resid_parts = {"A": np.full(len(time), np.nan), "B": np.full(len(time), np.nan), "C": np.full(len(time), np.nan)}

    return RK4LsqResult(
        reaction_type="sequential", order=1.0, k=k1_fit, k2=k2_fit,
        r2=r2, rmse=rmse, success=success, message=msg,
        t_pred=t_pred,
        c_pred={"A": cA_pred, "B": cB_pred, "C": cC_pred},
        residuals=resid_parts,
    )


def _estimate_k2_sequential(
    time: np.ndarray,
    concB: np.ndarray,
    maskB: np.ndarray,
    k1: float,
) -> float:
    """Estimate k2 from B peak time for sequential reaction."""
    if maskB.sum() < 3:
        return k1 * 0.5
    B_vals = concB[maskB]
    T_vals = time[maskB]
    peak_idx = int(np.argmax(B_vals))
    if 0 < peak_idx < len(B_vals) - 1:
        t_peak = float(T_vals[peak_idx])
        if k1 * t_peak > 0:
            k2_est = k1 / max(np.exp(k1 * t_peak * 0.7), 1.01)
            return max(k2_est, 1e-6)
    return k1 * 0.4


# ---------------------------------------------------------------------------
# RK4 + Least Squares: parallel reaction A→B + A→C
# ---------------------------------------------------------------------------

def run_rk4lsq_parallel(df: pd.DataFrame) -> RK4LsqResult:
    """
    Fit A→B + A→C via RK45 + least_squares.
    Handles NaN in any species column (different time points OK).
    Requires concentration_B and concentration_C.
    """
    if "concentration_B" not in df.columns:
        raise ValueError("並列反応解析には concentration_B 列が必要です。")
    if "concentration_C" not in df.columns:
        raise ValueError("並列反応解析には concentration_C 列が必要です。")

    time  = df["time"].to_numpy(dtype=float)
    concA = df["concentration"].to_numpy(dtype=float)
    concB = df["concentration_B"].to_numpy(dtype=float)
    concC = df["concentration_C"].to_numpy(dtype=float)

    maskA = np.isfinite(concA)
    maskB = np.isfinite(concB)
    maskC = np.isfinite(concC)

    if maskA.sum() < 2:
        raise ValueError(f"有効な濃度A データが {maskA.sum()} 点のみです。")
    if maskB.sum() < 2:
        raise ValueError(f"有効な濃度B データが {maskB.sum()} 点のみです。")
    if maskC.sum() < 2:
        raise ValueError(f"有効な濃度C データが {maskC.sum()} 点のみです。")

    c0A = float(concA[maskA][0])
    c0B = float(concB[maskB][0]) if maskB.any() else 0.0
    c0C = float(concC[maskC][0]) if maskC.any() else 0.0
    t_start, t_end = float(time[0]), float(time[-1])

    n_res = int(maskA.sum() + maskB.sum() + maskC.sum())

    k_total = _initial_k_from_A(df)
    k1_init, k2_init = _estimate_k_parallel(time, concA, concB, concC, maskA, maskB, maskC, k_total)

    def residual_fn(params):
        k1, k2 = params
        if k1 <= 0 or k2 <= 0:
            return np.full(n_res, 1e6)
        sol_y = _solve_and_predict(
            _ode_parallel, (t_start, t_end), [c0A, c0B, c0C], time, (k1, k2)
        )
        if sol_y is None:
            return np.full(n_res, 1e6)
        rA = sol_y[0][maskA] - concA[maskA]
        rB = sol_y[1][maskB] - concB[maskB]
        rC = sol_y[2][maskC] - concC[maskC]
        return np.concatenate([rA, rB, rC])

    try:
        res = least_squares(
            residual_fn,
            [k1_init, k2_init],
            bounds=([1e-9, 1e-9], [np.inf, np.inf]),
            method="trf",
            ftol=1e-8, xtol=1e-8, gtol=1e-8,
            max_nfev=1000,
        )
        success = bool(res.success or res.cost < 1e-6)
        msg = res.message
        k1_fit, k2_fit = float(res.x[0]), float(res.x[1])
    except Exception as e:
        success = False
        msg = str(e)
        k1_fit, k2_fit = k1_init, k2_init

    t_pred = np.linspace(t_start, t_end, 300)
    sol_pred = _solve_and_predict(_ode_parallel, (t_start, t_end), [c0A, c0B, c0C], t_pred, (k1_fit, k2_fit))
    if sol_pred is not None:
        cA_pred, cB_pred, cC_pred = sol_pred[0], sol_pred[1], sol_pred[2]
    else:
        cA_pred = cB_pred = cC_pred = np.full(300, np.nan)

    sol_obs = _solve_and_predict(_ode_parallel, (t_start, t_end), [c0A, c0B, c0C], time, (k1_fit, k2_fit))
    if sol_obs is not None:
        r2 = float(np.mean([
            _safe_r2(concA[maskA], sol_obs[0][maskA]),
            _safe_r2(concB[maskB], sol_obs[1][maskB]),
            _safe_r2(concC[maskC], sol_obs[2][maskC]),
        ]))
        rmse = float(np.sqrt(np.nanmean((concA[maskA] - sol_obs[0][maskA]) ** 2)))
        resid_parts = {
            "A": concA - sol_obs[0],
            "B": concB - sol_obs[1],
            "C": concC - sol_obs[2],
        }
    else:
        r2, rmse = 0.0, np.nan
        resid_parts = {sp: np.full(len(time), np.nan) for sp in ("A", "B", "C")}

    return RK4LsqResult(
        reaction_type="parallel", order=1.0, k=k1_fit, k2=k2_fit,
        r2=r2, rmse=rmse, success=success, message=msg,
        t_pred=t_pred,
        c_pred={"A": cA_pred, "B": cB_pred, "C": cC_pred},
        residuals=resid_parts,
    )


def _estimate_k_parallel(
    time, concA, concB, concC, maskA, maskB, maskC, k_total: float
) -> tuple[float, float]:
    """Estimate k1 and k2 from branching ratio of B and C."""
    try:
        dA = float(concA[maskA][0]) - float(concA[maskA][-1])
        dB = float(concB[maskB][-1]) - float(concB[maskB][0])
        dC = float(concC[maskC][-1]) - float(concC[maskC][0])
        total_prod = dB + dC
        if total_prod > 0 and dA > 0:
            ratio_B = dB / (dB + dC)
            k1 = max(k_total * ratio_B, 1e-9)
            k2 = max(k_total * (1 - ratio_B), 1e-9)
            return k1, k2
    except Exception:
        pass
    return k_total * 0.7, k_total * 0.3


# ---------------------------------------------------------------------------
# Arrhenius analysis
# ---------------------------------------------------------------------------

def run_arrhenius_analysis(
    temp_groups: dict[float, pd.DataFrame],
    method: str = "integral",
    reaction_type: str = "simple",
    k_index: int = 1,
) -> ArrheniusResult | None:
    """
    Fit Arrhenius equation ln(k) = ln(A) - Ea/R * 1/T from multiple temperatures.

    Parameters
    ----------
    temp_groups  : {T_celsius: sub_df}
    method       : "integral" | "lsq" | "rk4"
    reaction_type: "simple" | "sequential" | "parallel"
    k_index      : 1 for k (or k1), 2 for k2
    """
    valid_temps: list[float] = []
    valid_ks:    list[float] = []

    for T_c, sub_df in temp_groups.items():
        mask_A = sub_df["concentration"].notna()
        if mask_A.sum() < 3:
            continue
        try:
            k_val = _extract_k_for_arrhenius(sub_df, method, reaction_type, k_index)
            if k_val is not None and np.isfinite(k_val) and k_val > 0:
                valid_temps.append(T_c + 273.15)
                valid_ks.append(k_val)
        except Exception:
            continue

    if len(valid_temps) < 2:
        return None

    T_K  = np.array(valid_temps)
    k_arr = np.array(valid_ks)
    inv_T = 1.0 / T_K
    ln_k  = np.log(k_arr)

    res = stats.linregress(inv_T, ln_k)
    slope: float     = float(res.slope)
    intercept: float = float(res.intercept)
    r2: float        = float(res.rvalue ** 2)
    stderr: float    = float(res.stderr)

    Ea       = float(-slope * R_GAS)
    A_factor = float(np.exp(intercept))

    n = len(T_K)
    t_crit   = stats.t.ppf(0.975, df=n - 2) if n > 2 else 1.96
    ci_half  = t_crit * abs(stderr) * R_GAS
    Ea_lower = Ea - ci_half
    Ea_upper = Ea + ci_half

    k_label = "k" if (reaction_type == "simple" or k_index == 1) else "k2"
    if reaction_type in ("sequential", "parallel") and k_index == 1:
        k_label = "k1"

    return ArrheniusResult(
        temperatures_K=T_K,
        k_values=k_arr,
        k_method=method,
        k_label=k_label,
        Ea=Ea,
        A=A_factor,
        r2=r2,
        Ea_ci_lower=Ea_lower,
        Ea_ci_upper=Ea_upper,
        n_temperatures=n,
    )


def _extract_k_for_arrhenius(
    sub_df: pd.DataFrame,
    method: str,
    reaction_type: str,
    k_index: int,
) -> float | None:
    """Extract appropriate rate constant for one temperature group."""
    if reaction_type == "simple":
        if method == "rk4":
            rk = run_rk4lsq_simple(sub_df)
            return rk.k if rk.success else None
        elif method == "lsq":
            lsq = run_lsq_simple(sub_df)
            return lsq.k if lsq.success else None
        else:  # integral
            integral = run_integral_analysis(sub_df)
            best_ord = select_best_order(integral)
            return integral[best_ord].k

    # Multi-species: always use RK4
    has_B = "concentration_B" in sub_df.columns and sub_df["concentration_B"].notna().any()
    has_C = "concentration_C" in sub_df.columns and sub_df["concentration_C"].notna().any()

    if not has_B:
        integral = run_integral_analysis(sub_df)
        best_ord = select_best_order(integral)
        return integral[best_ord].k

    if reaction_type == "sequential":
        rk = run_rk4lsq_sequential(sub_df)
    else:  # parallel
        if not has_C:
            integral = run_integral_analysis(sub_df)
            best_ord = select_best_order(integral)
            return integral[best_ord].k
        rk = run_rk4lsq_parallel(sub_df)

    if not rk.success:
        return None
    return rk.k if k_index == 1 else rk.k2


# ---------------------------------------------------------------------------
# Per-temperature analysis (multi-temp datasets)
# ---------------------------------------------------------------------------

def run_per_temperature_analysis(
    temp_groups: dict[float, pd.DataFrame],
    method: str,          # "integral" | "lsq" | "rk4"
    reaction_type: str = "simple",
) -> list[PerTempResult]:
    """Analyse each temperature group independently to estimate n and k."""
    results: list[PerTempResult] = []

    for T_c, sub_df in temp_groups.items():
        T_K = T_c + 273.15
        mask_A = sub_df["concentration"].notna()
        n_valid = int(mask_A.sum())

        if n_valid < 3:
            results.append(PerTempResult(
                temperature_C=T_c, temperature_K=T_K,
                n=float("nan"), k=float("nan"), r2=float("nan"),
                method=method, success=False,
                message=f"有効データが {n_valid} 点のみです（最低3点必要）。",
            ))
            continue

        try:
            if reaction_type == "simple":
                if method == "rk4":
                    rk = run_rk4lsq_simple(sub_df)
                    if rk.success:
                        results.append(PerTempResult(
                            temperature_C=T_c, temperature_K=T_K,
                            n=rk.order, k=rk.k, r2=rk.r2,
                            method=method, success=True,
                        ))
                    else:
                        results.append(PerTempResult(
                            temperature_C=T_c, temperature_K=T_K,
                            n=float("nan"), k=float("nan"), r2=float("nan"),
                            method=method, success=False, message=rk.message,
                        ))
                elif method == "lsq":
                    lsq = run_lsq_simple(sub_df)
                    if lsq.success:
                        results.append(PerTempResult(
                            temperature_C=T_c, temperature_K=T_K,
                            n=lsq.n, k=lsq.k, r2=lsq.r2,
                            method=method, success=True,
                        ))
                    else:
                        results.append(PerTempResult(
                            temperature_C=T_c, temperature_K=T_K,
                            n=float("nan"), k=float("nan"), r2=float("nan"),
                            method=method, success=False, message=lsq.message,
                        ))
                else:  # integral
                    integral = run_integral_analysis(sub_df)
                    best_ord = select_best_order(integral)
                    res = integral[best_ord]
                    results.append(PerTempResult(
                        temperature_C=T_c, temperature_K=T_K,
                        n=float(best_ord), k=res.k, r2=res.r2,
                        method=method, success=True,
                    ))
            else:
                # Non-simple: fall back to integral for n estimation
                integral = run_integral_analysis(sub_df)
                best_ord = select_best_order(integral)
                res = integral[best_ord]
                results.append(PerTempResult(
                    temperature_C=T_c, temperature_K=T_K,
                    n=float(best_ord), k=res.k, r2=res.r2,
                    method="integral", success=True,
                ))
        except Exception as e:
            results.append(PerTempResult(
                temperature_C=T_c, temperature_K=T_K,
                n=float("nan"), k=float("nan"), r2=float("nan"),
                method=method, success=False, message=str(e),
            ))

    return results


def determine_optimal_order(
    per_temp: list[PerTempResult],
) -> tuple[float, str]:
    """Compute R²-weighted average reaction order from per-temperature results."""
    valid = [r for r in per_temp if r.success and np.isfinite(r.n) and np.isfinite(r.r2)]

    if not valid:
        return 1.0, "有効な温度別解析結果がありません。デフォルト値 n=1.0 を使用します。"

    n_vals = np.array([r.n for r in valid])
    weights = np.array([r.r2 for r in valid])
    if weights.sum() == 0:
        weights = np.ones(len(valid))

    optimal_n = float(np.average(n_vals, weights=weights))

    note = ""
    if optimal_n < 0:
        optimal_n = 0.0
        note = " （計算値が負のため 0.0 にクランプ）"
    elif optimal_n > 5:
        optimal_n = 5.0
        note = " （計算値が5超のため 5.0 にクランプ）"

    detail_lines = [f"  {r.temperature_C:.1f}°C: n={r.n:.3f}, R²={r.r2:.4f}" for r in valid]
    explanation = (
        f"有効温度数: {len(valid)} / {len(per_temp)}\n"
        + "\n".join(detail_lines)
        + f"\nR²加重平均 n = {optimal_n:.4f}{note}"
    )

    return optimal_n, explanation


# ---------------------------------------------------------------------------
# Differential method (log-log)
# ---------------------------------------------------------------------------

def run_differential_analysis(df: pd.DataFrame) -> DifferentialResult | None:
    """
    Estimate reaction order and rate constant by the differential method.

    Uses numerical differentiation of concentration_A vs time, then fits
    log(-dC/dt) = log(k) + n * log(C) by linear regression.

    Returns None when there are fewer than 3 valid points.
    """
    sub = df[["time", "concentration_A"]].dropna()
    if len(sub) < 3:
        return None

    t = sub["time"].to_numpy(dtype=float)
    c = sub["concentration_A"].to_numpy(dtype=float)

    # Central differences for interior points, one-sided at edges
    rate = np.gradient(c, t)  # dC/dt (negative for decay)
    neg_rate = -rate           # should be positive for A→products

    # Keep only points where C > 0 and rate > 0
    mask = (c > 0) & (neg_rate > 0)
    if mask.sum() < 3:
        return None

    log_c    = np.log(c[mask])
    log_rate = np.log(neg_rate[mask])

    slope, intercept, r, *_ = stats.linregress(log_c, log_rate)
    n = slope
    k = float(np.exp(intercept))
    r2 = float(r ** 2)

    return DifferentialResult(n=float(n), k=k, r2=r2, log_c=log_c, log_rate=log_rate)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(
    df: pd.DataFrame,
    reaction_type: str = "simple",
    enable_lsq: bool = True,
    enable_rk4: bool = True,
    enable_arrhenius: bool = True,
    temp_groups: dict[float, pd.DataFrame] | None = None,
) -> FullAnalysisResult:
    """
    Run complete kinetic analysis.

    Parameters
    ----------
    reaction_type    : "simple" | "sequential" | "parallel"
    enable_lsq       : run analytical-solution LSQ (simple reaction only)
    enable_rk4       : run RK4+LSQ ODE fitting
    enable_arrhenius : run Arrhenius (requires temp_groups with ≥2 entries)
    temp_groups      : output of get_temperature_groups()
    """
    extra_warnings: list[str] = []

    # Auto-detect reaction type (informational)
    detected_type, detected_reason = auto_detect_reaction_type(df)

    # Integral (always on A data)
    integral   = run_integral_analysis(df)
    best_order = select_best_order(integral)
    best_result = integral[best_order]

    # Warn when multi-temperature data is combined into a single df
    is_multi_temp = temp_groups is not None and len(temp_groups) >= 2
    if is_multi_temp:
        arr_method = "rk4" if enable_rk4 else ("lsq" if enable_lsq else "integral")
        extra_warnings.append(
            "多温度データが検出されました。温度別の反応次数・速度定数はアレニウス解析タブで確認できます。"
        )

    # Note if user-selected type differs from auto-detected
    if detected_type != reaction_type:
        extra_warnings.append(
            f"自動判定では「{_type_label(detected_type)}」を推奨しますが、"
            f"「{_type_label(reaction_type)}」で解析します。"
        )

    # Differential method
    differential: DifferentialResult | None = None
    try:
        differential = run_differential_analysis(df)
    except Exception as e:
        extra_warnings.append(f"微分法でエラーが発生しました: {e}")

    # LSQ (analytical solution, simple reaction only)
    lsq_result: LsqSimpleResult | None = None
    if enable_lsq and reaction_type == "simple":
        try:
            lsq_result = run_lsq_simple(df)
            if not lsq_result.success:
                extra_warnings.append(f"最小二乗法（解析解）: 収束しませんでした — {lsq_result.message}")
        except Exception as e:
            extra_warnings.append(f"最小二乗法（解析解）でエラーが発生しました: {e}")

    # RK4 + LSQ (ODE-based)
    rk4lsq: RK4LsqResult | None = None
    if enable_rk4:
        has_B = "concentration_B" in df.columns and df["concentration_B"].notna().any()
        has_C = "concentration_C" in df.columns and df["concentration_C"].notna().any()

        try:
            if reaction_type == "sequential":
                if not has_B:
                    extra_warnings.append(
                        "逐次反応: 濃度Bデータがありません。単純反応RK4にフォールバックします。"
                    )
                    rk4lsq = run_rk4lsq_simple(df, order_hint=float(best_order))
                else:
                    rk4lsq = run_rk4lsq_sequential(df)
            elif reaction_type == "parallel":
                if not has_B or not has_C:
                    extra_warnings.append(
                        "並列反応: 濃度BまたはCデータが不足しています。単純反応RK4にフォールバックします。"
                    )
                    rk4lsq = run_rk4lsq_simple(df, order_hint=float(best_order))
                else:
                    rk4lsq = run_rk4lsq_parallel(df)
            else:
                order_hint = lsq_result.n if lsq_result and lsq_result.success else None
                rk4lsq = run_rk4lsq_simple(df, order_hint=order_hint)
        except Exception as e:
            extra_warnings.append(f"RK4法でエラーが発生しました: {e}")

    # Per-temperature analysis
    per_temp_results: list[PerTempResult] = []
    optimal_order_multi_temp: float | None = None
    optimal_order_explanation: str = ""

    if is_multi_temp and reaction_type == "simple":
        try:
            per_temp_results = run_per_temperature_analysis(
                temp_groups, method=arr_method, reaction_type=reaction_type
            )
            optimal_order_multi_temp, optimal_order_explanation = determine_optimal_order(per_temp_results)
        except Exception as e:
            extra_warnings.append(f"温度別反応次数解析でエラーが発生しました: {e}")

    # Arrhenius
    arrhenius:    ArrheniusResult | None = None
    arrhenius_k2: ArrheniusResult | None = None

    if enable_arrhenius and temp_groups and len(temp_groups) >= 2:
        if not is_multi_temp:
            arr_method = "rk4" if enable_rk4 else ("lsq" if enable_lsq else "integral")

        try:
            arrhenius = run_arrhenius_analysis(
                temp_groups,
                method=arr_method,
                reaction_type=reaction_type,
                k_index=1,
            )
            if arrhenius is None:
                extra_warnings.append(
                    "アレニウス解析: 有効な温度グループが2点未満のためスキップしました。"
                )
        except Exception as e:
            extra_warnings.append(f"アレニウス解析 (k1) でエラー: {e}")

        if reaction_type in ("sequential", "parallel") and enable_rk4:
            try:
                arrhenius_k2 = run_arrhenius_analysis(
                    temp_groups,
                    method="rk4",
                    reaction_type=reaction_type,
                    k_index=2,
                )
            except Exception as e:
                extra_warnings.append(f"アレニウス解析 (k2) でエラー: {e}")

    return FullAnalysisResult(
        integral=integral,
        lsq=lsq_result,
        best_order=best_order,
        best_result=best_result,
        warnings=extra_warnings,
        rk4lsq=rk4lsq,
        differential=differential,
        arrhenius=arrhenius,
        arrhenius_k2=arrhenius_k2,
        detected_reaction_type=detected_type,
        detected_reaction_reason=detected_reason,
        per_temp_results=per_temp_results,
        optimal_order_multi_temp=optimal_order_multi_temp,
        optimal_order_explanation=optimal_order_explanation,
    )


def _type_label(t: str) -> str:
    return {"simple": "単純反応", "sequential": "逐次反応", "parallel": "並列反応"}.get(t, t)
