"""RK4+LSQ fitting for reaction kinetics with multistart and CI computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import pinv
from scipy.optimize import least_squares

from src.models import FitResult
from src.ode_systems import (
    _ode_parallel,
    _ode_sequential,
    _ode_simple,
    solve_and_predict,
)

_MIN_POINTS_RK4 = 4


# ---------------------------------------------------------------------------
# Helpers: R² and RMSE
# ---------------------------------------------------------------------------

def _safe_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return 0.0
    o, p = obs[valid], pred[valid]
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Initial estimate helpers
# ---------------------------------------------------------------------------

def _estimate_k_simple(time: np.ndarray, conc: np.ndarray) -> float:
    """Heuristic: k ≈ -ln(C_ref/C_first) / dt using the last positive concentration point."""
    try:
        c_first = float(conc[0])
        if c_first <= 0:
            return 1e-3
        pos_mask = conc > 0
        if not pos_mask.any():
            return 1e-3
        # Use the last positive point (robust against trailing zeros from rounding)
        last_pos_idx = int(np.where(pos_mask)[0][-1])
        c_ref = float(conc[last_pos_idx])
        dt = float(time[last_pos_idx]) - float(time[0])
        if dt > 0 and c_ref > 0:
            k = -np.log(c_ref / c_first) / dt
            return max(float(k), 1e-6)
    except Exception:
        pass
    return 1e-3


def _estimate_order_simple(time: np.ndarray, conc: np.ndarray) -> float:
    """Compare Pearson |r| for n=0, 1, 2; return best-fitting order."""
    pos_mask = conc > 0
    best_order = 1.0
    best_r = 0.0
    # n=0: C vs t (use all points)
    if len(time) >= 2:
        try:
            r0, _ = stats.pearsonr(time, conc)
            if abs(r0) > best_r:
                best_r = abs(r0)
                best_order = 0.0
        except Exception:
            pass
    # n=1: ln(C) vs t  (positive points only — robust against zeros from rounding)
    if pos_mask.sum() >= 2:
        t_pos, c_pos = time[pos_mask], conc[pos_mask]
        try:
            r1, _ = stats.pearsonr(t_pos, np.log(c_pos))
            if abs(r1) > best_r:
                best_r = abs(r1)
                best_order = 1.0
        except Exception:
            pass
    # n=2: 1/C vs t  (positive points only)
    if pos_mask.sum() >= 2:
        t_pos, c_pos = time[pos_mask], conc[pos_mask]
        try:
            r2, _ = stats.pearsonr(t_pos, 1.0 / c_pos)
            if abs(r2) > best_r:
                best_r = abs(r2)
                best_order = 2.0
        except Exception:
            pass
    return best_order


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
# CI from Jacobian
# ---------------------------------------------------------------------------

def _compute_ci_from_jacobian(
    res,
    n_obs: int,
    param_indices: list[int],
) -> list[tuple[float, float]]:
    """
    Compute 95% CI for selected parameters from least_squares Jacobian.

    cov = pinv(J^T J) * MSE
    CI_i = param_i ± t_{alpha/2, dof} * sqrt(cov[i,i])

    Returns list of (lower, upper) for each param_index.
    """
    try:
        J = res.jac
        n_params = J.shape[1]
        dof = max(n_obs - n_params, 1)
        mse = 2.0 * res.cost / max(n_obs, 1)
        JtJ = J.T @ J
        cov = pinv(JtJ) * mse
        t_crit = stats.t.ppf(0.975, df=dof)
        result = []
        for idx in param_indices:
            if idx < cov.shape[0] and cov[idx, idx] >= 0:
                half_width = t_crit * np.sqrt(cov[idx, idx])
            else:
                half_width = np.nan
            param_val = float(res.x[idx])
            result.append((param_val - half_width, param_val + half_width))
        return result
    except Exception:
        return [(np.nan, np.nan)] * len(param_indices)


# ---------------------------------------------------------------------------
# Simple reaction fitting (with multistart)
# ---------------------------------------------------------------------------

def _run_fit_simple_multistart(
    df: pd.DataFrame,
    n_starts: int = 3,
) -> tuple[bool, float, float, float, float, float, object | None]:
    """
    Try multiple starting points, return best (by R²) fit result.

    Returns
    -------
    (success, k_fit, n_fit, c0_fit, r2_best, rmse, res_obj_or_None)
    """
    mask_A = df["concentration"].notna()
    time = df.loc[mask_A, "time"].to_numpy(dtype=float)
    conc = df.loc[mask_A, "concentration"].to_numpy(dtype=float)

    k_heuristic = _estimate_k_simple(time, conc)
    n_heuristic = _estimate_order_simple(time, conc)
    c0_init = max(float(conc[0]), 1e-9)

    # 3 starting points: heuristic, force 1st order, slower reaction
    starts = [
        (k_heuristic, n_heuristic, c0_init),
        (k_heuristic, 1.0, c0_init),
        (k_heuristic * 0.1, n_heuristic, c0_init),
    ]

    best_r2 = -np.inf
    best_result = None
    best_params = (k_heuristic, n_heuristic, c0_init)
    best_success = False

    def residual_fn(params):
        k, n, c0 = params
        if k <= 0 or c0 <= 0:
            return np.full(len(time), 1e6)
        sol_y = solve_and_predict(
            _ode_simple, (time[0], time[-1]), [c0], time, (k, max(n, 0.0))
        )
        if sol_y is None:
            return np.full(len(time), 1e6)
        return sol_y[0] - conc

    for k_s, n_s, c0_s in starts[:n_starts]:
        try:
            res = least_squares(
                residual_fn,
                [k_s, n_s, c0_s],
                bounds=([1e-9, 0.0, 1e-9], [np.inf, 5.0, np.inf]),
                method="trf",
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
                max_nfev=1000,
            )
            k_t, n_t, c0_t = float(res.x[0]), float(res.x[1]), float(res.x[2])
            sol_obs = solve_and_predict(
                _ode_simple, (time[0], time[-1]), [c0_t], time, (k_t, n_t)
            )
            if sol_obs is not None:
                r2_t = _safe_r2(conc, sol_obs[0])
            else:
                r2_t = -np.inf
            if r2_t > best_r2:
                best_r2 = r2_t
                best_params = (k_t, n_t, c0_t)
                best_success = bool(res.success or res.cost < 1e-6)
                best_result = res
        except Exception:
            continue

    if best_result is None:
        # All starts failed: return NaN
        return False, np.nan, np.nan, np.nan, 0.0, np.nan, None

    k_fit, n_fit, c0_fit = best_params
    sol_obs = solve_and_predict(
        _ode_simple, (time[0], time[-1]), [c0_fit], time, (k_fit, n_fit)
    )
    if sol_obs is not None:
        r2 = _safe_r2(conc, sol_obs[0])
        rmse = float(np.sqrt(np.nanmean((conc - sol_obs[0]) ** 2)))
    else:
        r2, rmse = 0.0, np.nan

    return best_success, k_fit, n_fit, c0_fit, r2, rmse, best_result


# ---------------------------------------------------------------------------
# Public fitting functions
# ---------------------------------------------------------------------------

def run_fit_simple(df: pd.DataFrame) -> FitResult:
    """Fit A→products via RK45 + least_squares with multistart."""
    mask_A = df["concentration"].notna()
    n_valid = int(mask_A.sum())

    _fail_result = FitResult(
        reaction_type="simple", order=1.0, k=np.nan, k2=None,
        k_ci_lower=np.nan, k_ci_upper=np.nan,
        k2_ci_lower=None, k2_ci_upper=None,
        r2=0.0, rmse=np.nan, success=False,
        message=f"有効データが {n_valid} 点のみです（最低 {_MIN_POINTS_RK4} 点必要）。",
        t_pred=np.array([]), c_pred={"A": np.array([])},
        residuals={"A": np.array([])}, n_points=n_valid,
    )

    if n_valid < _MIN_POINTS_RK4:
        return _fail_result

    time = df.loc[mask_A, "time"].to_numpy(dtype=float)
    conc = df.loc[mask_A, "concentration"].to_numpy(dtype=float)

    success, k_fit, n_fit, c0_fit, r2, rmse, res_obj = _run_fit_simple_multistart(df)

    if not success and np.isnan(k_fit):
        return FitResult(
            reaction_type="simple", order=1.0, k=np.nan, k2=None,
            k_ci_lower=np.nan, k_ci_upper=np.nan,
            k2_ci_lower=None, k2_ci_upper=None,
            r2=0.0, rmse=np.nan, success=False,
            message="全ての初期値で収束しませんでした。",
            t_pred=np.array([]), c_pred={"A": np.array([])},
            residuals={"A": np.array([])}, n_points=n_valid,
        )

    # CI from Jacobian
    k_ci_lower, k_ci_upper = np.nan, np.nan
    if res_obj is not None:
        cis = _compute_ci_from_jacobian(res_obj, n_obs=n_valid, param_indices=[0])
        k_ci_lower, k_ci_upper = cis[0]

    # Dense prediction
    t_pred = np.linspace(time[0], time[-1], 300)
    sol_pred = solve_and_predict(_ode_simple, (time[0], time[-1]), [c0_fit], t_pred, (k_fit, n_fit))
    c_pred_A = sol_pred[0] if sol_pred is not None else np.full(300, np.nan)

    # Residuals at observations
    sol_obs = solve_and_predict(_ode_simple, (time[0], time[-1]), [c0_fit], time, (k_fit, n_fit))
    resid_A = (conc - sol_obs[0]) if sol_obs is not None else np.full(len(time), np.nan)

    msg = res_obj.message if res_obj is not None else "収束"

    return FitResult(
        reaction_type="simple", order=n_fit, k=k_fit, k2=None,
        k_ci_lower=float(k_ci_lower), k_ci_upper=float(k_ci_upper),
        k2_ci_lower=None, k2_ci_upper=None,
        r2=r2, rmse=rmse, success=success, message=msg,
        t_pred=t_pred, c_pred={"A": c_pred_A},
        residuals={"A": resid_A}, n_points=n_valid,
    )


def run_fit_sequential(df: pd.DataFrame) -> FitResult:
    """Fit A→B→C via RK45 + least_squares with n1, n2 estimation (multistart)."""
    if "concentration_B" not in df.columns:
        raise ValueError("逐次反応解析には concentration_B 列が必要です。")

    time  = df["time"].to_numpy(dtype=float)
    concA = df["concentration"].to_numpy(dtype=float)
    concB = df["concentration_B"].to_numpy(dtype=float)
    has_C = "concentration_C" in df.columns
    concC = df["concentration_C"].to_numpy(dtype=float) if has_C else np.full(len(time), np.nan)

    maskA = np.isfinite(concA)
    maskB = np.isfinite(concB)
    maskC = np.isfinite(concC)

    if maskA.sum() < 2:
        raise ValueError(f"有効な濃度A データが {maskA.sum()} 点のみです（最低2点必要）。")
    if maskB.sum() < 2:
        raise ValueError(f"有効な濃度B データが {maskB.sum()} 点のみです（最低2点必要）。")

    n_obs = int(maskA.sum() + maskB.sum() + (maskC.sum() if has_C else 0))
    c0A = float(concA[maskA][0])
    c0B = float(concB[maskB][0]) if maskB.any() else 0.0
    c0C = float(concC[maskC][0]) if maskC.any() else 0.0
    t_start, t_end = float(time[0]), float(time[-1])

    k1_init = _estimate_k_simple(time[maskA], concA[maskA])
    k2_init = _estimate_k2_sequential(time, concB, maskB, k1_init)
    n1_init = _estimate_order_simple(time[maskA], concA[maskA])
    # Estimate n2 from B data; use 1.0 when B starts at 0 (typical in sequential)
    # or when insufficient B data
    B_start = float(concB[maskB][0]) if maskB.any() else 0.0
    if maskB.sum() >= 3 and B_start > 1e-6 * max(float(concA[maskA][0]), 1e-9):
        n2_init = _estimate_order_simple(time[maskB], concB[maskB])
    else:
        n2_init = 1.0

    n_res = n_obs

    def residual_fn(params):
        k1, k2, n1, n2 = params
        if k1 <= 0 or k2 <= 0:
            return np.full(n_res, 1e6)
        sol_y = solve_and_predict(
            _ode_sequential, (t_start, t_end), [c0A, c0B, c0C], time,
            (k1, k2, max(n1, 0.0), max(n2, 0.0))
        )
        if sol_y is None:
            return np.full(n_res, 1e6)
        rA = sol_y[0][maskA] - concA[maskA]
        rB = sol_y[1][maskB] - concB[maskB]
        parts = [rA, rB]
        if has_C and maskC.any():
            parts.append(sol_y[2][maskC] - concC[maskC])
        return np.concatenate(parts)

    starts = [
        [k1_init, k2_init, n1_init, n2_init],
        [k1_init, k2_init, 1.0, 1.0],
        [k1_init * 0.1, k2_init * 0.1, n1_init, n2_init],
    ]

    best_r2 = -np.inf
    res_obj = None
    k1_fit = k2_fit = n1_fit = n2_fit = np.nan
    success = False
    msg = "収束失敗"

    for x0 in starts:
        try:
            res_t = least_squares(
                residual_fn, x0,
                bounds=([1e-9, 1e-9, 0.0, 0.0], [np.inf, np.inf, 5.0, 5.0]),
                method="trf",
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
                max_nfev=2000,
            )
            k1_t, k2_t, n1_t, n2_t = (
                float(res_t.x[0]), float(res_t.x[1]),
                float(res_t.x[2]), float(res_t.x[3]),
            )
            sol_t = solve_and_predict(
                _ode_sequential, (t_start, t_end), [c0A, c0B, c0C], time[maskA],
                (k1_t, k2_t, n1_t, n2_t)
            )
            r2_t = _safe_r2(concA[maskA], sol_t[0]) if sol_t is not None else -np.inf
            if r2_t > best_r2:
                best_r2 = r2_t
                res_obj = res_t
                k1_fit, k2_fit, n1_fit, n2_fit = k1_t, k2_t, n1_t, n2_t
                success = bool(res_t.success or res_t.cost < 1e-6)
                msg = res_t.message
        except Exception:
            continue

    # CI for k1, k2 only (param_indices=[0,1])
    k_ci_lower, k_ci_upper = np.nan, np.nan
    k2_ci_lower, k2_ci_upper = np.nan, np.nan
    if res_obj is not None and not np.isnan(k1_fit):
        cis = _compute_ci_from_jacobian(res_obj, n_obs=n_obs, param_indices=[0, 1])
        k_ci_lower, k_ci_upper = cis[0]
        k2_ci_lower, k2_ci_upper = cis[1]

    if np.isnan(k1_fit):
        return FitResult(
            reaction_type="sequential", order=1.0, k=np.nan, k2=np.nan,
            k_ci_lower=np.nan, k_ci_upper=np.nan,
            k2_ci_lower=np.nan, k2_ci_upper=np.nan,
            r2=0.0, rmse=np.nan, success=False, message=msg,
            t_pred=np.array([]), c_pred={}, residuals={}, n_points=n_obs,
            order2=None,
        )

    # Dense prediction
    t_pred = np.linspace(t_start, t_end, 300)
    sol_pred = solve_and_predict(
        _ode_sequential, (t_start, t_end), [c0A, c0B, c0C], t_pred,
        (k1_fit, k2_fit, n1_fit, n2_fit)
    )
    if sol_pred is not None:
        cA_pred, cB_pred, cC_pred = sol_pred[0], sol_pred[1], sol_pred[2]
    else:
        cA_pred = cB_pred = cC_pred = np.full(300, np.nan)

    # R² and residuals
    sol_obs = solve_and_predict(
        _ode_sequential, (t_start, t_end), [c0A, c0B, c0C], time,
        (k1_fit, k2_fit, n1_fit, n2_fit)
    )
    if sol_obs is not None:
        r2_vals = [_safe_r2(concA[maskA], sol_obs[0][maskA]),
                   _safe_r2(concB[maskB], sol_obs[1][maskB])]
        if has_C and maskC.any():
            r2_vals.append(_safe_r2(concC[maskC], sol_obs[2][maskC]))
        r2 = float(np.mean(r2_vals))
        rmse = float(np.sqrt(np.nanmean((concA[maskA] - sol_obs[0][maskA]) ** 2)))
        resid_parts = {
            "A": concA - sol_obs[0],
            "B": concB - sol_obs[1],
            "C": (concC - sol_obs[2]) if has_C else np.zeros(len(time)),
        }
    else:
        r2, rmse = 0.0, np.nan
        resid_parts = {
            "A": np.full(len(time), np.nan),
            "B": np.full(len(time), np.nan),
            "C": np.full(len(time), np.nan),
        }

    return FitResult(
        reaction_type="sequential", order=n1_fit, k=k1_fit, k2=k2_fit,
        k_ci_lower=float(k_ci_lower), k_ci_upper=float(k_ci_upper),
        k2_ci_lower=float(k2_ci_lower), k2_ci_upper=float(k2_ci_upper),
        r2=r2, rmse=rmse, success=success, message=msg,
        t_pred=t_pred,
        c_pred={"A": cA_pred, "B": cB_pred, "C": cC_pred},
        residuals=resid_parts,
        n_points=n_obs,
        order2=n2_fit,
    )


def run_fit_parallel(df: pd.DataFrame) -> FitResult:
    """Fit A→B + A→C via RK45 + least_squares with shared n estimation (multistart)."""
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

    n_obs = int(maskA.sum() + maskB.sum() + maskC.sum())
    c0A = float(concA[maskA][0])
    c0B = float(concB[maskB][0]) if maskB.any() else 0.0
    c0C = float(concC[maskC][0]) if maskC.any() else 0.0
    t_start, t_end = float(time[0]), float(time[-1])

    k_total = _estimate_k_simple(time[maskA], concA[maskA])
    k1_init, k2_init = _estimate_k_parallel(
        time, concA, concB, concC, maskA, maskB, maskC, k_total
    )
    n_init = _estimate_order_simple(time[maskA], concA[maskA])

    n_res = n_obs

    def residual_fn(params):
        k1, k2, n = params
        if k1 <= 0 or k2 <= 0:
            return np.full(n_res, 1e6)
        sol_y = solve_and_predict(
            _ode_parallel, (t_start, t_end), [c0A, c0B, c0C], time,
            (k1, k2, max(n, 0.0))
        )
        if sol_y is None:
            return np.full(n_res, 1e6)
        rA = sol_y[0][maskA] - concA[maskA]
        rB = sol_y[1][maskB] - concB[maskB]
        rC = sol_y[2][maskC] - concC[maskC]
        return np.concatenate([rA, rB, rC])

    starts = [
        [k1_init, k2_init, n_init],
        [k1_init, k2_init, 1.0],
        [k1_init * 0.1, k2_init * 0.1, n_init],
    ]

    best_r2 = -np.inf
    res_obj = None
    k1_fit = k2_fit = n_fit = np.nan
    success = False
    msg = "収束失敗"

    for x0 in starts:
        try:
            res_t = least_squares(
                residual_fn, x0,
                bounds=([1e-9, 1e-9, 0.0], [np.inf, np.inf, 5.0]),
                method="trf",
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
                max_nfev=2000,
            )
            k1_t, k2_t, n_t = (
                float(res_t.x[0]), float(res_t.x[1]), float(res_t.x[2])
            )
            sol_t = solve_and_predict(
                _ode_parallel, (t_start, t_end), [c0A, c0B, c0C], time[maskA],
                (k1_t, k2_t, n_t)
            )
            r2_t = _safe_r2(concA[maskA], sol_t[0]) if sol_t is not None else -np.inf
            if r2_t > best_r2:
                best_r2 = r2_t
                res_obj = res_t
                k1_fit, k2_fit, n_fit = k1_t, k2_t, n_t
                success = bool(res_t.success or res_t.cost < 1e-6)
                msg = res_t.message
        except Exception:
            continue

    # CI for k1, k2 only (param_indices=[0,1])
    k_ci_lower, k_ci_upper = np.nan, np.nan
    k2_ci_lower, k2_ci_upper = np.nan, np.nan
    if res_obj is not None and not np.isnan(k1_fit):
        cis = _compute_ci_from_jacobian(res_obj, n_obs=n_obs, param_indices=[0, 1])
        k_ci_lower, k_ci_upper = cis[0]
        k2_ci_lower, k2_ci_upper = cis[1]

    if np.isnan(k1_fit):
        return FitResult(
            reaction_type="parallel", order=1.0, k=np.nan, k2=np.nan,
            k_ci_lower=np.nan, k_ci_upper=np.nan,
            k2_ci_lower=np.nan, k2_ci_upper=np.nan,
            r2=0.0, rmse=np.nan, success=False, message=msg,
            t_pred=np.array([]), c_pred={}, residuals={}, n_points=n_obs,
        )

    # Dense prediction
    t_pred = np.linspace(t_start, t_end, 300)
    sol_pred = solve_and_predict(
        _ode_parallel, (t_start, t_end), [c0A, c0B, c0C], t_pred,
        (k1_fit, k2_fit, n_fit)
    )
    if sol_pred is not None:
        cA_pred, cB_pred, cC_pred = sol_pred[0], sol_pred[1], sol_pred[2]
    else:
        cA_pred = cB_pred = cC_pred = np.full(300, np.nan)

    sol_obs = solve_and_predict(
        _ode_parallel, (t_start, t_end), [c0A, c0B, c0C], time,
        (k1_fit, k2_fit, n_fit)
    )
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

    return FitResult(
        reaction_type="parallel", order=n_fit, k=k1_fit, k2=k2_fit,
        k_ci_lower=float(k_ci_lower), k_ci_upper=float(k_ci_upper),
        k2_ci_lower=float(k2_ci_lower), k2_ci_upper=float(k2_ci_upper),
        r2=r2, rmse=rmse, success=success, message=msg,
        t_pred=t_pred,
        c_pred={"A": cA_pred, "B": cB_pred, "C": cC_pred},
        residuals=resid_parts,
        n_points=n_obs,
    )


def run_fit(df: pd.DataFrame, reaction_type: str) -> FitResult:
    """Dispatcher: run appropriate fit based on reaction_type."""
    if reaction_type == "sequential":
        has_B = "concentration_B" in df.columns and df["concentration_B"].notna().any()
        if not has_B:
            return run_fit_simple(df)
        return run_fit_sequential(df)
    elif reaction_type == "parallel":
        has_B = "concentration_B" in df.columns and df["concentration_B"].notna().any()
        has_C = "concentration_C" in df.columns and df["concentration_C"].notna().any()
        if not has_B or not has_C:
            return run_fit_simple(df)
        return run_fit_parallel(df)
    else:
        return run_fit_simple(df)
