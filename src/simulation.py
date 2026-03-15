"""Forward simulation of reaction kinetics for arbitrary conditions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.models import ArrheniusResult
from src.ode_systems import (
    _ode_parallel,
    _ode_sequential,
    _ode_simple,
    solve_and_predict,
)

R_GAS = 8.314  # J/(mol·K)


@dataclass
class SimulationCondition:
    label: str
    reaction_type: str        # "simple" | "sequential" | "parallel"
    k: float
    n: float
    k2: Optional[float] = None   # sequential/parallel のみ
    n2: Optional[float] = None   # sequential のみ (B→C 次数)
    A0: float = 1.0
    t_end: float = 60.0
    n_points: int = 500


def k_from_arrhenius(arr: ArrheniusResult, T_celsius: float) -> float:
    """k = A * exp(-Ea / (R * T_K))。T_celsius <= -273.15 は ValueError。"""
    T_K = T_celsius + 273.15
    if T_K <= 0:
        raise ValueError(f"絶対温度が 0 以下になります: T = {T_K:.2f} K")
    try:
        val = arr.A * math.exp(-arr.Ea / (R_GAS * T_K))
    except OverflowError:
        raise ValueError(f"k の計算でオーバーフローが発生しました (T = {T_celsius}°C)")
    return val


def run_simulation(
    cond: SimulationCondition,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """ODE 求解。(t_eval, {"A": ..., "B": ..., "C": ...}) を返す。失敗時 None。"""
    t_eval = np.linspace(0.0, cond.t_end, cond.n_points)
    t_span = (0.0, cond.t_end)

    rt = cond.reaction_type

    if rt == "simple":
        y0   = [cond.A0]
        args = (cond.k, cond.n)
        y    = solve_and_predict(_ode_simple, t_span, y0, t_eval, args)
        if y is None:
            return None
        return t_eval, {"A": np.maximum(y[0], 0.0)}

    elif rt == "sequential":
        k2 = cond.k2 if (cond.k2 is not None and np.isfinite(cond.k2)) else 0.0
        n2 = cond.n2 if (cond.n2 is not None and np.isfinite(cond.n2)) else 1.0
        y0   = [cond.A0, 0.0, 0.0]
        args = (cond.k, k2, cond.n, n2)
        y    = solve_and_predict(_ode_sequential, t_span, y0, t_eval, args)
        if y is None:
            return None
        return t_eval, {
            "A": np.maximum(y[0], 0.0),
            "B": np.maximum(y[1], 0.0),
            "C": np.maximum(y[2], 0.0),
        }

    elif rt == "parallel":
        k2 = cond.k2 if (cond.k2 is not None and np.isfinite(cond.k2)) else 0.0
        y0   = [cond.A0, 0.0, 0.0]
        args = (cond.k, k2, cond.n)
        y    = solve_and_predict(_ode_parallel, t_span, y0, t_eval, args)
        if y is None:
            return None
        return t_eval, {
            "A": np.maximum(y[0], 0.0),
            "B": np.maximum(y[1], 0.0),
            "C": np.maximum(y[2], 0.0),
        }

    return None


def run_all_simulations(
    conditions: list[SimulationCondition],
) -> list[tuple[SimulationCondition, np.ndarray | None, dict | None]]:
    """全条件を実行。結果は (cond, t_eval, c_dict) のリスト。"""
    results = []
    for cond in conditions:
        out = run_simulation(cond)
        if out is None:
            results.append((cond, None, None))
        else:
            t_eval, c_dict = out
            results.append((cond, t_eval, c_dict))
    return results


def build_csv(sim_results) -> str:
    """全条件の結果をスタック形式 CSV に変換（utf-8-sig 用の str）。
    列: condition_label, time, A (,B ,C for multi-species)"""
    rows: list[str] = []

    # Determine which species columns are needed
    has_B = any(c_dict is not None and "B" in c_dict for _, _, c_dict in sim_results)
    has_C = any(c_dict is not None and "C" in c_dict for _, _, c_dict in sim_results)

    header_parts = ["condition_label", "time", "A"]
    if has_B:
        header_parts.append("B")
    if has_C:
        header_parts.append("C")
    rows.append(",".join(header_parts))

    for cond, t_eval, c_dict in sim_results:
        if t_eval is None or c_dict is None:
            continue
        for i, t in enumerate(t_eval):
            parts = [cond.label, f"{t:.6g}", f"{c_dict['A'][i]:.6g}"]
            if has_B:
                parts.append(f"{c_dict['B'][i]:.6g}" if "B" in c_dict else "")
            if has_C:
                parts.append(f"{c_dict['C'][i]:.6g}" if "C" in c_dict else "")
            rows.append(",".join(parts))

    return "\n".join(rows)
