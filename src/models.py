"""Data model definitions for reaction kinetics analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FitResult:
    """RK4+LSQ の1回分のフィッティング結果"""
    reaction_type: str          # "simple" | "sequential" | "parallel"
    order: float                # n (simple), n1 (sequential A→B), n (parallel)
    k: float                    # k または k1
    k2: Optional[float]         # k2（逐次・並列のみ）
    k_ci_lower: float           # k の 95% CI 下限
    k_ci_upper: float           # k の 95% CI 上限
    k2_ci_lower: Optional[float]
    k2_ci_upper: Optional[float]
    r2: float
    rmse: float
    success: bool
    message: str
    t_pred: np.ndarray
    c_pred: dict                # dict[str, np.ndarray]
    residuals: dict             # dict[str, np.ndarray]
    n_points: int
    order2: Optional[float] = None  # n2 (sequential B→C のみ; 他は None)


@dataclass
class ArrheniusResult:
    """アレニウス解析結果"""
    temps_celsius: list
    k_values: list
    ln_k: list
    inv_T: list
    Ea: float                   # J/mol
    Ea_kJmol: float             # kJ/mol
    A: float
    r2: float
    slope: float
    intercept: float
    k_method: str = "RK4+LSQ"  # 常に RK4+LSQ


@dataclass
class AnalysisResult:
    """解析結果全体"""
    fit: FitResult
    reaction_type: str
    detected_reaction_type: str
    detected_reaction_reason: str
    is_multi_temp: bool
    per_temp_fits: list         # list[tuple[float, FitResult]]
    arrhenius: Optional[ArrheniusResult]
    arrhenius_k2: Optional[ArrheniusResult]  # 逐次・並列の k2 用
    optimal_order: Optional[float]
    warnings: list
