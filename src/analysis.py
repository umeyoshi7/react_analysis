"""Top-level analysis pipeline for reaction kinetics."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.arrhenius import run_arrhenius
from src.data_loader import auto_detect_reaction_type
from src.fitting import run_fit
from src.models import AnalysisResult, ArrheniusResult, FitResult


def run_analysis(
    df: pd.DataFrame,
    reaction_type: str,
    temp_groups: Optional[dict] = None,
) -> AnalysisResult:
    """
    Run complete kinetic analysis pipeline.

    Steps
    -----
    1. auto_detect_reaction_type(df)
    2. run_fit(df, reaction_type) → FitResult (all data combined)
    3. If multi-temp: run_fit() for each temperature → per_temp_fits
    4. run_arrhenius() → arrhenius (and arrhenius_k2 if sequential/parallel)
    5. optimal_order = R²-weighted mean of per_temp_fits[i].order

    Parameters
    ----------
    df            : experiment DataFrame
    reaction_type : "simple" | "sequential" | "parallel"
    temp_groups   : {T_celsius: sub_df} or None
    """
    warnings: list[str] = []

    # 1. Auto-detect
    detected_type, detected_reason = auto_detect_reaction_type(df)
    if detected_type != reaction_type:
        warnings.append(
            f"自動判定では「{_type_label(detected_type)}」を推奨しますが、"
            f"「{_type_label(reaction_type)}」で解析します。"
        )

    # 2. Fit all data
    fit: FitResult = run_fit(df, reaction_type)
    if not fit.success:
        warnings.append(f"フィッティング収束エラー: {fit.message}")

    is_multi_temp = temp_groups is not None and len(temp_groups) >= 2

    if is_multi_temp:
        warnings.append(
            "多温度データが検出されました。温度別解析・Arrheniusパラメータは"
            "「Arrheniusパラメータ」タブで確認できます。"
        )

    # 3. Per-temperature fits
    per_temp_fits: list[tuple[float, FitResult]] = []
    if is_multi_temp:
        for T_c, sub_df in temp_groups.items():
            try:
                sub_fit = run_fit(sub_df, reaction_type)
                per_temp_fits.append((T_c, sub_fit))
            except Exception as e:
                warnings.append(f"{T_c:.1f}°C の解析でエラー: {e}")

    # 4. Arrhenius (use pre-computed per_temp_fits to avoid redundant ODE fitting)
    arrhenius: Optional[ArrheniusResult] = None
    arrhenius_k2: Optional[ArrheniusResult] = None
    if is_multi_temp:
        fits_for_arrhenius = per_temp_fits if per_temp_fits else None
        try:
            arrhenius = run_arrhenius(
                temp_groups, reaction_type, k_index=1,
                per_temp_fits=fits_for_arrhenius,
            )
            if arrhenius is None:
                warnings.append(
                    "アレニウス解析: 有効な温度グループが2点未満のためスキップしました。"
                )
        except Exception as e:
            warnings.append(f"アレニウス解析 (k1) でエラー: {e}")

        if reaction_type in ("sequential", "parallel"):
            try:
                arrhenius_k2 = run_arrhenius(
                    temp_groups, reaction_type, k_index=2,
                    per_temp_fits=fits_for_arrhenius,
                )
            except Exception as e:
                warnings.append(f"アレニウス解析 (k2) でエラー: {e}")

    # 5. Optimal order (R²-weighted mean over per-temp fits)
    optimal_order: Optional[float] = None
    if per_temp_fits:
        valid = [
            (fit_i.order, fit_i.r2)
            for _, fit_i in per_temp_fits
            if fit_i.success and np.isfinite(fit_i.order) and np.isfinite(fit_i.r2)
        ]
        if valid:
            orders, weights = zip(*valid)
            w = np.array(weights, dtype=float)
            if w.sum() > 0:
                optimal_order = float(np.average(orders, weights=w))
            else:
                optimal_order = float(np.mean(orders))
            # Clamp to [0, 5]
            optimal_order = float(np.clip(optimal_order, 0.0, 5.0))

    return AnalysisResult(
        fit=fit,
        reaction_type=reaction_type,
        detected_reaction_type=detected_type,
        detected_reaction_reason=detected_reason,
        is_multi_temp=is_multi_temp,
        per_temp_fits=per_temp_fits,
        arrhenius=arrhenius,
        arrhenius_k2=arrhenius_k2,
        optimal_order=optimal_order,
        warnings=warnings,
    )


def _type_label(t: str) -> str:
    return {
        "simple": "単純反応",
        "sequential": "逐次反応",
        "parallel": "並列反応",
    }.get(t, t)
