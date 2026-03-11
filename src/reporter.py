"""Excel report generation for kinetics analysis results."""

from __future__ import annotations

import io
import math
from datetime import datetime

import pandas as pd
import xlsxwriter

from src.models import AnalysisResult

REACTION_TYPE_LABELS = {
    "simple":     "単純反応 A→products",
    "sequential": "逐次反応 A→B→C",
    "parallel":   "並列反応 A→B + A→C",
}


def generate_excel_report(
    df: pd.DataFrame,
    result: AnalysisResult,
) -> bytes:
    """
    Generate an Excel report as bytes.

    Sheets:
        1. サマリー        – RK4+LSQ results, Arrhenius, k 95% CI
        2. 温度別解析      – per-temperature table (multi-temp only)
        3. 生データ        – original experiment data
    """
    output = io.BytesIO()
    wb = xlsxwriter.Workbook(output, {"in_memory": True, "nan_inf_to_errors": True})

    bold       = wb.add_format({"bold": True})
    header_fmt = wb.add_format({"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1})
    border     = wb.add_format({"border": 1})
    number4    = wb.add_format({"num_format": "0.0000", "border": 1})
    highlight  = wb.add_format({"bold": True, "bg_color": "#E2EFDA", "border": 1, "num_format": "0.0000"})

    fit = result.fit

    # =================================================================
    # Sheet 1: サマリー
    # =================================================================
    ws1 = wb.add_worksheet("サマリー")
    ws1.set_column("A:A", 32)
    ws1.set_column("B:B", 24)

    row = 0
    ws1.write(row, 0, "反応速度解析レポート", bold)
    ws1.write(row, 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    row += 2

    # Initial concentration
    t0_rows = df[df["time"] == df["time"].min()]
    c0 = t0_rows["concentration"].iloc[0] if not t0_rows.empty else ""
    ws1.write(row, 0, "【実験条件】", bold); row += 1
    ws1.write(row, 0, "初期濃度 C0 (mol/L)", border)
    ws1.write(row, 1, round(float(c0), 6) if c0 != "" else "", border); row += 1
    row += 1

    # Auto-detected reaction type
    ws1.write(row, 0, "【自動判定】", bold); row += 1
    ws1.write(row, 0, "推奨反応タイプ", border)
    ws1.write(row, 1, REACTION_TYPE_LABELS.get(result.detected_reaction_type, result.detected_reaction_type), border); row += 1
    ws1.write(row, 0, "判定理由", border)
    ws1.write(row, 1, result.detected_reaction_reason, border); row += 2

    # RK4+LSQ results
    rtype_label = REACTION_TYPE_LABELS.get(fit.reaction_type, fit.reaction_type)
    ws1.write(row, 0, f"【RK4+最小二乗法: {rtype_label}】", bold); row += 1

    def _fmt_val(v) -> str:
        if v is None:
            return "N/A"
        try:
            if math.isnan(float(v)) or math.isinf(float(v)):
                return "N/A"
            return f"{float(v):.6f}"
        except (TypeError, ValueError):
            return str(v)

    rk_rows: list[tuple[str, str]] = [
        ("速度定数 k1 (min⁻¹)", _fmt_val(fit.k)),
        ("k 95%CI 下限",        _fmt_val(fit.k_ci_lower)),
        ("k 95%CI 上限",        _fmt_val(fit.k_ci_upper)),
    ]
    if fit.k2 is not None:
        rk_rows += [
            ("速度定数 k2 (min⁻¹)", _fmt_val(fit.k2)),
            ("k2 95%CI 下限",       _fmt_val(fit.k2_ci_lower)),
            ("k2 95%CI 上限",       _fmt_val(fit.k2_ci_upper)),
        ]
    if fit.reaction_type == "simple":
        rk_rows.append(("推定反応次数 n", _fmt_val(fit.order)))
    elif fit.reaction_type == "sequential":
        rk_rows.append(("推定次数 n1 (A→B)", _fmt_val(fit.order)))
        rk_rows.append(("推定次数 n2 (B→C)", _fmt_val(fit.order2)))
    elif fit.reaction_type == "parallel":
        rk_rows.append(("推定反応次数 n", _fmt_val(fit.order)))

    rk_rows += [
        ("R²",          _fmt_val(fit.r2)),
        ("RMSE",        _fmt_val(fit.rmse)),
        ("データ点数", str(fit.n_points)),
        ("収束",       "成功" if fit.success else f"失敗: {fit.message}"),
    ]

    for label, val in rk_rows:
        ws1.write(row, 0, label, border)
        ws1.write(row, 1, val, border)
        row += 1

    # Arrhenius
    for arr, label in [(result.arrhenius, "k1"), (result.arrhenius_k2, "k2")]:
        if arr is None:
            continue
        row += 1
        ws1.write(row, 0, f"【アレニウス解析 ({label})】", bold); row += 1
        for lbl, val in [
            ("温度点数",                        str(len(arr.temps_celsius))),
            ("活性化エネルギー Ea (kJ/mol)",    f"{arr.Ea_kJmol:.3f}"),
            ("頻度因子 A",                       f"{arr.A:.4e}"),
            ("R² (アレニウス)",                  f"{arr.r2:.6f}"),
            ("解析手法",                         arr.k_method),
        ]:
            ws1.write(row, 0, lbl, border)
            ws1.write(row, 1, val, border)
            row += 1

    # Optimal order (multi-temp)
    if result.optimal_order is not None:
        row += 1
        ws1.write(row, 0, "【温度別 R²加重平均 反応次数】", bold); row += 1
        ws1.write(row, 0, "最適反応次数 n", border)
        ws1.write(row, 1, f"{result.optimal_order:.4f}", border); row += 1

    # Warnings
    if result.warnings:
        row += 1
        ws1.write(row, 0, "【警告】", bold); row += 1
        for w in result.warnings:
            ws1.write(row, 0, w); row += 1

    # =================================================================
    # Sheet 2: 温度別解析 (multi-temp only)
    # =================================================================
    if result.per_temp_fits:
        ws2 = wb.add_worksheet("温度別解析")
        ws2.set_column("A:H", 18)

        headers2 = [
            "温度 (°C)", "温度 (K)",
            "k", "k 95%CI 下限", "k 95%CI 上限",
            "n", "R²", "収束",
        ]
        for col, h in enumerate(headers2):
            ws2.write(0, col, h, header_fmt)

        for r_idx, (T_c, fit_i) in enumerate(result.per_temp_fits, start=1):
            T_K = T_c + 273.15
            ws2.write(r_idx, 0, T_c, border)
            ws2.write(r_idx, 1, T_K, border)
            ws2.write(r_idx, 2, _fmt_val(fit_i.k), border)
            ws2.write(r_idx, 3, _fmt_val(fit_i.k_ci_lower), border)
            ws2.write(r_idx, 4, _fmt_val(fit_i.k_ci_upper), border)
            ws2.write(r_idx, 5, _fmt_val(fit_i.order), border)
            ws2.write(r_idx, 6, _fmt_val(fit_i.r2), border)
            ws2.write(r_idx, 7, "成功" if fit_i.success else "失敗", border)

        if result.optimal_order is not None:
            last_row = len(result.per_temp_fits) + 2
            ws2.write(last_row, 0, "R²加重平均 n", bold)
            ws2.write(last_row, 5, result.optimal_order, highlight)

    # =================================================================
    # Sheet 3: 生データ
    # =================================================================
    ws3 = wb.add_worksheet("生データ")
    raw_headers = ["時間 (min)", "濃度 [A] (mol/L)"]
    raw_cols    = ["time", "concentration"]
    for sp, col in [("B", "concentration_B"), ("C", "concentration_C")]:
        if col in df.columns:
            raw_headers.append(f"濃度 [{sp}] (mol/L)")
            raw_cols.append(col)
    raw_headers += ["温度 (°C)", "備考"]
    raw_cols    += ["temperature", "notes"]

    ws3.set_column(0, len(raw_headers) - 1, 18)
    for col, h in enumerate(raw_headers):
        ws3.write(0, col, h, header_fmt)

    for r_idx, row_data in df.iterrows():
        for col_idx, col_name in enumerate(raw_cols):
            val = row_data.get(col_name, "")
            if col_name == "notes":
                ws3.write(r_idx + 1, col_idx, str(val) if pd.notna(val) else "", border)
            else:
                try:
                    v = float(val)
                    ws3.write(r_idx + 1, col_idx, v if pd.notna(v) else "", border)
                except (ValueError, TypeError):
                    ws3.write(r_idx + 1, col_idx, str(val) if pd.notna(val) else "", border)

    wb.close()
    return output.getvalue()
