"""Excel report generation for kinetics analysis results."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd
import xlsxwriter

from src.kinetics import FullAnalysisResult


ORDER_LABELS = {0: "0次反応", 1: "1次反応", 2: "2次反応"}


def generate_excel_report(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    result: FullAnalysisResult,
) -> bytes:
    """
    Generate an Excel report as bytes.

    Sheets:
        1. サマリー  – integral, differential, RK4, Arrhenius
        2. 積分法詳細 – per-order regression
        3. 生データ   – original experiment data (incl. B/C)
    """
    output = io.BytesIO()
    wb = xlsxwriter.Workbook(output, {"in_memory": True, "nan_inf_to_errors": True})

    bold       = wb.add_format({"bold": True})
    header_fmt = wb.add_format({"bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1})
    border     = wb.add_format({"border": 1})
    number4    = wb.add_format({"num_format": "0.0000", "border": 1})
    highlight  = wb.add_format({"bold": True, "bg_color": "#E2EFDA", "border": 1, "num_format": "0.0000"})

    # =================================================================
    # Sheet 1: サマリー
    # =================================================================
    ws1 = wb.add_worksheet("サマリー")
    ws1.set_column("A:A", 30)
    ws1.set_column("B:B", 22)

    row = 0
    ws1.write(row, 0, "反応速度解析レポート", bold)
    ws1.write(row, 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    row += 2

    # Metadata
    ws1.write(row, 0, "【実験条件】", bold); row += 1
    meta_display = {
        "実験名":           metadata.get("experiment_name", ""),
        "反応物質":         metadata.get("substance", ""),
        "初期濃度 (mol/L)": metadata.get("initial_concentration", ""),
        "反応温度 (°C)":    metadata.get("temperature", ""),
        "実験日":           str(metadata.get("experiment_date", "")),
        "担当者":           metadata.get("operator", ""),
        "備考":             metadata.get("notes", ""),
    }
    for k, v in meta_display.items():
        ws1.write(row, 0, k, border); ws1.write(row, 1, str(v), border); row += 1

    row += 1

    # Auto-detected reaction type
    ws1.write(row, 0, "【自動判定】", bold); row += 1
    ws1.write(row, 0, "推奨反応タイプ", border)
    ws1.write(row, 1, result.detected_reaction_type, border); row += 1
    ws1.write(row, 0, "判定理由", border)
    ws1.write(row, 1, result.detected_reaction_reason, border); row += 2

    # Integral results
    ws1.write(row, 0, "【積分法 解析結果】", bold); row += 1
    best = result.best_result
    best_order = result.best_order
    for label, val in [
        ("推定反応次数 (自動)", ORDER_LABELS[best_order]),
        ("速度定数 k",          f"{best.k:.6f}"),
        ("単位",                "min⁻¹" if best_order == 1 else ("mol·L⁻¹·min⁻¹" if best_order == 0 else "L·mol⁻¹·min⁻¹")),
        ("R²",                  f"{best.r2:.6f}"),
        ("k 95%CI 下限",        f"{best.k_ci_lower:.6f}"),
        ("k 95%CI 上限",        f"{best.k_ci_upper:.6f}"),
        ("AIC",                 f"{best.aic:.4f}"),
        ("データ点数",          str(best.n_points)),
    ]:
        ws1.write(row, 0, label, border); ws1.write(row, 1, val, border); row += 1

    if result.lsq is not None:
        row += 1
        ws1.write(row, 0, "【最小二乗法（解析解）】", bold); row += 1
        lsq = result.lsq
        for label, val in [
            ("推定反応次数 n", f"{lsq.n:.4f}"),
            ("速度定数 k",     f"{lsq.k:.6f}"),
            ("初期濃度 C0",    f"{lsq.C0:.6f}"),
            ("R²",            f"{lsq.r2:.6f}"),
            ("RMSE",          f"{lsq.rmse:.6f}" if not __import__("math").isnan(lsq.rmse) else "N/A"),
            ("収束",          "成功" if lsq.success else f"失敗: {lsq.message}"),
        ]:
            ws1.write(row, 0, label, border); ws1.write(row, 1, val, border); row += 1

    if result.rk4lsq is not None:
        row += 1
        rk = result.rk4lsq
        rtype = {"simple": "単純反応", "sequential": "逐次反応 A→B→C", "parallel": "並列反応 A→B + A→C"}
        ws1.write(row, 0, f"【RK4+最小二乗法: {rtype.get(rk.reaction_type, '')}】", bold); row += 1
        rk_rows: list[tuple[str, str]] = [("速度定数 k1", f"{rk.k:.6f}")]
        if rk.k2 is not None:
            rk_rows.append(("速度定数 k2", f"{rk.k2:.6f}"))
        if rk.reaction_type == "simple":
            rk_rows.append(("推定反応次数 n", f"{rk.order:.4f}"))
        rk_rows += [("R²", f"{rk.r2:.6f}"), ("RMSE", f"{rk.rmse:.6f}"),
                    ("収束", "成功" if rk.success else f"失敗: {rk.message}")]
        for label, val in rk_rows:
            ws1.write(row, 0, label, border); ws1.write(row, 1, val, border); row += 1

    for arr, title in [(result.arrhenius, "k1"), (result.arrhenius_k2, "k2")]:
        if arr is None:
            continue
        row += 1
        ws1.write(row, 0, f"【アレニウス解析 ({arr.k_label})】", bold); row += 1
        for label, val in [
            ("温度点数",                        str(arr.n_temperatures)),
            ("活性化エネルギー Ea (kJ/mol)",    f"{arr.Ea / 1000:.3f}"),
            ("Ea 95%CI 下限 (kJ/mol)",          f"{arr.Ea_ci_lower / 1000:.3f}"),
            ("Ea 95%CI 上限 (kJ/mol)",          f"{arr.Ea_ci_upper / 1000:.3f}"),
            ("頻度因子 A",                       f"{arr.A:.4e}"),
            ("R² (アレニウス)",                  f"{arr.r2:.6f}"),
        ]:
            ws1.write(row, 0, label, border); ws1.write(row, 1, val, border); row += 1

    if result.warnings:
        row += 1
        ws1.write(row, 0, "【警告】", bold); row += 1
        for w in result.warnings:
            ws1.write(row, 0, w); row += 1

    # =================================================================
    # Sheet 2: 積分法詳細
    # =================================================================
    ws2 = wb.add_worksheet("積分法詳細")
    ws2.set_column("A:G", 16)
    for col, h in enumerate(["反応次数", "k", "R²", "slope", "intercept", "AIC", "推奨"]):
        ws2.write(0, col, h, header_fmt)
    for r_idx, (order, res) in enumerate(result.integral.items(), start=1):
        is_best = order == result.best_order
        fmt     = highlight if is_best else border
        fmt_num = highlight if is_best else number4
        ws2.write(r_idx, 0, ORDER_LABELS[order], fmt)
        ws2.write(r_idx, 1, res.k,         fmt_num)
        ws2.write(r_idx, 2, res.r2,        fmt_num)
        ws2.write(r_idx, 3, res.slope,     fmt_num)
        ws2.write(r_idx, 4, res.intercept, fmt_num)
        ws2.write(r_idx, 5, res.aic,       fmt_num)
        ws2.write(r_idx, 6, "★ 推奨" if is_best else "", fmt)

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

    # =================================================================
    # Sheet 4: 温度別反応次数 (multi-temp simple reaction only)
    # =================================================================
    if result.per_temp_results:
        ws4 = wb.add_worksheet("温度別反応次数")
        ws4.set_column("A:G", 18)

        headers4 = ["温度 (°C)", "温度 (K)", "推算次数 n", "速度定数 k", "R²", "解法", "状態"]
        for col, h in enumerate(headers4):
            ws4.write(0, col, h, header_fmt)

        for r_idx, r in enumerate(result.per_temp_results, start=1):
            fmt_row = border
            ws4.write(r_idx, 0, r.temperature_C, fmt_row)
            ws4.write(r_idx, 1, r.temperature_K, fmt_row)
            if r.success:
                ws4.write(r_idx, 2, r.n,  number4)
                ws4.write(r_idx, 3, r.k,  number4)
                ws4.write(r_idx, 4, r.r2, number4)
            else:
                ws4.write(r_idx, 2, "—", fmt_row)
                ws4.write(r_idx, 3, "—", fmt_row)
                ws4.write(r_idx, 4, "—", fmt_row)
            ws4.write(r_idx, 5, r.method,  fmt_row)
            ws4.write(r_idx, 6, "成功" if r.success else f"失敗: {r.message}", fmt_row)

        if result.optimal_order_multi_temp is not None:
            last_row = len(result.per_temp_results) + 2
            ws4.write(last_row, 0, "R²加重平均 n", bold)
            ws4.write(last_row, 2, result.optimal_order_multi_temp, highlight)

    wb.close()
    return output.getvalue()
