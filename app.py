"""
反応速度定数・反応次数推算アプリ
Streamlit entry point
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    check_mass_balance,
    get_temperature_groups,
    load_experiment_data,
)
from src.kinetics import (
    FullAnalysisResult,
    auto_detect_reaction_type,
    run_full_analysis,
)
from src.plotting import (
    plot_arrhenius,
    plot_best_fit_conc,
    plot_integral_fit,
    plot_lsq_fit,
    plot_multi_species,
    plot_order_per_temp,
    plot_raw,
    plot_raw_multi_temp,
    plot_residuals,
    plot_rk4lsq_fit,
)
from src.reporter import generate_excel_report

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="反応速度解析アプリ",
    page_icon="⚗️",
    layout="wide",
)

TEMPLATE_PATH = Path(__file__).parent / "template" / "experiment_template.xlsx"
ORDER_LABELS  = {0: "0次反応", 1: "1次反応", 2: "2次反応"}
REACTION_TYPE_LABELS = {
    "simple":     "単純反応 A→products",
    "sequential": "逐次反応 A→B→C",
    "parallel":   "並列反応 A→B + A→C",
}


def _ensure_template() -> None:
    if not TEMPLATE_PATH.exists():
        try:
            import create_template
            create_template.create_template()
        except Exception as e:
            st.warning(f"テンプレートの自動生成に失敗しました: {e}")


_ensure_template()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "file_key":          None,
        "uploaded_df":       None,
        "metadata":          {},
        "analysis_results":  None,
        "analysis_complete": False,
        "load_warnings":     [],
        "temp_groups":       {},
        "detected_type":     "simple",
        "detected_reason":   "",
        "mass_balance_ok":   None,
        "mass_balance_cv":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚗️ 反応速度解析")
    st.markdown("---")

    # Template download
    st.subheader("1. テンプレートDL")
    if TEMPLATE_PATH.exists():
        with open(TEMPLATE_PATH, "rb") as f:
            st.download_button(
                label="📥 Excelテンプレートをダウンロード",
                data=f.read(),
                file_name="experiment_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.warning("テンプレートファイルが見つかりません。")

    st.markdown("---")

    # File upload
    st.subheader("2. データアップロード")
    uploaded_file = st.file_uploader(
        "実験データ (.xlsx)",
        type=["xlsx"],
        help=(
            "テンプレートに従って入力したExcelファイルをアップロードしてください。\n\n"
            "**対応フォーマット:**\n"
            "- 単純反応: 濃度A列のみ\n"
            "- 逐次/並列反応: 濃度A+B（+C）列\n"
            "- 複数温度: temperature列に各行の温度を記入\n"
            "- 成分ごとに時間点が異なる場合: 他成分の欄を空白にしてください"
        ),
    )

    if uploaded_file is not None:
        file_key = uploaded_file.name + str(uploaded_file.size)
        if file_key != st.session_state["file_key"]:
            st.session_state["file_key"]          = file_key
            st.session_state["analysis_complete"] = False
            st.session_state["analysis_results"]  = None
            with st.spinner("ファイルを読み込み中…"):
                try:
                    df, metadata, load_warnings = load_experiment_data(
                        io.BytesIO(uploaded_file.getvalue())
                    )
                    temp_groups   = get_temperature_groups(df)
                    detected_type, detected_reason = auto_detect_reaction_type(df)
                    mb_ok, mb_cv  = check_mass_balance(df)

                    st.session_state.update({
                        "uploaded_df":   df,
                        "metadata":      metadata,
                        "load_warnings": load_warnings,
                        "temp_groups":   temp_groups,
                        "detected_type":   detected_type,
                        "detected_reason": detected_reason,
                        "mass_balance_ok": mb_ok,
                        "mass_balance_cv": mb_cv,
                    })

                    n_valid_A = df["concentration"].notna().sum()
                    st.success(f"✅ {len(df)} 行 (濃度A: {n_valid_A}点) 読み込みました。")

                    has_B = "concentration_B" in df.columns
                    has_C = "concentration_C" in df.columns
                    n_temps = len(temp_groups)

                    if has_B:
                        st.info(f"複数成分データを検出: 自動判定 → **{REACTION_TYPE_LABELS[detected_type]}**")
                    if n_temps > 1:
                        st.info(f"複数温度データを検出 ({n_temps} 温度点) → Arrhenius解析が可能です")
                    if not mb_ok and has_B:
                        st.warning(f"質量バランス変動係数 = {mb_cv:.3f} (>5%): データを確認してください")

                except ValueError as e:
                    st.error(f"❌ {e}")
                    st.session_state["uploaded_df"] = None

    st.markdown("---")

    # Analysis settings
    st.subheader("3. 解析設定")

    df_loaded: pd.DataFrame | None = st.session_state["uploaded_df"]
    has_B_data = df_loaded is not None and "concentration_B" in df_loaded.columns and df_loaded["concentration_B"].notna().any()
    has_C_data = df_loaded is not None and "concentration_C" in df_loaded.columns and df_loaded["concentration_C"].notna().any()
    detected   = st.session_state.get("detected_type", "simple")

    # Reaction type selectbox – available options based on data
    rt_options  = ["single"]         # always
    rt_labels   = ["単純反応 A→products"]
    rt_values   = ["simple"]
    if has_B_data:
        rt_options.append("seq")
        rt_labels.append("逐次反応 A→B→C")
        rt_values.append("sequential")
    if has_B_data and has_C_data:
        rt_options.append("par")
        rt_labels.append("並列反応 A→B + A→C")
        rt_values.append("parallel")

    # Suggest auto-detected type as default
    default_idx = rt_values.index(detected) if detected in rt_values else 0
    selected_label = st.selectbox(
        "反応タイプ",
        rt_labels,
        index=default_idx,
        help=(
            "自動判定の推奨タイプが選択済みです。\n"
            + st.session_state.get("detected_reason", "")
        ),
    )
    reaction_type = rt_values[rt_labels.index(selected_label)]

    # Solver
    if reaction_type == "simple":
        solver_options = [
            "全解法を実行",
            "積分法のみ",
            "最小二乗法（解析解）のみ",
            "RK4法（ODE）のみ",
        ]
        solver_label  = st.selectbox("解法", solver_options)
        enable_lsq    = solver_label in ("全解法を実行", "最小二乗法（解析解）のみ")
        enable_rk4lsq = solver_label in ("全解法を実行", "RK4法（ODE）のみ")
    else:
        st.selectbox("解法", ["RK4法（ODE）"], disabled=True)
        enable_lsq    = False
        enable_rk4lsq = True

    st.markdown("---")
    run_btn = st.button(
        "🔬 解析実行",
        type="primary",
        disabled=(st.session_state["uploaded_df"] is None),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

if run_btn and st.session_state["uploaded_df"] is not None:
    with st.spinner("解析中…"):
        try:
            tg = st.session_state.get("temp_groups", {})
            result = run_full_analysis(
                st.session_state["uploaded_df"],
                reaction_type=reaction_type,
                enable_lsq=enable_lsq,
                enable_rk4=enable_rk4lsq,
                enable_arrhenius=len(tg) >= 2,
                temp_groups=tg if len(tg) >= 2 else None,
            )
            st.session_state["analysis_results"]  = result
            st.session_state["analysis_complete"] = True
        except Exception as e:
            st.error(f"❌ 解析エラー: {e}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("⚗️ 反応速度定数・反応次数推算アプリ")

if st.session_state["uploaded_df"] is None:
    st.info("👈 サイドバーからExcelファイルをアップロードして解析を開始してください。")
    st.markdown(
        """
        **対応する解析タイプ:**
        | タイプ | 必要な列 | 解析手法 |
        |--------|----------|----------|
        | 単純反応 A→products | 濃度A | 積分法・微分法・RK4 |
        | 逐次反応 A→B→C | 濃度A + 濃度B (+ 濃度C) | RK4+最小二乗法 |
        | 並列反応 A→B+A→C | 濃度A + 濃度B + 濃度C | RK4+最小二乗法 |
        | アレニウス解析 | 上記 + 複数温度点 | 線形回帰 |

        **濃度データについて:**
        - A, B, Cが異なる時間点で測定されている場合、他成分の欄を空白にしてください
        - 濃度Aのみのデータは単純反応として解析します
        - 濃度B/Cが全欠損の場合は自動的に除外します
        """
    )
    st.stop()

df: pd.DataFrame     = st.session_state["uploaded_df"]
metadata: dict       = st.session_state["metadata"]
load_warnings: list  = st.session_state.get("load_warnings", [])
temp_groups: dict    = st.session_state.get("temp_groups", {})
has_multi_species    = "concentration_B" in df.columns
has_multi_temp       = len(temp_groups) > 1

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 データ確認", "🔬 解析結果", "🌡️ Arrheniusパラメータ", "📄 レポート出力"]
)

# ===========================================================================
# Tab 1: Data overview
# ===========================================================================
with tab1:
    # Data quality summary
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("総行数", len(df))
    col_info2.metric("濃度A 有効点数", int(df["concentration"].notna().sum()))
    col_info3.metric("温度グループ数", len(temp_groups))

    # Auto-detection info
    detected_type   = st.session_state.get("detected_type", "simple")
    detected_reason = st.session_state.get("detected_reason", "")
    mb_ok = st.session_state.get("mass_balance_ok")
    mb_cv = st.session_state.get("mass_balance_cv")

    if has_multi_species:
        st.info(f"🔍 自動判定: **{REACTION_TYPE_LABELS[detected_type]}** — {detected_reason}")

    if mb_ok is not None and not mb_ok and has_multi_species:
        st.warning(
            f"⚠️ 質量バランス: A+B+C の変動係数 = {mb_cv:.3f} (閾値5% 超過)。"
            "開放系反応または測定誤差の可能性があります。解析結果を参考値としてください。"
        )
    elif mb_ok and has_multi_species:
        st.success(f"✅ 質量バランス OK (変動係数 = {mb_cv:.3f})")

    if metadata:
        st.subheader("実験条件")
        meta_df = pd.DataFrame(
            [{"項目": k, "値": v} for k, v in {
                "実験名":           metadata.get("experiment_name", ""),
                "反応物質":         metadata.get("substance", ""),
                "初期濃度 (mol/L)": metadata.get("initial_concentration", ""),
                "反応温度 (°C)":    metadata.get("temperature", ""),
                "実験日":           str(metadata.get("experiment_date", "")),
                "担当者":           metadata.get("operator", ""),
                "備考":             metadata.get("notes", ""),
            }.items()]
        )
        st.dataframe(meta_df, use_container_width=True, hide_index=True)

    if load_warnings:
        st.subheader("⚠️ データ品質警告")
        for w in load_warnings:
            st.warning(w)

    st.subheader("生データ")
    display_cols  = ["time", "concentration"]
    display_names = ["時間 (min)", "濃度 [A] (mol/L)"]
    for sp, col in [("B", "concentration_B"), ("C", "concentration_C")]:
        if col in df.columns:
            display_cols.append(col)
            display_names.append(f"濃度 [{sp}] (mol/L)")
    display_cols  += ["temperature", "notes"]
    display_names += ["温度 (°C)", "備考"]

    disp_df = df[[c for c in display_cols if c in df.columns]].copy()
    disp_df.columns = display_names[: len(disp_df.columns)]
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    # Concentration plot
    st.subheader("濃度 vs. 時間")
    if has_multi_species:
        st.plotly_chart(plot_multi_species(df), use_container_width=True, key="tab1_multi_species")
        if has_multi_temp:
            st.subheader(f"複数温度データ ({len(temp_groups)} 温度)")
            st.plotly_chart(plot_raw_multi_temp(temp_groups), use_container_width=True, key="tab1_multi_temp_species")
    elif has_multi_temp:
        # Primary view for single-species multi-temperature data
        st.plotly_chart(plot_raw_multi_temp(temp_groups), use_container_width=True, key="tab1_multi_temp")
    else:
        st.plotly_chart(plot_raw(df), use_container_width=True, key="tab1_raw")


# ===========================================================================
# Tab 2: Analysis results
# ===========================================================================
with tab2:
    if not st.session_state["analysis_complete"]:
        st.info("サイドバーの「解析実行」ボタンを押してください。")
        st.stop()

    result: FullAnalysisResult = st.session_state["analysis_results"]
    all_warnings = load_warnings + result.warnings

    if all_warnings:
        with st.expander("⚠️ 警告メッセージ", expanded=len(result.warnings) > 0):
            for w in all_warnings:
                st.warning(w)

    # ---- Primary result: RK4 if available for non-simple, else integral ----
    rk = result.rk4lsq
    is_multi = rk is not None and rk.reaction_type in ("sequential", "parallel")

    if is_multi and rk.success:
        st.subheader("主要解析結果 (RK4+最小二乗法)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("反応タイプ", REACTION_TYPE_LABELS.get(rk.reaction_type, rk.reaction_type))
        m2.metric("速度定数 k1", f"{rk.k:.5f} min⁻¹")
        if rk.k2 is not None:
            m3.metric("速度定数 k2", f"{rk.k2:.5f} min⁻¹")
        else:
            m3.metric("推定次数 n", f"{rk.order:.4f}")
        m4.metric("R²", f"{rk.r2:.5f}")
        st.plotly_chart(plot_rk4lsq_fit(df, rk), use_container_width=True, key="tab2_primary_rk4")
    else:
        st.subheader("主要解析結果 (積分法)")
        best      = result.best_result
        best_order = result.best_order
        is_optimal_n = result.optimal_order_multi_temp is not None
        ncols = 5 if is_optimal_n else 4
        cols = st.columns(ncols)
        cols[0].metric("推定反応次数", ORDER_LABELS[best_order])
        cols[1].metric("速度定数 k",  f"{best.k:.5f} min⁻¹")
        cols[2].metric("R²",          f"{best.r2:.5f}")
        cols[3].metric("95%CI (k)",   f"[{best.k_ci_lower:.4f}, {best.k_ci_upper:.4f}]")
        if is_optimal_n:
            cols[4].metric("最適次数 n (多温度)", f"{result.optimal_order_multi_temp:.4f}")
        st.plotly_chart(plot_best_fit_conc(df, result), use_container_width=True, key="tab2_primary_integral")
        if is_optimal_n:
            st.info(
                f"多温度解析による最適反応次数: **n = {result.optimal_order_multi_temp:.3f}**  "
                "(詳細は「Arrheniusパラメータ」タブを参照)"
            )

    st.markdown("---")

    # ---- Method subtabs ----
    subtab_titles = ["📐 積分法"]
    if result.lsq is not None:
        subtab_titles.append("📊 最小二乗法（解析解）")
    if rk is not None:
        subtab_titles.append("🔄 RK4法（ODE）")

    subtabs = st.tabs(subtab_titles)
    subtab_idx = 0

    # Integral
    with subtabs[subtab_idx]:
        subtab_idx += 1
        best       = result.best_result
        best_order = result.best_order
        st.caption("積分法は常に濃度A（成分A）のみを対象に解析します。")
        cols = st.columns(3)
        for idx, (order, res) in enumerate(result.integral.items()):
            with cols[idx]:
                is_best = order == best_order
                prefix  = "★ " if is_best else ""
                st.markdown(f"**{prefix}{ORDER_LABELS[order]}**")
                st.metric("R²", f"{res.r2:.4f}")
                st.metric("k",  f"{res.k:.5f}")
                st.plotly_chart(plot_integral_fit(res), use_container_width=True, key=f"tab2_integral_fit_{order}")
                st.plotly_chart(plot_residuals(res),    use_container_width=True, key=f"tab2_residuals_{order}")

        st.subheader("モデル比較")
        comp_df = pd.DataFrame([{
            "反応次数": ORDER_LABELS[o],
            "k":        round(r.k, 6),
            "R²":       round(r.r2, 6),
            "AIC":      round(r.aic, 4),
            "推奨":     "★ 推奨" if o == best_order else "",
        } for o, r in result.integral.items()])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # LSQ (analytical solution)
    if result.lsq is not None:
        with subtabs[subtab_idx]:
            subtab_idx += 1
            lsq = result.lsq
            st.caption("解析解（dC/dt = -k·Cⁿ の厳密解）を使った非線形最小二乗フィット。単純反応 A→products のみ。")
            l1, l2, l3, l4 = st.columns(4)
            l1.metric("推定次数 n",  f"{lsq.n:.4f}")
            l2.metric("速度定数 k",  f"{lsq.k:.5f}")
            l3.metric("R²",          f"{lsq.r2:.5f}")
            l4.metric("RMSE",        f"{lsq.rmse:.5f}" if np.isfinite(lsq.rmse) else "N/A")
            l5, l6 = st.columns(2)
            l5.metric("初期濃度 C0 (フィット)", f"{lsq.C0:.5f}")
            l6.metric("収束", "✅ 成功" if lsq.success else "❌ 失敗")
            if not lsq.success:
                st.warning(f"最適化メッセージ: {lsq.message}")
            st.plotly_chart(plot_lsq_fit(df, lsq), use_container_width=True, key="tab2_lsq_detail")

            if rk is not None and rk.reaction_type == "simple":
                st.subheader("手法比較")
                cmp_data = {
                    "手法":   ["積分法 (best)", "最小二乗法（解析解）", "RK4法（ODE）"],
                    "k":      [round(result.best_result.k, 6), round(lsq.k, 6), round(rk.k, 6)],
                    "n/次数": [result.best_order, round(lsq.n, 4), round(rk.order, 4)],
                    "R²":     [round(result.best_result.r2, 6), round(lsq.r2, 6), round(rk.r2, 6)],
                }
                st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

    # RK4 (ODE)
    if rk is not None:
        with subtabs[subtab_idx]:
            rtype_label = REACTION_TYPE_LABELS.get(rk.reaction_type, rk.reaction_type)
            st.caption("RK45 ODE ソルバーで反応速度式を数値積分し、最小二乗法でパラメータを最適化します。")
            st.markdown(f"**反応タイプ:** {rtype_label}")

            m1, m2, m3 = st.columns(3)
            m1.metric("速度定数 k1", f"{rk.k:.5f}")
            if rk.k2 is not None:
                m2.metric("速度定数 k2", f"{rk.k2:.5f}")
            else:
                m2.metric("推定次数 n",  f"{rk.order:.4f}")
            m3.metric("R²", f"{rk.r2:.5f}")

            m4, m5 = st.columns(2)
            m4.metric("RMSE", f"{rk.rmse:.5f}" if np.isfinite(rk.rmse) else "N/A")
            m5.metric("収束", "✅ 成功" if rk.success else "❌ 失敗")

            if not rk.success:
                st.warning(f"最適化メッセージ: {rk.message}")

            st.plotly_chart(plot_rk4lsq_fit(df, rk), use_container_width=True, key="tab2_rk4_detail")

            if rk.reaction_type == "simple" and result.lsq is None:
                st.subheader("積分法 vs RK4法 比較")
                cmp_data = {
                    "手法":   ["積分法 (best)", "RK4法（ODE）"],
                    "k":      [round(result.best_result.k, 6), round(rk.k, 6)],
                    "n/次数": [result.best_order, round(rk.order, 4)],
                    "R²":     [round(result.best_result.r2, 6), round(rk.r2, 6)],
                }
                st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)


# ===========================================================================
# Tab 3: Arrhenius
# ===========================================================================
with tab3:
    if not has_multi_temp:
        st.info(
            "アレニウス解析には複数温度のデータが必要です。\n\n"
            "**設定方法:** Excelファイルの `temperature` 列に各行の測定温度 (°C) を記入してください。"
            "例: 25°Cと40°Cで測定した場合、それぞれの行に25.0または40.0と入力します。"
        )
    elif not st.session_state["analysis_complete"]:
        st.info("サイドバーの「解析実行」ボタンを押してください。")
    else:
        result: FullAnalysisResult = st.session_state["analysis_results"]
        arr    = result.arrhenius
        arr_k2 = result.arrhenius_k2

        if arr is None and arr_k2 is None:
            st.warning(
                "アレニウス解析が実行されませんでした。"
                "温度グループごとに3点以上の濃度Aデータが必要です。"
            )
        else:
            if arr is not None:
                st.subheader(f"アレニウス解析結果 ({arr.k_label})")
                a1, a2, a3 = st.columns(3)
                a1.metric("活性化エネルギー Ea", f"{arr.Ea / 1000:.2f} kJ/mol")
                a2.metric("頻度因子 A",           f"{arr.A:.3e}")
                a3.metric("R² (アレニウス)",       f"{arr.r2:.5f}")
                a4, a5 = st.columns(2)
                a4.metric("Ea 95%CI 下限 (kJ/mol)", f"{arr.Ea_ci_lower / 1000:.2f}")
                a5.metric("Ea 95%CI 上限 (kJ/mol)", f"{arr.Ea_ci_upper / 1000:.2f}")
                st.plotly_chart(plot_arrhenius(arr), use_container_width=True, key="tab3_arrhenius_k1")

                st.subheader("温度別速度定数")
                arr_tbl = pd.DataFrame({
                    "温度 (°C)":  arr.temperatures_K - 273.15,
                    "温度 (K)":   arr.temperatures_K,
                    "1/T (K⁻¹)":  1.0 / arr.temperatures_K,
                    f"{arr.k_label}": arr.k_values,
                    f"ln({arr.k_label})": np.log(arr.k_values),
                })
                st.dataframe(arr_tbl, use_container_width=True, hide_index=True)

            if arr_k2 is not None:
                st.markdown("---")
                st.subheader(f"アレニウス解析結果 ({arr_k2.k_label})")
                b1, b2, b3 = st.columns(3)
                b1.metric("活性化エネルギー Ea", f"{arr_k2.Ea / 1000:.2f} kJ/mol")
                b2.metric("頻度因子 A",           f"{arr_k2.A:.3e}")
                b3.metric("R² (アレニウス)",       f"{arr_k2.r2:.5f}")
                st.plotly_chart(plot_arrhenius(arr_k2), use_container_width=True, key="tab3_arrhenius_k2")

        if result.per_temp_results:
            st.markdown("---")
            st.subheader("温度別反応次数推算")

            if result.optimal_order_multi_temp is not None:
                oc1, oc2 = st.columns(2)
                oc1.metric("R²加重平均 反応次数 n", f"{result.optimal_order_multi_temp:.4f}")
                n_ok = sum(1 for r in result.per_temp_results if r.success)
                oc2.metric("解析成功温度数", f"{n_ok} / {len(result.per_temp_results)}")

            if result.optimal_order_explanation:
                with st.expander("推算詳細", expanded=False):
                    st.text(result.optimal_order_explanation)

            pt_rows = [
                {
                    "温度 (°C)": r.temperature_C,
                    "推算次数 n": f"{r.n:.4f}" if r.success else "—",
                    "速度定数 k": f"{r.k:.5f}" if r.success else "—",
                    "R²": f"{r.r2:.4f}" if r.success else "—",
                    "解法": r.method,
                    "状態": "成功" if r.success else f"失敗: {r.message}",
                }
                for r in result.per_temp_results
            ]
            st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)

            st.plotly_chart(
                plot_order_per_temp(result.per_temp_results, result.optimal_order_multi_temp),
                use_container_width=True, key="tab3_order_per_temp",
            )

        st.subheader("温度別 濃度プロファイル")
        st.plotly_chart(plot_raw_multi_temp(temp_groups), use_container_width=True, key="tab3_multi_temp")


# ===========================================================================
# Tab 4: Report
# ===========================================================================
with tab4:
    if not st.session_state["analysis_complete"]:
        st.info("先に解析を実行してください。")
        st.stop()

    result: FullAnalysisResult = st.session_state["analysis_results"]
    best       = result.best_result
    best_order = result.best_order

    st.subheader("解析結果サマリー")

    # Auto-detected type info
    st.markdown(
        f"**自動判定:** {REACTION_TYPE_LABELS.get(result.detected_reaction_type, '')} — "
        f"{result.detected_reaction_reason}"
    )

    st.markdown(
        f"""
| 項目 | 値 |
|------|-----|
| 推定反応次数 (積分法) | {ORDER_LABELS[best_order]} |
| 速度定数 k | {best.k:.6f} min⁻¹ |
| R² | {best.r2:.6f} |
| k 95%CI | [{best.k_ci_lower:.6f}, {best.k_ci_upper:.6f}] |
| AIC | {best.aic:.4f} |
| データ点数 (A) | {best.n_points} |
"""
    )

    if result.differential:
        diff = result.differential
        st.markdown(
            f"""
**微分法結果**

| 項目 | 値 |
|------|-----|
| 推定次数 n | {diff.n:.4f} |
| 速度定数 k | {diff.k:.6f} |
| R² (log-log) | {diff.r2:.4f} |
"""
        )

    rk = result.rk4lsq
    if rk is not None:
        k2_row   = f"| 速度定数 k2 | {rk.k2:.6f} |\n" if rk.k2 is not None else ""
        n_row    = f"| 推定次数 n | {rk.order:.4f} |\n" if rk.k2 is None else ""
        rmse_str = f"{rk.rmse:.6f}" if np.isfinite(rk.rmse) else "N/A"
        conv_str = "成功" if rk.success else "失敗"
        st.markdown(
            f"""
**RK4+最小二乗法結果** ({REACTION_TYPE_LABELS.get(rk.reaction_type, '')})

| 項目 | 値 |
|------|-----|
| 速度定数 k1 | {rk.k:.6f} |
{k2_row}{n_row}| R² | {rk.r2:.6f} |
| RMSE | {rmse_str} |
| 収束 | {conv_str} |
"""
        )

    for arr, title in [(result.arrhenius, "k1/k"), (result.arrhenius_k2, "k2")]:
        if arr is None:
            continue
        st.markdown(
            f"""
**アレニウス解析 ({arr.k_label})**

| 項目 | 値 |
|------|-----|
| Ea (kJ/mol) | {arr.Ea / 1000:.3f} |
| 頻度因子 A | {arr.A:.4e} |
| R² | {arr.r2:.6f} |
| 温度点数 | {arr.n_temperatures} |
"""
        )

    st.markdown("---")
    st.subheader("ダウンロード")

    try:
        excel_bytes = generate_excel_report(df, metadata, result)
        st.download_button(
            label="📥 Excelレポートをダウンロード",
            data=excel_bytes,
            file_name="kinetics_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(f"Excelレポート生成エラー: {e}")

    try:
        import plotly.io as pio
        rk = result.rk4lsq
        if rk is not None and rk.reaction_type in ("sequential", "parallel"):
            fig_main = plot_rk4lsq_fit(df, rk)
        else:
            fig_main = plot_best_fit_conc(df, result)
        png_bytes = pio.to_image(fig_main, format="png", width=900, height=500, scale=2)
        st.download_button(
            label="📸 主要グラフ (PNG) をダウンロード",
            data=png_bytes,
            file_name="main_plot.png",
            mime="image/png",
        )
    except Exception:
        st.info("PNG出力には kaleido パッケージが必要です (`pip install kaleido`)。")
