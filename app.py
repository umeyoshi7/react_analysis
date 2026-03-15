"""
反応速度定数・反応次数推算アプリ
Streamlit entry point
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import run_analysis
from src.data_loader import (
    auto_detect_reaction_type,
    check_mass_balance,
    get_temperature_groups,
    load_experiment_data,
)
from src.models import AnalysisResult
from src.plotting import (
    plot_arrhenius,
    plot_fit,
    plot_multi_species,
    plot_raw,
    plot_raw_multi_temp,
    plot_residuals_rk4,
    plot_simulation_results,
)
from src.reporter import generate_excel_report
from src.simulation import (
    SimulationCondition,
    build_csv,
    k_from_arrhenius,
    run_all_simulations,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="反応速度解析アプリ",
    page_icon="⚗️",
    layout="wide",
)

TEMPLATE_PATH = Path(__file__).parent / "template" / "experiment_template.xlsx"

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
        "analysis_results":  None,
        "analysis_complete": False,
        "load_warnings":     [],
        "temp_groups":       {},
        "detected_type":     "simple",
        "detected_reason":   "",
        "mass_balance_ok":   None,
        "mass_balance_cv":   None,
        "sim_conditions":    [],
        "sim_results":       None,
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
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if TEMPLATE_PATH.exists():
            with open(TEMPLATE_PATH, "rb") as f:
                st.download_button(
                    label="📥 Excel",
                    data=f.read(),
                    file_name="experiment_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
    with col_dl2:
        csv_header = "time,concentration,concentration_B,concentration_C,temperature,notes\n"
        csv_sample = (
            "0,1.0,,,25.0,開始\n"
            "5,0.778,,,25.0,\n"
            "10,0.607,,,25.0,\n"
            "20,0.368,,,25.0,\n"
            "30,0.223,,,25.0,\n"
            "60,0.050,,,25.0,終了\n"
        )
        st.download_button(
            label="📥 CSV",
            data=(csv_header + csv_sample).encode("utf-8"),
            file_name="experiment_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    # File upload
    st.subheader("2. データアップロード")
    uploaded_file = st.file_uploader(
        "実験データ (.xlsx / .csv)",
        type=["xlsx", "csv"],
        help=(
            "ExcelまたはCSVファイルをアップロードしてください。\n\n"
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
                    df, _, load_warnings = load_experiment_data(
                        io.BytesIO(uploaded_file.getvalue()),
                        filename=uploaded_file.name,
                    )
                    temp_groups   = get_temperature_groups(df)
                    detected_type, detected_reason = auto_detect_reaction_type(df)
                    mb_ok, mb_cv  = check_mass_balance(df)

                    st.session_state.update({
                        "uploaded_df":     df,
                        "load_warnings":   load_warnings,
                        "temp_groups":     temp_groups,
                        "detected_type":   detected_type,
                        "detected_reason": detected_reason,
                        "mass_balance_ok": mb_ok,
                        "mass_balance_cv": mb_cv,
                    })

                    n_valid_A = df["concentration"].notna().sum()
                    st.success(f"✅ {len(df)} 行 (濃度A: {n_valid_A}点) 読み込みました。")

                    has_B  = "concentration_B" in df.columns
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
    st.caption("解析手法: RK4+最小二乗法（数値積分法）")

    df_loaded: pd.DataFrame | None = st.session_state["uploaded_df"]
    has_B_data = df_loaded is not None and "concentration_B" in df_loaded.columns and df_loaded["concentration_B"].notna().any()
    has_C_data = df_loaded is not None and "concentration_C" in df_loaded.columns and df_loaded["concentration_C"].notna().any()
    detected   = st.session_state.get("detected_type", "simple")

    # Reaction type selectbox
    rt_labels = ["単純反応 A→products"]
    rt_values = ["simple"]
    if has_B_data:
        rt_labels.append("逐次反応 A→B→C")
        rt_values.append("sequential")
    if has_B_data and has_C_data:
        rt_labels.append("並列反応 A→B + A→C")
        rt_values.append("parallel")

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
            result = run_analysis(
                st.session_state["uploaded_df"],
                reaction_type=reaction_type,
                temp_groups=tg if len(tg) >= 2 else None,
            )
            st.session_state["analysis_results"]  = result
            st.session_state["analysis_complete"] = True
            st.session_state["sim_results"]       = None
            st.session_state["sim_conditions"]    = []
        except Exception as e:
            st.error(f"❌ 解析エラー: {e}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("⚗️ 反応速度定数・反応次数推算アプリ")

if st.session_state["uploaded_df"] is None:
    st.info("👈 サイドバーからファイルをアップロードして解析を開始してください。")
    st.markdown(
        """
        **対応する解析タイプ:**
        | タイプ | 必要な列 | 解析手法 |
        |--------|----------|----------|
        | 単純反応 A→products | 濃度A | RK4+最小二乗法（数値積分法） |
        | 逐次反応 A→B→C | 濃度A + 濃度B (+ 濃度C) | RK4+最小二乗法 |
        | 並列反応 A→B+A→C | 濃度A + 濃度B + 濃度C | RK4+最小二乗法 |
        | アレニウス解析 | 上記 + 複数温度点 | 線形回帰 |

        **対応ファイル形式:** Excel (.xlsx) / CSV (.csv)
        """
    )
    st.stop()

df: pd.DataFrame    = st.session_state["uploaded_df"]
load_warnings: list = st.session_state.get("load_warnings", [])
temp_groups: dict   = st.session_state.get("temp_groups", {})
has_multi_species   = "concentration_B" in df.columns
has_multi_temp      = len(temp_groups) > 1

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 データ確認", "🔬 解析結果", "🌡️ Arrheniusパラメータ",
    "📄 レポート出力", "🧪 シミュレーション", "📖 解析ロジック・結果の読み取り方"
])

# ===========================================================================
# Tab 1: Data overview
# ===========================================================================
with tab1:
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("総行数", len(df))
    col_info2.metric("濃度A 有効点数", int(df["concentration"].notna().sum()))
    col_info3.metric("温度グループ数", len(temp_groups))

    detected_type   = st.session_state.get("detected_type", "simple")
    detected_reason = st.session_state.get("detected_reason", "")
    mb_ok = st.session_state.get("mass_balance_ok")
    mb_cv = st.session_state.get("mass_balance_cv")

    if has_multi_species:
        st.info(f"🔍 自動判定: **{REACTION_TYPE_LABELS[detected_type]}** — {detected_reason}")

    if mb_ok is not None and not mb_ok and has_multi_species:
        st.warning(
            f"⚠️ 質量バランス: A+B+C の変動係数 = {mb_cv:.3f} (閾値5% 超過)。"
            "開放系反応または測定誤差の可能性があります。"
        )
    elif mb_ok and has_multi_species:
        st.success(f"✅ 質量バランス OK (変動係数 = {mb_cv:.3f})")

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

    st.subheader("濃度 vs. 時間")
    if has_multi_species:
        st.plotly_chart(plot_multi_species(df), use_container_width=True, key="tab1_multi_species")
        if has_multi_temp:
            st.subheader(f"複数温度データ ({len(temp_groups)} 温度)")
            st.plotly_chart(plot_raw_multi_temp(temp_groups), use_container_width=True, key="tab1_multi_temp_species")
    elif has_multi_temp:
        st.plotly_chart(plot_raw_multi_temp(temp_groups), use_container_width=True, key="tab1_multi_temp")
    else:
        st.plotly_chart(plot_raw(df), use_container_width=True, key="tab1_raw")


# ===========================================================================
# Tab 2: Analysis results
# ===========================================================================
with tab2:
    if not st.session_state["analysis_complete"]:
        st.info("サイドバーの「解析実行」ボタンを押してください。")
    else:
        result: AnalysisResult = st.session_state["analysis_results"]
        fit = result.fit

        all_warnings = load_warnings + result.warnings
        if all_warnings:
            with st.expander("⚠️ 警告メッセージ", expanded=len(result.warnings) > 0):
                for w in all_warnings:
                    st.warning(w)

        st.subheader("解析結果 (RK4+最小二乗法)")

        def _show_fit_metrics(fit_i, df_i, key_prefix: str) -> None:
            """Display metrics + fit/residual plots for a single FitResult."""
            is_multi_i = fit_i.reaction_type in ("sequential", "parallel")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("反応タイプ", REACTION_TYPE_LABELS.get(fit_i.reaction_type, fit_i.reaction_type))
            k_str = f"{fit_i.k:.5f}" if np.isfinite(fit_i.k) else "N/A"
            m2.metric("速度定数 k1", k_str)
            r2_str = f"{fit_i.r2:.5f}" if np.isfinite(fit_i.r2) else "N/A"
            m3.metric("R²", r2_str)
            m4.metric("収束", "✅ 成功" if fit_i.success else "❌ 失敗")

            ci1, ci2 = st.columns(2)
            ci1.metric("k 95%CI 下限", f"{fit_i.k_ci_lower:.5f}" if np.isfinite(fit_i.k_ci_lower) else "N/A")
            ci2.metric("k 95%CI 上限", f"{fit_i.k_ci_upper:.5f}" if np.isfinite(fit_i.k_ci_upper) else "N/A")

            if not is_multi_i:
                st.metric("推定反応次数 n", f"{fit_i.order:.4f}" if np.isfinite(fit_i.order) else "N/A")
            else:
                if fit_i.k2 is not None and np.isfinite(fit_i.k2):
                    mk1, mk2, mk3, mk4 = st.columns(4)
                    mk1.metric("速度定数 k2", f"{fit_i.k2:.5f}")
                    mk2.metric("k2 95%CI 下限", f"{fit_i.k2_ci_lower:.5f}" if fit_i.k2_ci_lower is not None and np.isfinite(fit_i.k2_ci_lower) else "N/A")
                    mk3.metric("k2 95%CI 上限", f"{fit_i.k2_ci_upper:.5f}" if fit_i.k2_ci_upper is not None and np.isfinite(fit_i.k2_ci_upper) else "N/A")
                    rmse_val = f"{fit_i.rmse:.5f}" if np.isfinite(fit_i.rmse) else "N/A"
                    mk4.metric("RMSE", rmse_val)
                if fit_i.reaction_type == "sequential":
                    nc1, nc2 = st.columns(2)
                    nc1.metric("推定反応次数 n1 (A→B)", f"{fit_i.order:.4f}" if np.isfinite(fit_i.order) else "N/A")
                    n2_val = fit_i.order2
                    nc2.metric("推定反応次数 n2 (B→C)", f"{n2_val:.4f}" if n2_val is not None and np.isfinite(n2_val) else "N/A")
                elif fit_i.reaction_type == "parallel":
                    st.metric("推定反応次数 n", f"{fit_i.order:.4f}" if np.isfinite(fit_i.order) else "N/A")

            if not fit_i.success:
                st.warning(f"収束メッセージ: {fit_i.message}")

            if len(fit_i.t_pred) > 0:
                st.plotly_chart(plot_fit(df_i, fit_i), use_container_width=True, key=f"{key_prefix}_fit")
                st.plotly_chart(plot_residuals_rk4(df_i, fit_i), use_container_width=True, key=f"{key_prefix}_residuals")

        if result.is_multi_temp:
            st.subheader("温度別解析結果")
            tg = st.session_state.get("temp_groups", {})
            for idx, (T_c, fit_i) in enumerate(result.per_temp_fits):
                with st.expander(f"{T_c:.1f}°C の解析結果", expanded=False):
                    df_i = tg.get(T_c, df)
                    _show_fit_metrics(fit_i, df_i, key_prefix=f"tab2_temp_{idx}")
            if result.optimal_order is not None:
                st.info(
                    f"温度別解析完了: Arrheniusパラメータは「Arrheniusパラメータ」タブを参照してください。"
                    f"（R²加重平均 最適反応次数 n = {result.optimal_order:.4f}）"
                )
        else:
            _show_fit_metrics(fit, df, key_prefix="tab2")


# ===========================================================================
# Tab 3: Arrhenius
# ===========================================================================
with tab3:
    if not has_multi_temp:
        st.info(
            "アレニウス解析には複数温度のデータが必要です。\n\n"
            "**設定方法:** データファイルの `temperature` 列に各行の測定温度 (°C) を記入してください。"
        )
    elif not st.session_state["analysis_complete"]:
        st.info("サイドバーの「解析実行」ボタンを押してください。")
    else:
        result: AnalysisResult = st.session_state["analysis_results"]
        arr    = result.arrhenius
        arr_k2 = result.arrhenius_k2

        if arr is None and arr_k2 is None:
            st.warning(
                "アレニウス解析が実行されませんでした。"
                "温度グループごとに3点以上の濃度Aデータが必要です。"
            )
        else:
            if arr is not None:
                st.subheader("アレニウス解析結果 (k1/k)")
                a1, a2, a3 = st.columns(3)
                a1.metric("活性化エネルギー Ea", f"{arr.Ea_kJmol:.2f} kJ/mol")
                a2.metric("頻度因子 A",           f"{arr.A:.3e}")
                a3.metric("R² (アレニウス)",       f"{arr.r2:.5f}")
                st.plotly_chart(plot_arrhenius(arr), use_container_width=True, key="tab3_arrhenius_k1")

                st.subheader("温度別速度定数")
                arr_tbl = pd.DataFrame({
                    "温度 (°C)":  arr.temps_celsius,
                    "温度 (K)":   [t + 273.15 for t in arr.temps_celsius],
                    "1/T (K⁻¹)":  arr.inv_T,
                    "k":          arr.k_values,
                    "ln(k)":      arr.ln_k,
                })
                st.dataframe(arr_tbl, use_container_width=True, hide_index=True)

            if arr_k2 is not None:
                st.markdown("---")
                st.subheader("アレニウス解析結果 (k2)")
                b1, b2, b3 = st.columns(3)
                b1.metric("活性化エネルギー Ea", f"{arr_k2.Ea_kJmol:.2f} kJ/mol")
                b2.metric("頻度因子 A",           f"{arr_k2.A:.3e}")
                b3.metric("R² (アレニウス)",       f"{arr_k2.r2:.5f}")
                st.plotly_chart(plot_arrhenius(arr_k2), use_container_width=True, key="tab3_arrhenius_k2")

        if result.per_temp_fits:
            st.markdown("---")
            st.subheader("温度別解析結果")

            if result.optimal_order is not None:
                oc1, oc2 = st.columns(2)
                oc1.metric("R²加重平均 反応次数 n", f"{result.optimal_order:.4f}")
                n_ok = sum(1 for _, f in result.per_temp_fits if f.success)
                oc2.metric("解析成功温度数", f"{n_ok} / {len(result.per_temp_fits)}")

            pt_rows = []
            for T_c, fit_i in result.per_temp_fits:
                row_d: dict = {
                    "温度 (°C)": T_c,
                    "温度 (K)":  T_c + 273.15,
                    "k":         f"{fit_i.k:.5f}" if np.isfinite(fit_i.k) else "—",
                    "k 95%CI 下限": f"{fit_i.k_ci_lower:.5f}" if np.isfinite(fit_i.k_ci_lower) else "—",
                    "k 95%CI 上限": f"{fit_i.k_ci_upper:.5f}" if np.isfinite(fit_i.k_ci_upper) else "—",
                    "n":         f"{fit_i.order:.4f}" if np.isfinite(fit_i.order) else "—",
                    "R²":        f"{fit_i.r2:.4f}"    if np.isfinite(fit_i.r2)    else "—",
                    "収束":      "成功" if fit_i.success else "失敗",
                }
                pt_rows.append(row_d)
            st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)

        st.subheader("温度別 濃度プロファイル")
        st.plotly_chart(plot_raw_multi_temp(temp_groups), use_container_width=True, key="tab3_multi_temp")


# ===========================================================================
# Tab 4: Report
# ===========================================================================
with tab4:
    if not st.session_state["analysis_complete"]:
        st.info("先に解析を実行してください。")
    else:
        result: AnalysisResult = st.session_state["analysis_results"]
        fit = result.fit

        st.subheader("解析結果サマリー")

        st.markdown(
            f"**自動判定:** {REACTION_TYPE_LABELS.get(result.detected_reaction_type, '')} — "
            f"{result.detected_reaction_reason}"
        )

        def _fmt(v) -> str:
            if v is None:
                return "N/A"
            try:
                if not np.isfinite(float(v)):
                    return "N/A"
                return f"{float(v):.6f}"
            except (TypeError, ValueError):
                return str(v)

        k2_row = f"| 速度定数 k2 | {_fmt(fit.k2)} |\n" if fit.k2 is not None else ""
        n_row  = f"| 推定次数 n | {_fmt(fit.order)} |\n" if fit.reaction_type == "simple" else ""
        conv_str = "成功" if fit.success else "失敗"
        st.markdown(
            f"""
**RK4+最小二乗法結果** ({REACTION_TYPE_LABELS.get(fit.reaction_type, '')})

| 項目 | 値 |
|------|-----|
| 速度定数 k1 | {_fmt(fit.k)} |
{k2_row}{n_row}| k 95%CI 下限 | {_fmt(fit.k_ci_lower)} |
| k 95%CI 上限 | {_fmt(fit.k_ci_upper)} |
| R² | {_fmt(fit.r2)} |
| RMSE | {_fmt(fit.rmse)} |
| 収束 | {conv_str} |
"""
        )

        for arr, title in [(result.arrhenius, "k1/k"), (result.arrhenius_k2, "k2")]:
            if arr is None:
                continue
            st.markdown(
                f"""
**アレニウス解析 ({title})**

| 項目 | 値 |
|------|-----|
| Ea (kJ/mol) | {arr.Ea_kJmol:.3f} |
| 頻度因子 A | {arr.A:.4e} |
| R² | {arr.r2:.6f} |
| 温度点数 | {len(arr.temps_celsius)} |
"""
            )

        st.markdown("---")
        st.subheader("ダウンロード")

        try:
            excel_bytes = generate_excel_report(df, result)
            st.download_button(
                label="📥 Excelレポートをダウンロード",
                data=excel_bytes,
                file_name="kinetics_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Excelレポート生成エラー: {e}")


# ===========================================================================
# Tab 5: Simulation
# ===========================================================================
with tab5:
    if not st.session_state["analysis_complete"]:
        st.info("先にサイドバーの「解析実行」ボタンを押してください。解析結果が自動的にシミュレーション条件に反映されます。")
    else:
        result_sim: AnalysisResult = st.session_state["analysis_results"]
        fit_sim = result_sim.fit
        arr_sim = result_sim.arrhenius
        arr_k2_sim = result_sim.arrhenius_k2
        has_arrhenius = arr_sim is not None
        rt_sim = fit_sim.reaction_type

        # ── デフォルト条件の生成 ──────────────────────────────────────────────
        if not st.session_state["sim_conditions"]:
            k_default = fit_sim.k if np.isfinite(fit_sim.k) else 0.01
            k2_default = fit_sim.k2 if (fit_sim.k2 is not None and np.isfinite(fit_sim.k2)) else None
            n_default  = fit_sim.order if np.isfinite(fit_sim.order) else 1.0
            n2_default = fit_sim.order2 if (fit_sim.order2 is not None and np.isfinite(fit_sim.order2)) else 1.0

            default_cond: dict = {
                "label":    "条件1",
                "k":        k_default,
                "k2":       k2_default,
                "n":        n_default,
                "n2":       n2_default,
                "A0":       1.0,
                "t_end":    60.0,
                "T_celsius": arr_sim.temps_celsius[0] if has_arrhenius else None,
                "use_arrhenius": has_arrhenius,
            }

            # k が NaN だった場合は警告
            if not np.isfinite(fit_sim.k):
                st.warning("解析結果の k が無効な値のため、k = 0.01 をデフォルト値に設定しました。")

            st.session_state["sim_conditions"] = [default_cond]

        sim_conds: list[dict] = st.session_state["sim_conditions"]

        # ── 条件エディタ ────────────────────────────────────────────────────
        st.subheader("シミュレーション条件")

        for idx, cond_d in enumerate(sim_conds):
            with st.expander(f"🔧 {cond_d['label']}", expanded=True):
                col_lbl, col_del = st.columns([4, 1])
                with col_lbl:
                    new_label = st.text_input("条件ラベル", value=cond_d["label"],
                                              key=f"sim_label_{idx}")
                    sim_conds[idx]["label"] = new_label
                with col_del:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ 削除", key=f"sim_del_{idx}",
                                 disabled=(len(sim_conds) <= 1)):
                        sim_conds.pop(idx)
                        st.session_state["sim_conditions"] = sim_conds
                        st.rerun()

                # Arrhenius あり: 温度入力 → k 自動計算
                if has_arrhenius:
                    T_val = st.number_input(
                        "温度 (°C)", value=float(cond_d.get("T_celsius") or 25.0),
                        step=1.0, format="%.1f", key=f"sim_T_{idx}",
                    )
                    sim_conds[idx]["T_celsius"] = T_val
                    try:
                        k_calc = k_from_arrhenius(arr_sim, T_val)
                        sim_conds[idx]["k"] = k_calc
                        st.info(f"k(T) = {k_calc:.5g}  (Arrhenius: Ea={arr_sim.Ea_kJmol:.1f} kJ/mol, A={arr_sim.A:.3e})")
                    except ValueError as e:
                        st.error(f"k 計算エラー: {e}")

                    if arr_k2_sim is not None:
                        try:
                            k2_calc = k_from_arrhenius(arr_k2_sim, T_val)
                            sim_conds[idx]["k2"] = k2_calc
                            st.info(f"k2(T) = {k2_calc:.5g}  (Arrhenius: Ea={arr_k2_sim.Ea_kJmol:.1f} kJ/mol)")
                        except ValueError as e:
                            st.error(f"k2 計算エラー: {e}")
                else:
                    # k を直接編集
                    k_val = st.number_input(
                        "速度定数 k", value=float(cond_d["k"]),
                        min_value=0.0, step=0.001, format="%.5f", key=f"sim_k_{idx}",
                    )
                    sim_conds[idx]["k"] = k_val

                # 反応次数 n
                n_val = st.number_input(
                    "反応次数 n", value=float(cond_d["n"]),
                    min_value=0.0, max_value=5.0, step=0.1, format="%.2f", key=f"sim_n_{idx}",
                )
                sim_conds[idx]["n"] = n_val

                # sequential / parallel の k2
                if rt_sim in ("sequential", "parallel"):
                    if not (has_arrhenius and arr_k2_sim is not None):
                        k2_val = st.number_input(
                            "速度定数 k2", value=float(cond_d["k2"] or 0.01),
                            min_value=0.0, step=0.001, format="%.5f", key=f"sim_k2_{idx}",
                        )
                        sim_conds[idx]["k2"] = k2_val

                    if rt_sim == "sequential":
                        n2_val = st.number_input(
                            "反応次数 n2 (B→C)", value=float(cond_d.get("n2") or 1.0),
                            min_value=0.0, max_value=5.0, step=0.1, format="%.2f", key=f"sim_n2_{idx}",
                        )
                        sim_conds[idx]["n2"] = n2_val

                # 初期濃度 / 終了時刻
                col_A0, col_tend = st.columns(2)
                with col_A0:
                    A0_val = st.number_input(
                        "[A]₀ (mol/L)", value=float(cond_d["A0"]),
                        min_value=0.0, step=0.1, format="%.3f", key=f"sim_A0_{idx}",
                    )
                    sim_conds[idx]["A0"] = A0_val
                with col_tend:
                    tend_val = st.number_input(
                        "終了時刻 t_end (min)", value=float(cond_d["t_end"]),
                        min_value=0.1, step=10.0, format="%.1f", key=f"sim_tend_{idx}",
                    )
                    sim_conds[idx]["t_end"] = tend_val

        st.session_state["sim_conditions"] = sim_conds

        # ── 条件追加ボタン ──────────────────────────────────────────────────
        if st.button("➕ 条件を追加"):
            last = dict(sim_conds[-1])
            last["label"] = f"条件{len(sim_conds) + 1}"
            sim_conds.append(last)
            st.session_state["sim_conditions"] = sim_conds
            st.rerun()

        st.markdown("---")

        # ── 実行ボタン ──────────────────────────────────────────────────────
        if st.button("▶ シミュレーション実行", type="primary"):
            conditions_objs = []
            for cond_d in sim_conds:
                conditions_objs.append(SimulationCondition(
                    label=cond_d["label"],
                    reaction_type=rt_sim,
                    k=cond_d["k"],
                    n=cond_d["n"],
                    k2=cond_d.get("k2"),
                    n2=cond_d.get("n2"),
                    A0=cond_d["A0"],
                    t_end=cond_d["t_end"],
                ))
            with st.spinner("シミュレーション中…"):
                st.session_state["sim_results"] = run_all_simulations(conditions_objs)

        # ── 結果表示 ────────────────────────────────────────────────────────
        sim_results = st.session_state.get("sim_results")
        if sim_results is not None:
            failed = [cond.label for cond, t, c in sim_results if t is None]
            if failed:
                st.warning(f"以下の条件で ODE 求解に失敗しました: {', '.join(failed)}")

            success_results = [(cond, t, c) for cond, t, c in sim_results if t is not None]
            if not success_results:
                st.error("すべての条件でシミュレーションに失敗しました。パラメータを確認してください。")
            else:
                fig_sim = plot_simulation_results(sim_results, rt_sim)
                st.plotly_chart(fig_sim, use_container_width=True, key="tab5_sim")

                csv_str = build_csv(sim_results)
                st.download_button(
                    label="📥 CSV ダウンロード",
                    data=csv_str.encode("utf-8-sig"),
                    file_name="simulation_results.csv",
                    mime="text/csv",
                )


# ===========================================================================
# Tab 6: Explanation — analysis logic and how to read results
# ===========================================================================
with tab6:
    st.header("解析ロジック及び解析結果の読み取り方")

    # -----------------------------------------------------------------------
    # 1. 解析手法の概要
    # -----------------------------------------------------------------------
    st.subheader("1. 解析手法: RK4+最小二乗法（数値積分法）")
    st.markdown(
        """
本アプリは **RK45 数値積分 + scipy.optimize.least_squares** の組み合わせで反応速度パラメータを推定します。

**処理フロー**

1. **ODE 数値積分 (RK45)** — 試行パラメータ $(k, n, C_0)$ を用いて反応の微分方程式を数値的に解き、濃度プロファイルを予測します。
2. **残差計算** — 予測値と実測値の差（残差）を計算します。
3. **最小二乗最適化** — 残差の二乗和を最小化するようにパラメータを更新します（Trust Region Reflective 法）。
4. **マルチスタート** — 3 種類の異なる初期値から最適化を実行し、最もよい結果（最大 R²）を採用します。局所解への収束を抑制します。
5. **信頼区間 (95%CI)** — 収束後のヤコビアン行列から共分散行列を計算し、95% 信頼区間を算出します。

**各反応タイプの ODE 系**

| 反応タイプ | 微分方程式 |
|-----------|-----------|
| 単純反応 A→P | $dA/dt = -k A^n$ |
| 逐次反応 A→B→C | $dA/dt = -k_1 A^{n_1}$, $dB/dt = k_1 A^{n_1} - k_2 B^{n_2}$, $dC/dt = k_2 B^{n_2}$ |
| 並列反応 A→B+A→C | $dA/dt = -(k_1+k_2) A^n$, $dB/dt = k_1 A^n$, $dC/dt = k_2 A^n$ |
"""
    )

    # -----------------------------------------------------------------------
    # 2. アレニウス解析
    # -----------------------------------------------------------------------
    st.subheader("2. アレニウス解析")
    st.markdown(
        r"""
複数温度のデータがある場合、各温度で推定した速度定数 $k$ からアレニウス式をフィットします。

$$
k = A \exp\!\left(-\frac{E_a}{RT}\right)
\quad\Longrightarrow\quad
\ln k = \ln A - \frac{E_a}{R} \cdot \frac{1}{T}
$$

| 記号 | 説明 | 単位 |
|------|------|------|
| $E_a$ | 活性化エネルギー | kJ/mol |
| $A$ | 頻度因子（前指数因子） | k と同じ単位 (min⁻¹ など) |
| $R$ | 気体定数 = 8.314 | J/(mol·K) |
| $T$ | 絶対温度 | K |

**グラフの読み方**

- **横軸 (1/T)** — 温度の逆数 (K⁻¹)。左ほど高温・右ほど低温。
- **縦軸 (ln k)** — 速度定数の自然対数。高いほど反応が速い。
- **回帰直線** — 傾きが $-E_a/R$。傾きが大きいほど温度依存性が強い（高 $E_a$）。

**条件**: 各温度グループで 3 点以上の有効データと R² ≥ 0.5 の良好なフィットが必要です。
条件を満たさない温度はアレニウスプロットから除外されます。
"""
    )

    # -----------------------------------------------------------------------
    # 3. 解析結果の読み取り方
    # -----------------------------------------------------------------------
    st.subheader("3. 解析結果の読み取り方")

    st.markdown("#### 速度パラメータ")
    st.markdown(
        """
| 指標 | 説明 | 目安 |
|------|------|------|
| **k (速度定数)** | 反応の速さを表す定数。大きいほど速い反応。 | 単位は min⁻¹（1次）や L/(mol·min)（2次）など |
| **n (反応次数)** | 反応次数。整数に近い値が物理的に解釈しやすい。 | 0次=濃度に無関係、1次=比例、2次=二乗比例 |
| **k2** | 逐次・並列反応の第2速度定数。 | k1 と k2 の比が選択性・収率に影響 |
| **95%CI** | パラメータの 95% 信頼区間。 | 区間が狭いほど推定精度が高い |
"""
    )

    st.markdown("#### フィット品質指標")
    st.markdown(
        """
| 指標 | 説明 | 目安 |
|------|------|------|
| **R²** | 決定係数。1 に近いほど実測値と予測値が一致。 | ≥ 0.99: 優秀、≥ 0.95: 良好、< 0.90: 要確認 |
| **RMSE** | 残差の二乗平均平方根（実測値との平均的なズレ）。 | 小さいほど良い。濃度スケールと同じ単位 |
| **収束** | 最適化が正常に収束したかどうか。 | ❌ 失敗の場合は k・n の値の信頼性が低い |
"""
    )

    st.markdown("#### アレニウスパラメータ")
    st.markdown(
        r"""
| 指標 | 説明 | 目安 |
|------|------|------|
| **$E_a$ (活性化エネルギー)** | 反応の温度感受性の指標。 | 化学反応: 40〜150 kJ/mol が典型的 |
| **頻度因子 $A$** | 分子衝突頻度に関係する定数。 | 温度 0 K での極限速度定数に相当（理論値） |
| **R² (アレニウス)** | アレニウスプロットの直線性。 | ≥ 0.99 が良好。低い場合は複数反応機構の可能性 |
"""
    )

    st.markdown("#### 残差プロットの見方")
    st.markdown(
        """
残差プロット（観測値 − 予測値）は**系統的なパターンがないこと**が理想です。

- **ランダムな分布** → フィットが適切（ODE モデルが実験をよく表現）
- **U 字・逆 U 字のパターン** → モデルの反応次数が合っていない可能性
- **特定時刻に大きな残差** → 外れ値・測定誤差、または反応機構の変化
"""
    )

    # -----------------------------------------------------------------------
    # 4. 反応シミュレーション
    # -----------------------------------------------------------------------
    st.subheader("4. 反応シミュレーション")
    st.markdown(
        r"""
「🧪 シミュレーション」タブでは、解析で得られたパラメータを用いて任意の条件下での濃度プロファイルを予測します。

**処理フロー**

1. **速度定数の決定** — アレニウスパラメータが得られている場合（複数温度データ）と、そうでない場合とで異なります（下表）。
2. **ODE 求解** — 決定した $k$, $n$, $[A]_0$ を ODE に代入し、RK45 で数値積分します。
3. **濃度プロファイル出力** — 時刻 $0$ から $t_\text{end}$ までの $[A]$（逐次・並列反応では $[B]$, $[C]$ も）を計算して描画します。

**速度定数 $k$ の決定方法**

| データ条件 | k の決定方法 |
|-----------|------------|
| 複数温度データあり（Arrhenius 解析成立） | $k(T) = A \exp\!\left(-\dfrac{E_a}{RT}\right)$ でユーザー指定温度から自動計算 |
| 一点温度のみ（Arrhenius 解析なし） | その温度でフィットした $k$ を初期値として手動入力 |

> **注意**: 一点温度の場合は $E_a$ と $A$ を一意に決定できないため、他の温度への外挿は行いません。
> 他の温度での $k$ を推定するには、文献値などから $E_a$ を別途入力してください。

**ODE 系（シミュレーション・フィッティングで共通）**

| 反応タイプ | 微分方程式 |
|-----------|-----------|
| 単純反応 A→P | $\dfrac{d[A]}{dt} = -k[A]^n$ |
| 逐次反応 A→B→C | $\dfrac{d[A]}{dt} = -k_1[A]^{n_1}$、$\dfrac{d[B]}{dt} = k_1[A]^{n_1} - k_2[B]^{n_2}$、$\dfrac{d[C]}{dt} = k_2[B]^{n_2}$ |
| 並列反応 A→B+A→C | $\dfrac{d[A]}{dt} = -(k_1+k_2)[A]^n$、$\dfrac{d[B]}{dt} = k_1[A]^n$、$\dfrac{d[C]}{dt} = k_2[A]^n$ |

**数値積分の設定**

| 設定項目 | 値 |
|---------|----|
| ソルバー | `scipy.integrate.solve_ivp`（`RK45` 法） |
| 相対許容誤差 `rtol` | $10^{-6}$ |
| 絶対許容誤差 `atol` | $10^{-9}$ |
| 出力点数 | 500点（$t=0$ から $t_\text{end}$ を等間隔） |
| 負値処理 | 各成分は $\max(\text{計算値},\, 0)$ でクリップ |

**複数条件の比較**

「条件を追加」ボタンで複数の条件を同時にシミュレートし、1つのグラフ上で重ね描きできます。
結果は CSV ダウンロードも可能です（列: `condition_label`, `time`, `A`, `B`, `C`）。
"""
    )

    # -----------------------------------------------------------------------
    # 5. 注意事項・よくある質問
    # -----------------------------------------------------------------------
    st.subheader("5. 注意事項・トラブルシューティング")
    st.markdown(
        """
**Q: 収束が失敗する**
→ データ点数が少ない（推奨: 8 点以上）、または初期濃度付近のデータが不足しています。
  時間 t=0 の測定値を含めることで安定します。

**Q: R² が低い (<0.90)**
→ 反応タイプの選択を確認してください。単純反応データに逐次反応モデルを適用すると R² が下がります。
  複数のピーク・屈曲点がある場合は逐次反応モデルが適切な場合があります。

**Q: アレニウスプロットに一部の温度が表示されない**
→ 該当温度のフィットが失敗（収束エラー）か R² < 0.5 のため除外されています。
  「解析結果」タブで各温度の解析状況を確認してください。

**Q: 活性化エネルギーが負になる**
→ 温度を上げると反応が遅くなるデータパターンです。拡散律速反応などで起こる場合があります。
  データの信頼性（測定温度の誤りなど）を確認してください。

**Q: 複数温度データの「全体フィット」結果は？**
→ 「解析結果」タブに表示される結果は全温度のデータを結合した解析です。
  温度別の速度定数は「Arrhenius パラメータ」タブの温度別テーブルを参照してください。
"""
    )
