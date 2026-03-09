"""Plotly figure generators for reaction kinetics analysis."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.kinetics import (
    ArrheniusResult,
    FullAnalysisResult,
    LsqSimpleResult,
    PerTempResult,
    RegressionResult,
    RK4LsqResult,
)


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
ORDER_COLORS = {0: "#636EFA", 1: "#EF553B", 2: "#00CC96"}
ORDER_LABELS = {0: "0次反応", 1: "1次反応", 2: "2次反応"}
TRANSFORM_LABELS = {
    0: "[A] (mol/L)",
    1: "ln[A]",
    2: "1/[A] (L/mol)",
}
SPECIES_COLORS = {"A": "#636EFA", "B": "#EF553B", "C": "#00CC96"}


# ---------------------------------------------------------------------------
# Raw concentration–time plot
# ---------------------------------------------------------------------------

def plot_raw(df) -> go.Figure:
    """Concentration vs. time scatter plot."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["concentration"],
            mode="markers+lines",
            marker=dict(size=8, color="#636EFA"),
            line=dict(color="#636EFA", dash="dot"),
            name="[A] (mol/L)",
        )
    )
    fig.update_layout(
        title="濃度 vs. 時間",
        xaxis_title="時間 (min)",
        yaxis_title="濃度 (mol/L)",
        template="plotly_white",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Integral method: transformed plot + fit line
# ---------------------------------------------------------------------------

def plot_integral_fit(result: RegressionResult) -> go.Figure:
    order = result.order
    color = ORDER_COLORS[order]
    label = ORDER_LABELS[order]
    y_label = TRANSFORM_LABELS[order]

    x = result.transform_x
    y = result.transform_y
    y_fit = result.slope * x + result.intercept

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=8, color=color),
            name="データ",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y_fit, mode="lines",
            line=dict(color=color, width=2),
            name=f"回帰直線 (R²={result.r2:.4f})",
        )
    )
    fig.update_layout(
        title=f"{label} – 積分法フィット",
        xaxis_title="時間 (min)",
        yaxis_title=y_label,
        template="plotly_white",
        height=350,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# Residual plot
# ---------------------------------------------------------------------------

def plot_residuals(result: RegressionResult) -> go.Figure:
    order = result.order
    color = ORDER_COLORS[order]
    label = ORDER_LABELS[order]

    x = result.transform_x
    resid = result.residuals

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=resid, mode="markers",
            marker=dict(size=7, color=color, opacity=0.8),
            name="残差",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"{label} – 残差",
        xaxis_title="時間 (min)",
        yaxis_title="残差",
        template="plotly_white",
        height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# LSQ (analytical solution) fit plot
# ---------------------------------------------------------------------------

def plot_lsq_fit(df, result: LsqSimpleResult) -> go.Figure:
    """Observed data (scatter) + analytical-solution LSQ prediction (line)."""
    mask = df["concentration"].notna()
    time = df.loc[mask, "time"].to_numpy(dtype=float)
    conc = df.loc[mask, "concentration"].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time, y=conc, mode="markers",
            marker=dict(size=9, color="#333333", symbol="circle"),
            name="実験データ",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.t_pred, y=result.c_pred, mode="lines",
            line=dict(color="#AB63FA", width=2.5),
            name=f"LSQ予測 (n={result.n:.3f}, k={result.k:.4f})",
        )
    )
    fig.update_layout(
        title=f"最小二乗法（解析解）: n={result.n:.3f}, k={result.k:.4f} min⁻¹, R²={result.r2:.4f}",
        xaxis_title="時間 (min)",
        yaxis_title="濃度 (mol/L)",
        template="plotly_white",
        height=420,
        legend=dict(x=0.5, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# Summary figure: best-fit concentration vs. time with model prediction
# ---------------------------------------------------------------------------

def plot_best_fit_conc(df, result: FullAnalysisResult) -> go.Figure:
    """Plot raw concentration data with best-fit model prediction."""
    time = df["time"].to_numpy(dtype=float)
    conc = df["concentration"].to_numpy(dtype=float)
    t_pred = np.linspace(time.min(), time.max(), 300)

    best = result.best_result
    order = result.best_order
    color = ORDER_COLORS[order]
    label = ORDER_LABELS[order]

    k = best.k

    if order == 0:
        c0 = best.intercept
        c_pred = np.maximum(c0 - k * t_pred, 0)
    elif order == 1:
        c_pred = np.exp(best.intercept) * np.exp(-k * t_pred)
    else:  # order == 2
        inv_c0 = best.intercept
        c_pred = 1.0 / (inv_c0 + k * t_pred)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time, y=conc, mode="markers",
            marker=dict(size=9, color="#333333", symbol="circle"),
            name="実験データ",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_pred, y=c_pred, mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{label} モデル (k={k:.4f})",
        )
    )
    fig.update_layout(
        title=f"ベストフィット: {label} (k={k:.4f} min⁻¹, R²={best.r2:.4f})",
        xaxis_title="時間 (min)",
        yaxis_title="濃度 (mol/L)",
        template="plotly_white",
        height=420,
        legend=dict(x=0.5, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# RK4+LSQ fit plot
# ---------------------------------------------------------------------------

def plot_rk4lsq_fit(df, result: RK4LsqResult) -> go.Figure:
    """Observed points (scatter) + ODE prediction (line) per species."""
    time = df["time"].to_numpy(dtype=float)

    species_obs = {"A": df["concentration"].to_numpy(dtype=float)}
    if "concentration_B" in df.columns:
        species_obs["B"] = df["concentration_B"].to_numpy(dtype=float)
    if "concentration_C" in df.columns:
        species_obs["C"] = df["concentration_C"].to_numpy(dtype=float)

    reaction_labels = {
        "simple": "単純反応",
        "sequential": "逐次反応 A→B→C",
        "parallel": "並列反応 A→B + A→C",
    }
    title_suffix = reaction_labels.get(result.reaction_type, result.reaction_type)

    fig = go.Figure()

    for sp, obs in species_obs.items():
        color = SPECIES_COLORS[sp]
        fig.add_trace(
            go.Scatter(
                x=time, y=obs,
                mode="markers",
                marker=dict(size=9, color=color, symbol="circle"),
                name=f"[{sp}] 観測",
            )
        )

    for sp, pred in result.c_pred.items():
        if sp not in species_obs and result.reaction_type == "simple":
            continue
        color = SPECIES_COLORS[sp]
        fig.add_trace(
            go.Scatter(
                x=result.t_pred, y=pred,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"[{sp}] RK45予測",
            )
        )

    k_str = f"k={result.k:.4f}"
    if result.k2 is not None:
        k_str += f", k₂={result.k2:.4f}"

    fig.update_layout(
        title=f"RK4+最小二乗法: {title_suffix} ({k_str}, R²={result.r2:.4f})",
        xaxis_title="時間 (min)",
        yaxis_title="濃度 (mol/L)",
        template="plotly_white",
        height=420,
        legend=dict(x=0.5, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-species raw data plot
# ---------------------------------------------------------------------------

def plot_multi_species(df) -> go.Figure:
    """[A], [B], [C] vs time on the same axis."""
    time = df["time"].to_numpy(dtype=float)
    fig = go.Figure()

    species_cols = [
        ("A", "concentration", "[A]"),
        ("B", "concentration_B", "[B]"),
        ("C", "concentration_C", "[C]"),
    ]
    for sp, col, label in species_cols:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=df[col].to_numpy(dtype=float),
                    mode="markers+lines",
                    marker=dict(size=7, color=SPECIES_COLORS[sp]),
                    line=dict(color=SPECIES_COLORS[sp], dash="dot"),
                    name=label,
                )
            )

    fig.update_layout(
        title="複数成分 濃度 vs. 時間",
        xaxis_title="時間 (min)",
        yaxis_title="濃度 (mol/L)",
        template="plotly_white",
        height=420,
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# Arrhenius plot
# ---------------------------------------------------------------------------

def plot_arrhenius(result: ArrheniusResult) -> go.Figure:
    """ln(k) vs 1/T scatter + regression line. Secondary x-axis in °C."""
    T_K = result.temperatures_K
    k_arr = result.k_values
    inv_T = 1.0 / T_K
    ln_k = np.log(k_arr)

    inv_T_fit = np.linspace(inv_T.min() * 0.99, inv_T.max() * 1.01, 200)
    ln_k_fit = np.log(result.A) - (result.Ea / 8.314) * inv_T_fit

    Ea_kJ = result.Ea / 1000

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=inv_T, y=ln_k,
            mode="markers+text",
            marker=dict(size=10, color="#EF553B"),
            text=[f"{t - 273.15:.0f}°C" for t in T_K],
            textposition="top center",
            name="ln(k) データ",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=inv_T_fit, y=ln_k_fit,
            mode="lines",
            line=dict(color="#636EFA", width=2),
            name=f"回帰 (Ea={Ea_kJ:.1f} kJ/mol, R²={result.r2:.4f})",
        )
    )
    fig.update_layout(
        title=f"アレニウスプロット (Ea={Ea_kJ:.2f} kJ/mol, A={result.A:.3e})",
        xaxis_title="1/T (K⁻¹)",
        yaxis_title="ln(k)",
        template="plotly_white",
        height=420,
        legend=dict(x=0.5, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-temperature raw data overlay
# ---------------------------------------------------------------------------

def plot_raw_multi_temp(temp_groups: dict[float, object]) -> go.Figure:
    """Overlay raw concentration data for each temperature, color-coded."""
    import plotly.express as px

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, (T_c, sub_df) in enumerate(temp_groups.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=sub_df["time"],
                y=sub_df["concentration"],
                mode="markers+lines",
                marker=dict(size=7, color=color),
                line=dict(color=color, dash="dot"),
                name=f"{T_c:.1f} °C",
            )
        )

    fig.update_layout(
        title="複数温度データ 原データ重ね描き",
        xaxis_title="時間 (min)",
        yaxis_title="濃度 (mol/L)",
        template="plotly_white",
        height=420,
        legend=dict(title="温度"),
    )
    return fig


# ---------------------------------------------------------------------------
# Per-temperature reaction order bar chart
# ---------------------------------------------------------------------------

def plot_order_per_temp(
    per_temp_results: list[PerTempResult],
    optimal_n: float | None = None,
) -> go.Figure:
    """Bar chart of estimated reaction order per temperature."""
    fig = go.Figure()

    valid = [r for r in per_temp_results if r.success and np.isfinite(r.n)]

    if not valid:
        fig.add_annotation(
            text="解析成功データなし",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title="温度別 推算反応次数",
            xaxis_title="温度 (°C)",
            yaxis_title="推算反応次数 n",
            template="plotly_white",
            height=400,
        )
        return fig

    temps = [r.temperature_C for r in valid]
    n_vals = [r.n for r in valid]
    r2_vals = [r.r2 for r in valid]
    text_labels = [f"R²={r:.3f}" for r in r2_vals]

    fig.add_trace(
        go.Bar(
            x=temps,
            y=n_vals,
            text=text_labels,
            textposition="outside",
            marker=dict(
                color=r2_vals,
                colorscale="Viridis",
                colorbar=dict(title="R²"),
                showscale=True,
            ),
            name="推算次数 n",
        )
    )

    if optimal_n is not None:
        fig.add_hline(
            y=optimal_n,
            line_dash="dash",
            line_color="red",
            annotation_text=f"最適 n={optimal_n:.3f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title="温度別 推算反応次数",
        xaxis_title="温度 (°C)",
        yaxis_title="推算反応次数 n",
        template="plotly_white",
        height=400,
    )
    return fig
