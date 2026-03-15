"""Plotly figure generators for reaction kinetics analysis."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.models import ArrheniusResult, FitResult


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
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
# RK4+LSQ fit plot
# ---------------------------------------------------------------------------

def plot_fit(df, fit: FitResult) -> go.Figure:
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
    title_suffix = reaction_labels.get(fit.reaction_type, fit.reaction_type)

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

    for sp, pred in fit.c_pred.items():
        if sp not in species_obs and fit.reaction_type == "simple":
            continue
        color = SPECIES_COLORS[sp]
        fig.add_trace(
            go.Scatter(
                x=fit.t_pred, y=pred,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"[{sp}] RK45予測",
            )
        )

    k_str = f"k={fit.k:.4f}" if np.isfinite(fit.k) else "k=N/A"
    if fit.k2 is not None and np.isfinite(fit.k2):
        k_str += f", k₂={fit.k2:.4f}"

    fig.update_layout(
        title=f"RK4+最小二乗法: {title_suffix} ({k_str}, R²={fit.r2:.4f})",
        xaxis_title="時間 (min)",
        yaxis_title="濃度 (mol/L)",
        template="plotly_white",
        height=420,
        legend=dict(x=0.5, y=0.99),
    )
    return fig


# ---------------------------------------------------------------------------
# Residuals plot (new)
# ---------------------------------------------------------------------------

def plot_residuals_rk4(df, fit: FitResult) -> go.Figure:
    """Residuals (observed - predicted) vs. time for each species."""
    time = df["time"].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    for sp, resid in fit.residuals.items():
        color = SPECIES_COLORS.get(sp, "#888888")
        # Only plot where we have observed data
        if sp == "A":
            mask = df["concentration"].notna().to_numpy()
        elif sp == "B":
            mask = df["concentration_B"].notna().to_numpy() if "concentration_B" in df.columns else np.zeros(len(time), dtype=bool)
        elif sp == "C":
            mask = df["concentration_C"].notna().to_numpy() if "concentration_C" in df.columns else np.zeros(len(time), dtype=bool)
        else:
            mask = np.ones(len(time), dtype=bool)

        valid_resid = resid[mask] if len(resid) == len(time) else resid
        valid_time  = time[mask]

        finite_mask = np.isfinite(valid_resid)
        if finite_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=valid_time[finite_mask],
                    y=valid_resid[finite_mask],
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.8),
                    name=f"[{sp}] 残差",
                )
            )

    fig.update_layout(
        title="残差プロット (観測 − 予測)",
        xaxis_title="時間 (min)",
        yaxis_title="残差 (mol/L)",
        template="plotly_white",
        height=300,
        legend=dict(x=0.01, y=0.99),
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
    """ln(k) vs 1/T scatter + regression line."""
    inv_T = np.array(result.inv_T)
    ln_k  = np.array(result.ln_k)

    inv_T_fit = np.linspace(inv_T.min() * 0.99, inv_T.max() * 1.01, 200)
    ln_k_fit  = result.intercept + result.slope * inv_T_fit

    Ea_kJ = result.Ea_kJmol
    temps_c = result.temps_celsius

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=inv_T, y=ln_k,
            mode="markers+text",
            marker=dict(size=10, color="#EF553B"),
            text=[f"{t:.0f}°C" for t in temps_c],
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
# Simulation results overlay
# ---------------------------------------------------------------------------

DASH_STYLES = ["solid", "dash", "dot", "dashdot", "longdash"]


def plot_simulation_results(sim_results, reaction_type: str) -> go.Figure:
    """複数条件の濃度プロファイルを1図に重ね描き。
    - 成分色: SPECIES_COLORS
    - 線種: DASH_STYLES[condition_index % 5] で条件を区別
    - トレース名: "[A] 条件1", "[B] 条件1", ...
    """
    fig = go.Figure()

    species_order = ["A", "B", "C"]
    if reaction_type == "simple":
        species_order = ["A"]

    for cond_idx, (cond, t_eval, c_dict) in enumerate(sim_results):
        if t_eval is None or c_dict is None:
            continue
        dash = DASH_STYLES[cond_idx % len(DASH_STYLES)]
        for sp in species_order:
            if sp not in c_dict:
                continue
            color = SPECIES_COLORS[sp]
            fig.add_trace(
                go.Scatter(
                    x=t_eval,
                    y=c_dict[sp],
                    mode="lines",
                    line=dict(color=color, width=2, dash=dash),
                    name=f"[{sp}] {cond.label}",
                )
            )

    fig.update_layout(
        title="シミュレーション結果",
        xaxis_title="時間 (min)",
        yaxis_title="濃度 (mol/L)",
        template="plotly_white",
        height=500,
        legend=dict(x=1.01, y=1.0, xanchor="left"),
        margin=dict(r=180),
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-temperature raw data overlay
# ---------------------------------------------------------------------------

def plot_raw_multi_temp(temp_groups: dict) -> go.Figure:
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
