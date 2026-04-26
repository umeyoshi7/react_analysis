"""反応熱推算アプリ — Streamlit エントリーポイント"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.thermochemistry import (
    COMMON_MOLECULES,
    REACTION_TEMPLATES,
    SOLVENT_DATA,
    calc_reaction_heat,
    get_mol_svg,
    svg_to_data_uri,
    validate_smiles,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="反応熱推算アプリ", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "reactants": [{"smiles": "", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0}],
        "products":  [{"smiles": "", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0}],
        "result":    None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("反応熱推算")
    st.markdown("---")

    # テンプレート選択
    st.subheader("反応テンプレート")
    template_options = ["(なし — 手動入力)"] + list(REACTION_TEMPLATES.keys())
    selected_template = st.selectbox(
        "反応タイプを選択",
        template_options,
        help="選択すると反応物・生成物が自動入力されます。",
    )

    if selected_template != "(なし — 手動入力)":
        tmpl = REACTION_TEMPLATES[selected_template]
        st.caption(f"**{tmpl['type']}** — {tmpl['description']}")
        if st.button("テンプレートを適用", use_container_width=True, type="primary"):
            import copy
            st.session_state["reactants"] = copy.deepcopy(tmpl["reactants"])
            st.session_state["products"]  = copy.deepcopy(tmpl["products"])
            st.session_state["result"]    = None
            st.rerun()

    st.markdown("---")

    # SMILES 早見表
    with st.expander("よく使う SMILES 一覧"):
        import pandas as pd
        ref_df = pd.DataFrame(
            [{"化合物名": n, "化学式": f, "SMILES": s} for f, s, n in COMMON_MOLECULES]
        )
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "**推算精度目安**\n"
        "- 文献値: 高精度\n"
        "- Joback 法: ±10〜20 kJ/mol（有機分子）\n"
        "- 無機物・小分子で失敗する場合は手動入力"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_molecule_svg(smiles: str, width: int = 180, height: int = 130) -> None:
    """SMILES を 2D 構造式として表示する."""
    if not smiles.strip():
        st.caption("—")
        return
    svg = get_mol_svg(smiles, width=width, height=height)
    if svg is None:
        st.error("無効な SMILES")
        return
    uri = svg_to_data_uri(svg)
    st.markdown(
        f'<img src="{uri}" width="{width}" style="border:1px solid #ddd;border-radius:4px;"/>',
        unsafe_allow_html=True,
    )


def _render_compound_rows(rows: list[dict], role: str, key_prefix: str) -> None:
    """反応物 / 生成物の入力行を描画し、rows を in-place で更新する."""
    header = st.columns([3.5, 1.2, 1.8, 2.0, 0.7])
    header[0].caption("SMILES")
    header[1].caption("係数")
    header[2].caption("化学式")
    header[3].caption("ΔHf° 手動入力 (kJ/mol)")
    header[4].caption("削除")

    for idx, row in enumerate(rows):
        c1, c2, c3, c4, c5 = st.columns([3.5, 1.2, 1.8, 2.0, 0.7])

        with c1:
            smiles_val = st.text_input(
                "SMILES",
                value=row["smiles"],
                key=f"{key_prefix}_smiles_{idx}",
                placeholder="例: CCO",
                label_visibility="collapsed",
            )
            rows[idx]["smiles"] = smiles_val

        with c2:
            coeff_val = st.number_input(
                "係数",
                value=float(row["coeff"]),
                min_value=0.01,
                step=0.5,
                format="%.2f",
                key=f"{key_prefix}_coeff_{idx}",
                label_visibility="collapsed",
            )
            rows[idx]["coeff"] = coeff_val

        with c3:
            if smiles_val.strip():
                valid, _, formula = validate_smiles(smiles_val)
                if valid:
                    st.markdown(f"**{formula}**")
                else:
                    st.error("無効")
            else:
                st.caption("—")

        with c4:
            use_manual = st.checkbox(
                "手動入力",
                value=bool(row.get("use_manual", False)),
                key=f"{key_prefix}_useman_{idx}",
                label_visibility="collapsed",
            )
            rows[idx]["use_manual"] = use_manual
            if use_manual:
                manual_val = st.number_input(
                    "ΔHf°",
                    value=float(row.get("manual_hf", 0.0)),
                    step=10.0,
                    format="%.2f",
                    key=f"{key_prefix}_mhf_{idx}",
                    label_visibility="collapsed",
                )
                rows[idx]["manual_hf"] = manual_val
            else:
                st.caption("自動推算")

        with c5:
            st.write("")
            if st.button("x", key=f"{key_prefix}_del_{idx}",
                         disabled=(len(rows) <= 1), use_container_width=True):
                rows.pop(idx)
                st.session_state[f"{role}"] = rows
                st.rerun()


def _build_scheme_html(
    reactants: list[dict],
    products: list[dict],
    img_w: int = 160,
    img_h: int = 120,
) -> str:
    """反応スキーム (構造式 → 矢印 → 構造式) の HTML を生成する."""

    def _mol_block(row: dict) -> str:
        smi = row["smiles"].strip()
        coeff = row["coeff"]
        coeff_str = f"{coeff:.0f}" if coeff == int(coeff) else f"{coeff:.2f}"

        if not smi:
            return "<div style='text-align:center;padding:8px;color:#aaa;'>?</div>"

        svg = get_mol_svg(smi, width=img_w, height=img_h)
        _, _, formula = validate_smiles(smi)
        if svg is None:
            return f"<div style='text-align:center;color:red;'>{smi}<br>(無効)</div>"

        uri = svg_to_data_uri(svg)
        return (
            f"<div style='text-align:center;'>"
            f"<div style='font-size:1.1em;font-weight:bold;color:#555;'>{coeff_str}</div>"
            f"<img src='{uri}' width='{img_w}' style='border:1px solid #ddd;border-radius:4px;'/>"
            f"<div style='font-size:0.85em;color:#333;margin-top:2px;'>{formula}</div>"
            f"</div>"
        )

    def _side_html(rows: list[dict]) -> str:
        parts = []
        for i, row in enumerate(rows):
            if i > 0:
                parts.append(
                    "<div style='display:flex;align-items:center;font-size:1.6em;"
                    "color:#555;padding:0 4px;'>+</div>"
                )
            parts.append(_mol_block(row))
        return "".join(parts)

    reactant_html = _side_html([r for r in reactants if r["smiles"].strip()])
    product_html  = _side_html([p for p in products  if p["smiles"].strip()])

    arrow = (
        "<div style='display:flex;align-items:center;font-size:2.5em;"
        "color:#444;padding:0 12px;'>→</div>"
    )

    return (
        "<div style='display:flex;align-items:center;flex-wrap:wrap;"
        "gap:4px;padding:12px;background:#fafafa;border-radius:8px;"
        "border:1px solid #e0e0e0;'>"
        + reactant_html
        + arrow
        + product_html
        + "</div>"
    )


def _make_energy_diagram(
    delta_H: float,
    delta_H_gas: float | None,
    kirchhoff: float,
    solvent: float,
) -> go.Figure:
    """エンタルピーレベル図を生成する."""
    color = "#d62728" if delta_H < 0 else "#1f77b4"
    reactant_y = 0.0
    product_y = delta_H

    shapes = [
        dict(type="line", x0=0.1, x1=0.4, y0=reactant_y, y1=reactant_y,
             line=dict(color="#555", width=3)),
        dict(type="line", x0=0.6, x1=0.9, y0=product_y, y1=product_y,
             line=dict(color="#555", width=3)),
        dict(type="line", x0=0.5, x1=0.5, y0=reactant_y, y1=product_y,
             line=dict(color=color, width=2, dash="dot")),
    ]

    annotations = [
        dict(x=0.25, y=reactant_y + abs(delta_H) * 0.05 if delta_H > 0 else reactant_y - abs(delta_H) * 0.05,
             text="反応物", showarrow=False, font=dict(size=13)),
        dict(x=0.75, y=product_y + abs(delta_H) * 0.05 if delta_H < 0 else product_y - abs(delta_H) * 0.05,
             text="生成物", showarrow=False, font=dict(size=13)),
        dict(x=0.55, y=(reactant_y + product_y) / 2,
             text=f"ΔH = {delta_H:+.2f} kJ/mol",
             showarrow=False,
             font=dict(size=14, color=color),
             xanchor="left"),
    ]

    y_pad = max(abs(delta_H) * 0.3, 20)
    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(
            title="エンタルピー (kJ/mol)",
            range=[min(reactant_y, product_y) - y_pad, max(reactant_y, product_y) + y_pad],
        ),
        height=300,
        margin=dict(l=60, r=20, t=20, b=20),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("反応熱推算アプリ")
st.caption("SMILES 入力 → 反応スキーム可視化 → ΔH_rxn 推算 (Joback 基団寄与法 / 文献値)")

# ── 入力セクション ──────────────────────────────────────────────────────────
col_r, col_p = st.columns(2)

with col_r:
    st.subheader("反応物")
    thermo_r: list[dict] = st.session_state["reactants"]
    _render_compound_rows(thermo_r, "reactants", "thr")
    st.session_state["reactants"] = thermo_r
    if st.button("反応物を追加", key="add_r"):
        thermo_r.append({"smiles": "", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0})
        st.session_state["reactants"] = thermo_r
        st.rerun()

with col_p:
    st.subheader("生成物")
    thermo_p: list[dict] = st.session_state["products"]
    _render_compound_rows(thermo_p, "products", "thp")
    st.session_state["products"] = thermo_p
    if st.button("生成物を追加", key="add_p"):
        thermo_p.append({"smiles": "", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0})
        st.session_state["products"] = thermo_p
        st.rerun()

st.markdown("---")

# ── 反応スキーム可視化 ──────────────────────────────────────────────────────
has_any_smiles = any(r["smiles"].strip() for r in thermo_r) or any(p["smiles"].strip() for p in thermo_p)

if has_any_smiles:
    st.subheader("反応スキーム")
    scheme_html = _build_scheme_html(thermo_r, thermo_p)
    st.markdown(scheme_html, unsafe_allow_html=True)
    st.markdown("---")

# ── 反応条件 ────────────────────────────────────────────────────────────────
st.subheader("反応条件")
cond_col1, cond_col2 = st.columns(2)

with cond_col1:
    st.markdown("**温度 (Kirchhoff 補正)**")
    use_temp_corr = st.checkbox(
        "298.15 K 以外の温度で計算する",
        value=False,
        help="Kirchhoff則 ΔH(T) ≈ ΔH°(298.15K) + ΔCp × (T − 298.15K) を適用します",
    )
    if use_temp_corr:
        temp_celsius = st.number_input(
            "反応温度 (°C)", value=25.0, min_value=-200.0, max_value=2000.0,
            step=10.0, format="%.1f",
        )
        temperature_K = temp_celsius + 273.15
        st.caption(f"= {temperature_K:.2f} K")
    else:
        temperature_K = 298.15
        st.caption("標準状態 (298.15 K / 25 °C)")

with cond_col2:
    st.markdown("**溶媒 (補正)**")
    solvent_name = st.selectbox(
        "溶媒を選択",
        list(SOLVENT_DATA.keys()),
        help="溶媒効果の定性的な情報を表示します。定量補正は下の手動入力を使用してください。",
        label_visibility="collapsed",
    )
    solvent_info = SOLVENT_DATA[solvent_name]
    st.caption(
        f"ε = {solvent_info['dielectric']} | {solvent_info['note']}"
    )
    solvent_correction = st.number_input(
        "溶媒補正値 ΔH_solv (kJ/mol)",
        value=0.0,
        step=1.0,
        format="%.2f",
        help=(
            "溶媒中での反応エンタルピー補正値を手動入力します。\n"
            "ΔH_rxn(溶液) ≈ ΔH_rxn(気相) + ΔH_solv"
        ),
    )

st.markdown("---")

# ── 計算ボタン ───────────────────────────────────────────────────────────────
col_calc, col_clear = st.columns([3, 1])
with col_calc:
    calc_disabled = (
        not any(r["smiles"].strip() for r in thermo_r)
        or not any(p["smiles"].strip() for p in thermo_p)
    )
    calc_btn = st.button(
        "反応熱を計算", type="primary", use_container_width=True, disabled=calc_disabled
    )
with col_clear:
    if st.button("リセット", use_container_width=True):
        st.session_state["reactants"] = [{"smiles": "", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0}]
        st.session_state["products"]  = [{"smiles": "", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0}]
        st.session_state["result"]    = None
        st.rerun()

if calc_btn:
    reactants_in = [
        (r["coeff"], r["smiles"], r["manual_hf"] if r["use_manual"] else None)
        for r in thermo_r if r["smiles"].strip()
    ]
    products_in = [
        (p["coeff"], p["smiles"], p["manual_hf"] if p["use_manual"] else None)
        for p in thermo_p if p["smiles"].strip()
    ]
    with st.spinner("計算中…"):
        st.session_state["result"] = calc_reaction_heat(
            reactants_in,
            products_in,
            temperature_K=temperature_K,
            solvent_correction_kJ=solvent_correction,
        )

# ── 結果表示 ─────────────────────────────────────────────────────────────────
result = st.session_state.get("result")
if result is not None:
    st.markdown("---")
    st.subheader("計算結果")

    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    if result.success and result.delta_H_kJ_mol is not None:
        dH = result.delta_H_kJ_mol
        color = "#d62728" if dH < 0 else ("#1f77b4" if dH > 0 else "#555555")
        rxn_label = "発熱反応 (exothermic)" if dH < 0 else ("吸熱反応 (endothermic)" if dH > 0 else "熱中性")

        st.markdown(
            f"<h2 style='color:{color};'>ΔH_rxn = {dH:+.2f} kJ/mol</h2>"
            f"<p style='color:{color};font-size:1.1em;'>{rxn_label}"
            f" @ {result.temperature_K - 273.15:.1f} °C"
            + (f"  |  溶媒: {solvent_name}" if solvent_name != "なし (気相・標準状態)" else "")
            + "</p>",
            unsafe_allow_html=True,
        )

        # 補正内訳
        if abs(result.kirchhoff_correction_kJ) > 0.01 or abs(result.solvent_correction_kJ) > 0.01:
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("気相標準値 ΔH° (298K)", f"{result.delta_H_gas_kJ_mol:+.2f} kJ/mol")
            dc2.metric("Kirchhoff 温度補正", f"{result.kirchhoff_correction_kJ:+.2f} kJ/mol",
                       delta=None if result.temperature_K == 298.15 else f"{result.temperature_K - 273.15:.0f} °C")
            dc3.metric("溶媒補正", f"{result.solvent_correction_kJ:+.2f} kJ/mol")

        # エンタルピーレベル図
        fig = _make_energy_diagram(
            dH,
            result.delta_H_gas_kJ_mol,
            result.kirchhoff_correction_kJ,
            result.solvent_correction_kJ,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 明細テーブル
        import pandas as pd
        rows_data = []
        for coeff, cr in result.reactant_results:
            rows_data.append({
                "区分": "反応物",
                "化学式": cr.formula,
                "SMILES": cr.canonical_smiles,
                "係数 ν": f"−{coeff:.2f}",
                "ΔHf° (kJ/mol)": f"{cr.hf_kJ_mol:.2f}",
                "寄与 (kJ/mol)": f"{-coeff * cr.hf_kJ_mol:+.2f}",
                "計算手法": cr.method + (f" [{cr.known_name}]" if cr.known_name else ""),
            })
        for coeff, cr in result.product_results:
            rows_data.append({
                "区分": "生成物",
                "化学式": cr.formula,
                "SMILES": cr.canonical_smiles,
                "係数 ν": f"+{coeff:.2f}",
                "ΔHf° (kJ/mol)": f"{cr.hf_kJ_mol:.2f}",
                "寄与 (kJ/mol)": f"{coeff * cr.hf_kJ_mol:+.2f}",
                "計算手法": cr.method + (f" [{cr.known_name}]" if cr.known_name else ""),
            })
        st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)

        st.markdown(
            r"""
**計算式:** ΔH_rxn = Σ(ν_生成物 × ΔHf°_生成物) − Σ(ν_反応物 × ΔHf°_反応物)

**精度目安:** 文献値: 高精度 | Joback 法: ±10〜20 kJ/mol（有機分子）
"""
        )
    else:
        st.error("一部の化合物で ΔHf° を取得できませんでした。上記の警告を確認して手動入力してください。")

# ── 解析ロジック説明 ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("解析ロジック・計算手法の説明"):
    st.markdown(
        r"""
### 計算手法

**1. ΔHf° の取得優先順位**

| 優先度 | 手法 | 説明 |
|--------|------|------|
| 1 | 手動入力 | ユーザーが指定した値をそのまま使用 |
| 2 | 文献値 | 主要無機物・小分子の NIST 値 |
| 3 | Joback 基団寄与法 | ugropy ライブラリで官能基を分解・積算 |

**2. 反応熱の計算式**

$$\Delta H_{rxn} = \sum_i \nu_i \Delta H_{f,i}^\circ(\text{生成物}) - \sum_j \nu_j \Delta H_{f,j}^\circ(\text{反応物})$$

**3. Kirchhoff 温度補正**

$$\Delta H(T) \approx \Delta H^\circ(298.15\,\text{K}) + \Delta C_p \cdot (T - 298.15)$$

$\Delta C_p$ は文献値または Joback 法の $C_p$ から算出（定数近似）。データ不足の化合物は $\Delta C_p = 0$ と仮定。

**4. 溶媒補正**

$$\Delta H_{rxn}(\text{溶液}) \approx \Delta H_{rxn}(\text{気相}) + \Delta H_{solv}$$

$\Delta H_{solv}$ はユーザー手動入力（溶媒和エンタルピーは化合物・溶媒により大きく異なるため）。

### 精度の目安

| 手法 | 精度 |
|------|------|
| 文献値 | < ±1 kJ/mol |
| Joback 法 | ±10〜20 kJ/mol（有機分子） |
| Kirchhoff 補正（定数 $C_p$）| ±5〜20 kJ/mol（±100 K 以内） |
"""
    )
