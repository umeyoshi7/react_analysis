"""反応熱推算モジュール (Joback基団寄与法 + 文献値フォールバック)"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# 標準生成エンタルピー文献値 ΔHf° (kJ/mol, 298.15 K, 理想気体)
# ---------------------------------------------------------------------------
_KNOWN_HF: dict[str, tuple[float, str]] = {
    "O":              (-241.826, "H₂O(g)"),
    "O=C=O":          (-393.509, "CO₂(g)"),
    "[C-]#[O+]":      (-110.527, "CO(g)"),
    "[H][H]":         (0.0,      "H₂(g)"),
    "N#N":            (0.0,      "N₂(g)"),
    "[O][O]":         (0.0,      "O₂(g)"),
    "O=O":            (0.0,      "O₂(g)"),
    "O=S=O":          (-296.830, "SO₂(g)"),
    "O=S(=O)=O":      (-395.765, "SO₃(g)"),
    "Cl":             (-92.307,  "HCl(g)"),
    "N":              (-46.11,   "NH₃(g)"),
    "C":              (-74.81,   "CH₄(g)"),
    "C=C":            (52.47,    "C₂H₄(g)"),
    "F":              (-273.3,   "HF(g)"),
    "Br":             (-36.29,   "HBr(g)"),
    "S":              (-20.6,    "H₂S(g)"),
    "ClCl":           (0.0,      "Cl₂(g)"),
    "[N]=O":          (91.29,    "NO(g)"),
    "O=[N+][O-]":     (33.18,    "NO₂(g)"),
    "[N-]=[N+]=O":    (82.05,    "N₂O(g)"),
    "O=[N+]([O-])O":  (-133.9,   "HNO₃(g)"),
    "C=O":            (-108.57,  "HCHO(g)"),
    "O=CO":           (-378.7,   "HCOOH(g)"),
    "OO":             (-136.3,   "H₂O₂(g)"),
    "BrBr":           (30.91,    "Br₂(g)"),
    "FF":             (0.0,      "F₂(g)"),
    "CC=O":           (-166.19,  "CH₃CHO(g)"),
}

# 定圧熱容量 Cp (J/mol/K, 理想気体 ~298 K, 定数近似)
# キーは RDKit 正規化 SMILES (モジュール末尾の _build_cp_table で正規化済み)
_CP_RAW: dict[str, float] = {
    "O":          33.59,   # H2O(g)
    "O=C=O":      37.11,   # CO2(g)
    "[H][H]":     28.84,   # H2(g)
    "[O][O]":     29.38,   # O2(g)
    "O=O":        29.38,
    "N#N":        29.12,   # N2(g)
    "C":          35.71,   # CH4(g)
    "[C-]#[O+]":  29.14,   # CO(g)
    "N":          35.65,   # NH3(g)
    "Cl":         29.12,   # HCl(g)
    "O=S=O":      39.87,   # SO2(g)
    "C=C":        43.56,   # C2H4(g)
    "CO":         44.06,   # CH3OH(g)
    "CCO":        65.56,   # C2H5OH(g)
    "CC=O":       57.32,   # CH3CHO(g)
    "CC(=O)C":    74.92,   # acetone(g)
    "CC(=O)O":    86.5,    # CH3COOH(g)
    "c1ccccc1":   82.44,   # benzene(g)
    "Cc1ccccc1":  103.6,   # toluene(g)
    "CCCl":       73.6,    # chloroethane(g)
    "BrBr":       75.73,   # Br2(g)
    "ClCl":       33.91,   # Cl2(g)
    "OO":         43.1,    # H2O2(g)
    "BrCCBr":     93.4,    # 1,2-dibromoethane(g)
    "CCOC(C)=O":  113.7,   # ethyl acetate(g)
    "OC(=O)/C=C\\C(=O)O": 120.0,  # maleic acid (approx)
}


def _build_cp_table() -> dict[str, float]:
    """起動時にキーを RDKit 正規化 SMILES へ変換する."""
    result: dict[str, float] = {}
    for smi, cp in _CP_RAW.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            result[Chem.MolToSmiles(mol)] = cp
    return result


_KNOWN_CP: dict[str, float] = _build_cp_table()

# ---------------------------------------------------------------------------
# 溶媒データ
# ---------------------------------------------------------------------------
SOLVENT_DATA: dict[str, dict] = {
    "なし (気相・標準状態)": {
        "dielectric": 1.0,
        "note": "気体状態でのΔH_rxn。標準状態 (298.15 K, 1 bar)。",
    },
    "水 (H₂O)": {
        "dielectric": 78.4,
        "note": "極性プロトン性。水素結合・水和の影響が大きい。",
    },
    "メタノール": {
        "dielectric": 32.7,
        "note": "極性プロトン性。",
    },
    "エタノール": {
        "dielectric": 24.6,
        "note": "極性プロトン性。",
    },
    "アセトン": {
        "dielectric": 20.7,
        "note": "極性非プロトン性。",
    },
    "DMSO": {
        "dielectric": 46.7,
        "note": "極性非プロトン性。強い溶媒和効果。",
    },
    "THF": {
        "dielectric": 7.6,
        "note": "極性非プロトン性。環状エーテル。",
    },
    "ジクロロメタン (DCM)": {
        "dielectric": 8.93,
        "note": "極性非プロトン性。塩素化溶媒。",
    },
    "ヘキサン": {
        "dielectric": 1.88,
        "note": "非極性炭化水素溶媒。溶媒和効果は小さい。",
    },
    "トルエン": {
        "dielectric": 2.38,
        "note": "非極性芳香族溶媒。",
    },
}

# ---------------------------------------------------------------------------
# 反応テンプレート
# ---------------------------------------------------------------------------
REACTION_TEMPLATES: dict[str, dict] = {
    "付加反応: エチレン + HCl → クロロエタン": {
        "type": "付加反応 (Addition)",
        "description": "アルケンへの HCl 付加（マルコフニコフ則）",
        "reactants": [
            {"smiles": "C=C",  "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "Cl",   "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
        "products": [
            {"smiles": "CCCl", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
    },
    "付加反応: エチレン + Br₂ → ジブロモエタン": {
        "type": "付加反応 (Addition)",
        "description": "アルケンへの Br₂ 付加（ハロゲン化）",
        "reactants": [
            {"smiles": "C=C",   "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "BrBr",  "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
        "products": [
            {"smiles": "BrCCBr", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
    },
    "置換反応: エタノール + HCl → クロロエタン + H₂O": {
        "type": "置換反応 (SN)",
        "description": "アルコールのハロゲン化（求核置換）",
        "reactants": [
            {"smiles": "CCO",  "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "Cl",   "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
        "products": [
            {"smiles": "CCCl", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "O",    "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
    },
    "脱離反応: クロロエタン → エチレン + HCl": {
        "type": "脱離反応 (E2)",
        "description": "ハロゲン化アルキルからの脱ハロゲン化水素",
        "reactants": [
            {"smiles": "CCCl", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
        "products": [
            {"smiles": "C=C", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "Cl",  "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
    },
    "脱離反応: エタノール → エチレン + H₂O": {
        "type": "脱離反応 (E1)",
        "description": "アルコールの脱水（酸触媒）",
        "reactants": [
            {"smiles": "CCO", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
        "products": [
            {"smiles": "C=C", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "O",   "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
    },
    "縮合反応: 酢酸 + エタノール → 酢酸エチル + H₂O": {
        "type": "縮合反応 / エステル化",
        "description": "カルボン酸とアルコールのエステル化",
        "reactants": [
            {"smiles": "CC(=O)O", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "CCO",     "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
        "products": [
            {"smiles": "CCOC(C)=O", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "O",         "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
    },
    "分子内環化: マレイン酸 → 無水マレイン酸 + H₂O": {
        "type": "分子内環化 / 脱水",
        "description": "シス-ジカルボン酸の環状無水物形成",
        "reactants": [
            {"smiles": "OC(=O)/C=C\\C(=O)O", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
        "products": [
            {"smiles": "O=C1OC(=O)C=C1", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "O",              "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
        ],
    },
    "燃焼反応: メタン + 2O₂ → CO₂ + 2H₂O": {
        "type": "燃焼反応",
        "description": "炭化水素の完全燃焼",
        "reactants": [
            {"smiles": "C",     "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "[O][O]","coeff": 2.0, "use_manual": False, "manual_hf": 0.0},
        ],
        "products": [
            {"smiles": "O=C=O", "coeff": 1.0, "use_manual": False, "manual_hf": 0.0},
            {"smiles": "O",     "coeff": 2.0, "use_manual": False, "manual_hf": 0.0},
        ],
    },
}

# よく使う分子の SMILES 早見表 (UI 表示用)
COMMON_MOLECULES: list[tuple[str, str, str]] = [
    ("H₂O",      "O",             "水"),
    ("CO₂",      "O=C=O",         "二酸化炭素"),
    ("O₂",       "[O][O]",        "酸素"),
    ("H₂",       "[H][H]",        "水素"),
    ("N₂",       "N#N",           "窒素"),
    ("CO",       "[C-]#[O+]",     "一酸化炭素"),
    ("CH₄",      "C",             "メタン"),
    ("NH₃",      "N",             "アンモニア"),
    ("SO₂",      "O=S=O",         "二酸化硫黄"),
    ("HCl",      "Cl",            "塩化水素"),
    ("HBr",      "Br",            "臭化水素"),
    ("Br₂",      "BrBr",          "臭素"),
    ("Cl₂",      "ClCl",          "塩素"),
    ("C₂H₄",     "C=C",           "エチレン"),
    ("HCHO",     "C=O",           "ホルムアルデヒド"),
    ("CH₃OH",    "CO",            "メタノール"),
    ("C₂H₅OH",   "CCO",           "エタノール"),
    ("CH₃COCH₃", "CC(=O)C",       "アセトン"),
    ("CH₃COOH",  "CC(=O)O",       "酢酸"),
    ("C₆H₆",     "c1ccccc1",      "ベンゼン"),
    ("C₇H₈",     "Cc1ccccc1",     "トルエン"),
]


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def validate_smiles(smiles: str) -> tuple[bool, str, str]:
    """SMILES を検証し (valid, canonical_smiles, formula) を返す."""
    smiles = smiles.strip()
    if not smiles:
        return False, "", ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, smiles, ""
    canon = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return True, canon, formula


def get_mol_svg(smiles: str, width: int = 200, height: int = 150) -> str | None:
    """SMILES から RDKit で 2D 構造式 SVG を生成する."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def svg_to_data_uri(svg: str) -> str:
    """SVG 文字列を data URI に変換する."""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


# ---------------------------------------------------------------------------
# 計算結果データクラス
# ---------------------------------------------------------------------------

@dataclass
class CompoundResult:
    smiles_input: str
    canonical_smiles: str
    formula: str
    hf_kJ_mol: float | None
    method: str
    known_name: str = ""
    error: str = ""


@dataclass
class ReactionHeatResult:
    delta_H_kJ_mol: float | None          # 補正後の最終値
    delta_H_gas_kJ_mol: float | None       # 気相標準値 (補正前)
    kirchhoff_correction_kJ: float         # 温度補正量 (kJ/mol)
    solvent_correction_kJ: float           # 溶媒補正量 (kJ/mol)
    temperature_K: float
    reactant_results: list[tuple[float, CompoundResult]]
    product_results: list[tuple[float, CompoundResult]]
    warnings: list[str]
    success: bool


# ---------------------------------------------------------------------------
# ΔHf° 取得
# ---------------------------------------------------------------------------

def get_hf(smiles: str, manual_hf: float | None = None) -> CompoundResult:
    """SMILES から ΔHf° (kJ/mol) を取得する.

    優先順位:
        1. manual_hf が指定されている場合
        2. 文献値ルックアップ
        3. Joback 基団寄与法 (ugropy)
        4. 失敗
    """
    valid, canon, formula = validate_smiles(smiles)
    if not valid:
        return CompoundResult(
            smiles_input=smiles,
            canonical_smiles=smiles,
            formula="?",
            hf_kJ_mol=None,
            method="推算失敗",
            error="無効な SMILES です",
        )

    if manual_hf is not None:
        return CompoundResult(
            smiles_input=smiles,
            canonical_smiles=canon,
            formula=formula,
            hf_kJ_mol=manual_hf,
            method="手動入力",
        )

    if canon in _KNOWN_HF:
        hf, name = _KNOWN_HF[canon]
        return CompoundResult(
            smiles_input=smiles,
            canonical_smiles=canon,
            formula=formula,
            hf_kJ_mol=hf,
            method="文献値",
            known_name=name,
        )

    try:
        from ugropy import joback
        result = joback.get_groups(smiles, identifier_type="smiles")
        hf_q = result.ig_enthalpy_formation
        if hf_q is not None:
            return CompoundResult(
                smiles_input=smiles,
                canonical_smiles=canon,
                formula=formula,
                hf_kJ_mol=float(hf_q.magnitude),
                method="Joback法",
            )
    except Exception:
        pass

    return CompoundResult(
        smiles_input=smiles,
        canonical_smiles=canon,
        formula=formula,
        hf_kJ_mol=None,
        method="推算失敗",
        error=(
            "Joback 法でグループ分解に失敗しました。"
            "手動で ΔHf° を入力してください。"
        ),
    )


# ---------------------------------------------------------------------------
# Cp 推定
# ---------------------------------------------------------------------------

def get_cp_estimate(smiles: str) -> tuple[float | None, str]:
    """SMILES から Cp (J/mol/K, ~298 K) を推定する. (値, 手法) を返す."""
    valid, canon, _ = validate_smiles(smiles)
    if not valid:
        return None, "無効なSMILES"

    if canon in _KNOWN_CP:
        return _KNOWN_CP[canon], "文献値 (定数近似)"

    try:
        from ugropy import joback
        result = joback.get_groups(smiles, identifier_type="smiles")
        # ugropy が Cp 係数を提供している場合 (a + b*T + c*T² + d*T³)
        for a_attr in ("ig_heat_capacity_a", "heat_capacity_a", "Cp_a"):
            if hasattr(result, a_attr) and getattr(result, a_attr) is not None:
                a = float(getattr(result, a_attr))
                def _get(attr: str) -> float:
                    v = getattr(result, attr, None)
                    return float(v) if v is not None else 0.0
                b = _get(a_attr.replace("_a", "_b"))
                c = _get(a_attr.replace("_a", "_c"))
                d = _get(a_attr.replace("_a", "_d"))
                T = 298.15
                cp = a + b * T + c * T**2 + d * T**3
                return cp, "Joback法 (298 K)"
    except Exception:
        pass

    return None, "推算不可"


# ---------------------------------------------------------------------------
# Kirchhoff 温度補正
# ---------------------------------------------------------------------------

def calc_kirchhoff_correction(
    reactants: list[tuple[float, str, float | None]],
    products: list[tuple[float, str, float | None]],
    T_K: float,
) -> tuple[float, list[str]]:
    """Kirchhoff 則による温度補正 (kJ/mol) を計算する.

    ΔH(T) ≈ ΔH°(298.15 K) + ΔCp_const × (T − 298.15)
    """
    T0 = 298.15
    warnings: list[str] = []
    delta_cp = 0.0

    all_species = (
        [(coeff, smiles, +1) for coeff, smiles, _ in products]
        + [(coeff, smiles, -1) for coeff, smiles, _ in reactants]
    )

    missing: list[str] = []
    for coeff, smiles, sign in all_species:
        if not smiles.strip():
            continue
        cp, _ = get_cp_estimate(smiles)
        if cp is None:
            _, canon, formula = validate_smiles(smiles)
            missing.append(formula or smiles)
        else:
            delta_cp += sign * coeff * cp

    if missing:
        warnings.append(
            f"Cp 推算不可の化合物 ({', '.join(missing)}) は ΔCp=0 と仮定しています。"
            "温度補正の精度が低下する可能性があります。"
        )

    correction = delta_cp * (T_K - T0) / 1000.0  # J → kJ
    return correction, warnings


# ---------------------------------------------------------------------------
# 反応熱計算 (メイン関数)
# ---------------------------------------------------------------------------

def calc_reaction_heat(
    reactants: list[tuple[float, str, float | None]],
    products: list[tuple[float, str, float | None]],
    temperature_K: float = 298.15,
    solvent_correction_kJ: float = 0.0,
) -> ReactionHeatResult:
    """反応熱を計算する.

    Args:
        reactants:              [(係数, SMILES, manual_hf or None), ...]
        products:               [(係数, SMILES, manual_hf or None), ...]
        temperature_K:          計算温度 (K)。デフォルト 298.15 K。
        solvent_correction_kJ:  溶媒補正値 (kJ/mol)。ユーザー手動入力。

    Returns:
        ReactionHeatResult
    """
    warnings_all: list[str] = []
    reactant_results: list[tuple[float, CompoundResult]] = []
    product_results: list[tuple[float, CompoundResult]] = []
    all_ok = True

    for coeff, smiles, manual_hf in reactants:
        cr = get_hf(smiles, manual_hf)
        reactant_results.append((coeff, cr))
        if cr.hf_kJ_mol is None:
            all_ok = False
            warnings_all.append(
                f"反応物 {cr.formula or smiles}: ΔHf° を取得できませんでした。手動入力してください。"
            )

    for coeff, smiles, manual_hf in products:
        cr = get_hf(smiles, manual_hf)
        product_results.append((coeff, cr))
        if cr.hf_kJ_mol is None:
            all_ok = False
            warnings_all.append(
                f"生成物 {cr.formula or smiles}: ΔHf° を取得できませんでした。手動入力してください。"
            )

    if not all_ok:
        return ReactionHeatResult(
            delta_H_kJ_mol=None,
            delta_H_gas_kJ_mol=None,
            kirchhoff_correction_kJ=0.0,
            solvent_correction_kJ=solvent_correction_kJ,
            temperature_K=temperature_K,
            reactant_results=reactant_results,
            product_results=product_results,
            warnings=warnings_all,
            success=False,
        )

    delta_H_gas = (
        sum(c * r.hf_kJ_mol for c, r in product_results)
        - sum(c * r.hf_kJ_mol for c, r in reactant_results)
    )

    # Kirchhoff 温度補正
    kirchhoff = 0.0
    if abs(temperature_K - 298.15) > 1.0:
        kirchhoff, k_warns = calc_kirchhoff_correction(reactants, products, temperature_K)
        warnings_all.extend(k_warns)

    delta_H_final = delta_H_gas + kirchhoff + solvent_correction_kJ

    return ReactionHeatResult(
        delta_H_kJ_mol=delta_H_final,
        delta_H_gas_kJ_mol=delta_H_gas,
        kirchhoff_correction_kJ=kirchhoff,
        solvent_correction_kJ=solvent_correction_kJ,
        temperature_K=temperature_K,
        reactant_results=reactant_results,
        product_results=product_results,
        warnings=warnings_all,
        success=True,
    )
