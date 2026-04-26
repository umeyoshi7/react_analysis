"""反応熱推算モジュール (Joback基団寄与法 + 文献値フォールバック)"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# 標準生成エンタルピー文献値 ΔHf° (kJ/mol, 298.15 K, 理想気体)
# キー: RDKit 正規化 SMILES
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
    ("C₂H₄",     "C=C",           "エチレン"),
    ("HCHO",     "C=O",           "ホルムアルデヒド"),
    ("CH₃OH",    "CO",            "メタノール"),
    ("C₂H₅OH",   "CCO",           "エタノール"),
    ("CH₃COCH₃", "CC(=O)C",       "アセトン"),
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


# ---------------------------------------------------------------------------
# 計算結果データクラス
# ---------------------------------------------------------------------------

@dataclass
class CompoundResult:
    smiles_input: str
    canonical_smiles: str
    formula: str
    hf_kJ_mol: float | None
    method: str          # '文献値', 'Joback法', '手動入力', '推算失敗'
    known_name: str = ""
    error: str = ""


@dataclass
class ReactionHeatResult:
    delta_H_kJ_mol: float | None
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
        1. manual_hf が指定されている場合 → そのまま使用
        2. 文献値ルックアップ (_KNOWN_HF)
        3. Joback 基団寄与法 (ugropy)
        4. 失敗 → hf_kJ_mol = None
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

    # 1. 手動入力
    if manual_hf is not None:
        return CompoundResult(
            smiles_input=smiles,
            canonical_smiles=canon,
            formula=formula,
            hf_kJ_mol=manual_hf,
            method="手動入力",
        )

    # 2. 文献値
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

    # 3. Joback 法
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
            "無機物・シンプルな小分子は手動で ΔHf° を入力してください。"
        ),
    )


# ---------------------------------------------------------------------------
# 反応熱計算
# ---------------------------------------------------------------------------

def calc_reaction_heat(
    reactants: list[tuple[float, str, float | None]],
    products:  list[tuple[float, str, float | None]],
) -> ReactionHeatResult:
    """反応熱を計算する.

    Args:
        reactants: [(係数, SMILES, manual_hf or None), ...]
        products:  [(係数, SMILES, manual_hf or None), ...]

    Returns:
        ReactionHeatResult
            ΔH_rxn = Σ(ν_prod × ΔHf°_prod) − Σ(ν_react × ΔHf°_react)
    """
    warnings: list[str] = []
    reactant_results: list[tuple[float, CompoundResult]] = []
    product_results:  list[tuple[float, CompoundResult]] = []
    all_ok = True

    for coeff, smiles, manual_hf in reactants:
        cr = get_hf(smiles, manual_hf)
        reactant_results.append((coeff, cr))
        if cr.hf_kJ_mol is None:
            all_ok = False
            warnings.append(
                f"反応物 {cr.formula or smiles}: ΔHf° を取得できませんでした。"
                "手動入力欄に値を入力してください。"
            )

    for coeff, smiles, manual_hf in products:
        cr = get_hf(smiles, manual_hf)
        product_results.append((coeff, cr))
        if cr.hf_kJ_mol is None:
            all_ok = False
            warnings.append(
                f"生成物 {cr.formula or smiles}: ΔHf° を取得できませんでした。"
                "手動入力欄に値を入力してください。"
            )

    if not all_ok:
        return ReactionHeatResult(
            delta_H_kJ_mol=None,
            reactant_results=reactant_results,
            product_results=product_results,
            warnings=warnings,
            success=False,
        )

    delta_H = sum(c * r.hf_kJ_mol for c, r in product_results) - \
              sum(c * r.hf_kJ_mol for c, r in reactant_results)

    return ReactionHeatResult(
        delta_H_kJ_mol=delta_H,
        reactant_results=reactant_results,
        product_results=product_results,
        warnings=warnings,
        success=True,
    )
