"""反応熱推算モジュール (文献値 → Gani法 → Joback法 のフォールバック)"""
from __future__ import annotations

import base64
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, inchi, rdMolDescriptors
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
    "CO":             (-200.66,  "CH₃OH(g)"),
    "CCO":            (-234.8,   "C₂H₅OH(g)"),
    "c1ccccc1":       (82.9,     "ベンゼン(g)"),
    "CC(=O)O":        (-432.2,   "CH₃COOH(g)"),
    "CC(C)=O":        (-217.1,   "アセトン(g)"),
    "Cc1ccccc1":      (50.4,     "トルエン(g)"),
    "C1CC1":          (53.3,     "シクロプロパン(g)"),
    "ClC(Cl)Cl":      (-102.7,   "CHCl₃(g)"),
    "O=C1C=CC(=O)O1": (-398.3,   "無水マレイン酸(g)"),
}

# ---------------------------------------------------------------------------
# CSV 文献値 DB (data/hf_gas.csv, InChIKey 照合)
# ---------------------------------------------------------------------------
_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "hf_gas.csv"


def _load_hf_db() -> tuple[dict[str, tuple[float, str, float, float | None]], dict[str, list[str]]]:
    """(InChIKey → (ΔHf, 名称, 不確かさ, ΔHvap or None), 接続層(先頭14文字) → InChIKey 一覧) を返す."""
    full: dict[str, tuple[float, str, float, float | None]] = {}
    by_skeleton: dict[str, list[str]] = {}
    try:
        with _DB_PATH.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row["inchikey"].strip()
                full[key] = (
                    float(row["hf_kJ_mol"]),
                    row["name"].strip(),
                    float(row.get("uncertainty_kJ") or 1.0),
                    float(row["hvap_kJ_mol"]) if row.get("hvap_kJ_mol") else None,
                )
                by_skeleton.setdefault(key[:14], []).append(key)
    except (OSError, ValueError, KeyError):
        pass  # DB が無くても Gani/Joback で動作する
    return full, by_skeleton


_HF_DB, _HF_DB_SKELETON = _load_hf_db()


def _lookup_hf_db(canon: str) -> tuple[float, str, float, float | None, bool] | None:
    """InChIKey で DB を引く. (ΔHf, 名称, 不確かさ, ΔHvap, 立体異性体まで一致か)."""
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return None
    try:
        key = inchi.MolToInchiKey(mol)
    except Exception:
        return None
    if not key:
        return None
    if key in _HF_DB:
        hf, name, unc, hvap = _HF_DB[key]
        return hf, name, unc, hvap, True
    # 立体情報だけが違う場合は、接続層が一意に一致するときのみ採用
    cands = _HF_DB_SKELETON.get(key[:14], [])
    if len(cands) == 1:
        hf, name, unc, hvap = _HF_DB[cands[0]]
        return hf, name, unc, hvap, False
    return None


# 推算法ごとの不確かさ目安 (kJ/mol, 1σ 相当の概算)
METHOD_UNCERTAINTY: dict[str, float] = {
    "手動入力": 0.0,
    "文献値": 1.0,
    "Gani法": 10.0,
    "Joback法": 25.0,
}
_SMALL_RING_MAX = 4  # 環ひずみ補正のない基団寄与法が苦手な環サイズ

# 定圧熱容量 Cp (J/mol/K, 理想気体 ~298 K, 定数近似)
# キーは任意の SMILES でよい (_build_cp_table が RDKit 正規化 SMILES に変換する)
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
    "CC(=O)O":    63.4,    # CH3COOH(g, 単量体)
    "c1ccccc1":   82.44,   # benzene(g)
    "Cc1ccccc1":  103.6,   # toluene(g)
    "CCCl":       62.6,    # chloroethane(g)
    "BrBr":       36.0,    # Br2(g)
    "ClCl":       33.91,   # Cl2(g)
    "OO":         43.1,    # H2O2(g)
    "BrCCBr":     85.3,    # 1,2-dibromoethane(g)
    "CCOC(C)=O":  113.7,   # ethyl acetate(g)
    "OC(=O)/C=C\\C(=O)O": 120.0,  # maleic acid (approx)
}


# 上の文献 Cp 表 (25 化合物) への線形回帰 Cp ≈ a + b × 原子数(H 含む).
# 独立データ (Poling 344 化合物, 298 K) との比較: 平均誤差 11%, 上位 10% は約 30%.
_CP_ATOM_INTERCEPT = 19.6
_CP_PER_ATOM = 6.27
CP_ROUGH_LABEL = "原子数による概算 (概ね±10〜30%)"


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
class ManualHf:
    """手動入力した ΔHf° (kJ/mol) と、その不確かさ・出典メモ."""
    value: float
    uncertainty_kJ: float = 0.0
    note: str = ""


@dataclass
class CompoundResult:
    smiles_input: str
    canonical_smiles: str
    formula: str
    hf_kJ_mol: float | None
    method: str
    known_name: str = ""
    error: str = ""
    uncertainty_kJ: float | None = None
    warnings: list[str] = field(default_factory=list)
    phase: str = "g"                    # "g": 気体, "l": 液体
    hvap_kJ_mol: float | None = None    # 液相補正に使った ΔHvap (298.15 K)


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
    uncertainty_kJ: float | None = None  # ΔH_rxn の不確かさ (二乗和平方根)


# ---------------------------------------------------------------------------
# ΔHf° 取得
# ---------------------------------------------------------------------------

def _get_hf_gas(smiles: str, manual_hf: float | ManualHf | None = None) -> CompoundResult:
    """SMILES から気相 ΔHf° (kJ/mol) を取得する.

    優先順位:
        1. manual_hf が指定されている場合
        2. 文献値ルックアップ (組込み表 → data/hf_gas.csv を InChIKey で照合)
        3. Gani 法 (ugropy, 環・近接効果を扱える)
        4. Joback 基団寄与法 (ugropy)
        5. 失敗
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
        if not isinstance(manual_hf, ManualHf):
            manual_hf = ManualHf(float(manual_hf), METHOD_UNCERTAINTY["手動入力"])
        return CompoundResult(
            smiles_input=smiles,
            canonical_smiles=canon,
            formula=formula,
            hf_kJ_mol=manual_hf.value,
            method="手動入力",
            known_name=manual_hf.note,
            uncertainty_kJ=manual_hf.uncertainty_kJ,
        )

    if manual_hf is None and canon in _KNOWN_HF:
        hf, name = _KNOWN_HF[canon]
        return CompoundResult(
            smiles_input=smiles,
            canonical_smiles=canon,
            formula=formula,
            hf_kJ_mol=hf,
            method="文献値",
            known_name=name,
            uncertainty_kJ=METHOD_UNCERTAINTY["文献値"],
        )

    if manual_hf is None:
        hit = _lookup_hf_db(canon)
        if hit is not None:
            hf, name, unc, _, exact = hit
            return CompoundResult(
                smiles_input=smiles,
                canonical_smiles=canon,
                formula=formula,
                hf_kJ_mol=hf,
                method="文献値",
                known_name=f"{name}(g)",
                uncertainty_kJ=unc,
                warnings=[] if exact else [
                    f"{formula}: 立体異性体が DB と異なるため {name} の値を流用しています。"
                ],
            )

    non_free = _non_free_form_reason(canon)
    if non_free:
        return CompoundResult(
            smiles_input=smiles,
            canonical_smiles=canon,
            formula=formula,
            hf_kJ_mol=None,
            method="推算失敗",
            error=(
                f"{non_free}は基団寄与法の対象外です。反応熱の計算にはフリー体 (中性の単一分子) の "
                "SMILES を使ってください。塩・溶媒和物そのものの値は、DFT (dft/ ディレクトリ) などで求めた値を "
                "手動入力してください。"
            ),
        )

    ring_warns = _small_ring_warnings(canon, formula)

    for method, estimator in (("Gani法", _estimate_gani), ("Joback法", _estimate_joback)):
        try:
            hf = estimator(smiles)
        except Exception:
            hf = None
        if hf is not None:
            return CompoundResult(
                smiles_input=smiles,
                canonical_smiles=canon,
                formula=formula,
                hf_kJ_mol=hf,
                method=method,
                uncertainty_kJ=METHOD_UNCERTAINTY[method],
                warnings=ring_warns,
            )

    return CompoundResult(
        smiles_input=smiles,
        canonical_smiles=canon,
        formula=formula,
        hf_kJ_mol=None,
        method="推算失敗",
        error=(
            "Gani 法・Joback 法ともにグループ分解に失敗しました。"
            "手動で ΔHf° を入力してください。"
        ),
    )


def _non_free_form_reason(canon: str) -> str:
    """塩・溶媒和物・水和物・イオンなら理由を返す. 中性の単一分子なら空文字."""
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return ""
    if len(Chem.GetMolFrags(mol)) > 1:
        return "複数成分 (塩・溶媒和物・水和物など)"
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return "電荷を持つイオン"
    return ""


def _estimate_gani(smiles: str) -> float | None:
    from ugropy import Groups
    q = Groups(smiles, "smiles").agani.ig_formation_enthalpy
    return float(q.magnitude) if q is not None else None


def _estimate_joback(smiles: str) -> float | None:
    from ugropy import joback
    q = joback.get_groups(smiles, identifier_type="smiles").ig_enthalpy_formation
    return float(q.magnitude) if q is not None else None


def _small_ring_warnings(canon: str, formula: str) -> list[str]:
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return []
    sizes = {len(r) for r in mol.GetRingInfo().AtomRings()}
    small = sorted(n for n in sizes if n <= _SMALL_RING_MAX)
    if not small:
        return []
    return [
        f"{formula}: {'・'.join(str(n) for n in small)}員環を含みます。基団寄与法は環ひずみを"
        "十分に考慮できず、数十 kJ/mol ずれる場合があります。文献値の手動入力を推奨します。"
    ]


# 298.15 K での蒸発エンタルピー (kJ/mol). 組込み文献値表にある化合物用
_KNOWN_HVAP: dict[str, float] = {
    "O": 44.0,
    "CO": 37.4,
    "CCO": 42.3,
    "c1ccccc1": 33.9,
    "Cc1ccccc1": 38.0,
    "CC(C)=O": 31.0,
    "ClC(Cl)Cl": 31.4,
    "BrBr": 30.91,
}


def get_hvap(smiles: str) -> tuple[float | None, float, str]:
    """298.15 K の蒸発エンタルピー (kJ/mol), 不確かさ, 由来を返す.

    優先順位: 組込み表 → data/hf_gas.csv → Joback 法の ΔHvap(Tb) を Watson 式で 298 K へ換算.
    """
    valid, canon, _ = validate_smiles(smiles)
    if not valid:
        return None, 0.0, ""
    if canon in _KNOWN_HVAP:
        return _KNOWN_HVAP[canon], 1.0, "文献値"
    hit = _lookup_hf_db(canon)
    if hit is not None and hit[3] is not None:
        return hit[3], 1.0, "文献値"
    try:
        from ugropy import joback
        r = joback.get_groups(smiles, identifier_type="smiles")
        hb, tb, tc = r.vaporization_enthalpy, r.normal_boiling_point, r.critical_temperature
        if hb is None or tb is None or tc is None:
            return None, 0.0, ""
        hb_v, tb_v, tc_v = float(hb.magnitude), float(tb.magnitude), float(tc.magnitude)
        t0 = 298.15
        if tc_v <= t0:
            return None, 0.0, ""
        # Watson 式: ΔHvap(T) = ΔHvap(Tb) × ((1 − T/Tc) / (1 − Tb/Tc))^0.38
        hv = hb_v * ((1 - t0 / tc_v) / (1 - tb_v / tc_v)) ** 0.38
        return hv, max(4.0, 0.15 * hv), "Joback+Watson"
    except Exception:
        return None, 0.0, ""


def get_hf(
    smiles: str,
    manual_hf: float | None = None,
    phase: str = "g",
) -> CompoundResult:
    """ΔHf° (kJ/mol) を取得する. phase="l" のときは気相値から ΔHvap を引いて液相値にする.

    手動入力値は指定した相の値としてそのまま使う (相補正はしない).
    """
    cr = _get_hf_gas(smiles, manual_hf)
    if phase != "l" or manual_hf is not None or cr.hf_kJ_mol is None:
        cr.phase = "l" if phase == "l" else "g"
        return cr

    cr.phase = "l"
    hvap, hvap_unc, origin = get_hvap(smiles)
    if hvap is None:
        cr.hf_kJ_mol = None
        cr.method = "推算失敗"
        cr.error = "蒸発エンタルピーを取得できず、液相の ΔHf° を求められません。手動入力してください。"
        return cr
    cr.hf_kJ_mol -= hvap
    cr.hvap_kJ_mol = hvap
    cr.uncertainty_kJ = math.hypot(cr.uncertainty_kJ or 0.0, hvap_unc)
    if origin != "文献値":
        cr.warnings.append(
            f"{cr.formula}: ΔHvap を Joback 法 + Watson 式で推算しています (約 ±15%)。"
        )
    if cr.canonical_smiles and cr.method != "推算失敗":
        cr.warnings.extend(_gas_at_room_temp_warning(smiles, cr.formula))
    return cr


def _gas_at_room_temp_warning(smiles: str, formula: str) -> list[str]:
    try:
        from ugropy import joback
        tb = joback.get_groups(smiles, identifier_type="smiles").normal_boiling_point
    except Exception:
        return []
    if tb is not None and float(tb.magnitude) < 298.15:
        return [f"{formula}: 常圧の沸点が 298 K 未満のため、液体として存在するには加圧・冷却が必要です。"]
    return []


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

    if _non_free_form_reason(canon):
        return None, "推算不可"

    try:
        from ugropy import joback
        result = joback.get_groups(smiles, identifier_type="smiles")
        params = result.heat_capacity_ideal_gas_params  # a + bT + cT² + dT³
        if params is not None and len(params) == 4:
            a_, b_, c_, d_ = (float(x) for x in params)
            T = 298.15
            return a_ + b_ * T + c_ * T**2 + d_ * T**3, "Joback法 (298 K)"
    except Exception:
        pass

    # Joback 法に基団がない化合物 (硫黄系など): 原子数による概算
    mol = Chem.AddHs(Chem.MolFromSmiles(canon))
    return _CP_ATOM_INTERCEPT + _CP_PER_ATOM * mol.GetNumAtoms(), CP_ROUGH_LABEL


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
    rough: list[str] = []
    for coeff, smiles, sign in all_species:
        if not smiles.strip():
            continue
        cp, cp_method = get_cp_estimate(smiles)
        if cp is None:
            _, canon, formula = validate_smiles(smiles)
            missing.append(formula or smiles)
        else:
            delta_cp += sign * coeff * cp
            if cp_method == CP_ROUGH_LABEL:
                rough.append(validate_smiles(smiles)[2] or smiles)

    if rough:
        warnings.append(
            f"Cp を原子数から概算した化合物 ({', '.join(rough)}) があります。"
            "温度補正は目安として扱ってください。"
        )

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
    reactant_phases: list[str] | None = None,
    product_phases: list[str] | None = None,
) -> ReactionHeatResult:
    """反応熱を計算する.

    Args:
        reactants:              [(係数, SMILES, ManualHf | float | None), ...]
        products:               [(係数, SMILES, ManualHf | float | None), ...]
        temperature_K:          計算温度 (K)。デフォルト 298.15 K。
        solvent_correction_kJ:  溶媒補正値 (kJ/mol)。ユーザー手動入力。
        reactant_phases:        反応物ごとの相 ("g" 気体 / "l" 液体)。省略時は全て気体。
        product_phases:         生成物ごとの相。省略時は全て気体。

    Returns:
        ReactionHeatResult
    """
    warnings_all: list[str] = []
    reactant_results: list[tuple[float, CompoundResult]] = []
    product_results: list[tuple[float, CompoundResult]] = []
    all_ok = True

    r_phases = reactant_phases or ["g"] * len(reactants)
    p_phases = product_phases or ["g"] * len(products)

    for (coeff, smiles, manual_hf), ph in zip(reactants, r_phases):
        cr = get_hf(smiles, manual_hf, ph)
        reactant_results.append((coeff, cr))
        if cr.hf_kJ_mol is None:
            all_ok = False
            warnings_all.append(
                f"反応物 {cr.formula or smiles}: ΔHf° を取得できませんでした。" + (cr.error or "手動入力してください。")
            )

    for (coeff, smiles, manual_hf), ph in zip(products, p_phases):
        cr = get_hf(smiles, manual_hf, ph)
        product_results.append((coeff, cr))
        if cr.hf_kJ_mol is None:
            all_ok = False
            warnings_all.append(
                f"生成物 {cr.formula or smiles}: ΔHf° を取得できませんでした。" + (cr.error or "手動入力してください。")
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

    for _, r in reactant_results + product_results:
        warnings_all.extend(r.warnings)

    methods = {r.method for _, r in reactant_results + product_results} - {"手動入力"}
    estimated = methods - {"文献値"}
    if estimated and len(methods) > 1:
        warnings_all.append(
            "推算手法が混在しています (" + "・".join(sorted(methods)) + ")。"
            "手法ごとの系統誤差が打ち消されないため、ΔH_rxn の誤差が大きくなる可能性があります。"
        )

    unc = math.sqrt(sum(
        (c * (r.uncertainty_kJ or 0.0)) ** 2
        for c, r in reactant_results + product_results
    ))

    if any(r.phase == "l" for _, r in reactant_results + product_results):
        if abs(temperature_K - 298.15) > 1.0:
            warnings_all.append(
                "液相の化合物を含みますが、温度補正 (Kirchhoff) は気相 Cp で計算しています。"
                "液相 Cp は気相より大きいため、補正量は目安です。"
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
        uncertainty_kJ=unc,
    )


# ---------------------------------------------------------------------------
# プロセス安全評価（MTSR / TD24 / Stoessel 5段階リスク評価）
# ---------------------------------------------------------------------------

_STOESSEL_CLASS_INFO: dict[int, dict] = {
    1: {
        "label": "クラス 1 — 非常に低リスク",
        "color": "#2ca02c",
        "description": "MTSR < MTT。冷却失敗で到達する最高温度が沸点以下であり、二次分解の懸念なし。",
        "action": "通常の操作手順で安全に実施可能。",
    },
    2: {
        "label": "クラス 2 — 低リスク",
        "color": "#98df8a",
        "description": "MTT < MTSR かつ TD24 > MTSR。沸点は超えるが、24 h 自己加速分解温度には届かない。",
        "action": "沸点超過に対する対策（還流設備・ベント設計）を検討。",
    },
    3: {
        "label": "クラス 3 — 中リスク",
        "color": "#ff7f0e",
        "description": "MTT < TD24 < MTSR。MTSR に至る過程で TD24 を超え、二次分解が誘発される可能性がある。",
        "action": "冷却失敗シナリオの DIERS 解析と緊急冷却設備の検討が必要。",
    },
    4: {
        "label": "クラス 4 — 高リスク",
        "color": "#d62728",
        "description": "TD24 ≤ MTT < MTSR。沸点以下でも 24 h 以内に自己加速分解が起きる温度域。",
        "action": "プロセスの抜本的見直し。断熱条件での安全性試験（ARC/DSC）が必須。",
    },
    5: {
        "label": "クラス 5 — 非常に高リスク",
        "color": "#7b241c",
        "description": "TD24 ≤ Tp。通常のプロセス温度がすでに自己加速分解温度以上。即座に危険な状態。",
        "action": "直ちに操業停止を検討。プロセス全体の再設計が必要。",
    },
}


@dataclass
class ProcessSafetyResult:
    tp_C: float
    delta_tad_K: float
    mtsr_C: float
    td24_C: float | None
    td24_method: str
    mtt_C: float
    stoessel_class: int
    stoessel_label: str
    stoessel_color: str
    stoessel_description: str
    stoessel_action: str


def calc_td24_simple(tonset_C: float) -> float:
    """TD24 を DSC 開始温度から簡易推算する（経験則: Tonset − 100 °C）."""
    return tonset_C - 100.0


def _tmr_seconds(
    T_K: float,
    ea_J_mol: float,
    A_1_s: float,
    qd_J_kg: float,
    cp_J_kgK: float,
) -> float:
    """断熱誘導期間 TMR (s) を Arrhenius モデルで計算する.

    TMR_ad = Cp * R * T² / (Qd * A * exp(-Ea/RT) * Ea)
    """
    R = 8.314
    exponent = -ea_J_mol / (R * T_K)
    if exponent < -700:
        return float("inf")
    k = A_1_s * math.exp(exponent)
    if k <= 0 or qd_J_kg <= 0 or ea_J_mol <= 0:
        return float("inf")
    return (cp_J_kgK * R * T_K**2) / (qd_J_kg * k * ea_J_mol)


def calc_td24_arrhenius(
    ea_kJ_mol: float,
    A_1_s: float,
    qd_kJ_kg: float,
    cp_J_gK: float,
    t_search_range_C: tuple[float, float] = (-50.0, 500.0),
) -> float | None:
    """TD24 を Arrhenius 動力学パラメータから二分法で計算する（TMR = 24 h の温度）."""
    TARGET_S = 86400.0
    ea_J = ea_kJ_mol * 1000.0
    qd_J_kg = qd_kJ_kg * 1000.0
    cp_J_kgK = cp_J_gK * 1000.0

    T_lo = t_search_range_C[0] + 273.15
    T_hi = t_search_range_C[1] + 273.15

    try:
        tmr_lo = _tmr_seconds(T_lo, ea_J, A_1_s, qd_J_kg, cp_J_kgK)
        tmr_hi = _tmr_seconds(T_hi, ea_J, A_1_s, qd_J_kg, cp_J_kgK)
    except Exception:
        return None

    if tmr_hi > TARGET_S or tmr_lo < TARGET_S:
        return None

    for _ in range(80):
        T_mid = (T_lo + T_hi) / 2.0
        tmr_mid = _tmr_seconds(T_mid, ea_J, A_1_s, qd_J_kg, cp_J_kgK)
        if tmr_mid > TARGET_S:
            T_lo = T_mid
        else:
            T_hi = T_mid
        if T_hi - T_lo < 0.005:
            break

    return (T_lo + T_hi) / 2.0 - 273.15


def assess_process_safety(
    tp_C: float,
    delta_tad_K: float,
    mtt_C: float,
    td24_C: float | None,
    td24_method: str = "なし",
) -> ProcessSafetyResult:
    """MTSR と Stoessel 5段階リスク評価を実施する.

    温度の大小関係:
        Class 1: MTSR ≤ MTT
        Class 2: MTSR > MTT かつ TD24 > MTSR (または TD24 未入力)
        Class 3: MTSR > MTT かつ MTT < TD24 ≤ MTSR
        Class 4: MTSR > MTT かつ TD24 ≤ MTT
        Class 5: TD24 ≤ Tp (最優先判定)
    """
    mtsr_C = tp_C + delta_tad_K

    if td24_C is not None and td24_C <= tp_C:
        cls = 5
    elif mtsr_C <= mtt_C:
        cls = 1
    elif td24_C is None or td24_C > mtsr_C:
        cls = 2
    elif td24_C > mtt_C:
        cls = 3
    else:
        cls = 4

    info = _STOESSEL_CLASS_INFO[cls]
    return ProcessSafetyResult(
        tp_C=tp_C,
        delta_tad_K=delta_tad_K,
        mtsr_C=mtsr_C,
        td24_C=td24_C,
        td24_method=td24_method,
        mtt_C=mtt_C,
        stoessel_class=cls,
        stoessel_label=info["label"],
        stoessel_color=info["color"],
        stoessel_description=info["description"],
        stoessel_action=info["action"],
    )
