"""計算時間・メモリ・費用の概算 (実行前の見積もり用).

ここの係数は、WSL (8 コア) の小分子で測った所要時間から外挿した目安で、
大きい分子・別マシンでの実測ではない。桁 (時間か日か) の判断に使い、
厳密な予算計算には使わないこと。
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from rdkit import Chem

# 基底関数の数 (球面調和関数): 元素 → 個数
_NBF = {
    "def2-svp": {"H": 5, "C": 14, "N": 14, "O": 14, "F": 14, "S": 18, "Cl": 18, "Br": 32, "P": 18},
    "def2-tzvp": {"H": 6, "C": 31, "N": 31, "O": 31, "F": 31, "S": 37, "Cl": 37, "Br": 32, "P": 37},
    "6-31g*": {"H": 2, "C": 15, "N": 15, "O": 15, "F": 15, "S": 19, "Cl": 19, "Br": 33, "P": 19},
}

# マシンタイプ: (vCPU, メモリ GB, オンデマンド $/h, Spot $/h)
# n2-highmem-32 は米国リージョンの GCE 公開価格 (Vantage の掲載値: $2.0962 / Spot $0.6781)。
# 他のタイプは vCPU 数・メモリ量の比で換算した値で、Spot は n2-highmem-32 と同じ割引率 (約 68%) を仮定した。
# Vertex AI カスタムジョブは管理料が上乗せされ、リージョンによっても変わるため、必ず最新の料金表で確認する。
_SPOT_RATIO = 0.6781 / 2.0962
MACHINES: dict[str, tuple[int, int, float, float]] = {
    "n2-highmem-32": (32, 256, 2.0962, 0.6781),
    "n2-standard-32": (32, 128, 1.5539, 1.5539 * _SPOT_RATIO),
    "n2-highmem-16": (16, 128, 1.0481, 1.0481 * _SPOT_RATIO),
    "n2-standard-16": (16, 64, 0.7769, 0.7769 * _SPOT_RATIO),
}

# --- 時間モデルの基準値 (基準マシン: 32 vCPU) ---
_T_DFT_REF_MIN = 12.0       # def2-SVP, 1330 基底関数での DFT 一点計算 [分]
_NBF_REF = 1330
_T_XTB_GRAD_REF_S = 0.4     # 131 原子での xTB 勾配 1 回 [秒]
_N_ATOM_REF = 131
_XTB_STEPS = 250            # 1 配座あたりの最適化ステップ数 (目安)


@dataclass
class Estimate:
    formula: str
    n_atoms: int
    n_heavy: int
    n_electrons: int
    nbf: int
    df_tensor_gb: float
    t_xtb_min: float
    t_dft_min: float
    t_total_min: float
    cost_usd_ondemand: float
    cost_usd_spot: float
    notes: list[str]


def count_basis(smiles: str, basis: str) -> tuple[int, Counter]:
    table = _NBF.get(basis.lower())
    if table is None:
        raise ValueError(f"見積もり未対応の基底関数: {basis} (対応: {', '.join(_NBF)})")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
    missing = [e for e in counts if e not in table]
    if missing:
        raise ValueError(f"見積もり未対応の元素: {', '.join(missing)}")
    return sum(table[e] * n for e, n in counts.items()), counts


def estimate(smiles: str, basis: str = "def2-svp", machine: str = "n2-highmem-32",
             n_keep: int = 5, n_conformers: int = 30) -> Estimate:
    from rdkit.Chem import rdMolDescriptors

    vcpu, mem_gb, price, spot = MACHINES[machine]
    mol = Chem.MolFromSmiles(smiles)
    nbf, counts = count_basis(smiles, basis)
    n_atoms = sum(counts.values())
    n_el = sum(Chem.GetPeriodicTable().GetAtomicNumber(e) * n for e, n in counts.items())

    # xTB: 配座最適化 n_keep 個 + ヘッセ行列 (3N 座標 × 両方向)
    t_grad = _T_XTB_GRAD_REF_S * (n_atoms / _N_ATOM_REF) ** 2 * (32 / vcpu) ** 0.3
    n_grad = n_keep * _XTB_STEPS + 6 * n_atoms
    t_xtb = n_grad * t_grad / 60.0

    # DFT 一点計算: 基底関数の ~2.5 乗、vCPU 数に対して並列効率 ~0.8 乗
    t_dft = _T_DFT_REF_MIN * (nbf / _NBF_REF) ** 2.5 * (32 / vcpu) ** 0.8
    # 配座探索に RDKit の時間 (数分) を加算
    t_total = t_xtb + t_dft + 2.0

    naux = 3.0 * nbf
    df_gb = nbf * (nbf + 1) / 2 * naux * 8 / 1e9

    notes: list[str] = []
    if df_gb > 0.7 * mem_gb:
        notes.append(f"密度フィッティングの 3 中心積分だけで約 {df_gb:.0f} GB で、メモリ {mem_gb} GB を超える恐れがあります (ディスク退避で大幅に遅くなる)。")
    if n_atoms > 100:
        notes.append("100 原子超の柔軟分子は、配座ごとのエネルギー差が数十 kJ/mol あり、配座探索の不足が誤差の主因になります。")
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if n_rot > 10:
        notes.append(f"回転可能結合が {n_rot} 個あります。n_conformers={n_conformers} では配座空間を十分に探索できません。")

    return Estimate(
        formula=rdMolDescriptors.CalcMolFormula(mol), n_atoms=n_atoms,
        n_heavy=mol.GetNumHeavyAtoms(), n_electrons=n_el, nbf=nbf, df_tensor_gb=df_gb,
        t_xtb_min=t_xtb, t_dft_min=t_dft, t_total_min=t_total,
        cost_usd_ondemand=t_total / 60 * price, cost_usd_spot=t_total / 60 * spot, notes=notes,
    )


def format_estimate(e: Estimate, machine: str) -> str:
    lo, hi = e.t_total_min / 2, e.t_total_min * 3
    return (
        f"{e.formula}: {e.n_atoms} 原子 (重原子 {e.n_heavy}), 電子 {e.n_electrons}, 基底関数 {e.nbf}\n"
        f"  マシン {machine}\n"
        f"  xTB {e.t_xtb_min:.0f} 分 + DFT {e.t_dft_min:.0f} 分 = 約 {e.t_total_min:.0f} 分 "
        f"(幅: {lo:.0f}〜{hi:.0f} 分)\n"
        f"  費用: オンデマンド 約 ${e.cost_usd_ondemand:.2f} / Spot 約 ${e.cost_usd_spot:.2f} (Vertex 管理料・リージョン差は含まず)\n"
        f"  3 中心積分: 約 {e.df_tensor_gb:.0f} GB\n"
        + "".join(f"  ! {n}\n" for n in e.notes)
    )


def round_up(x: float, step: float = 0.01) -> float:
    return math.ceil(x / step) * step
