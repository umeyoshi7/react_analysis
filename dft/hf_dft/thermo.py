"""原子当量 (AE) + 結合補正 (BAC) による ΔHf° の校正.

    ΔHf°(298) = E_DFT + H_corr + Σ n_e·c_e + Σ m_b·d_b

E_DFT は電子エネルギー、H_corr は xTB の RRHO による H(298) − E_el、n_e は元素 e の原子数、
m_b は結合タイプ b (例 C=O, C-H) の本数。c_e, d_b は ΔHf° が既知の参照分子への最小二乗
(結合項だけリッジ正則化) でフィットする。原子化エネルギー法そのものより、DFT の
元素ごと・結合ごとの系統誤差を吸収できる。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem


def element_counts(smiles: str) -> Counter:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return Counter(a.GetSymbol() for a in mol.GetAtoms())


def bond_counts(smiles: str) -> Counter:
    """結合タイプ (元素対 + 結合次数) の個数. 例: 'C-H', 'C=O', 'C:C' (芳香族)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    sym = {Chem.BondType.SINGLE: "-", Chem.BondType.DOUBLE: "=", Chem.BondType.TRIPLE: "#",
           Chem.BondType.AROMATIC: ":"}
    out: Counter = Counter()
    for b in mol.GetBonds():
        e1, e2 = sorted([b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()])
        out[f"{e1}{sym.get(b.GetBondType(), '?')}{e2}"] += 1
    return out


def features(smiles: str, use_bonds: bool) -> Counter:
    f = Counter({f"el:{k}": v for k, v in element_counts(smiles).items()})
    if use_bonds:
        f.update({f"bd:{k}": v for k, v in bond_counts(smiles).items()})
    return f


@dataclass
class Calibration:
    level: dict
    use_bonds: bool
    coefficients: dict[str, float]           # "el:C" / "bd:C=O" → 係数 (kJ/mol)
    rmse_fit: float
    rmse_loo: float                          # leave-one-out の RMSE
    n: int
    mean_heavy: float
    max_heavy: int
    ridge: float = 0.0
    entries: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    def apply(self, e_plus_h_kJ: float, smiles: str) -> tuple[float, list[str]]:
        """(ΔHf°, 校正セットにない特徴のリスト). 元素が未知なら例外."""
        feats = features(smiles, self.use_bonds)
        missing_el = [k[3:] for k in feats if k.startswith("el:") and k not in self.coefficients]
        if missing_el:
            raise ValueError(f"校正セットに含まれない元素: {', '.join(missing_el)}")
        missing = [k[3:] for k in feats if k.startswith("bd:") and k not in self.coefficients]
        total = e_plus_h_kJ + sum(n * self.coefficients.get(k, 0.0) for k, n in feats.items())
        return total, missing

    def uncertainty(self, n_heavy: int, n_missing_bonds: int = 0) -> float:
        """不確かさ (kJ/mol) の目安: LOO RMSE を、参照分子より大きい分だけ √ でスケール.

        校正にない結合タイプがあれば、その分の加算誤差 (1 タイプあたり LOO RMSE) を二乗和で足す。
        """
        scale = math.sqrt(max(n_heavy / max(self.mean_heavy, 1.0), 1.0))
        return math.hypot(self.rmse_loo * scale, self.rmse_loo * n_missing_bonds)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def from_dict(d: dict) -> "Calibration":
        return Calibration(**d)


def _solve(A: np.ndarray, y: np.ndarray, pen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A'A + diag(pen)) θ = A'y を解き、(θ, ハット行列の対角) を返す."""
    M = A.T @ A + np.diag(pen)
    Minv = np.linalg.pinv(M)
    theta = Minv @ (A.T @ y)
    h = np.einsum("ij,jk,ik->i", A, Minv, A)
    return theta, np.clip(h, 0, 0.999)


def fit_calibration(entries: list[dict], level: dict, use_bonds: bool = True, ridge: float = 3.0,
                    min_count: int = 3, reject_loo_kJ: float = 30.0) -> Calibration:
    """entries: [{smiles, hf_ref_kJ, e_plus_h_kJ, n_heavy, name?}, ...] から係数をフィットする.

    - 出現が min_count 分子未満の特徴 (希少な元素・結合) は係数を持たせない。希少な元素を含む分子は除く。
    - 1 回目のフィットで |LOO 残差| > reject_loo_kJ の分子は、参照値の誤りか DFT の適用外とみなして
      除き、フィットし直す (除いた分子は rejected に記録する)。
    """
    if len(entries) < 8:
        raise ValueError("校正には 8 分子以上の計算結果が必要です")

    def build(ents):
        feats = [features(x["smiles"], use_bonds) for x in ents]
        n_with = Counter(k for f in feats for k in f)
        el_keys = sorted(k for k, n in n_with.items() if k.startswith("el:") and n >= min_count)
        rare_el = {k for k, n in n_with.items() if k.startswith("el:") and n < min_count}
        keep = [i for i, f in enumerate(feats) if not (set(f) & rare_el)]
        ents2 = [ents[i] for i in keep]
        feats2 = [feats[i] for i in keep]
        n_with2 = Counter(k for f in feats2 for k in f)
        bd_keys = sorted(k for k, n in n_with2.items() if k.startswith("bd:") and n >= min_count)
        keys = el_keys + bd_keys
        A = np.array([[f.get(k, 0) for k in keys] for f in feats2], float)
        y = np.array([x["hf_ref_kJ"] - x["e_plus_h_kJ"] for x in ents2], float)
        pen = np.array([0.0 if k.startswith("el:") else ridge for k in keys])
        return ents2, keys, A, y, pen

    ents, keys, A, y, pen = build(entries)
    rejected: list[dict] = []
    theta, h = _solve(A, y, pen)
    loo = (y - A @ theta) / (1 - h)
    bad = np.abs(loo) > reject_loo_kJ
    if bad.any() and (~bad).sum() >= 8:
        rejected = [{"smiles": x["smiles"], "name": x.get("name", ""), "resid_loo_kJ": float(l)}
                    for x, l, b in zip(ents, loo, bad) if b]
        ents = [x for x, b in zip(ents, bad) if not b]
        ents, keys, A, y, pen = build(ents)
        theta, h = _solve(A, y, pen)

    resid = y - A @ theta
    loo = resid / (1 - h)
    out = [
        {**x, "hf_calc_kJ": x["hf_ref_kJ"] - float(r), "resid_kJ": float(r), "resid_loo_kJ": float(l)}
        for x, r, l in zip(ents, resid, loo)
    ]
    return Calibration(
        level=level, use_bonds=use_bonds,
        coefficients={k: float(t) for k, t in zip(keys, theta)},
        rmse_fit=float(np.sqrt(np.mean(resid ** 2))),
        rmse_loo=float(np.sqrt(np.mean(loo ** 2))),
        n=len(ents),
        mean_heavy=float(np.mean([x["n_heavy"] for x in ents])),
        max_heavy=int(max(x["n_heavy"] for x in ents)),
        ridge=ridge, entries=out, rejected=rejected,
    )
