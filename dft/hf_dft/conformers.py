"""RDKit による配座生成と、複数成分 (塩・溶媒和物) の初期配置."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def _mmff_or_uff(mol: Chem.Mol, conf_ids: list[int]) -> list[tuple[int, float]]:
    """(conf_id, エネルギー) を返す. MMFF が使えなければ UFF."""
    if AllChem.MMFFHasAllMoleculeParams(mol):
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
    else:
        res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000, numThreads=0)
    return [(cid, e) for cid, (_, e) in zip(conf_ids, res)]


def single_molecule_confs(smiles: str, n_conformers: int, n_keep: int, seed: int = 7
                          ) -> tuple[list[str], list[np.ndarray]]:
    """(元素記号, 座標[Å]) の候補を最大 n_keep 個返す (MMFF/UFF エネルギー昇順)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    params.numThreads = 0
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not conf_ids:
        params.useRandomCoords = True
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not conf_ids:
        raise RuntimeError(f"3D 構造を生成できませんでした: {smiles}")
    try:
        ranked = sorted(_mmff_or_uff(mol, conf_ids), key=lambda x: x[1])
    except Exception:
        ranked = [(c, 0.0) for c in conf_ids]
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    return symbols, [mol.GetConformer(cid).GetPositions() for cid, _ in ranked[:n_keep]]


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d],
    ])


def packed_complex_confs(components: list[str], n_orient: int, seed: int = 7,
                         contact: float = 2.6) -> tuple[list[str], list[np.ndarray]]:
    """成分を 1 つずつ、既存の集合体の周りにランダムな向きで接触させて置く.

    xTB の構造最適化で最安定の配置へ緩和させる前提の初期構造。
    """
    rng = np.random.default_rng(seed)
    parts = [single_molecule_confs(s, 5, 1, seed) for s in components]
    symbols: list[str] = []
    for sym, _ in parts:
        symbols += sym

    out: list[np.ndarray] = []
    for _ in range(n_orient):
        placed = parts[0][1][0] - parts[0][1][0].mean(axis=0)
        for _, confs in parts[1:]:
            xyz = confs[0] - confs[0].mean(axis=0)
            for _try in range(200):
                xyz_r = xyz @ _rotation(rng).T
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                # 集合体の外側から近づけ、最短原子間距離が contact になる位置を探す
                shift = direction * (np.linalg.norm(placed, axis=1).max() + np.linalg.norm(xyz_r, axis=1).max() + 4.0)
                cand = xyz_r + shift
                for _step in range(80):
                    d = np.linalg.norm(placed[:, None, :] - cand[None, :, :], axis=2).min()
                    if d <= contact:
                        break
                    cand = cand - direction * 0.25
                d = np.linalg.norm(placed[:, None, :] - cand[None, :, :], axis=2).min()
                if d >= 1.6:
                    placed = np.vstack([placed, cand])
                    break
            else:
                raise RuntimeError("成分の初期配置に失敗しました")
        out.append(placed)
    return symbols, out
