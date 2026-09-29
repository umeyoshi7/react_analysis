"""計算エンジン: xTB (tblite + ASE) による構造最適化・振動、PySCF による DFT 一点計算.

重い依存 (tblite, ase, pyscf) はここでだけ import する。
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np

from .units import EV_TO_KJ, HARTREE_TO_KJ, R_KJ

IMAG_TOL_CM = 50.0   # これ未満の虚振動は実振動として扱う


@dataclass
class XtbResult:
    symbols: list[str]
    coords: np.ndarray        # 最適化後の座標 [Å]
    energy_eV: float          # GFN2-xTB エネルギー
    converged: bool


@dataclass
class ThermoCorr:
    h_corr_kJ: float          # H(T) − E_el = ZPE + 熱エネルギー + RT (kJ/mol)
    zpe_kJ: float
    n_imag: int               # 有意な虚振動の数
    lowest_freq_cm: float     # 最も低い振動数


def set_threads(n: int | None) -> None:
    n = n or os.cpu_count() or 1
    os.environ["OMP_NUM_THREADS"] = str(n)
    try:
        from pyscf import lib
        lib.num_threads(n)
    except ImportError:
        pass


def _atoms(symbols, coords, charge, multiplicity, method):
    from ase import Atoms
    from tblite.ase import TBLite

    atoms = Atoms(symbols=symbols, positions=coords)
    atoms.calc = TBLite(method=method, charge=charge, multiplicity=multiplicity,
                        verbosity=0, cache_api=True)
    return atoms


def is_linear(coords: np.ndarray, tol: float = 1e-3) -> bool:
    if len(coords) <= 2:
        return len(coords) == 2
    v = coords - coords[0]
    return int(np.linalg.matrix_rank(v, tol=tol)) <= 1


def xtb_optimize(symbols, coords, charge: int, multiplicity: int,
                 method: str = "GFN2-xTB", fmax: float = 0.005, steps: int = 3000) -> XtbResult:
    from ase.optimize import LBFGS

    atoms = _atoms(symbols, coords, charge, multiplicity, method)
    if len(symbols) == 1:
        return XtbResult(list(symbols), atoms.get_positions(), float(atoms.get_potential_energy()), True)
    converged = bool(LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps))
    return XtbResult(list(symbols), atoms.get_positions(), float(atoms.get_potential_energy()), converged)


def xtb_thermo(symbols, coords, charge: int, multiplicity: int, method: str = "GFN2-xTB",
               temperature_K: float = 298.15, delta: float = 0.01) -> ThermoCorr:
    """xTB のヘッセ行列から調和振動 (RRHO) 近似で H(T) − E_el を求める (対称数は H に影響しない)."""
    from ase.vibrations import Vibrations

    n_atoms = len(symbols)
    linear = is_linear(coords)
    n_vib = 3 * n_atoms - (5 if linear else 6)
    kT = R_KJ * temperature_K

    if n_atoms == 1:
        return ThermoCorr(1.5 * kT + kT, 0.0, 0, 0.0)

    atoms = _atoms(symbols, coords, charge, multiplicity, method)
    with tempfile.TemporaryDirectory() as tmp:
        vib = Vibrations(atoms, name=os.path.join(tmp, "vib"), delta=delta)
        vib.run()
        energies = np.asarray(vib.get_energies())   # eV。虚振動は虚数
        freqs = np.asarray(vib.get_frequencies())   # cm^-1
        vib.clean()

    # 大きさの小さい 6 (直線分子は 5) 個は並進・回転由来なので、残りを振動とみなす
    order = np.argsort(np.abs(energies))
    idx = order[-n_vib:]
    e_modes = energies[idx]
    f_modes = freqs[idx]

    # 虚振動のうち小さいもの (|ν| < 50 cm^-1) はメチル基や柔らかいねじれ由来の数値ノイズなので、
    # 実振動として扱う (熱エネルギーへの寄与は約 RT)。それ以上のものだけを「虚振動」と数える。
    is_imag = np.imag(e_modes) != 0
    mag_cm = np.abs(f_modes)
    significant = is_imag & (mag_cm >= IMAG_TOL_CM)
    n_imag = int(np.sum(significant))
    eps_eV = np.abs(e_modes)[~significant]
    eps_eV = eps_eV[eps_eV > 0]
    lowest = float(mag_cm[~significant].min()) if np.any(~significant) else 0.0

    # H(T) − E_el = ZPE + Σ ε/(exp(ε/kT)−1) + (3/2)kT [並進] + 回転 + kT [pV]
    eps = np.maximum(eps_eV * EV_TO_KJ, 1e-6)    # kJ/mol
    zpe = 0.5 * float(np.sum(eps))
    e_vib = float(np.sum(eps / np.expm1(eps / kT)))
    rot = kT if linear else 1.5 * kT
    return ThermoCorr(zpe + e_vib + 1.5 * kT + rot + kT, zpe, n_imag, lowest)


@dataclass
class DftResult:
    energy_hartree: float
    converged: bool
    seconds: float
    nbf: int


def dft_single_point(symbols, coords, charge: int, multiplicity: int,
                     xc: str = "b3lyp", basis: str = "def2-svp", disp: str | None = "d3bj",
                     density_fit: bool = True, max_memory_mb: int | None = None,
                     verbose: int = 3) -> DftResult:
    from pyscf import dft, gto

    t0 = time.time()
    mol = gto.Mole()
    mol.atom = [(s, tuple(map(float, c))) for s, c in zip(symbols, coords)]
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.verbose = verbose
    if max_memory_mb:
        mol.max_memory = max_memory_mb
    mol.build()

    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf.xc = xc
    if disp:
        mf.disp = disp
    if density_fit:
        mf = mf.density_fit()
    mf.conv_tol = 1e-8
    mf.max_cycle = 150
    e = mf.kernel()
    if not mf.converged:
        # 収束しなかった場合はレベルシフト、それでもだめなら 2 次収束法で再試行する
        mf.level_shift = 0.3
        e = mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            e = mf.kernel()
    return DftResult(float(e), bool(mf.converged), time.time() - t0, int(mol.nao))


def hartree_to_kJ(e: float) -> float:
    return e * HARTREE_TO_KJ
