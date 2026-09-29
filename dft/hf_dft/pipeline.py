"""1 化合物の計算: 配座 → xTB 最適化 → 熱補正 → DFT 一点計算 → ΔHf°."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import conformers
from .species import Species, analyze
from .thermo import Calibration
from .units import HARTREE_TO_KJ


@dataclass
class Level:
    xc: str = "b3lyp"
    basis: str = "def2-svp"
    disp: str | None = "d3bj"
    xtb_method: str = "GFN2-xTB"
    temperature_K: float = 298.15
    n_conformers: int = 30
    n_keep: int = 5
    n_orient: int = 8
    max_memory_mb: int | None = None

    def key(self) -> dict:
        return {"xc": self.xc, "basis": self.basis, "disp": self.disp,
                "xtb_method": self.xtb_method, "temperature_K": self.temperature_K}


@dataclass
class State:
    """1 つの分子 (または複合体) の計算結果."""
    smiles: str
    charge: int
    multiplicity: int
    n_heavy: int
    symbols: list[str]
    coords: np.ndarray
    e_dft_kJ: float
    h_corr_kJ: float
    n_imag: int
    lowest_freq_cm: float
    xtb_converged: bool
    dft_converged: bool
    seconds: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def e_plus_h_kJ(self) -> float:
        return self.e_dft_kJ + self.h_corr_kJ

    def xyz(self, comment: str = "") -> str:
        lines = [str(len(self.symbols)), comment or self.smiles]
        lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(self.symbols, self.coords)]
        return "\n".join(lines) + "\n"


def compute_state(sp: Species, level: Level, log=print, engines=None) -> State:
    """配座探索から DFT 一点計算までを 1 つの Species について実行する."""
    if engines is None:
        from . import engines as engines_mod
        engines = engines_mod
    t = {}
    warnings: list[str] = []

    t0 = time.time()
    if len(sp.components) == 1:
        symbols, cands = conformers.single_molecule_confs(sp.smiles, level.n_conformers, level.n_keep)
    else:
        symbols, cands = conformers.packed_complex_confs([c.smiles for c in sp.components], level.n_orient)
    t["conformers"] = time.time() - t0
    log(f"  初期配座 {len(cands)} 個 ({t['conformers']:.0f} 秒)")

    t0 = time.time()
    best = None
    for i, xyz in enumerate(cands):
        r = engines.xtb_optimize(symbols, xyz, sp.charge, sp.multiplicity, level.xtb_method)
        log(f"    xTB 最適化 {i + 1}/{len(cands)}: E = {r.energy_eV:.4f} eV{'' if r.converged else ' (未収束)'}")
        # 収束したものを優先し、その中でエネルギー最小を選ぶ
        rank = (0 if r.converged else 1, r.energy_eV)
        if best is None or rank < best[0]:
            best = (rank, r)
    xtb = best[1]
    if not xtb.converged:
        warnings.append("xTB の構造最適化が収束しませんでした。")
    t["xtb_opt"] = time.time() - t0

    t0 = time.time()
    th = engines.xtb_thermo(xtb.symbols, xtb.coords, sp.charge, sp.multiplicity, level.xtb_method, level.temperature_K)
    t["xtb_thermo"] = time.time() - t0
    log(f"  熱補正 H_corr = {th.h_corr_kJ:.1f} kJ/mol, ZPE = {th.zpe_kJ:.1f}, 虚振動 {th.n_imag} ({t['xtb_thermo']:.0f} 秒)")
    if th.n_imag > 0:
        warnings.append(f"xTB 構造に虚振動が {th.n_imag} 個あります (極小点でない可能性)。")
    if len(sp.components) > 1:
        warnings.append("複合体の熱補正は xTB の調和振動近似で、分子間の低振動は精度が低いです。")

    t0 = time.time()
    dft = engines.dft_single_point(xtb.symbols, xtb.coords, sp.charge, sp.multiplicity,
                                   level.xc, level.basis, level.disp, max_memory_mb=level.max_memory_mb)
    t["dft"] = time.time() - t0
    log(f"  DFT {level.xc}/{level.basis}: E = {dft.energy_hartree:.6f} Eh, 基底関数 {dft.nbf} ({t['dft']:.0f} 秒)")
    if not dft.converged:
        warnings.append("DFT の SCF が収束しませんでした。")

    return State(
        smiles=sp.smiles, charge=sp.charge, multiplicity=sp.multiplicity, n_heavy=sp.n_heavy,
        symbols=list(xtb.symbols), coords=np.asarray(xtb.coords),
        e_dft_kJ=dft.energy_hartree * HARTREE_TO_KJ, h_corr_kJ=th.h_corr_kJ,
        n_imag=th.n_imag, lowest_freq_cm=th.lowest_freq_cm,
        xtb_converged=xtb.converged, dft_converged=dft.converged, seconds=t, warnings=warnings,
    )


class StateCache:
    """同じ (SMILES, 電荷, 多重度) を job 内で 2 度計算しない."""

    def __init__(self, level: Level, log=print, engines=None):
        self.level, self.log, self.engines = level, log, engines
        self._d: dict[tuple, State] = {}

    def get(self, smiles: str, charge: int | None = None, multiplicity: int | None = None) -> State:
        sp = analyze(smiles, charge=charge, multiplicity=multiplicity)
        key = (sp.smiles, sp.charge, sp.multiplicity)
        if key not in self._d:
            self.log(f"[計算] {sp.smiles}  (電荷 {sp.charge}, 多重度 {sp.multiplicity})")
            self._d[key] = compute_state(sp, self.level, self.log, self.engines)
        return self._d[key]


def run_compound(entry: dict, level: Level, cal: Calibration, cache: StateCache, log=print) -> dict:
    """job.yaml の 1 エントリを処理して結果 dict を返す."""
    sp = analyze(entry["smiles"], entry.get("kind"), entry.get("free_form_smiles"),
                 entry.get("charge"), entry.get("multiplicity"))
    warnings = list(sp.warnings)
    res: dict = {
        "id": entry["id"], "name": entry.get("name", ""), "input_smiles": entry["smiles"],
        "canonical_smiles": sp.smiles, "kind": sp.kind, "charge": sp.charge, "multiplicity": sp.multiplicity,
        "n_heavy": sp.n_heavy, "components": [c.__dict__ for c in sp.components],
        "level": level.key(),
    }

    # --- アプリで使うフリー体 (先に計算し、複合体側の失敗と切り離す) ---
    ff_state = cache.get(sp.free_form)
    warnings += ff_state.warnings
    hf_ff, miss_ff = cal.apply(ff_state.e_plus_h_kJ, sp.free_form)
    unc_ff = cal.uncertainty(ff_state.n_heavy, len(miss_ff))
    if miss_ff:
        warnings.append(f"フリー体に校正セットにない結合タイプ: {', '.join(miss_ff)}。この分の補正が入っておらず誤差が大きくなります。")
    res["free_form"] = {"smiles": sp.free_form, "hf_gas_kJ_mol": hf_ff, "uncertainty_kJ": unc_ff}

    # --- 入力どおりの化学種 (塩・溶媒和物は複合体として) ---
    hf_whole = None
    if sp.smiles == sp.free_form and sp.charge == 0:
        whole = ff_state
        hf_whole, unc_whole = hf_ff, unc_ff
    else:
        whole = cache.get(sp.smiles, sp.charge, sp.multiplicity)
        warnings += [w for w in whole.warnings if w not in warnings]
        try:
            hf_whole, miss_whole = cal.apply(whole.e_plus_h_kJ, sp.smiles)
            unc_whole = cal.uncertainty(sp.n_heavy, len(miss_whole))
            if miss_whole:
                warnings.append(f"複合体に校正セットにない結合タイプ: {', '.join(miss_whole)}。誤差が大きくなります。")
        except ValueError as exc:
            warnings.append(f"複合体の ΔHf° は求められません ({exc})。フリー体の値のみ出力します。")
    if hf_whole is not None:
        res["whole"] = {"hf_gas_kJ_mol": hf_whole, "uncertainty_kJ": unc_whole,
                        "e_dft_kJ": whole.e_dft_kJ, "h_corr_kJ": whole.h_corr_kJ,
                        "n_imag": whole.n_imag, "seconds": whole.seconds}

    if sp.kind == "salt":
        warnings.append("塩の気相イオン対の値です。固体の標準生成エンタルピーではありません "
                        "(固体にするには sublimation_enthalpy_kJ_mol が必要)。イオン対は校正の対象外のため誤差が大きくなります。")
    if sp.charge != 0 and sp.kind != "salt":
        warnings.append("電荷を持つ単一種です。校正は中性分子で行っているため、値は参考程度です。")

    if hf_whole is not None and sp.kind == "solvate" and all(c.charge == 0 for c in sp.components):
        parts = [cache.get(c.smiles, c.charge, c.multiplicity) for c in sp.components]
        assoc = whole.e_plus_h_kJ - sum(p.e_plus_h_kJ for p in parts)
        res["association_enthalpy_kJ_mol"] = assoc
        res["association_note"] = ("ΔH(複合体) − ΣΔH(成分) の気相値。基底関数重なり誤差 (BSSE) を補正していないため、"
                                   "小さい基底では結合を過大評価します。")

    sub = entry.get("sublimation_enthalpy_kJ_mol")
    if sub is not None and hf_whole is not None:
        res["hf_solid_kJ_mol"] = hf_whole - float(sub)
        res["hf_solid_note"] = "ΔHf(固体) = ΔHf(気相複合体) − ΔH(固体→気相複合体)。後者は入力値。"

    if sp.n_heavy > 2 * cal.max_heavy:
        warnings.append(f"校正に使った参照分子 (最大重原子数 {cal.max_heavy}) より大きい分子です "
                        f"(重原子 {sp.n_heavy})。不確かさは外挿で、実際の誤差はこれより大きい可能性があります。")
    n_conf = level.n_keep
    if ff_state.n_heavy > 30:
        warnings.append(f"柔軟な大分子の配座探索は限定的です (RDKit {level.n_conformers} 個から xTB 最適化 {n_conf} 個)。"
                        "配座による違いだけで数十 kJ/mol 動きます。")

    res["app_input"] = {
        "smiles": sp.free_form,
        "hf_kJ_mol": round(hf_ff, 1),
        "uncertainty_kJ": round(unc_ff, 1),
        "note": f"DFT {level.xc}-{level.disp or ''}/{level.basis}//{level.xtb_method}, AE{'+結合' if cal.use_bonds else ''}校正 (LOO {cal.rmse_loo:.1f} kJ/mol)",
    }
    res["warnings"] = warnings
    res["structures"] = {"free_form": ff_state.xyz(f"{entry['id']} free form {sp.free_form}")}
    if whole is not ff_state:
        res["structures"]["whole"] = whole.xyz(f"{entry['id']} {sp.smiles}")
    return res
