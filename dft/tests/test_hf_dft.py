"""PySCF / tblite なしで動く単体テスト (計算エンジンは偽物に差し替える).

    cd dft && python tests/test_hf_dft.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hf_dft import conformers, cost, runner, thermo  # noqa: E402
from hf_dft.species import analyze  # noqa: E402

EXAMPLE = ("CC[C@@]1(O)C(=O)OCC2=C1C=C3N(Cc4c5[C@H](CCc6c(C)c(F)cc(nc34)c56)NC(=O)COCNC(=O)CNC(=O)"
           "[C@H](Cc7ccccc7)NC(=O)CNC(=O)CNC(=O)CCCCCN8C(=O)C=CC8=O)C2=O")


def test_species():
    s = analyze("CCO")
    assert s.kind == "neutral" and s.free_form == "CCO"
    s = analyze("CC(N)=O.O")
    assert s.kind == "solvate" and s.free_form == "CC(N)=O" and s.charge == 0
    s = analyze("CC[NH3+].[Cl-]")
    assert s.kind == "salt" and s.charge == 0 and s.free_form == "CCN"
    s = analyze("CC(=O)[O-].[Na+]")
    assert s.kind == "salt" and s.free_form == "CC(=O)O"
    s = analyze("CC[NH3+].[Cl-]", free_form="CCN")
    assert s.free_form == "CCN"
    try:
        analyze("CC.CC", free_form="CC.CC")
    except ValueError:
        pass
    else:
        raise AssertionError("複数成分の free_form は拒否されるべき")
    s = analyze("[CH3]")
    assert s.multiplicity == 2


def Chem_heavy(smi: str) -> int:
    from rdkit import Chem
    return Chem.MolFromSmiles(smi).GetNumHeavyAtoms()


def test_calibration_exact_when_model_holds():
    """ΔHf が (元素数の線形和 − E_plus_H) で厳密に表せるデータなら、残差はほぼ 0."""
    smiles = ["C", "CC", "CCC", "CCCC", "CO", "CCO", "CCCO", "C=C", "C=O", "CC=O", "CC(C)=O", "O", "N",
              "CN", "CCN", "C#C", "c1ccccc1", "CC(=O)O", "COC", "CCOC", "CNC", "C=CC", "OCCO", "CCCCO"]
    coef = {"C": 123.0, "H": -45.0, "O": 300.0, "N": -80.0}
    entries = []
    rng = np.random.default_rng(1)
    for smi in smiles:
        ec = thermo.element_counts(smi)
        e_plus_h = float(rng.uniform(-5e5, -1e5))
        hf_ref = e_plus_h + sum(n * coef[e] for e, n in ec.items())
        entries.append({"smiles": smi, "hf_ref_kJ": hf_ref, "e_plus_h_kJ": e_plus_h,
                        "n_heavy": Chem_heavy(smi), "name": smi})
    cal = thermo.fit_calibration(entries, {"x": 1}, use_bonds=False)
    assert cal.rmse_fit < 1e-6 and not cal.rejected, (cal.rmse_fit, cal.rejected)
    for e, c in coef.items():
        assert abs(cal.coefficients[f"el:{e}"] - c) < 1e-4
    hf, _ = cal.apply(-2e5, "CCO")
    assert abs(hf - (-2e5 + 2 * 123.0 + 6 * -45.0 + 300.0)) < 1e-3
    try:
        cal.apply(0.0, "CCS")
    except ValueError as e:
        assert "S" in str(e)
    else:
        raise AssertionError("未知の元素は例外にするべき")


def test_calibration_rejects_outlier():
    smiles = ["C", "CC", "CCC", "CCCC", "CO", "CCO", "CCCO", "C=C", "C=O", "CC=O", "CC(C)=O", "O", "N",
              "CN", "CCN", "COC", "CCOC", "CNC", "C=CC", "OCCO", "CCCCO", "CC(=O)O"]
    coef = {"C": 10.0, "H": -5.0, "O": 30.0, "N": -8.0}
    rng = np.random.default_rng(2)
    entries = []
    for smi in smiles:
        ec = thermo.element_counts(smi)
        e = float(rng.uniform(-5e5, -1e5))
        entries.append({"smiles": smi, "hf_ref_kJ": e + sum(n * coef[k] for k, n in ec.items()),
                        "e_plus_h_kJ": e, "n_heavy": Chem_heavy(smi), "name": smi})
    entries[5]["hf_ref_kJ"] += 150.0        # 参照値の誤りを模す
    cal = thermo.fit_calibration(entries, {"x": 1}, use_bonds=False)
    assert [r["smiles"] for r in cal.rejected] == ["CCO"], cal.rejected
    assert cal.rmse_fit < 1e-6


def test_packed_complex():
    symbols, confs = conformers.packed_complex_confs(["CC(N)=O", "O"], n_orient=4)
    assert len(symbols) == 9 + 3 and len(confs) == 4
    for xyz in confs:
        assert xyz.shape == (12, 3)
        d = np.linalg.norm(xyz[:9, None, :] - xyz[None, 9:, :], axis=2)
        assert 1.5 < d.min() < 3.5, d.min()


def test_cost_example():
    e = cost.estimate(EXAMPLE, "def2-svp", "n2-highmem-32")
    assert e.n_atoms == 131 and e.nbf == 1330 and e.n_heavy == 75
    e2 = cost.estimate(EXAMPLE, "def2-tzvp", "n2-highmem-32")
    assert e2.df_tensor_gb > 0.7 * 256 and e2.notes, "TZVP は 3 中心積分がメモリを超える警告が出るはず"


class FakeEngines:
    """計算エンジンの代わり: 原子数と元素から決まる決定的な値を返す."""
    calls = 0

    @staticmethod
    def set_threads(n):
        pass

    @staticmethod
    def xtb_optimize(symbols, coords, charge, multiplicity, method="GFN2-xTB"):
        FakeEngines.calls += 1
        return SimpleNamespace(symbols=list(symbols), coords=np.asarray(coords),
                               energy_eV=-float(len(symbols)), converged=True)

    @staticmethod
    def xtb_thermo(symbols, coords, charge, multiplicity, method="GFN2-xTB", temperature_K=298.15):
        return SimpleNamespace(h_corr_kJ=10.0 * len(symbols), zpe_kJ=5.0 * len(symbols), n_imag=0, lowest_freq_cm=50.0)

    @staticmethod
    def dft_single_point(symbols, coords, charge, multiplicity, xc, basis, disp, max_memory_mb=None):
        z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11, "Cl": 17}
        return SimpleNamespace(energy_hartree=-0.5 * sum(z[s] for s in symbols), converged=True,
                               seconds=0.0, nbf=len(symbols))


def test_run_job_with_fake_engines():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        level = {"xc": "b3lyp", "basis": "def2-svp", "disp": "d3bj", "xtb_method": "GFN2-xTB", "temperature_K": 298.15}
        refs = ["C", "CC", "CCC", "CO", "CCO", "CN", "CCN", "C=C", "C=O", "CC=O", "O", "N", "COC", "CNC",
                "CC(N)=O", "CCCO", "CC(=O)O", "OCCO", "CCCN", "CC(C)=O", "CCl", "CCCl", "ClCCl", "CC(C)Cl"]
        entries = []
        for smi in refs:
            from hf_dft.pipeline import Level, StateCache
            st = StateCache(Level(), log=lambda *_: None, engines=FakeEngines).get(smi)
            entries.append({"smiles": smi, "hf_ref_kJ": -100.0 - 3.0 * st.n_heavy, "e_plus_h_kJ": st.e_plus_h_kJ,
                            "n_heavy": st.n_heavy, "name": smi})
        cal = thermo.fit_calibration(entries, level, use_bonds=False)
        cal_path = tmp / "cal.json"
        cal.save(cal_path)

        job = {
            "job_name": "t", "level": {}, "sampling": {"n_conformers": 4, "n_keep": 2, "n_orient": 2},
            "resources": {"threads": 1}, "calibration": str(cal_path),
            "compounds": [
                {"id": "a", "smiles": "CC(N)=O.O"},
                {"id": "b", "smiles": "CC[NH3+].[Cl-]", "free_form_smiles": "CCN", "sublimation_enthalpy_kJ_mol": 100.0},
                {"id": "c", "smiles": "CCO"},
                {"id": "d", "smiles": "CC[NH3+].[Na+]", "free_form_smiles": "CCN"},
            ],
        }
        out = str(tmp / "out")
        FakeEngines.calls = 0
        res = runner.run_job(job, out, log=lambda *_: None, engines=FakeEngines)
        by_id = {r["id"]: r for r in res}
        assert set(by_id) == {"a", "b", "c", "d"}, set(by_id)
        assert not any("error" in r for r in res), [r.get("error") for r in res]
        a, b = by_id["a"], by_id["b"]
        assert a["kind"] == "solvate" and "association_enthalpy_kJ_mol" in a
        assert a["free_form"]["smiles"] == "CC(N)=O"
        assert a["app_input"]["hf_kJ_mol"] == round(a["free_form"]["hf_gas_kJ_mol"], 1)
        assert b["kind"] == "salt" and "association_enthalpy_kJ_mol" not in b
        assert abs(b["hf_solid_kJ_mol"] - (b["whole"]["hf_gas_kJ_mol"] - 100.0)) < 1e-9
        assert any("イオン対" in w for w in b["warnings"])
        d = by_id["d"]     # Na は校正セットにないので複合体は出せないが、フリー体は出る
        assert "whole" not in d and d["free_form"]["smiles"] == "CCN" and d["app_input"]["hf_kJ_mol"] is not None
        assert any("求められません" in w for w in d["warnings"])
        assert (tmp / "out" / "structures" / "a_whole.xyz").exists()
        assert not (tmp / "out" / "structures" / "c_whole.xyz").exists()
        assert (tmp / "out" / "results.csv").exists()

        # 再実行 (Spot の再起動を想定): 計算済みは飛ばされ、エンジンが呼ばれない
        calls_before = FakeEngines.calls
        runner.run_job(job, out, log=lambda *_: None, engines=FakeEngines)
        assert FakeEngines.calls == calls_before

        # レベル不一致の校正ファイルは拒否される
        job["level"] = {"basis": "def2-tzvp"}
        try:
            runner.run_job(job, str(tmp / "out2"), log=lambda *_: None, engines=FakeEngines)
        except ValueError as e:
            assert "一致しません" in str(e)
        else:
            raise AssertionError("レベル不一致は拒否されるべき")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} tests passed")
