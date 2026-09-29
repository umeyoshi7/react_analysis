"""job.yaml を読み、run / calibrate / estimate を実行して結果を保存する."""
from __future__ import annotations

import csv
import io
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import yaml

from . import cost, storage
from .pipeline import Level, StateCache, run_compound
from .thermo import Calibration, fit_calibration

PKG_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = PKG_DIR / "reference_set.csv"


def load_job(path: str) -> dict:
    job = yaml.safe_load(storage.read_text(path)) or {}
    job.setdefault("job_name", "hf-dft")
    job.setdefault("level", {})
    job.setdefault("sampling", {})
    job.setdefault("resources", {})
    return job


def level_from_job(job: dict) -> Level:
    lv, sm, rs = job["level"], job["sampling"], job["resources"]
    return Level(
        xc=lv.get("xc", "b3lyp"), basis=lv.get("basis", "def2-svp"), disp=lv.get("disp", "d3bj"),
        xtb_method=lv.get("xtb_method", "GFN2-xTB"), temperature_K=float(lv.get("temperature_K", 298.15)),
        n_conformers=int(sm.get("n_conformers", 30)), n_keep=int(sm.get("n_keep", 5)),
        n_orient=int(sm.get("n_orient", 8)),
        max_memory_mb=int(rs["max_memory_gb"] * 1024) if rs.get("max_memory_gb") else None,
    )


def _validate_entries(entries: list[dict]) -> None:
    ids = [e.get("id") for e in entries]
    if any(not i for i in ids):
        raise ValueError("compounds の各要素に id が必要です")
    if len(set(ids)) != len(ids):
        raise ValueError("compounds の id が重複しています")
    for e in entries:
        if not e.get("smiles"):
            raise ValueError(f"{e['id']}: smiles がありません")


def calibration_path(job: dict, level: Level, out: str) -> str:
    if job.get("calibration"):
        return job["calibration"]
    tag = f"{level.xc}_{level.basis}_{level.disp or 'nodisp'}".replace("*", "s")
    return storage.join(out, f"calibration_{tag}.json")


# ---------------------------------------------------------------- estimate

def run_estimate(job: dict, machine: str) -> str:
    level = level_from_job(job)
    lines = []
    for e in job.get("compounds", []):
        from .species import analyze
        sp = analyze(e["smiles"])
        for smi in {sp.smiles, sp.free_form}:
            est = cost.estimate(smi, level.basis, machine, level.n_keep, level.n_conformers)
            lines.append(f"[{e['id']}] {smi[:70]}{'…' if len(smi) > 70 else ''}\n" + cost.format_estimate(est, machine))
    return "\n".join(lines)


# ---------------------------------------------------------------- calibrate

def _calibrate_one(args: tuple) -> dict:
    smiles, hf_ref, name, level, threads = args
    from . import engines
    from .species import analyze
    engines.set_threads(threads)
    t0 = time.time()
    try:
        cache = StateCache(level, log=lambda *_: None)
        st = cache.get(smiles)
        return {"smiles": smiles, "name": name, "hf_ref_kJ": hf_ref, "e_plus_h_kJ": st.e_plus_h_kJ,
                "n_heavy": analyze(smiles).n_heavy, "seconds": time.time() - t0,
                "ok": st.dft_converged and st.xtb_converged and st.n_imag == 0,
                "reason": f"xTB収束={st.xtb_converged}, DFT収束={st.dft_converged}, 虚振動={st.n_imag}"}
    except Exception as exc:   # 1 分子の失敗で校正全体を止めない
        return {"smiles": smiles, "name": name, "error": f"{type(exc).__name__}: {exc}"}


def run_calibrate(job: dict, out: str, log=print) -> Calibration:
    level = level_from_job(job)
    ref_path = job.get("reference_set") or str(DEFAULT_REFERENCE)
    refs = list(csv.DictReader(io.StringIO(storage.read_text(str(ref_path)))))
    max_heavy = int(job.get("reference_max_heavy", 14))
    refs = [r for r in refs if int(r["n_heavy"]) <= max_heavy]
    total = job["resources"].get("threads") or os.cpu_count() or 1
    workers = max(1, int(job["resources"].get("workers", 1)))
    threads = max(1, total // workers)
    log(f"校正: 参照分子 {len(refs)} 個, ワーカー {workers}, 各 {threads} スレッド, レベル {level.key()}")

    tag = f"{level.xc}_{level.basis}_{level.disp or 'nodisp'}".replace("*", "s")
    raw_path = job.get("raw_energies")     # 計算済みエネルギーから校正だけやり直す場合
    if raw_path:
        results = json.loads(storage.read_text(raw_path))
    else:
        tasks = [(r["smiles"], float(r["hf_kJ_mol"]), r["name"], level, threads) for r in refs]
        if workers == 1:
            results = [_calibrate_one(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_calibrate_one, tasks))
        storage.write_text(storage.join(out, f"calibration_raw_{tag}.json"),
                           json.dumps(results, ensure_ascii=False, indent=2, default=float))

    good = [r for r in results if "error" not in r and r["ok"]]
    bad = [r for r in results if r not in good]
    for r in bad:
        log(f"  除外: {r['name']} ({r.get('error') or r.get('reason')})")
    cal = fit_calibration(good, level.key(), use_bonds=bool(job.get("calibration_bonds", True)),
                          ridge=float(job.get("calibration_ridge", 3.0)))
    log(f"校正完了: n={cal.n}, RMSE(fit)={cal.rmse_fit:.1f}, RMSE(LOO)={cal.rmse_loo:.1f} kJ/mol, "
        f"結合補正={'あり' if cal.use_bonds else 'なし'}, 参照値から外れとして除外 {len(cal.rejected)} 件")
    for r in cal.rejected:
        log(f"  外れ値: {r['name']} {r['smiles']} (LOO 残差 {r['resid_loo_kJ']:+.0f} kJ/mol)")
    path = calibration_path(job, level, out)
    storage.write_text(path, json.dumps(asdict(cal), ensure_ascii=False, indent=2))
    storage.write_text(storage.join(out, "calibration_excluded.json"), json.dumps(bad, ensure_ascii=False, indent=2))
    log(f"保存: {path}")
    return cal


# ---------------------------------------------------------------- run

def _load_results(path: str) -> dict:
    if storage.exists(path):
        return {r["id"]: r for r in json.loads(storage.read_text(path)).get("results", [])}
    return {}


def _flat_rows(results: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "name", "kind", "input_smiles", "free_form_smiles", "free_form_hf_kJ_mol",
                "free_form_unc_kJ", "whole_hf_gas_kJ_mol", "whole_unc_kJ", "association_kJ_mol",
                "hf_solid_kJ_mol", "warnings"])
    for r in results:
        if "error" in r:
            w.writerow([r["id"], r.get("name", ""), "", r.get("input_smiles", ""), "", "", "", "", "", "", "", r["error"]])
            continue
        w.writerow([r["id"], r["name"], r["kind"], r["input_smiles"], r["free_form"]["smiles"],
                    f"{r['free_form']['hf_gas_kJ_mol']:.1f}", f"{r['free_form']['uncertainty_kJ']:.1f}",
                    f"{r['whole']['hf_gas_kJ_mol']:.1f}" if "whole" in r else "",
                    f"{r['whole']['uncertainty_kJ']:.1f}" if "whole" in r else "",
                    "" if "association_enthalpy_kJ_mol" not in r else f"{r['association_enthalpy_kJ_mol']:.1f}",
                    "" if "hf_solid_kJ_mol" not in r else f"{r['hf_solid_kJ_mol']:.1f}",
                    " | ".join(r["warnings"])])
    return buf.getvalue()


def run_job(job: dict, out: str, log=print, engines=None) -> list[dict]:
    from . import engines as engines_mod
    level = level_from_job(job)
    (engines or engines_mod).set_threads(job["resources"].get("threads"))

    entries = job.get("compounds", [])
    _validate_entries(entries)
    cal_path = calibration_path(job, level, out)
    if not storage.exists(cal_path):
        raise FileNotFoundError(
            f"校正ファイルがありません: {cal_path}\n先に mode=calibrate のジョブを同じレベルで実行してください。")
    cal = Calibration.from_dict(json.loads(storage.read_text(cal_path)))
    if cal.level != level.key():
        raise ValueError(f"校正ファイルのレベル {cal.level} が job の計算レベル {level.key()} と一致しません")

    results_path = storage.join(out, "results.json")
    done = _load_results(results_path)          # Spot の再起動時に計算済みを飛ばす
    cache = StateCache(level, log, engines)
    results: list[dict] = list(done.values())

    for e in entries:
        if e["id"] in done and "error" not in done[e["id"]]:
            log(f"[スキップ] {e['id']} は計算済み")
            continue
        log(f"===== {e['id']} {e.get('name', '')} =====")
        t0 = time.time()
        try:
            r = run_compound(e, level, cal, cache, log)
            structures = r.pop("structures")
            for label, xyz in structures.items():
                storage.write_text(storage.join(out, "structures", f"{e['id']}_{label}.xyz"), xyz)
            r["seconds_total"] = time.time() - t0
        except Exception as exc:
            log(traceback.format_exc())
            r = {"id": e["id"], "name": e.get("name", ""), "input_smiles": e["smiles"],
                 "error": f"{type(exc).__name__}: {exc}"}
        results = [x for x in results if x["id"] != e["id"]] + [r]
        payload = {"job_name": job["job_name"], "level": level.key(), "calibration": cal_path,
                   "results": results}
        storage.write_text(results_path, json.dumps(payload, ensure_ascii=False, indent=2, default=float))
        storage.write_text(storage.join(out, "results.csv"), _flat_rows(results))
    log(f"完了: {results_path}")
    return results
