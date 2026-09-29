"""校正用の参照分子セット dft/reference_set.csv を生成する.

DFT の値から ΔHf° を求めるときの原子当量 (AE) 補正を、既知の ΔHf° を持つ小分子で
フィットする。その参照値は `chemicals` パッケージ同梱の ATcT / API TDB / Yaws の
気相 ΔHf° (298 K) から作る。実行時の依存ではなく、CSV を作り直すときだけ使う。

    pip install chemicals
    python dft/build_reference_set.py [--chemicals-dir <site-packages>/chemicals]

採用条件:
  - 2 つ以上の出典が 3 kJ/mol 以内で一致 → 不確かさ = 出典の幅/2 + 0.5 (最小 1)
  - 出典が 1 つだけ → 不確かさ 4 kJ/mol で採用
  - 出典間で食い違うものは採用しない
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

ROOT = Path(__file__).resolve().parent.parent

# データ層 (data/hf_gas.csv) 以外に加える参照分子: SMILES 名称
EXTRA = """FC fluoromethane
CCF fluoroethane
CC(F)F 1,1-difluoroethane
Fc1ccccc1 fluorobenzene
FC(F)(F)F tetrafluoromethane
CC(N)=O acetamide
CNC(C)=O N-methylacetamide
CC(=O)N(C)C N,N-dimethylacetamide
O=C1CCCN1 2-pyrrolidone
CN1CCCC1=O N-methylpyrrolidone
O=C1CCCO1 gamma-butyrolactone
O=C1CCC(=O)N1 succinimide
c1ccc2ncccc2c1 quinoline
Cc1ccccn1 2-methylpyridine
OCc1ccccc1 benzyl alcohol
c1ccc2[nH]ccc2c1 indole
O=C(N)c1ccccc1 benzamide
CC(=O)Nc1ccccc1 acetanilide
C1CCOCC1 tetrahydropyran
C1COCCO1 1,4-dioxane
CSC dimethyl sulfide
CS methanethiol
CCS ethanethiol
C1=CC(=O)NC1=O maleimide
CCC(=O)OC methyl propionate
CC(=O)OC(C)C isopropyl acetate
CC(C)(C)O tert-butanol
CCCCCC=O hexanal
CC(=O)CC(C)=O acetylacetone
C1CCNCC1 piperidine
C1CCNC1 pyrrolidine
CCC#N propionitrile""".splitlines()

MAX_HEAVY = 14

# 参照値が出典間で怪しい、または B3LYP 系の単参照配置で扱いにくく校正を乱すもの
EXCLUDE = {
    "CS(C)=O",        # DMSO: API/Yaws は -209.2 だが、気相値は約 -151 (液相値に近い)
    "O=[O+][O-]",     # オゾン: 多参照性が強い
    "S=C=S",          # 二硫化炭素: 同上
    "C#N",            # HCN: ATcT 129.3 と他 135 で割れている
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chemicals-dir", type=Path, default=None)
    args = ap.parse_args()

    import chemicals
    from chemicals.identifiers import search_chemical

    base = args.chemicals_dir or Path(chemicals.__file__).parent

    def load(name: str) -> pd.DataFrame:
        return pd.read_csv(base / name, sep="\t").drop_duplicates("CAS").set_index("CAS")

    srcs = {
        "ATcT": load("Reactions/ATcT 1.112 (g).tsv"),
        "API": load("Reactions/API TDB Albahri Hf (g).tsv"),
        "Yaws": load("Reactions/Yaws Hf S0 (g).tsv"),
    }

    cands = []
    with (ROOT / "data" / "hf_gas.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cands.append((r["smiles"], r["name"]))
    cands += [tuple(line.split(" ", 1)) for line in EXTRA]

    seen: set[str] = set()
    rows = []
    for smi, name in cands:
        mol = Chem.MolFromSmiles(smi)
        can = Chem.MolToSmiles(mol)
        if can in seen or can in EXCLUDE or mol.GetNumHeavyAtoms() > MAX_HEAVY:
            continue
        seen.add(can)
        try:
            cas = search_chemical("InChI=" + inchi.MolToInchi(mol)[6:]).CASs
        except Exception:
            continue
        refs = {k: float(df.loc[cas, "Hfg"]) / 1000 for k, df in srcs.items() if cas in df.index}
        if not refs:
            continue
        vals = sorted(refs.values())
        med = vals[len(vals) // 2]
        agree = [v for v in vals if abs(v - med) <= 3.0]
        if len(vals) == 1:
            hf, unc, used = vals[0], 4.0, list(refs)
        elif len(agree) >= 2:
            atct = refs.get("ATcT")
            hf = atct if atct is not None and abs(atct - med) <= 3.0 else sum(agree) / len(agree)
            unc = max(1.0, (max(agree) - min(agree)) / 2 + 0.5)
            used = [k for k, v in refs.items() if abs(v - med) <= 3.0]
        else:
            continue
        rows.append((can, name, round(hf, 1), round(unc, 1), "+".join(used), mol.GetNumHeavyAtoms()))

    out = ROOT / "dft" / "reference_set.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "name", "hf_kJ_mol", "uncertainty_kJ", "sources", "n_heavy"])
        w.writerows(rows)
    print(f"{len(rows)} compounds -> {out}")


if __name__ == "__main__":
    main()
