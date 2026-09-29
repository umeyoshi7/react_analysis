"""data/hf_gas.csv を生成する (SMILES → InChIKey 付与).

値は理想気体 ΔHf° (298.15 K, kJ/mol)。出典は NIST Chemistry WebBook / Pedley 等の
一般的な評価値。hvap_kJ_mol は 298.15 K の蒸発エンタルピーで、空欄は未登録。行を追加・修正する場合はこのファイルを編集して再実行する。
    python scripts/build_hf_db.py
"""
import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi

# (SMILES, 名称, ΔHf°(g) kJ/mol, 不確かさ kJ/mol)
ROWS = [
    ("CC", "エタン", -84.0, 0.4),
    ("CCC", "プロパン", -104.7, 0.5),
    ("CCCC", "n-ブタン", -125.7, 0.7),
    ("CC(C)C", "イソブタン", -134.2, 0.7),
    ("CCCCC", "n-ペンタン", -146.9, 0.8),
    ("CCCCCC", "n-ヘキサン", -167.2, 0.9),
    ("C1CCCCC1", "シクロヘキサン", -123.4, 0.8),
    ("C1CCCC1", "シクロペンタン", -76.4, 0.7),
    ("C1=CCCCC1", "シクロヘキセン", -5.0, 1.0),
    ("C=CC", "プロペン", 20.0, 0.5),
    ("C=CC=C", "1,3-ブタジエン", 110.0, 1.0),
    ("CC(=C)C=C", "イソプレン", 75.5, 1.0),
    ("C#C", "アセチレン", 226.7, 0.8),
    ("CCc1ccccc1", "エチルベンゼン", 29.9, 1.0),
    ("C=Cc1ccccc1", "スチレン", 147.9, 1.0),
    ("c1ccc2ccccc2c1", "ナフタレン", 150.6, 1.5),
    ("Oc1ccccc1", "フェノール", -96.4, 0.9),
    ("Nc1ccccc1", "アニリン", 87.5, 1.0),
    ("Clc1ccccc1", "クロロベンゼン", 52.0, 1.0),
    ("O=Cc1ccccc1", "ベンズアルデヒド", -36.7, 1.5),
    ("OC(=O)c1ccccc1", "安息香酸", -292.0, 4.0),
    ("CC(=O)c1ccccc1", "アセトフェノン", -86.7, 1.5),
    ("O=[N+]([O-])c1ccccc1", "ニトロベンゼン", 67.5, 1.5),
    ("CCCO", "1-プロパノール", -255.1, 1.0),
    ("CC(C)O", "2-プロパノール", -272.6, 1.0),
    ("CCCCO", "1-ブタノール", -274.9, 1.0),
    ("CCOCC", "ジエチルエーテル", -252.1, 1.0),
    ("C1CCOC1", "THF", -184.2, 1.0),
    ("C1CO1", "エチレンオキシド", -52.6, 0.7),
    ("CCC(C)=O", "2-ブタノン (MEK)", -238.5, 1.0),
    ("COC(C)=O", "酢酸メチル", -411.9, 1.0),
    ("CCOC(C)=O", "酢酸エチル", -444.5, 1.0),
    ("COC=O", "ギ酸メチル", -357.8, 2.5),
    ("CC(=O)OC(C)=O", "無水酢酸", -573.5, 4.0),
    ("CC(=O)Cl", "塩化アセチル", -243.5, 1.5),
    ("CN(C)C=O", "DMF", -192.1, 1.0),
    ("NC=O", "ホルムアミド", -186.2, 1.0),
    ("CC#N", "アセトニトリル", 74.0, 1.0),
    ("C=CC#N", "アクリロニトリル", 180.6, 1.5),
    ("C#N", "シアン化水素", 135.1, 8.0),
    ("CS(C)=O", "DMSO", -151.3, 3.0),
    ("CN", "メチルアミン", -22.5, 0.5),
    ("CNC", "ジメチルアミン", -18.8, 0.8),
    ("CN(C)C", "トリメチルアミン", -23.7, 0.5),
    ("CCN", "エチルアミン", -47.5, 0.8),
    ("CCNCC", "ジエチルアミン", -72.2, 1.0),
    ("c1ccncc1", "ピリジン", 140.4, 0.7),
    ("NN", "ヒドラジン", 95.4, 0.2),
    ("C[N+](=O)[O-]", "ニトロメタン", -74.7, 1.0),
    ("CCl", "クロロメタン", -81.9, 1.0),
    ("ClCCl", "ジクロロメタン", -95.4, 1.5),
    ("ClC(Cl)(Cl)Cl", "四塩化炭素", -95.8, 2.0),
    ("CCCl", "クロロエタン", -112.1, 1.0),
    ("ClCCCl", "1,2-ジクロロエタン", -129.5, 2.5),
    ("C=CCl", "塩化ビニル", 28.4, 7.0),
    ("CBr", "ブロモメタン", -35.4, 1.0),
    ("S=C=S", "二硫化炭素", 116.7, 0.7),
    ("O=[O+][O-]", "オゾン", 142.7, 1.0),
]


# 298.15 K での蒸発エンタルピー (kJ/mol)。液相 ΔHf° = 気相 ΔHf° − ΔHvap
HVAP = {
    "CCCCC": 26.4,
    "CCCCCC": 31.6,
    "C1CCCCC1": 33.0,
    "Cc1ccccc1": 38.0,
    "CCCO": 47.5,
    "CC(C)O": 45.4,
    "CCCCO": 52.4,
    "CCOCC": 27.1,
    "CC(C)=O": 31.0,
    "COC(C)=O": 32.3,
    "CCOC(C)=O": 35.6,
    "CC#N": 32.9,
    "CN(C)C=O": 46.9,
    "c1ccncc1": 40.2,
    "ClCCl": 28.8,
    "ClC(Cl)(Cl)Cl": 32.4,
    "ClCCCl": 35.2,
    "c1ccccc1": 33.9,
}


SOURCE = "NIST WebBook/Pedley 等。ATcT/API TDB/Yaws/CRC と照合済み (2026-09-30)"
# 出典間で値が食い違い、不確かさを広げた行 (要確認)
DISPUTED = {"OC(=O)c1ccccc1", "COC=O", "CC(=O)OC(C)=O", "CS(C)=O", "ClCCCl", "C=CCl"}


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "hf_gas.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["inchikey", "smiles", "name", "hf_kJ_mol", "uncertainty_kJ", "hvap_kJ_mol", "source"])
        for smi, name, hf, unc in ROWS:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, smi
            hvap = HVAP.get(smi)
            w.writerow([inchi.MolToInchiKey(mol), Chem.MolToSmiles(mol), name, hf, unc,
                        "" if hvap is None else hvap,
                        SOURCE if smi not in DISPUTED else "出典間で不一致。不確かさを拡大 (要確認)"])
    print(f"wrote {len(ROWS)} rows -> {out}")


if __name__ == "__main__":
    main()
