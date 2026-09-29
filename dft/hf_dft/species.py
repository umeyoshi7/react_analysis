"""SMILES の成分分解・電荷/スピン・フリー体の決定."""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass
class Component:
    smiles: str          # RDKit 正規化 SMILES
    charge: int
    multiplicity: int
    n_heavy: int
    formula: str


@dataclass
class Species:
    smiles: str                      # 入力 SMILES (正規化済み)
    components: list[Component]
    kind: str                        # neutral / salt / solvate
    charge: int
    multiplicity: int
    free_form: str                   # アプリで使うフリー体の SMILES (中性・単一分子)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_heavy(self) -> int:
        return sum(c.n_heavy for c in self.components)


def _multiplicity(mol: Chem.Mol) -> int:
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return radicals + 1


def _component(mol: Chem.Mol) -> Component:
    from rdkit.Chem import rdMolDescriptors

    return Component(
        smiles=Chem.MolToSmiles(mol),
        charge=sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        multiplicity=_multiplicity(mol),
        n_heavy=mol.GetNumHeavyAtoms(),
        formula=rdMolDescriptors.CalcMolFormula(mol),
    )


def neutral_form(mol: Chem.Mol) -> Chem.Mol:
    """電荷を中和した分子 (カルボン酸塩 → カルボン酸、アンモニウム → アミン など)."""
    return rdMolStandardize.Uncharger().uncharge(Chem.Mol(mol))


def analyze(smiles: str, kind: str | None = None, free_form: str | None = None,
            charge: int | None = None, multiplicity: int | None = None) -> Species:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"無効な SMILES: {smiles}")
    frags = [Chem.Mol(f) for f in Chem.GetMolFrags(mol, asMols=True)]
    comps = [_component(f) for f in frags]
    warnings: list[str] = []

    detected = "neutral"
    if len(frags) > 1:
        ionic = any(c.charge != 0 for c in comps)
        detected = "salt" if ionic else "solvate"
    if kind and kind != detected:
        warnings.append(f"kind={kind} が指定されましたが、SMILES からは {detected} と判定されます。指定を優先します。")
    kind = kind or detected

    total_charge = sum(c.charge for c in comps) if charge is None else charge
    if multiplicity is None:
        total_radicals = sum(c.multiplicity - 1 for c in comps)
        multiplicity = total_radicals + 1

    if free_form:
        ff = Chem.MolFromSmiles(free_form)
        if ff is None or len(Chem.GetMolFrags(ff)) != 1:
            raise ValueError(f"free_form_smiles は単一分子の SMILES にしてください: {free_form}")
        ff_smiles = Chem.MolToSmiles(ff)
    else:
        # 重原子数が最大の成分を中和したものをフリー体とする
        biggest = max(frags, key=lambda f: f.GetNumHeavyAtoms())
        ff_smiles = Chem.MolToSmiles(neutral_form(biggest))
        if len(frags) > 1:
            warnings.append(f"フリー体を自動判定しました: {ff_smiles} (違う場合は free_form_smiles で指定)")

    return Species(
        smiles=Chem.MolToSmiles(mol),
        components=comps,
        kind=kind,
        charge=total_charge,
        multiplicity=multiplicity,
        free_form=ff_smiles,
        warnings=warnings,
    )
