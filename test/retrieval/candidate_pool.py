"""Build deterministic, structure-filtered PubChem candidate pools."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem import inchi as rd_inchi


MAX_NUM_HEAVY_ATOMS = 128
ALLOWED_ELEMENTS = {
    "C",
    "H",
    "O",
    "N",
    "P",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "Se",
    "Si",
}


def _connectivity_key(molecule: Chem.Mol) -> str:
    return rd_inchi.MolToInchiKey(molecule).split("-", maxsplit=1)[0]


def filter_candidates(
    database_rows: Iterable[tuple[int, str, str, str, float]],
    target_smiles: str,
    target_mol_id: int,
    top_k: int | None = None,
    morgan_radius: int = 2,
) -> tuple[int, str, pd.DataFrame]:
    """Filter, deduplicate, and rank a mass-matched candidate collection.

    The true target is injected before canonical-SMILES and connectivity-key
    deduplication. Only neutral, closed-shell, single-component molecules with
    supported elements and at most 128 heavy atoms are retained.
    """

    target_molecule = Chem.MolFromSmiles(target_smiles)
    if target_molecule is None:
        raise ValueError(f"Invalid target SMILES: {target_smiles!r}")

    target_formula = rdMolDescriptors.CalcMolFormula(target_molecule)
    target_mass = rdMolDescriptors.CalcExactMolWt(target_molecule)
    target_inchikey = rd_inchi.MolToInchiKey(target_molecule)

    rows = list(database_rows)
    rows.append(
        (
            -1,
            target_inchikey,
            target_smiles,
            target_formula,
            target_mass,
        )
    )

    frame = pd.DataFrame(
        rows,
        columns=("mol_id", "inchikey", "smiles", "formula", "mw"),
    )
    frame = frame.loc[
        frame["smiles"].notna()
        & ~frame["smiles"].astype(str).str.contains(".", regex=False)
    ].copy()

    frame["mol"] = frame["smiles"].map(Chem.MolFromSmiles)
    frame = frame.loc[frame["mol"].notna()].copy()
    frame["smiles"] = frame["mol"].map(
        lambda molecule: Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=False,
        )
    )
    frame["inchikey"] = frame["mol"].map(rd_inchi.MolToInchiKey)
    frame["inchikey_s"] = frame["mol"].map(_connectivity_key)
    frame["formula"] = frame["mol"].map(rdMolDescriptors.CalcMolFormula)
    frame["mw"] = frame["mol"].map(rdMolDescriptors.CalcExactMolWt)

    frame = frame.drop_duplicates("smiles", keep="last")
    frame = frame.drop_duplicates("inchikey_s", keep="last")

    frame["num_heavy_atoms"] = frame["mol"].map(
        rdMolDescriptors.CalcNumHeavyAtoms
    )
    frame["num_radicals"] = frame["mol"].map(
        lambda molecule: sum(
            atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
        )
    )
    frame["num_out_set_elements"] = frame["mol"].map(
        lambda molecule: len(
            {atom.GetSymbol() for atom in molecule.GetAtoms()}
            - ALLOWED_ELEMENTS
        )
    )
    frame["charge"] = frame["mol"].map(Chem.GetFormalCharge)

    frame = frame.loc[
        (frame["num_heavy_atoms"] <= MAX_NUM_HEAVY_ATOMS)
        & (frame["num_radicals"] == 0)
        & (frame["num_out_set_elements"] == 0)
        & (frame["charge"] == 0)
    ].copy()

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(morgan_radius)
    )
    target_fingerprint = fingerprint_generator.GetFingerprint(target_molecule)
    frame["tanimoto"] = frame["mol"].map(
        lambda molecule: DataStructs.TanimotoSimilarity(
            target_fingerprint,
            fingerprint_generator.GetFingerprint(molecule),
        )
    )

    frame["mol_id"] = pd.to_numeric(frame["mol_id"], errors="coerce")
    frame = frame.sort_values(
        ["tanimoto", "inchikey_s", "mol_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if top_k is not None:
        frame = frame.head(int(top_k))

    frame = frame.reset_index(drop=True)
    if _connectivity_key(target_molecule) not in set(frame["inchikey_s"]):
        raise RuntimeError("The true target was removed from its candidate pool")

    return int(target_mol_id), target_smiles, frame
