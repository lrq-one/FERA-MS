from __future__ import annotations

from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]

SPEC_PATH = (
    ROOT
    / "data/proc/nist20_qtof_cid_safe19659/"
      "spec_df.pkl"
)

MOL_PATH = (
    ROOT
    / "data/proc/nist20_qtof_cid_safe19659/"
      "mol_df.pkl"
)

OUT_DIR = (
    ROOT
    / "runs/experiments/reviewer_analysis/"
      "similarity_acyclic"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOLS = {
    "random": {
        "directory":
            "molecule_disjoint_three_seeds",
        "train_ids":
            ROOT
            / "data/split/"
              "nist20_qtof_cid_safe19659_qcv1_trainonly/"
              "train_ids.csv",
        "test_ids":
            ROOT
            / "data/split/"
              "nist20_qtof_cid_safe19659_qcv1_trainonly/"
              "test_ids.csv",
    },
    "scaffold": {
        "directory":
            "scaffold_disjoint_three_seeds",
        "train_ids":
            ROOT
            / "data/split/"
              "nist20_qtof_cid_safe19659_scaffold60_20_20_seed42/"
              "train_ids.csv",
        "test_ids":
            ROOT
            / "data/split/"
              "nist20_qtof_cid_safe19659_scaffold60_20_20_seed42/"
              "test_ids.csv",
    },
}

SEEDS = [42, 43, 44]

SIMILARITY_BINS = [
    -1e-12,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0000001,
]

SIMILARITY_LABELS = [
    "[0.0,0.2)",
    "[0.2,0.4)",
    "[0.4,0.6)",
    "[0.6,0.8)",
    "[0.8,1.0]",
]


def choose_score_file(
    protocol_directory: str,
    seed: int,
) -> Path:
    seed_root = (
        ROOT
        / "runs/experiments"
        / protocol_directory
        / f"seed_{seed}"
    )

    candidates = list(
        seed_root.rglob(
            "test_per_spectrum.csv"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"No test_per_spectrum.csv under "
            f"{seed_root}"
        )

    def score(path):
        text = str(path).lower()

        value = 0

        if "final_locked_evaluation" in text:
            value += 100

        if "full_fera_ms" in text:
            value += 20

        if "final" in text:
            value += 5

        return (
            value,
            -len(text),
        )

    return sorted(
        candidates,
        key=score,
        reverse=True,
    )[0]


def find_smiles_column(
    mol_df: pd.DataFrame,
) -> str:
    candidates = [
        "smiles",
        "canonical_smiles",
        "smiles_canonical",
        "mol_smiles",
    ]

    for column in candidates:
        if column in mol_df.columns:
            return column

    raise RuntimeError(
        "Cannot locate SMILES column. "
        f"mol_df columns={mol_df.columns.tolist()}"
    )


def summarize_group(
    group: pd.DataFrame,
) -> dict:
    molecule_means = (
        group.groupby(
            "mol_id",
            sort=True,
        )[["cbin", "jss"]]
        .mean()
    )

    return {
        "spectra":
            int(group["spec_id"].nunique()),
        "molecules":
            int(group["mol_id"].nunique()),
        "micro_cbin":
            float(group["cbin"].mean()),
        "micro_jss":
            float(group["jss"].mean()),
        "macro_cbin":
            float(
                molecule_means[
                    "cbin"
                ].mean()
            ),
        "macro_jss":
            float(
                molecule_means[
                    "jss"
                ].mean()
            ),
    }


def main():
    spec_df = pd.read_pickle(
        SPEC_PATH
    )

    mol_df = pd.read_pickle(
        MOL_PATH
    ).copy()

    if "mol_id" not in mol_df.columns:
        raise RuntimeError(
            f"mol_df missing mol_id: "
            f"{mol_df.columns.tolist()}"
        )

    smiles_column = find_smiles_column(
        mol_df
    )

    mol_df["mol_id"] = (
        mol_df["mol_id"].astype(int)
    )

    fp_generator = (
        rdFingerprintGenerator
        .GetMorganGenerator(
            radius=2,
            fpSize=2048,
        )
    )

    output_rows = []
    per_molecule_frames = []
    per_spectrum_frames = []

    for split_name, protocol in (
        PROTOCOLS.items()
    ):
        train_ids = pd.read_csv(
            protocol["train_ids"]
        )
        test_ids = pd.read_csv(
            protocol["test_ids"]
        )

        train_mol_ids = sorted(
            spec_df[
                spec_df["spec_id"].isin(
                    train_ids["spec_id"]
                )
            ]["mol_id"]
            .astype(int)
            .unique()
        )

        test_mol_ids = sorted(
            spec_df[
                spec_df["spec_id"].isin(
                    test_ids["spec_id"]
                )
            ]["mol_id"]
            .astype(int)
            .unique()
        )

        required_ids = set(
            train_mol_ids
        ) | set(
            test_mol_ids
        )

        molecule_table = mol_df[
            mol_df["mol_id"].isin(
                required_ids
            )
        ][
            [
                "mol_id",
                smiles_column,
            ]
        ].drop_duplicates(
            "mol_id"
        )

        molecule_map = {
            int(row.mol_id):
                str(
                    getattr(
                        row,
                        smiles_column,
                    )
                )
            for row in molecule_table.itertuples(
                index=False
            )
        }

        missing_ids = (
            required_ids
            - set(molecule_map)
        )

        if missing_ids:
            raise RuntimeError(
                f"{split_name}: missing mol_df rows: "
                f"{sorted(missing_ids)[:20]}"
            )

        molecule_objects = {}
        fingerprints = {}
        ring_counts = {}

        for mol_id in sorted(
            required_ids
        ):
            mol = Chem.MolFromSmiles(
                molecule_map[mol_id]
            )

            if mol is None:
                raise RuntimeError(
                    f"Cannot parse mol_id={mol_id}"
                )

            molecule_objects[mol_id] = mol
            fingerprints[mol_id] = (
                fp_generator.GetFingerprint(
                    mol
                )
            )
            ring_counts[mol_id] = int(
                rdMolDescriptors.CalcNumRings(
                    mol
                )
            )

        train_fingerprints = [
            fingerprints[mol_id]
            for mol_id in train_mol_ids
        ]

        similarity_rows = []

        for mol_id in test_mol_ids:
            similarities = (
                DataStructs
                .BulkTanimotoSimilarity(
                    fingerprints[mol_id],
                    train_fingerprints,
                )
            )

            similarity_rows.append({
                "split":
                    split_name,
                "mol_id":
                    mol_id,
                "max_train_tanimoto":
                    float(
                        max(similarities)
                    ),
                "ring_count":
                    ring_counts[mol_id],
                "acyclic":
                    int(
                        ring_counts[mol_id]
                        == 0
                    ),
            })

        similarity_df = pd.DataFrame(
            similarity_rows
        )

        seed_scores = []

        for seed in SEEDS:
            score_path = choose_score_file(
                protocol["directory"],
                seed,
            )

            print(
                split_name,
                seed,
                score_path,
            )

            score_frame = pd.read_csv(
                score_path
            )

            required = {
                "spec_id",
                "cos",
                "jss",
            }

            missing = (
                required
                - set(score_frame.columns)
            )

            if missing:
                raise RuntimeError(
                    f"{score_path} missing "
                    f"{sorted(missing)}"
                )

            score_frame = score_frame[
                [
                    "spec_id",
                    "cos",
                    "jss",
                ]
            ].copy()

            score_frame = score_frame.rename(
                columns={
                    "cos":
                        f"cbin_{seed}",
                    "jss":
                        f"jss_{seed}",
                }
            )

            seed_scores.append(
                score_frame
            )

        score_df = seed_scores[0]

        for seed_frame in seed_scores[1:]:
            score_df = score_df.merge(
                seed_frame,
                on="spec_id",
                how="inner",
                validate="one_to_one",
            )

        score_df["cbin"] = score_df[
            [
                f"cbin_{seed}"
                for seed in SEEDS
            ]
        ].mean(axis=1)

        score_df["jss"] = score_df[
            [
                f"jss_{seed}"
                for seed in SEEDS
            ]
        ].mean(axis=1)

        score_df = score_df[
            [
                "spec_id",
                "cbin",
                "jss",
            ]
        ]

        test_metadata = spec_df[
            spec_df["spec_id"].isin(
                test_ids["spec_id"]
            )
        ][
            [
                "spec_id",
                "mol_id",
                "ace",
                "prec_mz",
            ]
        ].copy()

        test_metadata["mol_id"] = (
            test_metadata["mol_id"]
            .astype(int)
        )

        per_spectrum = (
            test_metadata
            .merge(
                score_df,
                on="spec_id",
                how="inner",
                validate="one_to_one",
            )
            .merge(
                similarity_df,
                on="mol_id",
                how="left",
                validate="many_to_one",
            )
        )

        per_spectrum[
            "similarity_bin"
        ] = pd.cut(
            per_spectrum[
                "max_train_tanimoto"
            ],
            bins=SIMILARITY_BINS,
            labels=SIMILARITY_LABELS,
            include_lowest=True,
            right=False,
        )

        per_molecule = (
            per_spectrum
            .groupby(
                [
                    "mol_id",
                    "max_train_tanimoto",
                    "ring_count",
                    "acyclic",
                    "similarity_bin",
                ],
                observed=True,
                sort=True,
            )
            .agg(
                spectra=(
                    "spec_id",
                    "nunique",
                ),
                macro_cbin=(
                    "cbin",
                    "mean",
                ),
                macro_jss=(
                    "jss",
                    "mean",
                ),
            )
            .reset_index()
        )

        per_molecule.insert(
            0,
            "split",
            split_name,
        )

        if "split" in per_spectrum.columns:
            per_spectrum["split"] = split_name
        else:
            per_spectrum.insert(
                0,
                "split",
                split_name,
            )

        per_molecule_frames.append(
            per_molecule
        )
        per_spectrum_frames.append(
            per_spectrum
        )

        for label, group in (
            per_spectrum.groupby(
                "similarity_bin",
                observed=True,
                sort=True,
            )
        ):
            output_rows.append({
                "split":
                    split_name,
                "subgroup_type":
                    "max_train_tanimoto",
                "subgroup":
                    str(label),
                **summarize_group(
                    group
                ),
            })

        for acyclic_value, group in (
            per_spectrum.groupby(
                "acyclic",
                sort=True,
            )
        ):
            output_rows.append({
                "split":
                    split_name,
                "subgroup_type":
                    "ring_status",
                "subgroup":
                    (
                        "acyclic"
                        if int(
                            acyclic_value
                        ) == 1
                        else "cyclic"
                    ),
                **summarize_group(
                    group
                ),
            })

    per_molecule_all = pd.concat(
        per_molecule_frames,
        ignore_index=True,
    )

    per_spectrum_all = pd.concat(
        per_spectrum_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        output_rows
    )

    per_molecule_all.to_csv(
        OUT_DIR
        / "test_molecule_similarity_and_performance.csv",
        index=False,
    )

    per_spectrum_all.to_csv(
        OUT_DIR
        / "test_spectrum_similarity_and_performance.csv.gz",
        index=False,
        compression="gzip",
    )

    summary.to_csv(
        OUT_DIR
        / "similarity_acyclic_summary.csv",
        index=False,
    )

    print("\nSUMMARY")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nOUTPUT:", OUT_DIR)


if __name__ == "__main__":
    main()
