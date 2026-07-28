from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "figure" / "supp_ce_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_first_existing(patterns: List[str]) -> Optional[Path]:
    for pattern in patterns:
        hits = list(PROJECT_ROOT.rglob(pattern))
        if hits:
            hits = sorted(hits)
            return hits[0]
    return None


def find_split_csv(keyword: str) -> Optional[Path]:
    csvs = sorted(PROJECT_ROOT.rglob("*.csv"))
    for path in csvs:
        p = str(path).lower()
        if keyword.lower() in p and "test" in p:
            return path
    return None


def pick_column(df: pd.DataFrame, candidates: List[str], required=True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    if required:
        raise KeyError(f"Cannot find any of columns {candidates} in {list(df.columns)}")
    return None


def ace_to_stratum(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x <= 20:
        return "low"
    if x <= 40:
        return "middle"
    return "high"


def target_ace(stratum: str) -> float:
    return {"low": 15.0, "middle": 30.0, "high": 45.0}[stratum]


def main() -> None:
    print("=" * 80)
    print("BUILDING SUPPLEMENTARY CE-RESOLVED CASE MANIFEST")
    print("=" * 80)

    spec_df_path = find_first_existing([
        "spec_df.pkl",
        "spec_df.pickle",
    ])
    if spec_df_path is None:
        raise FileNotFoundError("Cannot locate spec_df.pkl under project root.")

    print("spec_df:", spec_df_path)
    spec_df = pd.read_pickle(spec_df_path)

    spec_id_col = pick_column(spec_df, ["spec_id", "spectrum_id", "id"])
    mol_id_col = pick_column(spec_df, ["mol_id", "molecule_id", "compound_id"])
    ace_col = pick_column(spec_df, ["ace", "collision_energy", "ce", "nce"])
    smiles_col = pick_column(spec_df, ["smiles", "canon_smiles", "canonical_smiles"], required=False)
    inchikey_col = pick_column(spec_df, ["inchikey", "inchi_key"], required=False)

    # 尝试找峰数列；如果没有，就置空
    peak_count_col = pick_column(
        spec_df,
        ["num_peaks", "n_peaks", "peak_count", "observed_peak_count"],
        required=False,
    )

    # 找 random / scaffold test split 文件
    random_test_csv = find_split_csv("qcv1") or find_split_csv("random")
    scaffold_test_csv = find_split_csv("scaffold")

    if random_test_csv is None:
        raise FileNotFoundError("Cannot locate random test split CSV.")
    if scaffold_test_csv is None:
        raise FileNotFoundError("Cannot locate scaffold test split CSV.")

    print("random split csv   :", random_test_csv)
    print("scaffold split csv :", scaffold_test_csv)

    def load_test_ids(path: Path) -> set:
        df = pd.read_csv(path)
        c = pick_column(df, ["spec_id", "spectrum_id", "id"])
        return set(df[c].tolist())

    random_ids = load_test_ids(random_test_csv)
    scaffold_ids = load_test_ids(scaffold_test_csv)

    work = spec_df.copy()
    work["spec_id_x"] = work[spec_id_col]
    work["mol_id_x"] = work[mol_id_col]
    work["ace_x"] = pd.to_numeric(work[ace_col], errors="coerce")
    work["ace_stratum"] = work["ace_x"].apply(ace_to_stratum)
    if peak_count_col is not None:
        work["peak_count_x"] = pd.to_numeric(work[peak_count_col], errors="coerce")
    else:
        work["peak_count_x"] = np.nan
    if smiles_col is not None:
        work["smiles_x"] = work[smiles_col]
    else:
        work["smiles_x"] = ""
    if inchikey_col is not None:
        work["inchikey_x"] = work[inchikey_col]
    else:
        work["inchikey_x"] = ""

    work = work[work["ace_stratum"].isin(["low", "middle", "high"])].copy()

    split_rows = []
    for split_name, idset in [("random", random_ids), ("scaffold", scaffold_ids)]:
        sub = work[work["spec_id_x"].isin(idset)].copy()
        if sub.empty:
            continue

        # 只保留在 low/middle/high 三个区间都至少有一张谱的分子
        grp = sub.groupby("mol_id_x")["ace_stratum"].nunique()
        valid_mols = set(grp[grp >= 3].index.tolist())
        sub = sub[sub["mol_id_x"].isin(valid_mols)].copy()

        if sub.empty:
            continue

        # 为每个分子、每个 stratum 选一张代表谱：
        # 先优先峰数多，再优先ACE更接近该区间中心
        keep_rows = []
        for mol_id, mol_df in sub.groupby("mol_id_x"):
            chosen = []
            ok = True
            for stratum in ["low", "middle", "high"]:
                cur = mol_df[mol_df["ace_stratum"] == stratum].copy()
                if cur.empty:
                    ok = False
                    break
                cur["target_dist"] = (cur["ace_x"] - target_ace(stratum)).abs()
                cur["peak_count_fill"] = cur["peak_count_x"].fillna(-1)
                cur = cur.sort_values(
                    by=["peak_count_fill", "target_dist"],
                    ascending=[False, True],
                )
                chosen.append(cur.iloc[0])
            if ok:
                for row in chosen:
                    keep_rows.append(dict(row))

        rep = pd.DataFrame(keep_rows)
        if rep.empty:
            continue

        # 给分子打分：三张谱平均峰数
        score_df = (
            rep.groupby("mol_id_x")["peak_count_x"]
            .mean()
            .reset_index()
            .rename(columns={"peak_count_x": "mean_peak_count"})
        )
        rep = rep.merge(score_df, on="mol_id_x", how="left")

        # 分子级排序
        mol_rank = (
            rep[["mol_id_x", "mean_peak_count", "smiles_x", "inchikey_x"]]
            .drop_duplicates()
            .sort_values(by=["mean_peak_count", "mol_id_x"], ascending=[False, True])
            .reset_index(drop=True)
        )
        mol_rank["split"] = split_name
        mol_rank["candidate_rank"] = np.arange(1, len(mol_rank) + 1)

        rep = rep.merge(
            mol_rank[["mol_id_x", "candidate_rank", "split"]],
            on="mol_id_x",
            how="left",
        )
        split_rows.append(rep)

    all_rep = pd.concat(split_rows, ignore_index=True)
    if all_rep.empty:
        raise RuntimeError("No valid low/middle/high ACE molecules found.")

    all_rep = all_rep[
        [
            "split",
            "candidate_rank",
            "mol_id_x",
            "spec_id_x",
            "ace_x",
            "ace_stratum",
            "peak_count_x",
            "mean_peak_count",
            "smiles_x",
            "inchikey_x",
        ]
    ].rename(
        columns={
            "mol_id_x": "mol_id",
            "spec_id_x": "spec_id",
            "ace_x": "ace",
            "peak_count_x": "peak_count",
            "smiles_x": "smiles",
            "inchikey_x": "inchikey",
        }
    )

    all_rep = all_rep.sort_values(
        by=["split", "candidate_rank", "ace"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    all_csv = OUT_DIR / "ce_case_candidates.csv"
    all_rep.to_csv(all_csv, index=False)

    # 默认各选 top-1 random 和 top-1 scaffold
    selected_rows = []
    for split_name in ["random", "scaffold"]:
        cur = all_rep[all_rep["split"] == split_name].copy()
        cur = cur[cur["candidate_rank"] == 1].copy()
        selected_rows.append(cur)

    selected = pd.concat(selected_rows, ignore_index=True)
    selected = selected.sort_values(by=["split", "ace"]).reset_index(drop=True)

    selected_csv = OUT_DIR / "ce_case_selected_default.csv"
    selected.to_csv(selected_csv, index=False)

    # 同时导出补充表格模板
    table_csv = OUT_DIR / "supp_ce_table_template.csv"
    selected_table = selected.copy()
    selected_table["CBIN"] = ""
    selected_table["JSS"] = ""
    selected_table["observed_peaks"] = selected_table["peak_count"]
    selected_table = selected_table[
        [
            "split",
            "mol_id",
            "spec_id",
            "ace_stratum",
            "ace",
            "observed_peaks",
            "CBIN",
            "JSS",
            "smiles",
            "inchikey",
        ]
    ]
    selected_table.to_csv(table_csv, index=False)

    tex_path = OUT_DIR / "supp_ce_table_template.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[!t]
\centering
\caption{Collision-energy-resolved representative cases used in the supplementary spectrum comparison figure.}
\begin{tabular}{llllllll}
\toprule
Split & Molecule & Spec ID & ACE stratum & ACE (eV) & Observed peaks & CBIN & JSS \\
\midrule
""")
        for _, row in selected_table.iterrows():
            split_name = str(row["split"]).capitalize()
            mol_name = str(row["mol_id"])
            spec_id = str(row["spec_id"])
            stratum = str(row["ace_stratum"]).capitalize()
            ace = f'{row["ace"]:.1f}' if pd.notna(row["ace"]) else ""
            peaks = "" if pd.isna(row["observed_peaks"]) else str(int(row["observed_peaks"]))
            f.write(
                f"{split_name} & {mol_name} & {spec_id} & {stratum} & {ace} & {peaks} &  &  \\\\\n"
            )
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

    summary = {
        "spec_df": str(spec_df_path),
        "random_test_csv": str(random_test_csv),
        "scaffold_test_csv": str(scaffold_test_csv),
        "candidate_csv": str(all_csv),
        "selected_csv": str(selected_csv),
        "table_csv": str(table_csv),
        "table_tex": str(tex_path),
    }
    (OUT_DIR / "build_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("candidate csv :", all_csv)
    print("selected csv  :", selected_csv)
    print("table csv     :", table_csv)
    print("table tex     :", tex_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
