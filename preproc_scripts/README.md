# Preprocessing

These scripts convert a licensed NIST20 MSP/MOL export into model inputs. Run them from the repository root after `pip install -e .`.

1. `prepare_dataframes.py` parses MSP spectra and MOL structures into a tabular export.
2. `prepare_processed_data.py` standardizes spectra and molecules and writes `spec_df.pkl`, `mol_df.pkl`, and `ann_df.pkl`.
3. `prepare_dag_features.py` builds the fragment DAG cache. FERA-MS uses `--max_depth 3`; H-transfer candidates and neutral-loss support are rendered by the tracked fragment utilities and locked training configuration.
4. `prepare_split.py` constructs the broad eligible pool used before cohort QC.
5. `final/generate_cohort_qc.py` generates the seven locked QC fields for every eligible spectrum from the licensed spectrum table and tracked DAG cache.
6. `final/make_random_split.py` applies cohort-wide QC, fixes 19,659 spectra and 2,274 molecular connectivity groups, writes `cohort_ids.csv`, and then creates the molecule-disjoint random split.
7. `prepare_scaffold_split.py` consumes that same `cohort_ids.csv`, groups it by Bemis–Murcko scaffold, treats acyclic molecules as molecule-specific singleton groups, and makes the locked seed-42 60/20/20 scaffold split.

The exact commands are in the root README. All outputs belong under ignored `data/df`, `data/proc`, `data/frag`, and `data/split` paths. Raw or derived NIST20 data must not be committed.
