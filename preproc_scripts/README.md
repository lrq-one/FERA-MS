# Preprocessing

These scripts convert a licensed NIST20 MSP/MOL export into model inputs. Run them from the repository root after `pip install -e .`.

1. `01_prepare_df.py` parses MSP spectra and MOL structures into a tabular export.
2. `02_prepare_proc.py` standardizes spectra and molecules and writes `spec_df.pkl`, `mol_df.pkl`, and `ann_df.pkl`.
3. `03_prepare_dag_feats.py` builds the fragment DAG cache. FERA-MS uses `--max_depth 3`; H-transfer candidates and neutral-loss support are rendered by the tracked fragment utilities and locked training configuration.
4. `04_prepare_split.py` constructs molecule-disjoint random splits from processed data and DAG eligibility.
5. `final/make_random_split.py` applies the locked QC rule to training IDs only; validation and test IDs are copied unchanged.
6. `05_prepare_scaffold_split_safe19659.py` groups the fixed cohort by Bemis–Murcko scaffold, treats acyclic molecules as molecule-specific singleton groups, and makes the locked seed-42 60/20/20 scaffold split.

The exact commands are in the root README. All outputs belong under ignored `data/df`, `data/proc`, `data/frag`, and `data/split` paths. Raw or derived NIST20 data must not be committed.
