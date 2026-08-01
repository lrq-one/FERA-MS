# Preprocessing

These scripts convert a licensed NIST20 MSP/MOL export into model inputs. Run them from the repository root after `pip install -e .`.

1. `prepare_dataframes.py` parses MSP spectra and MOL structures into a tabular export.
2. `prepare_processed_data.py` standardizes spectra and molecules and writes `spec_df.pkl`, `mol_df.pkl`, and `ann_df.pkl`.
3. `prepare_dag_features.py` builds the fragment DAG cache. FERA-MS uses `--max_depth 3`; H-transfer candidates and neutral-loss support are rendered by the tracked fragment utilities and locked training configuration.
4. `final/make_random_split.py` restores the archived `safe19707` random assignment, excludes the three training molecules whose required depth-3 DAGs were unavailable, writes the fixed 19,659-spectrum/2,274-molecule cohort, and verifies every formal file against `config/paper_experiment_identity.json`.
5. `prepare_scaffold_split.py` consumes that same `cohort_ids.csv`, groups it by Bemis–Murcko scaffold, treats acyclic molecules as molecule-specific singleton groups, and requires the exact paper counts and SHA-256 values.

The exact commands are in the root README. The former weighted cohort-wide QC reconstruction was removed because it was not the historical paper protocol. All outputs belong under ignored `data/df`, `data/proc`, `data/frag`, and `data/split` paths. Raw or derived NIST20 data must not be committed.
