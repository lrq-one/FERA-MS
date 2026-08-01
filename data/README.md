# Local data contract

This directory contains documentation only. NIST20 data, derived tables, record-level split IDs, fragment DAGs, PubChem caches, and model artifacts are excluded from Git. The release does not distribute train/validation/test record lists.

Required processed files:

- `spec_df.pkl`: one row per spectrum, including `spec_id`, `mol_id`, precursor type/mass, instrument/fragmentation fields, ACE, and peak arrays.
- `mol_df.pkl`: one row per molecule, including `mol_id`, canonical structure, connectivity key, and Murcko scaffold.
- `ann_df.pkl`: available peak/formula annotations keyed to the processed spectra.
- `{train,val,test}_ids.csv`: columns `spec_id`, `mol_id`, and `group_id`; molecule identities must be disjoint across partitions.
- `dags/<mol_id>.json.bz2` (or the cache format emitted by `prepare_dag_features.py`): depth-3 candidate DAGs used by the locked config.
- `cohort_ids.csv`: the cohort-wide QC manifest consumed by both random and scaffold split builders.

The formal configuration expects the paths shown in `config/train.yml`. Use `MS2_SPLIT_DP` to select another compatible split. Split files are not distributed until their redistribution status is confirmed. Regenerate them from a licensed local NIST20 export with the tracked scripts: seed 42, molecule grouping by connectivity key, the documented 19,659-spectrum cohort, and pairwise train/validation/test overlap assertions.

`test/align_nist_native_formula_annotations.py` reads the licensed native annotation table from `data/proc/nist20_full/ann_df.pkl` by default. Set `FERA_MS_NATIVE_ANNOTATION_PATH` to an explicit local file when it is stored elsewhere; the script never searches a sibling checkout.
