# FERA-MS

FERA-MS is a fragmentation- and energy-resolved model for predicting tandem mass spectra from molecular structure, precursor type, and absolute collision energy (ACE).

The implementation combines a structural GINE backbone and cut-chemistry features with retained-control continuation, formula/H-aware and multi-level ACE refinement, local m/z rendering, a rendered-entry gate, a LightGBM candidate reranker, and a spectrum-wise residual allocator.

## Repository layout

- `code/src/ms2spectra/`: FERA-MS model, losses, data utilities, and fragment generation.
- `config/train.yml`: locked mainline configuration bundle. Runtime YAML files are generated under `runs/_config/`.
- `preproc_scripts/`: NIST export parsing, processed data construction, random/scaffold splits, and depth-3 DAG construction.
- `train/`: single-seed and three-seed mainline entrypoints.
- `test/`: locked evaluation, CHUN, ACE perturbation, robustness, candidate-space, and molecular-identification analyses.
- `ablation_studies/`: validated global-only CE, m/z-offset, rendered-gate, and no-reranker controls.
- `baseline_rebuild/`: independent baseline source packages, locked random/scaffold configs, launchers, evaluation/retrieval adapters, and aggregation utilities. These sources are kept separate from `code/src/ms2spectra/`.
- `docs/`: pipeline and reproducibility details.

## Environment and installation

The recorded environment uses Python 3.10, PyTorch 2.3.0, CUDA 12.1-compatible DGL/PyG builds, RDKit 2022.9.4, and Lightning 2.6.x. GPU-specific DGL and PyG wheels may need to be installed from their official wheel indexes before the remaining requirements.

```bash
conda env create -f environment.yml
conda activate fera-ms
pip install -e .
python scripts/check_release.py
```

The root editable install compiles only `ms2spectra.frag.compute_frags` from its tracked Cython source; platform-specific `.so` files are intentionally excluded. The independent FraGNNet baseline package has its own `setup.py`, which compiles both `fragnnet.frag.compute_frags` and `fragnnet.massformer.algos` from the sources under `baseline_rebuild/baseline/source/fragnnet/`.

## Data availability and layout

NIST20 is licensed data and cannot be redistributed in this repository. Obtain and export the NIST 2020 MS/MS library under your own license. No MSP, MOL, processed pickle, fragment DAG, PubChem cache, checkpoint, prediction array, or result table is included.

Expected local layout after local regeneration (none of these data or split files are tracked):

```text
data/raw/nist_20/hr_nist_msms.MSP
data/raw/nist_20/hr_nist_msms.MOL/
data/proc/nist20_qtof_cid_safe19659/{spec_df,mol_df,ann_df}.pkl
data/split/nist20_qtof_cid_safe19659_qcv1_trainonly/{train,val,test}_ids.csv
data/split/nist20_qtof_cid_safe19659_scaffold60_20_20_seed42/{train,val,test}_ids.csv
data/frag/nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1/dags/
```

See `data/README.md` for schemas and `preproc_scripts/README.md` for the checked commands. Record-level train/validation/test CSV files are not included because their redistribution status has not been confirmed. They are deterministically regenerated from a licensed local NIST20 export using the tracked seeds, grouping rules, cohort counts, and overlap assertions.

## Preprocessing, splits, and depth-3 DAGs

The abbreviated build sequence is:

```bash
python preproc_scripts/prepare_dataframes.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --input_format msp+mol --output_format csv --output_dp data/df --output_name nist20_hr
python preproc_scripts/prepare_processed_data.py --df_dp data/df --dsets nist20_hr --proc_dp data/proc/nist20
python preproc_scripts/prepare_dag_features.py --max_depth 3 --frag_dp data/frag/nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1 --proc_dp data/proc/nist20_qtof_cid_safe19659 --prec_types "[M+H]+" --inst_types QTOF --frag_modes CID --ion_modes P --wandb_mode disabled
python preproc_scripts/prepare_split.py --split_type random --split_key inchikey_s --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types QTOF --frag_modes CID --ion_modes P --ces ace --proc_dp data/proc/nist20_qtof_cid_safe19659 --frag_dp data/frag/nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1 --split_dp data/split/nist20_qtof_cid_safe19659_random_base
python preproc_scripts/final/generate_cohort_qc.py --spec-fp data/proc/nist20_qtof_cid_safe19659/spec_df.pkl --frag-dp data/frag/nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1/dags --eligible-split data/split/nist20_qtof_cid_safe19659_random_base --out data/proc/nist20_qtof_cid_safe19659/cohort_qc.csv
python preproc_scripts/final/make_random_split.py --base_split_dp data/split/nist20_qtof_cid_safe19659_random_base --qc_csv data/proc/nist20_qtof_cid_safe19659/cohort_qc.csv --mol_df data/proc/nist20_qtof_cid_safe19659/mol_df.pkl --out_split_dp data/split/nist20_qtof_cid_safe19659_qcv1_trainonly --target_count 19659 --target_molecules 2274 --seed 42
python preproc_scripts/prepare_scaffold_split.py --source-split data/split/nist20_qtof_cid_safe19659_qcv1_trainonly --mol-df data/proc/nist20_qtof_cid_safe19659/mol_df.pkl --output-split data/split/nist20_qtof_cid_safe19659_scaffold60_20_20_seed42 --seed 42
```

QC metrics are generated locally from the licensed spectra and tracked depth-3 candidate DAGs. The cohort-wide QC rule first fixes exactly 19,659 spectra from 2,274 molecular connectivity groups; only then are the molecule-disjoint random and scaffold-disjoint partitions generated from the same `cohort_ids.csv`. The historical `qcv1_trainonly` directory name is retained solely as the locked artifact identifier. No test spectrum is used for fitting, early stopping, hyperparameter selection, or ablation selection.

## Training

Set portable paths when the repository or run directory is elsewhere:

```bash
export FERA_MS_ROOT="$PWD"
export FERA_MS_RUNS_DIR="$PWD/runs"
export MS2_GLOBAL_SEED=42
python train/train.py base
python train/train.py control
python train/train.py refinement
python train/train.py evaluation
```

`python train/train.py all` runs the same four top-level phases: base → control → refinement → evaluation. Inputs come from `config/train.yml`; runtime configurations and checkpoints are created under `$FERA_MS_RUNS_DIR`. The repository does not ship checkpoints.

Three paired seeds for the molecule-disjoint and scaffold-disjoint protocols:

```bash
bash train/run_molecule_disjoint_three_seeds.sh
bash train/run_scaffold_disjoint_three_seeds.sh
```

Both scripts set `MS2_GLOBAL_SEED` to 42, 43, and 44 and archive each completed seed beneath `$FERA_MS_RUNS_DIR/experiments/`. `MS2_SPLIT_DP` selects the scaffold split without changing the locked scientific configuration.

## Evaluation

The mainline command is `python train/train.py evaluation`. With a completed seed directory, additional formal evaluations are:

```bash
python test/evaluate_chun.py --seed-dir runs/experiments/molecule_disjoint_three_seeds/seed_42 --seed 42
python test/evaluate_ace_perturbation.py --seed-dir runs/experiments/molecule_disjoint_three_seeds/seed_42 --seed 42 --ace-mode median
python test/evaluate_ace_perturbation.py --seed-dir runs/experiments/molecule_disjoint_three_seeds/seed_42 --seed 42 --ace-mode shuffled
python test/evaluate_metric_robustness.py --seed-dir runs/experiments/molecule_disjoint_three_seeds/seed_42 --seed 42
python test/run_candidate_space_coverage.py
```

Each evaluator reads model artifacts from the supplied seed directory and writes under that directory or `runs/experiments/`; it does not train or select a checkpoint on the test set. Median-ACE replacement computes the median from the training loader unless a locked training value is explicitly supplied with `--ace-median`; it never falls back to the test distribution.

For molecular identification, first build the licensed/local PubChem candidate pools with `test/build_retrieval_candidates.py`, then run:

```bash
python test/run_molecular_retrieval.py --splits random scaffold --seeds 42 43 44
```

The builder derives target membership directly from the two tracked split contracts and uses the repository-local `test/retrieval/candidate_pool.py` implementation for structure validation, connectivity deduplication, true-target injection, and Morgan ranking. It has no dependency on a sibling or historical checkout. Set `FERA_MS_RETRIEVAL_ROOT` if the candidate pools are outside the default run directory. Candidate caches and retrieval outputs are intentionally not versioned.

Statistical inference follows the manuscript protocol. Spectrum-prediction metrics are averaged across the three seeds before 20,000 molecule-clustered bootstrap replicates, with one Holm correction over the required 32 comparisons. Retrieval uses 20,000 paired molecule bootstrap replicates and one independent Holm correction over the required 48 comparisons (3 baselines x 2 splits x 4 metrics x 2 aggregations). Both analysis scripts reject incomplete comparison families instead of silently correcting a smaller set.

## Ablations

Global-only CE (global ACE embedding retained; local/multi-level neural ACE controls disabled):

```bash
bash ablation_studies/fera_ms_global_ace_ablation/run_all_seeds.sh --dry-run
bash ablation_studies/fera_ms_global_ace_ablation/run_all_seeds.sh
```

The first command is a non-training wiring check. The second performs the formal seeds 42/43/44 run.

The two rendering-component controls, candidate-reranker control, and spectrum-allocator control are:

```bash
bash ablation_studies/fera_ms_panelb_ablation/scripts/run_remaining_panelb_ablation.sh
bash ablation_studies/fera_ms_core_ablation/scripts/run_no_candidate_reranker.sh
bash ablation_studies/fera_ms_core_ablation/scripts/run_no_spectrum_allocator.sh
```

These scripts preserve the locked split, seed, upstream checkpoints, losses, and evaluation metrics; only their documented ablation switch is changed.

## Baselines

Complete baseline runtime sources are retained under `baseline_rebuild/baseline/source/`, separately from the FERA-MS `ms2spectra` package. NEIMS-ACE, MassFormer-ACE, FraGNNet-D3-ACE, ICEBERG-ACE and GrAFF-MS use the independent `fragnnet` runtime with model-specific locked configs; FIORA retains its own MIT-licensed source and requires its external official weight. Licensed data, checkpoints, predictions and results are not included. See `baseline_rebuild/baseline/README.md`, `docs/BASELINE_PROVENANCE.md` and `THIRD_PARTY_NOTICES.md`.

## Citation and license

Citation metadata are in `CITATION.cff`; no DOI, journal metadata, ORCID, or preferred publication citation is asserted. FERA-MS-authored code is licensed under BSD-3-Clause. The separately retained baseline source remains governed by its own preserved licenses and notices; see `LICENSE`, `THIRD_PARTY_NOTICES.md`, `docs/SOURCE_PROVENANCE.md` and `docs/BASELINE_PROVENANCE.md`. NIST20 remains governed by its separate license and is not distributed.
