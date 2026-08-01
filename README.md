# FERA-MS

FERA-MS is a fragmentation- and energy-resolved model for predicting tandem mass spectra from molecular structure, precursor type, and absolute collision energy (ACE).

The implementation combines a structural GINE backbone and cut-chemistry features with retained-control continuation, formula/H-aware and multi-level ACE refinement, local m/z rendering, a rendered-entry gate, a LightGBM candidate reranker, and a spectrum-wise residual allocator.

## Repository layout

- `code/src/ms2spectra/`: model, losses, data utilities, fragment generation, and baseline model components.
- `config/train.yml`: locked mainline configuration bundle. Runtime YAML files are generated under `runs/_config/`.
- `preproc_scripts/`: NIST export parsing, processed data construction, random/scaffold splits, and depth-3 DAG construction.
- `train/`: single-seed and three-seed mainline entrypoints.
- `test/`: locked evaluation, CHUN, ACE perturbation, robustness, candidate-space, and molecular-identification analyses.
- `ablation_studies/`: validated global-only CE, m/z-offset, rendered-gate, and no-reranker controls.
- `baseline_rebuild/`: baseline configurations, launchers, and aggregation utilities. The integrated NEIMS, MassFormer, FraGNNet-D3, GrAFF-MS, and ICEBERG implementations remain under `code/src/ms2spectra/`; FIORA is invoked through its separately installed CLI.
- `docs/`: pipeline and reproducibility details.

## Environment and installation

The recorded environment uses Python 3.10, PyTorch 2.3.0, CUDA 12.1-compatible DGL/PyG builds, RDKit 2022.9.4, and Lightning 2.6.x. GPU-specific DGL and PyG wheels may need to be installed from their official wheel indexes before the remaining requirements.

```bash
conda env create -f environment.yml
conda activate fera-ms
pip install -e .
python scripts/check_release.py
```

The editable install compiles `ms2spectra.frag.compute_frags` and `ms2spectra.massformer.algos` from their tracked Cython sources; platform-specific `.so` files are intentionally excluded.

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
python preproc_scripts/01_prepare_df.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --input_format msp+mol --output_format csv --output_dp data/df --output_name nist20_hr
python preproc_scripts/02_prepare_proc.py --df_dp data/df --dsets nist20_hr --proc_dp data/proc/nist20
python preproc_scripts/03_prepare_dag_feats.py --max_depth 3 --frag_dp data/frag/nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1 --proc_dp data/proc/nist20_qtof_cid_safe19659 --prec_types "[M+H]+" --inst_types QTOF --frag_modes CID --ion_modes P --wandb_mode disabled
python preproc_scripts/04_prepare_split.py --split_type random --split_key inchikey_s --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types QTOF --frag_modes CID --ion_modes P --ces ace --proc_dp data/proc/nist20_qtof_cid_safe19659 --frag_dp data/frag/nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1 --split_dp data/split/nist20_qtof_cid_safe19659_random_base
python preproc_scripts/final/16_make_random_qcv1_trainonly_split.py --base_split_dp data/split/nist20_qtof_cid_safe19659_random_base --qc_csv /path/to/licensed_local_train_qc.csv --out_split_dp data/split/nist20_qtof_cid_safe19659_qcv1_trainonly
python preproc_scripts/05_prepare_scaffold_split_safe19659.py --source-split data/split/nist20_qtof_cid_safe19659_qcv1_trainonly --mol-df data/proc/nist20_qtof_cid_safe19659/mol_df.pkl --output-split data/split/nist20_qtof_cid_safe19659_scaffold60_20_20_seed42 --seed 42
```

The QC script only filters the random training partition; validation and test IDs remain unchanged. The exact cohort and QC inputs must come from the same licensed export. No test spectrum is used for fitting, early stopping, hyperparameter selection, or ablation selection.

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

`python train/train.py all` runs the same five-stage sequence. Inputs come from `config/train.yml`; runtime configurations and checkpoints are created under `$FERA_MS_RUNS_DIR`. The repository does not ship checkpoints.

Three paired seeds for the molecule-disjoint and scaffold-disjoint protocols:

```bash
bash train/run_molecule_disjoint_3seeds.sh
bash train/run_scaffold_disjoint_3seeds.sh
```

Both scripts set `MS2_GLOBAL_SEED` to 42, 43, and 44 and archive each completed seed beneath `$FERA_MS_RUNS_DIR/experiments/`. `MS2_SPLIT_DP` selects the scaffold split without changing the locked scientific configuration.

## Evaluation

The mainline command is `python train/train.py evaluation`. With a completed seed directory, additional formal evaluations are:

```bash
python test/evaluate_chun_10ppm.py --seed-dir runs/experiments/molecule_disjoint_3seeds/seed_42 --seed 42
python test/evaluate_ace_perturbation.py --seed-dir runs/experiments/molecule_disjoint_3seeds/seed_42 --seed 42 --ace-mode shuffled
python test/evaluate_metric_robustness.py --seed-dir runs/experiments/molecule_disjoint_3seeds/seed_42 --seed 42
python test/run_candidate_space_coverage.py
python test/run_cumulative_refinement_analysis.py
```

Each evaluator reads model artifacts from the supplied seed directory and writes under that directory or `runs/experiments/`; it does not train or select a checkpoint on the test set.

For molecular identification, first build the licensed/local PubChem candidate pools with `test/build_experiment5_pubchem_candidates.py`, then run:

```bash
python test/run_experiment5_ours.py --splits random scaffold --seeds 42 43 44
```

Set `FERA_MS_RETRIEVAL_ROOT` if the candidate pools are outside the default run directory. Candidate caches and retrieval outputs are intentionally not versioned.

## Ablations

Global-only CE (global ACE embedding retained; local/multi-level neural ACE controls disabled):

```bash
bash ablation_studies/fera_ms_global_ace_ablation_20260730/run_all_seeds.sh --dry-run
bash ablation_studies/fera_ms_global_ace_ablation_20260730/run_all_seeds.sh
```

The first command is a non-training wiring check. The second performs the formal seeds 42/43/44 run.

The two rendering-component controls and the candidate-reranker control are:

```bash
bash ablation_studies/fera_ms_panelb_ablation/scripts/run_remaining_panelb_ablation.sh
bash ablation_studies/fera_ms_core_ablation/scripts/01_run_no_candidate_reranker.sh
```

These scripts preserve the locked split, seed, upstream checkpoints, losses, and evaluation metrics; only their documented ablation switch is changed.

## Baselines

Baseline configs are under `baseline_rebuild/baseline/`. The repository retains the baseline model implementations that are integrated into the original FERA-MS codebase; licensed data, checkpoints, predictions, and nested Git checkouts are not included. The historical baseline wrappers use `FERA_MS_BASELINE_SOURCE` and `FERA_MS_ICEBERG_SOURCE` to identify the package-compatible local source snapshot used for the reported runs. FIORA remains a separately installed command-line package selected with `FERA_MS_FIORA_WORKSPACE`. See the baseline README and `THIRD_PARTY_NOTICES.md` for the exact current provenance limitations.

## Citation and license

Citation metadata are in `CITATION.cff`; no DOI, journal metadata, ORCID, or preferred publication citation is asserted. The FERA-MS license remains pending until the integrated baseline-source provenance audit is closed; see `LICENSE_PENDING.md`, `THIRD_PARTY_NOTICES.md`, and `docs/SOURCE_PROVENANCE.md`. NIST20 and third-party components remain governed by their separate terms.
