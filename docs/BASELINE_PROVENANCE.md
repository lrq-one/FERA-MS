# Baseline source and execution provenance

Audit date: 2026-08-01
Branch compared: `main` → `paper-release`

## Separation invariant

The manuscript baseline runtime is the independent Python package at
`baseline_rebuild/baseline/source/fragnnet/src/fragnnet`. FERA-MS itself is the
separate package at `code/src/ms2spectra`. The baseline launchers put only the
former on `PYTHONPATH`; no baseline launcher imports `ms2spectra` as a
replacement for FraGNNet-D3 or another baseline.

ICEBERG-ACE is not a second source repository. It selects the ICEBERG module in
the same independent `fragnnet` baseline package through `model_type:
iceberg_inten` and the locked ACE cohort configuration. The two source files
used by the formal run are canonical under
`source/fragnnet/src/fragnnet/iceberg/`; the excluded experimental CE-gate
difference is recorded at
`source/fragnnet/patches/iceberg_remove_experimental_ce_gate.patch`.

## Executable source chains

All five trained baselines enter through
`source/fragnnet/scripts/run_pl_model_fit.py`, which calls
`fragnnet.runner.init_run`. `runner.py` loads the locked YAML, constructs the
model-specific dataset/Lightning module, trains, and evaluates. The
FERA-MS-authored `tools_local/run_formal_one.py` resolves licensed data paths,
selects the validation-best checkpoint, calls `export_formal_test.py`, and
writes new outputs outside the tracked source tree.

| Baseline | Exact model/dataset source selected by `runner.py` | Shared executed source | Local adaptation/config | External artifacts |
|---|---|---|---|---|
| NEIMS-ACE | `src/fragnnet/model.py` (`NeimsModel`), `pl_model.py` (`NeimsPL`), `dataset.py` (`SpecMolDataset`) | `loss.py`, `form_embedder.py`, `utils/**`, `runner.py` | `neims/configs/{random,scaffold}/seed_*.yml`, common formal/export/retrieval tools | processed NIST20 cohort, split IDs, generated checkpoint |
| MassFormer-ACE | `src/fragnnet/massformer/{model,pl_model,nn_utils,data_utils}.py`, `algos.pyx`, shared `SpecMolDataset` | same shared runtime | `massformer/configs/{random,scaffold}/seed_*.yml` | processed cohort, split IDs, generated checkpoint |
| FraGNNet-D3-ACE | `src/fragnnet/model.py` (`FragGNNModel`), `pl_model.py` (`FragGNNPL`), `dataset.py` (`SpecMolFragDataset`), `frag/compute_frags.pyx` | formula, fragment and spectrum utilities | `fragnnet_d3/configs/{random,scaffold}/seed_*.yml` | processed cohort, depth-3 DAG cache, split IDs, generated checkpoint |
| ICEBERG-ACE | `src/fragnnet/iceberg/{model,pl_model,dataset,fragmentation}.py`, `iceberg/common/**`, `iceberg/nn_utils/**` | shared runtime; `model_type: iceberg_inten` | `iceberg/configs/{random,scaffold}/seed_*.yml`; formal two-file variant described above | processed cohort, MAGMa cache, split IDs, generated checkpoint |
| GrAFF-MS | `src/fragnnet/graff/{model,pl_model,dataset,data_utils,nn_utils}.py` | shared loss/formula/runtime utilities | `graff_ms/configs/{random,scaffold}/seed_*.yml`, local MAGMa-to-GrAFF annotation input | processed cohort, generated GrAFF annotation pickle, split IDs, generated checkpoint |
| FIORA | `source/fiora/fiora/GNN/**`, `MOL/**`, `MS/**`, `IO/**`, `cli/predict.py` | native FIORA package only | `fiora/build_inputs.py`, `run_final.sh`, `eval_fiora_against_library_csv.py` | licensed cohort and external `fiora_OS_v1.0.0.pt` |

Package-native prediction/retrieval exporters are retained under
`source/fragnnet/scripts/` and `source/fragnnet/scripts/ms2c/`. FERA-MS fixed
candidate-pool orchestration is retained under `baseline_rebuild/baseline/tools_local/`.

## Source snapshot fidelity

The retained `fragnnet` Python/Cython runtime was compared file-by-file with
the local manuscript execution snapshot. Every file matches byte-for-byte
except the documented selections below:

1. `iceberg/model.py` and `iceberg/pl_model.py` match the formal ICEBERG run
   snapshot rather than the main snapshot containing an experimental CE-gate.
2. `pyproject.toml` changes the inconsistent `MIT` metadata string to
   `BSD-2-Clause`, matching the unmodified bundled `LICENSE`.

The retained FIORA Python files match the local FIORA 1.0.1 snapshot
byte-for-byte. Model `.pt` files were deliberately not copied.

The local historical FraGNNet remote branch exposes commit
`f86390399f1660219479011937d0386786c5b933`. File hashing confirms that several
low-level files match it, but the executed `model.py`, `pl_model.py`, and
`runner.py` snapshot matches no available revision of those paths. This commit
cannot be reported as the exact manuscript baseline commit.

## `main` versus `paper-release` baseline file classification

No baseline-related deletion remains unclassified.

| Files changed or removed from `main` | Classification | Release action |
|---|---|---|
| `runs/**/_frozen_inputs/models/{neims,massformer,fragnnet_d3}/random/seed_*/config.yml` | `REQUIRED_CONFIG` | Restored as canonical `baseline_rebuild/baseline/<model>/configs/random/seed_*.yml`; formal-run versions retain JSS/selection settings and only private output paths were made portable |
| `runs/**/_frozen_inputs/models/{neims,massformer,fragnnet_d3}/scaffold/seed_*/config.yml` | `REQUIRED_CONFIG` | Restored as canonical `configs/scaffold/seed_*.yml` |
| Local formal ICEBERG and GrAFF random/scaffold configs | `REQUIRED_CONFIG` | Canonical copies added under their model directories |
| Local formal `fragnnet` and FIORA runtime snapshots, previously outside Git | `REQUIRED_SOURCE` and `REQUIRED_PROVENANCE` | Filtered source-only canonical copies restored under `baseline_rebuild/baseline/source/`; data/results/binaries omitted; original licenses retained |
| `runs/.../legacy_filter_candidates_source.py` | `DUPLICATE` | Not restored: the complete executable implementation is retained at `source/fragnnet/preproc_scripts/pubchem_ms2c/02_prepare_ms2c_candidates.py` |
| `code/src/ms2spectra/frag/compute_frags.c`, compiled `.so`, `massformer/algos.c`, compiled `.so` | `GENERATED_OUTPUT` | Not restored; tracked `.pyx` sources build these locally |
| supervisor locks, success/readiness marker files and `.interrupted_before_resume` | `GENERATED_OUTPUT` | Not restored |
| `train/experiments/**/code_backup/**` | `DUPLICATE` / `OBSOLETE` | Not restored; these were historical FERA-MS mainline backups, not baseline source; canonical training/evaluation files remain under `train/_impl/` and `test/` |
| Historical result, checkpoint, prediction, log and cache directories | `GENERATED_OUTPUT` | Not restored and ignored by `.gitignore` |

## Config invariants

For each trained baseline, both partitions contain exact locked configs for
seeds 42, 43 and 44. Preflight asserts:

- the YAML seed equals its filename;
- random configs use `qcv1_trainonly`;
- scaffold configs use `scaffold60_20_20_seed42`;
- the expected `model_type` is selected;
- required source and, optionally, local data paths exist.

No test record is used for training, early stopping or checkpoint selection.

## Upstream and license blockers

| Baseline | Upstream URL | Commit/tag | License evidence | Remaining blocker |
|---|---|---|---|---|
| NEIMS | `https://github.com/brain-research/deep-molecular-massspec` | Unknown | Snapshot-level BSD-2-Clause only | Exact upstream commit and license mapping for the adapted files |
| MassFormer | `https://github.com/Roestlab/massformer` | Unknown | Snapshot-level BSD-2-Clause; Microsoft MIT header in `algos.pyx` | Exact upstream commit and license mapping |
| FraGNNet-D3 | Historical local remote `https://github.com/lrq-one/fragnnet-main`; original upstream URL unknown | Unknown | BSD-2-Clause preserved | Exact executed commit and original public upstream URL |
| ICEBERG | `https://github.com/coleygroup/ms-pred` | Unknown | Snapshot-level BSD-2-Clause only | Exact upstream commit and license mapping |
| GrAFF-MS | `https://github.com/murphy17/graff-ms` | Unknown | Snapshot-level BSD-2-Clause only | Exact upstream commit and license mapping |
| FIORA | `https://github.com/BAMeScience/fiora` | Package version 1.0.1; commit unknown | MIT preserved | Exact source commit; official model weight remains external |

The runtime source routes are now complete, but these provenance fields prevent
`BASELINE_RELEASE_READY` and a root BSD-3-Clause license until confirmed.
