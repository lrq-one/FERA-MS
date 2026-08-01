# Manuscript baselines

This directory contains the complete local source implementations, locked
random/scaffold configurations, FERA-MS adapters, launchers, checkpoint
selection, prediction export, evaluation and aggregation code used for the
manuscript baselines. The independent baseline package is
`source/fragnnet/`; it is not `code/src/ms2spectra/` and the two packages must
not be interchanged.

No NIST20 data, record-level split CSV, MAGMa/DAG cache, PubChem cache,
checkpoint, prediction or result table is distributed. Generate those inputs
locally from a licensed NIST20 export as described in the root documentation.

## Provenance and executable routes

| Baseline | Manuscript name | Upstream repository | Exact commit/tag | Upstream/license evidence retained | Executed source | FERA-MS changes | Configs and seeds |
|---|---|---|---|---|---|---|---|
| NEIMS | NEIMS-ACE | `https://github.com/brain-research/deep-molecular-massspec` | Local execution snapshot retained | `source/fragnnet/LICENSE` (BSD-2-Clause) | `source/fragnnet/src/fragnnet/model.py`, `pl_model.py`, `dataset.py` | ACE input/config and common export/evaluation adapter | `neims/configs/locked.yml` |
| MassFormer | MassFormer-ACE | `https://github.com/Roestlab/massformer` | Local execution snapshot retained | BSD-2-Clause; Microsoft MIT header retained in `algos.pyx` | `source/fragnnet/src/fragnnet/massformer/` plus shared runtime | ACE input/config and common export/evaluation adapter | `massformer/configs/locked.yml` |
| FraGNNet | FraGNNet-D3-ACE | Historical local remote `https://github.com/lrq-one/fragnnet-main` | Complete local execution snapshot retained | `source/fragnnet/LICENSE` (BSD-2-Clause) | `source/fragnnet/src/fragnnet/frag/`, `model.py`, `pl_model.py`, `dataset.py` and utilities | Depth-3/ACE locked configuration and common protocol adapter | `fragnnet_depth_three/configs/locked.yml` |
| ICEBERG | ICEBERG-ACE | `https://github.com/coleygroup/ms-pred` | Formal-run source snapshot retained | `source/fragnnet/LICENSE` (BSD-2-Clause) | `source/fragnnet/src/fragnnet/iceberg/` plus shared runtime | ACE cohort/config; formal source difference recorded under `source/fragnnet/patches/` | `iceberg/configs/locked.yml` |
| GrAFF-MS | GrAFF-MS | `https://github.com/murphy17/graff-ms` | Local execution snapshot retained | `source/fragnnet/LICENSE` (BSD-2-Clause) | `source/fragnnet/src/fragnnet/graff/` plus shared runtime | MAGMa annotation/cohort adapter and common export/evaluation adapter | `graff_ms/configs/locked.yml` |
| FIORA | FIORA zero-shot control | `https://github.com/BAMeScience/fiora` | Local package version 1.0.1 retained | `source/fiora/LICENSE` (MIT) | `source/fiora/fiora/` | `fiora/build_inputs.py`, launcher and 0.01-Da evaluator | Zero-shot model; seed is not applicable |

Every trained baseline retains its native model class and configured objective.
The shared elements are the locked cohort partitions, ACE-bearing inputs,
checkpoint selection on validation data and final evaluation protocol. ICEBERG
is selected by `model_type: iceberg_inten` in the same independent baseline
runtime; there is no second `iceberg_core` source repository in this release.

Full file-level evidence is in `docs/BASELINE_PROVENANCE.md` and
`THIRD_PARTY_NOTICES.md`.

## Install and preflight

Install the two retained source packages as needed:

```bash
pip install -e baseline_rebuild/baseline/source/fragnnet
pip install -e baseline_rebuild/baseline/source/fiora
```

Run the non-training preflight for all models, or select one model:

```bash
python baseline_rebuild/baseline/tools_local/check_local_runtime.py --model all
python baseline_rebuild/baseline/tools_local/check_local_runtime.py --model fragnnet_d3
bash baseline_rebuild/baseline/check_package_ready.sh
```

Add `--require-data` to verify all licensed local input paths. The preflight
does not start training.

## Local data and output roots

Set the processed-data and output roots. Locked config paths beginning with
`data/` are resolved underneath `FERA_MS_BASELINE_DATA_DIR`; scientific config
values are not rewritten.

```bash
export FERA_MS_BASELINE_DATA_DIR=/path/to/local/fera-ms-data
export FERA_MS_BASELINE_OUTPUT_DIR=/path/to/new/baseline-runs
```

The data root must contain:

```text
proc/nist20_qtof_cid_safe19659/{spec_df.pkl,mol_df.pkl,ann_df.pkl}
split/nist20_qtof_cid_safe19659_qcv1_trainonly/{train,val,test}_ids.csv
split/nist20_qtof_cid_safe19659_scaffold60_20_20_seed42/{train,val,test}_ids.csv
frag/nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1/dags/
magma/gen/nist20_qtof_cid_safe19659_mhp_qtof_cid_v1/
```

GrAFF-MS additionally needs the locally generated
`ann_df_magma_fragbits_graff_nonempty.pkl`. None of these files are tracked.

## Training and spectrum evaluation

One exact locked job:

```bash
python baseline_rebuild/baseline/tools_local/run_formal_one.py \
  --model fragnnet_d3 --split random --seed 42
```

Valid model values are `neims`, `massformer`, `fragnnet_d3`, `iceberg` and
`graff_ms`; split is `random` or `scaffold`; seed is 42, 43 or 44. The runner
uses the tracked config, trains, selects the validation-best checkpoint,
exports test predictions and writes metrics beneath
`$FERA_MS_BASELINE_OUTPUT_DIR/formal_v1/<split>/<model>/seed<seed>/`.

All three seeds for one model:

```bash
bash baseline_rebuild/baseline/neims/run_all.sh random
bash baseline_rebuild/baseline/massformer/run_all.sh scaffold
bash baseline_rebuild/baseline/fragnnet_depth_three/run_all.sh random
bash baseline_rebuild/baseline/iceberg/run_all.sh random
bash baseline_rebuild/baseline/graff_ms/run_all.sh random
```

The full configured matrix and aggregation are launched with:

```bash
python baseline_rebuild/baseline/tools_local/run_formal_matrix.py
python baseline_rebuild/baseline/tools_local/aggregate_formal_results.py
```

## Fixed-pool molecular identification

The retained fixed-pool adapters are:

- `tools_local/build_retrieval_public_plan.py` for locally generated
  candidate structures, query spectra and DAGs;
- `tools_local/run_baseline_retrieval_formal.py` for NEIMS,
  MassFormer and FraGNNet-D3 checkpoint inference/ranking;
- `source/fragnnet/scripts/ms2c/` for the package-native FraGNNet/ICEBERG
  molecular-identification path;
- `tools_local/aggregate_formal_results.py` for generated-run summaries.

Run the scorer/ranker self-test without data or checkpoints:

```bash
python baseline_rebuild/baseline/tools_local/run_baseline_retrieval_formal.py --self-test
```

The candidate pools and record-level query/split files are regenerated locally
and are not included in Git.

## FIORA zero-shot control

Create a local FIORA reference/input table, then supply the external official
weight. The model used historically was `fiora_OS_v1.0.0.pt`, SHA-256
`273807127861ca0ac8404962f111ff8628ba02e6beb5d1142d8772ced07443a0`.

```bash
python baseline_rebuild/baseline/fiora/build_inputs.py \
  --processed-dir /path/to/proc/nist20_qtof_cid_safe19659 \
  --split-dir /path/to/split/nist20_qtof_cid_safe19659_qcv1_trainonly \
  --output /path/to/fiora_library.csv

export FERA_MS_FIORA_INPUT=/path/to/fiora_test_queries.csv
export FERA_MS_FIORA_REFERENCE=/path/to/fiora_library.csv
export FERA_MS_FIORA_MODEL=/path/to/fiora_OS_v1.0.0.pt
bash baseline_rebuild/baseline/fiora/run_final.sh
```

Generated checkpoints, predictions, logs and tables are ignored and must not
be committed.
