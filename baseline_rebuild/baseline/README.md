# Baseline reproduction

This directory keeps the locked configurations and orchestration used for NEIMS-ACE, MassFormer-ACE, FraGNNet-D3-ACE, GrAFF-MS, ICEBERG, and the FIORA zero-shot control. It intentionally excludes NIST20 data, checkpoints, third-party repositories, predictions, logs, and result tables.

Provide compatible external source checkouts and the same locally processed cohort:

```bash
export FERA_MS_BASELINE_ROOT="$PWD/baseline_rebuild/baseline"
export FERA_MS_BASELINE_SOURCE=/path/to/compatible/fragnnet-baseline-source
export FERA_MS_ICEBERG_SOURCE=/path/to/compatible/iceberg-source
export FERA_MS_BASELINE_OUTPUT_DIR=/path/to/baseline-runs
bash baseline_rebuild/baseline/check_package_ready.sh
```

The external checkouts must provide `benchmark_audit/run_one_pl_seed.sh` and accept the tracked YAML configs. Run a family with, for example, `bash baseline_rebuild/baseline/neims/run_all.sh`; analogous launchers exist for MassFormer, FraGNNet-D3, GrAFF-MS, and ICEBERG. Each launcher uses the reference/alpha/beta configs corresponding to seeds 42/43/44.

For FIORA, set `FERA_MS_FIORA_WORKSPACE` to a local workspace containing the licensed safe-cohort library/query CSVs and evaluation script, then run `bash baseline_rebuild/baseline/fiora/run_final.sh`.

`tools_local/run_formal_matrix.py` orchestrates the formal random/scaffold matrix, and `tools_local/aggregate_formal_results.py` aggregates newly generated outputs. Molecular-identification baseline tools require the locally generated PubChem candidate plan under `$FERA_MS_RUNS_DIR/experiments/molecular_retrieval/`.
