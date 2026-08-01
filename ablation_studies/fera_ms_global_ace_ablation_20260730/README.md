# Global-only CE control

This locked control retains ACE at the candidate-scoring MLP while disabling the local and multi-level neural ACE controllers. The tracked seed-42/43/44 configs and isolated pipeline cover V2A, retained control, R146–R160, LightGBM reranking, residual allocation, validation selection, and final test evaluation.

```bash
bash ablation_studies/fera_ms_global_ace_ablation_20260730/scripts/run_formal_preflight.sh
bash ablation_studies/fera_ms_global_ace_ablation_20260730/run_all_seeds.sh --dry-run
bash ablation_studies/fera_ms_global_ace_ablation_20260730/run_all_seeds.sh
```

The dry run checks wiring without training. Generated checkpoints, evaluation tables, audit logs, and aggregate results are excluded from the repository.
