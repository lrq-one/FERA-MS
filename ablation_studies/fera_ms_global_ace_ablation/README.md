# Global-only CE control

This locked control retains ACE at the candidate-scoring MLP while disabling the local and multi-level neural ACE controllers. A single functional configuration factors shared settings from seed and stage overrides. The isolated pipeline covers structural-backbone training, retained-control continuation, formula and collision-energy refinement, peak distillation, LightGBM reranking, residual allocation, validation selection, and final test evaluation.

```bash
bash ablation_studies/fera_ms_global_ace_ablation/scripts/run_formal_preflight.sh
bash ablation_studies/fera_ms_global_ace_ablation/run_all_seeds.sh --dry-run
bash ablation_studies/fera_ms_global_ace_ablation/run_all_seeds.sh
```

The dry run checks wiring without training. Generated checkpoints, evaluation tables, audit logs, and aggregate results are excluded from the repository.
