# Candidate-reranker ablation

This directory retains the valid **without chemical candidate reranking** control. It reuses each completed mainline final peak distillation checkpoint, removes the candidate reranker contribution by using the locked zero-weight route, retrains the spectrum allocator with the mainline hyperparameters, selects on validation, and evaluates test once.

```bash
bash ablation_studies/fera_ms_core_ablation/scripts/preflight.sh
bash ablation_studies/fera_ms_core_ablation/scripts/run_no_candidate_reranker.sh
```

The script expects completed random-split seed directories under `$FERA_MS_RUNS_DIR/experiments/`. Historical neural-ablation scripts that did not activate their intended architecture changes were removed from the release.
