# Candidate-reranker and spectrum-allocator ablations

This directory retains the valid **without chemical candidate reranking** control. It reuses each completed mainline final peak distillation checkpoint, removes the candidate reranker contribution by using the locked zero-weight route, retrains the spectrum allocator with the mainline hyperparameters, selects on validation, and evaluates test once.

```bash
bash ablation_studies/fera_ms_core_ablation/scripts/preflight.sh
bash ablation_studies/fera_ms_core_ablation/scripts/run_no_candidate_reranker.sh
bash ablation_studies/fera_ms_core_ablation/scripts/run_no_spectrum_allocator.sh
```

The no-reranker control sets the reranker contribution to zero and retrains the allocator. The no-allocator control retains the validation-selected candidate reranker and evaluates it directly, without loading or applying the spectrum allocator. Both scripts expect completed random-split seed directories under `$FERA_MS_RUNS_DIR/experiments/`. Historical neural-ablation scripts that did not activate their intended architecture changes were removed from the release.
