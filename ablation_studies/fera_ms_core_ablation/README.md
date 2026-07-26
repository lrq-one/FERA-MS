# FERA-MS core-module ablation

This directory contains the five remaining ablation experiments for the
FERA-MS manuscript.

## Experiments

| ID | Internal name | Formal name | Training scope |
|---|---|---|---|
| A1 | fragment_node_mlp | Fragment-wise node encoder | Full neural pipeline |
| A2 | topology_only_dag | Topology-only fragment DAG | Full neural pipeline |
| A3 | global_molecular_context | Global molecular context | Full neural pipeline |
| A4 | global_ace_only | Global ACE conditioning only | ACE/neural refinement pipeline |
| A5 | no_candidate_reranker | Without chemical candidate reranking | Allocator only |

## Shared protocol

- Random molecule-disjoint and scaffold-disjoint evaluation
- Seeds 42, 43 and 44
- Identical D3/H-transfer/NL candidate cache
- Identical train/validation/test split for the corresponding experiment
- Test set is never used for model selection
- 0.01-Da CBIN and JSS evaluation
- Main results: spectrum-micro mean ± sample standard deviation
- Molecule-macro results retained for supplementary reporting

## Directory policy

- `src/`: ablation-specific Python code
- `scripts/`: entrypoints and audit scripts
- `config/`: fixed experiment definitions
- `runs/`: checkpoints and per-run outputs
- `logs/`: terminal logs
- `results/`: final tables and summaries

The original model checkpoints and original training outputs are read-only.
