# FERA-MS pipeline

| Stage | Entrypoint | Configuration | Input | Output | Training and split use |
|---|---|---|---|---|---|
| MSP/MOL parsing | `preproc_scripts/prepare_dataframes.py` | CLI | licensed NIST20 MSP/MOL | `data/df/` | no training |
| Standardization | `preproc_scripts/prepare_processed_data.py` | CLI | parsed table | processed spectrum/molecule/annotation pickles | no training |
| Cohort and random split | `preproc_scripts/prepare_split.py`, `preproc_scripts/final/make_random_split.py` | CLI | processed tables, DAG eligibility, train QC | molecule-disjoint IDs | QC affects training only; validation/test are fixed |
| Scaffold split | `preproc_scripts/prepare_scaffold_split.py` | CLI, seed 42 | fixed cohort plus molecular scaffolds | 60/20/20 scaffold-disjoint IDs | no model fitting |
| Fragment candidates | `preproc_scripts/prepare_dag_features.py` | CLI, depth 3 | processed molecules/spectra | compressed DAG cache | no model fitting |
| Backbone | `python train/train.py base` | `config/train.yml` → `runs/_config/` | processed data, split, DAGs | GINE/cut-chemistry checkpoint | validation selects checkpoint; test is evaluated only after selection |
| Retained control | `python train/train.py control` | generated control config | frozen/selected base stage | control continuation checkpoint | initialized from base; validation only for selection |
| CE/formula/H refinement | `python train/train.py refinement` | locked refinement settings in `config/train.yml` | control checkpoint | formula-to-peak refinement checkpoints | upstream initialization retained; validation selects each stage |
| Local rendering and gate | same refinement entry | locked renderer/gate switches | neural candidate scores | rendered peak entries | trained within the locked refinement chain |
| Candidate reranker | `train/_impl/refinement_steps/candidate_reranker.py` via refinement | locked LightGBM settings | frozen final-peak candidate features | LightGBM reranker | fitted on training data, selected on validation |
| Residual allocator | `train/_impl/refinement_steps/spectrum_allocator.py` via refinement | locked allocator settings | frozen final-peak model plus reranker | allocator checkpoint | upstream model/reranker frozen; validation selects allocator |
| Locked evaluation | `python train/train.py evaluation` | materialized config and completed artifacts | selected final-peak, reranker and allocator artifacts | per-spectrum and aggregate metrics | test is evaluation-only |

The formal chain is therefore: licensed NIST20 export → processed 19,659-spectrum cohort → molecule- or scaffold-disjoint split → depth-3 DAGs → structural GINE/cut chemistry → retained-control continuation → formula/H and ACE refinement → local m/z rendering and rendered-entry gate → LightGBM reranking → residual allocation → locked evaluation. CHUN, ACE perturbation, robustness, retrieval, and ablation scripts consume completed artifacts and do not alter the main model.
