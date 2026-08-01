# FERA-MS pipeline

| Stage | Entrypoint | Configuration | Input | Output | Training and split use |
|---|---|---|---|---|---|
| MSP/MOL parsing | `preproc_scripts/01_prepare_df.py` | CLI | licensed NIST20 MSP/MOL | `data/df/` | no training |
| Standardization | `preproc_scripts/02_prepare_proc.py` | CLI | parsed table | processed spectrum/molecule/annotation pickles | no training |
| Cohort and random split | `preproc_scripts/04_prepare_split.py`, `preproc_scripts/final/16_make_random_qcv1_trainonly_split.py` | CLI | processed tables, DAG eligibility, train QC | molecule-disjoint IDs | QC affects training only; validation/test are fixed |
| Scaffold split | `preproc_scripts/05_prepare_scaffold_split_safe19659.py` | CLI, seed 42 | fixed cohort plus molecular scaffolds | 60/20/20 scaffold-disjoint IDs | no model fitting |
| Fragment candidates | `preproc_scripts/03_prepare_dag_feats.py` | CLI, depth 3 | processed molecules/spectra | compressed DAG cache | no model fitting |
| Backbone | `python train/train.py base` | `config/train.yml` → `runs/_config/` | processed data, split, DAGs | GINE/cut-chemistry checkpoint | validation selects checkpoint; test is evaluated only after selection |
| Retained control | `python train/train.py control` | generated control config | frozen/selected base stage | control continuation checkpoint | initialized from base; validation only for selection |
| CE/formula/H refinement | `python train/train.py refinement` | locked refinement settings in `config/train.yml` | control checkpoint | R146–R160 checkpoints | upstream initialization retained; validation selects each stage |
| Local rendering and gate | same refinement entry | locked renderer/gate switches | neural candidate scores | rendered peak entries | trained within the locked refinement chain |
| Candidate reranker | `train/_impl/refinement_steps/candidate_reranker.py` via refinement | locked LightGBM settings | frozen R160 candidate features | LightGBM reranker | fitted on training data, selected on validation |
| Residual allocator | `train/_impl/refinement_steps/spectrum_allocator.py` via refinement | locked allocator settings | frozen R160 plus reranker | allocator checkpoint | upstream model/reranker frozen; validation selects allocator |
| Locked evaluation | `python train/train.py evaluation` | materialized config and completed artifacts | selected R160/R172D/R184B artifacts | per-spectrum and aggregate metrics | test is evaluation-only |

The formal chain is therefore: licensed NIST20 export → processed 19,659-spectrum cohort → molecule- or scaffold-disjoint split → depth-3 DAGs → structural GINE/cut chemistry → retained-control continuation → formula/H and ACE refinement → local m/z rendering and rendered-entry gate → LightGBM reranking → residual allocation → locked evaluation. CHUN, ACE perturbation, robustness, retrieval, and ablation scripts consume completed artifacts and do not alter the main model.
