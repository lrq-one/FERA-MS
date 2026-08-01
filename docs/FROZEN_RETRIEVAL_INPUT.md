# Frozen molecular-identification input

The exact fixed-50 PubChem candidate package used for the paper is not distributed with this release and currently has no public download URL. Consequently, an external user without an author-provided copy can reproduce the retrieval method, but cannot exactly replay the paper's Table 3 candidate identities.

An exact replay requires these five files at the evaluator's locked relative paths (dated directory aliases are also accepted by the code):

- `frozen_manifest/experiment5_fixed50_targets.csv`
- `frozen_manifest/experiment5_fixed50_memberships.csv`
- `inference_ready_pools/experiment5_inference_ready_candidates.csv.gz`
- `query_candidate_manifest_20260723/experiment5_queries_fixed50.csv.gz`
- `query_candidate_manifest_20260723/experiment5_query_candidates_fixed50.csv.gz`

Their SHA-256 values and the final random/scaffold query counts are locked in `config/paper_experiment_identity.json`. `test/run_molecular_retrieval.py` refuses a missing or different package.

`test/build_retrieval_candidates.py` is a live PubChem source-drift audit. Its output is not an exact substitute for the frozen package, even when it produces 50 candidates per target, because PubChem contents and query results can change over time.
