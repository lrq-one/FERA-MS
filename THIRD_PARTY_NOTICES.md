# Third-party notices and baseline provenance

This file records the implementations actually used by the FERA-MS baseline
workflow. The complete baseline implementations integrated into the original
FERA-MS codebase are retained. This is not a claim that the FERA-MS authors
created the underlying methods, nor that a future root license supersedes the
licenses of externally derived files.

The historical baseline run manifests contain `code_commit: null`, and the
archived package-compatible source snapshots do not contain `.git` metadata.
Consequently, the exact source commit used for every reported baseline cannot
currently be reconstructed from local evidence. Each affected baseline is
therefore marked **BLOCKED** for the pre-merge provenance gate.

| Model | Method/upstream repository | Exact source used for the reported run | License evidenced locally | Source retained in FERA-MS | FERA-MS adapter/config | Manuscript citation | Status |
|---|---|---|---|---|---|---|---|
| NEIMS | `https://github.com/brain-research/deep-molecular-massspec` | Package-compatible `fragnnet-main-official` snapshot; exact commit not recorded | BSD-2-Clause for the integrated FraGNNet snapshot | Reimplementation integrated in `code/src/ms2spectra/model.py` and `training.py`; no Google checkout is bundled | `baseline_rebuild/baseline/neims/` | Wei et al., *Rapid prediction of electron-ionization mass spectrometry using neural networks*, ACS Central Science (2019) | **BLOCKED** — exact snapshot commit unavailable |
| MassFormer | `https://github.com/Roestlab/massformer` | Package-compatible `fragnnet-main-official` snapshot; exact commit not recorded | BSD-2-Clause for the integrated FraGNNet snapshot | Adapted implementation in `code/src/ms2spectra/massformer/` | `baseline_rebuild/baseline/massformer/` | Young, Wang and Röst, *Tandem mass spectrum prediction for small molecules using graph transformers*, Nature Machine Intelligence (2024) | **BLOCKED** — exact snapshot commit unavailable |
| FraGNNet-D3 | Local source remote recorded as `https://github.com/lrq-one/fragnnet-main`; public upstream URL/commit was not recorded in the run artifact | Package-compatible `fragnnet-main-official` snapshot; exact commit not recorded | BSD-2-Clause; original notice preserved in `licenses/FraGNNet-BSD-2-Clause.txt` | Integrated throughout `code/src/ms2spectra/`, including `frag/`, `model.py`, `data.py`, `training.py`, `utils/`, and `workflow.py` | `baseline_rebuild/baseline/fragnnet_d3/` | Young et al., *FraGNNet: A Deep Probabilistic Model for Tandem Mass Spectrum Prediction*, TMLR (2025) | **BLOCKED** — exact snapshot commit and public source URL unavailable |
| GrAFF-MS | `https://github.com/murphy17/graff-ms` | Package-compatible `fragnnet-main-official` snapshot; exact commit not recorded | BSD-2-Clause for the integrated FraGNNet snapshot | Adapted implementation in `code/src/ms2spectra/graff/` | `baseline_rebuild/baseline/graff_ms/` | Murphy et al., *Efficiently predicting high resolution mass spectra with graph neural networks*, ICML/PMLR (2023) | **BLOCKED** — exact snapshot commit unavailable |
| ICEBERG | `https://github.com/coleygroup/ms-pred` | Separate `fragnnet-main-iceberg-core-audit` snapshot; exact commit not recorded | BSD-2-Clause for the archived package-compatible snapshot | Integrated implementation in `code/src/ms2spectra/iceberg/`; the historical core-audit wrapper remains separately configurable | `baseline_rebuild/baseline/iceberg/` | Goldman, Li and Coley, *Generating molecular fragmentation graphs with autoregressive neural networks*, Analytical Chemistry (2024) | **BLOCKED** — exact snapshot commit unavailable |
| FIORA | `https://github.com/BAMeScience/fiora` | Local `fiora-main` package reports version 1.0.1; exact Git commit not retained | MIT; local source snapshot preserved its upstream LICENSE during the audit | No FIORA source file is copied into this repository; the official CLI is invoked externally | `baseline_rebuild/baseline/fiora/run_final.sh` | Nowatzky et al., *FIORA: Local neighborhood-based prediction of compound mass spectra from single fragmentation events*, Nature Communications (2025) | **BLOCKED** — exact source commit unavailable |

## License boundaries

- The BSD-2-Clause notice in `licenses/FraGNNet-BSD-2-Clause.txt` applies to
  code inherited or adapted from the integrated FraGNNet source snapshot.
- FERA-MS-authored orchestration, CE/formula/H extensions, refinement stages,
  release tooling, and documentation are intended for BSD-3-Clause, but that
  license is not activated until file-level provenance is signed off.
- FIORA is not vendored. Its separately installed package remains under its
  own MIT license.
- Method citations do not establish source-code provenance. The unresolved
  commit fields above must be filled from retained run records or confirmed by
  the authors before merging the release branch.
