# Third-party notices

The baseline implementations used by the manuscript are retained independently
under `baseline_rebuild/baseline/source/`. They are not relicensed by a future
FERA-MS root license and must not be represented as the FERA-MS
`code/src/ms2spectra` implementation.

The local execution snapshots did not contain `.git` metadata and their run
manifests recorded `code_commit: null`. Exact source commits therefore remain
unresolved and are marked **BLOCKED** for the pre-merge provenance gate.

| Model | Upstream repository recorded for the method | Exact executed source | License evidence preserved | Retained path | FERA-MS-authored adapter/config | Manuscript citation | Status |
|---|---|---|---|---|---|---|---|
| NEIMS | `https://github.com/brain-research/deep-molecular-massspec` | NEIMS classes in the local `fragnnet` execution snapshot; exact upstream commit not recorded | Snapshot-level BSD-2-Clause; exact upstream checkout/license mapping not independently recoverable | `baseline_rebuild/baseline/source/fragnnet/` | `baseline_rebuild/baseline/neims/`, `tools_local/` | Wei et al., *Rapid prediction of electron-ionization mass spectrometry using neural networks*, ACS Central Science (2019) | **BLOCKED** — exact upstream commit/license mapping missing |
| MassFormer | `https://github.com/Roestlab/massformer` | `fragnnet.massformer` from the local execution snapshot | Snapshot-level BSD-2-Clause; Microsoft MIT header retained in `algos.pyx`; exact MassFormer commit not recorded | `baseline_rebuild/baseline/source/fragnnet/src/fragnnet/massformer/` | `baseline_rebuild/baseline/massformer/`, `tools_local/` | Young, Wang and Röst, *Tandem mass spectrum prediction for small molecules using graph transformers*, Nature Machine Intelligence (2024) | **BLOCKED** — exact upstream commit/license mapping missing |
| FraGNNet-D3 | Historical local remote: `https://github.com/lrq-one/fragnnet-main`; original public upstream URL not established locally | Complete local execution snapshot | `baseline_rebuild/baseline/source/fragnnet/LICENSE` (BSD-2-Clause; Adamo Young and Fei Wang) | `baseline_rebuild/baseline/source/fragnnet/` | `baseline_rebuild/baseline/fragnnet_d3/`, `tools_local/` | Young et al., *FraGNNet: A Deep Probabilistic Model for Tandem Mass Spectrum Prediction*, TMLR (2025) | **BLOCKED** — exact commit and original public upstream URL missing |
| ICEBERG | `https://github.com/coleygroup/ms-pred` | `fragnnet.iceberg` files from the formal run snapshot, inside the same retained baseline package | Snapshot-level BSD-2-Clause; exact ms-pred commit/license mapping not recorded | `baseline_rebuild/baseline/source/fragnnet/src/fragnnet/iceberg/` | `baseline_rebuild/baseline/iceberg/`; exact removal of an unrelated experimental CE-gate recorded in `source/fragnnet/patches/` | Goldman, Li and Coley, *Generating molecular fragmentation graphs with autoregressive neural networks*, Analytical Chemistry (2024) | **BLOCKED** — exact upstream commit/license mapping missing |
| GrAFF-MS | `https://github.com/murphy17/graff-ms` | `fragnnet.graff` from the local execution snapshot | Snapshot-level BSD-2-Clause; exact GrAFF-MS commit/license mapping not recorded | `baseline_rebuild/baseline/source/fragnnet/src/fragnnet/graff/` | `baseline_rebuild/baseline/graff_ms/`, MAGMa annotation and common evaluation adapters | Murphy et al., *Efficiently predicting high resolution mass spectra with graph neural networks*, ICML/PMLR (2023) | **BLOCKED** — exact upstream commit/license mapping missing |
| FIORA | `https://github.com/BAMeScience/fiora` | Local package version 1.0.1; exact Git commit not retained | `baseline_rebuild/baseline/source/fiora/LICENSE` (MIT) | `baseline_rebuild/baseline/source/fiora/` | `baseline_rebuild/baseline/fiora/` | Nowatzky et al., *FIORA: Local neighborhood-based prediction of compound mass spectra from single fragmentation events*, Nature Communications (2025) | **BLOCKED** — exact source commit missing; external official model weight required |

## License boundaries

- `baseline_rebuild/baseline/source/fragnnet/**` remains governed by the
  preserved BSD-2-Clause notice plus any narrower file-level notice, including
  the Microsoft MIT header in `massformer/algos.pyx`.
- `baseline_rebuild/baseline/source/fiora/**` remains governed by its preserved
  MIT license.
- FERA-MS-authored configs, launchers, cohort adapters, evaluation adapters and
  documentation do not override those third-party licenses.
- `code/src/ms2spectra/**` is the FERA-MS implementation. It contains no
  baseline model packages and is not used as a substitute for any retained
  baseline implementation.
- Model checkpoints, NIST20 content, PubChem caches, predictions and result
  tables are not distributed.

## Retained low-level BSD-2 infrastructure in the FERA-MS package

FERA-MS is a separate manuscript model, not the FraGNNet-D3 baseline. A
file-level comparison nevertheless identifies common low-level implementation
ancestry from the BSD-2 licensed runtime whose notice is preserved at
`baseline_rebuild/baseline/source/fragnnet/LICENSE`:

- byte-identical infrastructure:
  `code/src/ms2spectra/frag/compute_frags.pyx`,
  `code/src/ms2spectra/utils/misc_utils.py`, and
  `code/src/ms2spectra/utils/profile_utils.py`;
- modified common infrastructure under `code/src/ms2spectra/utils/`, and
  retained runtime scaffolding in `model.py`, `training.py`, `data.py`, and
  `workflow.py`.

The inherited portions remain under BSD-2-Clause. FERA-MS-authored
modifications and new components are offered under BSD-3-Clause. The root
license does not remove the BSD-2 notice or change the licenses of the bundled
baseline source packages. Additional short adapted-code attributions already
present in source comments (SCARF, RetroSim, NetworkX, Pydantic and cited
implementation notes) are preserved unchanged.
