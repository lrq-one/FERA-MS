# Third-party notices

The baseline implementations used by the manuscript are retained independently
under `baseline_rebuild/baseline/source/`. They are not relicensed by a future
FERA-MS root license and must not be represented as the FERA-MS
`code/src/ms2spectra` implementation.

The local execution snapshots did not contain `.git` metadata and their run
manifests recorded `code_commit: null`. For this release, the complete executed
source snapshots and preserved license notices are the authoritative baseline
source records; exact upstream commit recovery is not a release requirement.

| Model | Upstream repository recorded for the method | Exact executed source | License evidence preserved | Retained path | FERA-MS-authored adapter/config | Manuscript citation | Status |
|---|---|---|---|---|---|---|---|
| NEIMS | `https://github.com/brain-research/deep-molecular-massspec` | NEIMS classes in the retained local `fragnnet` execution snapshot | BSD-2-Clause snapshot notice preserved | `baseline_rebuild/baseline/source/fragnnet/` | `baseline_rebuild/baseline/neims/`, `tools_local/` | Wei et al., *Rapid prediction of electron-ionization mass spectrometry using neural networks*, ACS Central Science (2019) | Complete retained-source route |
| MassFormer | `https://github.com/Roestlab/massformer` | `fragnnet.massformer` from the retained local execution snapshot | BSD-2-Clause snapshot notice; Microsoft MIT header retained in `algos.pyx` | `baseline_rebuild/baseline/source/fragnnet/src/fragnnet/massformer/` | `baseline_rebuild/baseline/massformer/`, `tools_local/` | Young, Wang and Röst, *Tandem mass spectrum prediction for small molecules using graph transformers*, Nature Machine Intelligence (2024) | Complete retained-source route |
| FraGNNet-D3 | Historical local remote: `https://github.com/lrq-one/fragnnet-main` | Complete local execution snapshot | `baseline_rebuild/baseline/source/fragnnet/LICENSE` (BSD-2-Clause; Adamo Young and Fei Wang) | `baseline_rebuild/baseline/source/fragnnet/` | `baseline_rebuild/baseline/fragnnet_d3/`, `tools_local/` | Young et al., *FraGNNet: A Deep Probabilistic Model for Tandem Mass Spectrum Prediction*, TMLR (2025) | Complete retained-source route |
| ICEBERG | `https://github.com/coleygroup/ms-pred` | Formal-run `fragnnet.iceberg` files in the retained baseline package | BSD-2-Clause snapshot notice preserved | `baseline_rebuild/baseline/source/fragnnet/src/fragnnet/iceberg/` | `baseline_rebuild/baseline/iceberg/`; formal source difference recorded in `source/fragnnet/patches/` | Goldman, Li and Coley, *Generating molecular fragmentation graphs with autoregressive neural networks*, Analytical Chemistry (2024) | Complete retained-source route |
| GrAFF-MS | `https://github.com/murphy17/graff-ms` | `fragnnet.graff` from the retained local execution snapshot | BSD-2-Clause snapshot notice preserved | `baseline_rebuild/baseline/source/fragnnet/src/fragnnet/graff/` | `baseline_rebuild/baseline/graff_ms/`, MAGMa annotation and common evaluation adapters | Murphy et al., *Efficiently predicting high resolution mass spectra with graph neural networks*, ICML/PMLR (2023) | Complete retained-source route |
| FIORA | `https://github.com/BAMeScience/fiora` | Retained local package version 1.0.1 | `baseline_rebuild/baseline/source/fiora/LICENSE` (MIT) | `baseline_rebuild/baseline/source/fiora/` | `baseline_rebuild/baseline/fiora/` | Nowatzky et al., *FIORA: Local neighborhood-based prediction of compound mass spectra from single fragmentation events*, Nature Communications (2025) | Complete source route; official model weight remains external |

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
