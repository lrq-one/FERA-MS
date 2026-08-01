# Source provenance report

Audit date: 2026-08-01
Release branch: `paper-release`

This report classifies every tracked source path by ownership scope. Generated
artifacts, licensed data and record-level split files are excluded from Git.
The complete `code/src/ms2spectra/**` scope is classified below by file role.

## Scope classification

| Tracked scope | Classification | License/provenance treatment |
|---|---|---|
| `code/src/ms2spectra/model.py`, `training.py`, `data.py`, `workflow.py`, `components/**`, `losses/**` | FERA-MS model and training implementation, including the FERA-specific formula/H, CE-control, rendering, gate and loss changes | FERA-MS-authored changes: BSD-3-Clause; retained BSD-2 scaffolding remains covered by the notice described below |
| `code/src/ms2spectra/frag/compute_frags.pyx`, `utils/misc_utils.py`, `utils/profile_utils.py` | Byte-identical low-level infrastructure shared with the independently retained BSD-2 snapshot | BSD-2-Clause notice preserved at `baseline_rebuild/baseline/source/fragnnet/LICENSE` |
| Other `code/src/ms2spectra/utils/**` | Common infrastructure with both retained BSD-2 portions and FERA-MS modifications | Inherited portions: BSD-2-Clause; FERA-MS-authored modifications: BSD-3-Clause |
| `train/**` | FERA-MS-authored training, refinement and checkpoint orchestration | BSD-3-Clause |
| `test/**` | FERA-MS-authored spectrum/retrieval evaluation | BSD-3-Clause |
| `preproc_scripts/**` | FERA-MS preprocessing, cohort, split and DAG-cache construction | BSD-3-Clause |
| `ablation_studies/**` | FERA-MS-authored formal ablation wiring/configs, with internal copies of FERA-MS refinement stages where required | BSD-3-Clause |
| `baseline_rebuild/baseline/source/fragnnet/**` | Independent third-party baseline execution snapshot used for NEIMS-ACE, MassFormer-ACE, FraGNNet-D3-ACE, ICEBERG-ACE and GrAFF-MS | BSD-2-Clause notice preserved in that directory; Microsoft MIT file header preserved; exact upstream commits unresolved |
| `baseline_rebuild/baseline/source/fiora/**` | Independent FIORA 1.0.1 source snapshot | MIT notice preserved in that directory; exact upstream commit unresolved |
| Other `baseline_rebuild/**` | FERA-MS-authored locked configs, launchers, cohort adapters, checkpoint selection, exporters, evaluators and aggregators | BSD-3-Clause |
| `config/**`, `scripts/**`, root packaging/environment files | FERA-MS release/configuration infrastructure | BSD-3-Clause |

The independent baseline source is intentionally not placed in or imported
from `code/src/ms2spectra`. Default baseline launchers resolve
`baseline_rebuild/baseline/source/fragnnet`, while the FERA-MS training path
resolves `code/src/ms2spectra`.

This provenance boundary does not identify FERA-MS as the FraGNNet-D3 model.
FERA-MS is the separate manuscript model and has its own architecture,
controllers, objectives and refinement pipeline. The BSD-2 classification
above records only retained implementation ancestry where the file comparison
shows identical or modified common scaffolding.

## Evidence and modifications

- The local baseline execution snapshots had no `.git` directory, and archived
  run manifests recorded `code_commit: null`.
- The only locally recorded FraGNNet remote is
  `https://github.com/lrq-one/fragnnet-main`; no commit matching the complete
  execution snapshot has been established.
- The retained independent FraGNNet baseline snapshot carries its own
  BSD-2-Clause notice attributed to Adamo Young and Fei Wang at
  `baseline_rebuild/baseline/source/fragnnet/LICENSE`.
- All retained `fragnnet` runtime files match the local formal-run snapshot,
  except that the two ICEBERG files match the formal ICEBERG run variant and
  the package metadata license string was aligned with the preserved
  BSD-2-Clause `LICENSE`. See `source/fragnnet/UPSTREAM.md` and its patch file.
- The retained FIORA source reports version 1.0.1 and MIT. The official model
  weight remains external; its historical checksum is recorded in
  `source/fiora/UPSTREAM.md`.
- Detailed model-to-file/import/config tracing is in
  `docs/BASELINE_PROVENANCE.md`.

## FERA-MS versus retained baseline file evidence

The release audit compared the FERA-MS package against the independent local
baseline execution snapshot. Representative line differences (added/deleted)
are: `model.py` 3103/422, `training.py` 2082/344, `data_utils.py` 50/15,
`feat_utils.py` 213/4 and `frag_utils.py` 831/17. The byte-identical files are
limited to `frag/compute_frags.pyx`, `utils/misc_utils.py` and
`utils/profile_utils.py`. Baseline-only packages (`massformer`, `graff`, and
`iceberg`) and baseline-only NEIMS/precursor/GNN classes were removed from
`code/src/ms2spectra` and retained only in the independent baseline package.

## License conclusion

The FERA-MS model source is structurally separated from all baseline model
implementations. FERA-MS-authored contributions are released under the root
BSD-3-Clause license. The root license does not relicense retained BSD-2
portions or `baseline_rebuild/baseline/source/**`; those files remain governed
by the licenses and file headers preserved in the repository. Missing exact
upstream commits remain release-provenance blockers for the bundled baseline
snapshots and are listed in `THIRD_PARTY_NOTICES.md`.
