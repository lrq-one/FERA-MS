# Source provenance report

Audit date: 2026-08-01

Release branch: `paper-release`

First repository snapshot containing the integrated implementations:
`a6f75e7bb83fc4e2adfd287a9cb2afeaafe86e91`

## Scope and classification

The audit covers all 214 tracked Python, Cython, Shell, YAML, and TOML source
or configuration files present before this compliance correction. Files are
classified by exhaustive path scope below. Documentation and metadata are
FERA-MS-authored unless a notice within the file states otherwise.

| Tracked scope | Count at audit | Classification | Provenance/license treatment |
|---|---:|---|---|
| `code/src/ms2spectra/**` | 58 | Mixed: integrated FraGNNet-derived code plus subsequent FERA-MS modifications | Preserve FraGNNet BSD-2-Clause notice; do not relicense inherited portions |
| `train/**` | 20 | FERA-MS-authored training and refinement orchestration | Intended BSD-3-Clause after audit closure |
| `test/**` | 25 | FERA-MS-authored evaluation and retrieval orchestration | Intended BSD-3-Clause after audit closure |
| `preproc_scripts/**` | 6 | Mixed: inherited preprocessing interfaces plus FERA-MS cohort/split changes | Preserve upstream notice; FERA-MS changes intended BSD-3-Clause |
| `ablation_studies/**` | 68 | FERA-MS-authored formal ablation configs and code, with internal copies of FERA-MS refinement stages | Intended BSD-3-Clause after audit closure |
| `baseline_rebuild/**` | 31 | FERA-MS-authored adapters, configs, launchers, and aggregators calling the integrated baseline implementations | Intended BSD-3-Clause after audit closure |
| Root/config/release source (`config/**`, `scripts/**`, `setup.py`, `pyproject.toml`, `sitecustomize.py`, environment YAML) | 6 | Mixed release/configuration infrastructure | Intended BSD-3-Clause where FERA-MS-authored |

Within `code/src/ms2spectra/**`, the following baseline-specific scopes require
explicit third-party treatment:

- `massformer/**`: adapted MassFormer implementation;
- `graff/**`: adapted GrAFF-MS implementation;
- `iceberg/**`: adapted ICEBERG implementation;
- `model.py` and `training.py`: include the integrated NEIMS reimplementation;
- `frag/**`, `data.py`, `model.py`, `training.py`, `utils/**`, and
  `workflow.py`: inherit from the FraGNNet codebase and contain later FERA-MS
  modifications.

The precise line-level boundary between the initial integrated snapshot and
later FERA-MS modifications can be inspected with:

```bash
git diff a6f75e7bb83fc4e2adfd287a9cb2afeaafe86e91..paper-release -- code/src/ms2spectra preproc_scripts
```

## Evidence

- Current-repository history places all integrated baseline modules in commit
  `a6f75e7bb83fc4e2adfd287a9cb2afeaafe86e91`.
- The retained local FraGNNet source history preserves a BSD-2-Clause notice
  credited to Adamo Young and Fei Wang; the exact notice is copied unchanged
  to `licenses/FraGNNet-BSD-2-Clause.txt`.
- Historical formal-run manifests for NEIMS and ICEBERG record
  `code_commit: null`; the package-compatible snapshots used for the reported
  runs have no `.git` directory.
- The archived FIORA package identifies itself as version 1.0.1 and MIT, but
  its exact Git commit was not retained.

## Audit conclusion

The repository cleanly separates generated/licensed data from source code and
identifies the integrated baseline scopes, but exact baseline source commits
remain unresolved. Therefore the provenance audit does **not** authorize the
root BSD-3-Clause license yet. `LICENSE_PENDING.md` must remain, `LICENSE` must
not be added, and `CITATION.cff` must not claim BSD-3-Clause until the blocker
table in `THIRD_PARTY_NOTICES.md` is resolved.
