# ICEBERG-ACE source record

- Manuscript implementation: ICEBERG-ACE.
- Upstream method repository: `https://github.com/coleygroup/ms-pred`.
- Execution route: the ICEBERG module already present in the independent
  baseline runtime, selected by `model_type: iceberg_inten` and the locked ACE
  configs in this directory. There is no separate `iceberg_core` repository.
- FERA-MS files: locked configs in `configs/`, `run_all.sh` and common adapters.
- License evidence: snapshot-level BSD-2-Clause notice at
  `../source/fragnnet/LICENSE`.
- Release decision: the formal retained source, locked ACE configs and
  preserved notice are sufficient; exact upstream commit recovery is not
  required.
