# FraGNNet baseline runtime provenance

This directory is an independent `fragnnet` package. It is not the FERA-MS
`code/src/ms2spectra` package and must not be substituted by it.

The files were copied byte-for-byte from the local source snapshot used for
the manuscript NEIMS-ACE, MassFormer-ACE, FraGNNet-D3-ACE and GrAFF-MS runs,
except for the following explicitly recorded release choices:

- `src/fragnnet/iceberg/model.py` and `pl_model.py` are the two files from the
  formal ICEBERG-ACE run snapshot. They remove an unrelated experimental
  ICEBERG CE-gate while retaining the shared ACE cohort/configuration protocol.
  The difference is recorded in `patches/iceberg_remove_experimental_ce_gate.patch`.
- `pyproject.toml` says `BSD-2-Clause` to match the preserved upstream
  `LICENSE`; the archived snapshot had an inconsistent `MIT` metadata string.

The local execution snapshots had no `.git` directory and their run manifests
recorded `code_commit: null`. The only locally recorded source remote is
`https://github.com/lrq-one/fragnnet-main`; an exact commit whose tree matches
the execution snapshot has not been established. The snapshot's BSD-2-Clause
license and copyright notice are preserved in this directory. Under the
selected release policy, this complete execution snapshot is the authoritative
source record and exact commit recovery is not required.

A local historical remote branch exposes commit
`f86390399f1660219479011937d0386786c5b933`. Several low-level files match
that revision, but the retained executed `model.py`, `pl_model.py`, and
`runner.py` match no available revision of those paths in the local Git
history. That commit is therefore partial ancestry evidence, not the exact
executed baseline commit.

The source modules used by each configured model are documented in
`docs/BASELINE_PROVENANCE.md`.
