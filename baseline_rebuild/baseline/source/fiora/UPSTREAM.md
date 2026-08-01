# FIORA source provenance

This directory retains the local FIORA 1.0.1 source snapshot used by the
manuscript control. Its source declares the upstream repository as
`https://github.com/BAMeScience/fiora` and is distributed under the MIT
license preserved in `LICENSE`.

The archived source had no `.git` directory, so the exact source commit is not
known. The FIORA model weights are intentionally not included. The manuscript
run used `fiora_OS_v1.0.0.pt` with SHA-256
`273807127861ca0ac8404962f111ff8628ba02e6beb5d1142d8772ced07443a0`.
Provide that external weight through `FERA_MS_FIORA_MODEL`.

The missing exact source commit remains a pre-merge provenance blocker even
though the complete Python implementation, FERA-MS input adapter and evaluator
are retained.
