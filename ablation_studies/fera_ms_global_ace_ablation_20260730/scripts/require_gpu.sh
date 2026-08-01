#!/usr/bin/env bash
# GPU hard gate for every future formal global-ACE stage.  Never fall back to CPU.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_NVML_BASED_CUDA_CHECK=0
for attempt in 1 2 3 4 5; do
  if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >/dev/null 2>&1 && \
     python - <<'PY'
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is false'
assert torch.cuda.device_count() >= 1, 'no CUDA device visible'
print(torch.cuda.get_device_name(0))
PY
  then
    return 0 2>/dev/null || exit 0
  fi
  sleep 5
done
echo 'GPU_REQUIRED_BUT_UNAVAILABLE: refusing CPU fallback' >&2
return 86 2>/dev/null || exit 86
