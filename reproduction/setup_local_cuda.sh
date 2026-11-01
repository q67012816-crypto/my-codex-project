#!/usr/bin/env bash
# Linux / WSL2 x86_64 only. CUDA installation has to be verified on the local GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
python3.10 -m venv .venv
.venv/bin/python -m pip install 'torch==2.0.1+cu118' --index-url https://download.pytorch.org/whl/cu118
.venv/bin/python -m pip install -r reproduction/requirements-lock.txt
.venv/bin/python -m pip install --force-reinstall --no-deps --no-index \
  --find-links https://data.dgl.ai/wheels/cu118/repo.html 'dgl==1.1.2+cu118'
DGLBACKEND=pytorch .venv/bin/python - <<'PY'
import torch, dgl, dgl.sparse
assert torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver and WSL GPU access'
g = dgl.graph(([0], [1])).to('cuda')
x = torch.ones(2, 2, device='cuda')
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'DGL:', dgl.__version__)
print('GPU:', torch.cuda.get_device_name(0), 'DGL device:', g.device, 'tensor:', x.device)
PY
