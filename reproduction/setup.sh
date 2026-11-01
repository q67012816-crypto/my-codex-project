#!/usr/bin/env bash
set -euo pipefail
cd /workspace/my-codex-project
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/python
uv python install 3.10.22 --system-certs
if [ ! -x /workspace/venvs/cgt/bin/python ]; then
  uv venv --python /workspace/python/cpython-3.10.22-linux-x86_64-gnu/bin/python3.10 /workspace/venvs/cgt
fi
uv pip install --python /workspace/venvs/cgt/bin/python --system-certs \
  'torch==2.0.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /workspace/venvs/cgt/bin/python --system-certs \
  -r reproduction/requirements-lock.txt
DGLBACKEND=pytorch /workspace/venvs/cgt/bin/python -c \
  'import torch, torch_geometric, dgl, dgl.sparse; print(torch.__version__, torch_geometric.__version__, dgl.__version__)'
