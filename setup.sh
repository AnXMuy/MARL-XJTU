#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON:-python}" -m pip install -r requirements.txt
# GPU-enabled PyTorch must match the host driver. No EPyMARL clone or imports.
"${PYTHON:-python}" -c 'import torch,torchrl,benchmarl; print("CUDA:",torch.cuda.is_available())'
