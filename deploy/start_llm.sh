#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-True}
exec vllm serve "${MODEL:-Qwen/Qwen3-8B}" --port 8002 --gpu-memory-utilization .42 --max-model-len 8192 "$@"
