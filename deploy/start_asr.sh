#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-True}
exec vllm serve "${MODEL:-Qwen/Qwen3-ASR-1.7B}" --port 8000 --gpu-memory-utilization .13  "$@"
