#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-True}
exec vllm serve "${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}" --port 8091 --gpu-memory-utilization .21 --omni "$@"
