#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/data/weight/HunyuanImage-3.0-Instruct-Distil}"
PORT="${PORT:-8011}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/hunyuan_image3_ar_dit_shm.yaml}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-2}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-TORCH_SDPA}"
export ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-0}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-19000-19063}"

vllm serve "${MODEL}" --omni --port "${PORT}" \
  --log-stats \
  --deploy-config "${DEPLOY_CONFIG}"
