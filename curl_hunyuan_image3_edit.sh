#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://localhost:8011}"
IMAGE_0="${IMAGE_0:-/tmp/hunyuanimage3_it2i_pr_inputs/input_1_0.png}"
IMAGE_1="${IMAGE_1:-/tmp/hunyuanimage3_it2i_pr_inputs/input_1_1.png}"
OUTPUT="${OUTPUT:-result.png}"
PROMPT="${PROMPT:-基于图一的logo，参考图二中冰箱贴的材质，制作一个新的冰箱贴}"
BOT_TASK="${BOT_TASK:-think_recaption}"
SIZE="${SIZE:-1280x720}"
STEPS="${STEPS:-8}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
SEED="${SEED:-42}"

curl -X POST "${SERVER_URL}/v1/images/edits" \
  -F "image=@${IMAGE_0}" \
  -F "image=@${IMAGE_1}" \
  -F "prompt=${PROMPT}" \
  -F "bot_task=${BOT_TASK}" \
  -F "n=1" \
  -F "num_inference_steps=${STEPS}" \
  -F "guidance_scale=${GUIDANCE_SCALE}" \
  -F "size=${SIZE}" \
  -F "seed=${SEED}" \
  | jq -r '.data[0].b64_json' \
  | base64 -d > "${OUTPUT}"
