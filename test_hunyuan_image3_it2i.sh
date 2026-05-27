#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./test_hunyuan_image3_common.sh
. "$SCRIPT_DIR/test_hunyuan_image3_common.sh"

TASK_OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/hunyuan_image3_it2i}"
LOG_FILE="$TASK_OUTPUT_DIR/logs/it2i.log"

usage() {
  cat <<'USAGE'
Usage:
  IMAGE_PATH=/path/to/image.jpg bash test_hunyuan_image3_it2i.sh

Optional environment variables:
  MODEL=/path/or/hf-id
  OUTPUT_DIR=/path/to/output
  PYTHON_BIN=python3
  PROMPT_IT2I="Make the scene snowy while preserving the main subject."
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_image_path
prepare_task_dir "$TASK_OUTPUT_DIR"
static_check_deploy_has_vit_dp "$FULL_DEPLOY"
static_check_siglip_hook
static_check_siglip_forward_stats_hook

echo "[run] HunyuanImage3 img2img via full AR+DiT deploy"
run_hunyuan_example "$LOG_FILE" \
  --model "$MODEL" \
  --modality img2img \
  --deploy-config "$FULL_DEPLOY" \
  --image-path "$IMAGE_PATH" \
  --output "$TASK_OUTPUT_DIR/output" \
  --prompts "$PROMPT_IT2I"

assert_regex "Deploy config: .*hunyuan_image_3_moe.yaml" "$LOG_FILE" \
  "img2img did not use the expected full deploy."
assert_literal "[Output] Saved image to" "$LOG_FILE" \
  "img2img finished without saving an image."
assert_literal "HunyuanImage3 SigLIP2 init: use_data_parallel=True, vit_tp_size=1" "$LOG_FILE" \
  "img2img log does not contain an AR-side ViT-DP init marker."
assert_ar_vit_dp_shard_log "$LOG_FILE"
assert_literal "HunyuanImage3 SigLIP2 forward local inputs:" "$LOG_FILE" \
  "img2img did not emit local ViT input stats."

echo "[done] HunyuanImage3 img2img check passed."
echo "Log: $LOG_FILE"
