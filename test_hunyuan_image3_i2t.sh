#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./test_hunyuan_image3_common.sh
. "$SCRIPT_DIR/test_hunyuan_image3_common.sh"

TASK_OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/hunyuan_image3_i2t}"
LOG_FILE="$TASK_OUTPUT_DIR/logs/i2t.log"

usage() {
  cat <<'USAGE'
Usage:
  IMAGE_PATH=/path/to/image.jpg bash test_hunyuan_image3_i2t.sh

Optional environment variables:
  MODEL=/path/or/hf-id
  OUTPUT_DIR=/path/to/output
  PYTHON_BIN=python3
  PROMPT_I2T="Describe this image in detail."
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_image_path
prepare_task_dir "$TASK_OUTPUT_DIR"
static_check_deploy_has_vit_dp "$AR_DEPLOY"
static_check_siglip_hook
static_check_siglip_forward_stats_hook

echo "[run] HunyuanImage3 img2text via AR-only deploy"
run_hunyuan_example "$LOG_FILE" \
  --model "$MODEL" \
  --modality img2text \
  --deploy-config "$AR_DEPLOY" \
  --image-path "$IMAGE_PATH" \
  --output "$TASK_OUTPUT_DIR/output" \
  --prompts "$PROMPT_I2T"

assert_regex "Deploy config: .*hunyuan_image3_ar.yaml" "$LOG_FILE" \
  "img2text did not use the expected AR deploy."
assert_literal "[Output] Text:" "$LOG_FILE" \
  "img2text finished without text output."
assert_literal "HunyuanImage3 SigLIP2 init: use_data_parallel=True, vit_tp_size=1" "$LOG_FILE" \
  "img2text did not emit the expected ViT-DP init log."
assert_literal "HunyuanImage3 SigLIP2 forward local inputs:" "$LOG_FILE" \
  "img2text did not emit local ViT input stats."

echo "[done] HunyuanImage3 img2text check passed."
echo "Log: $LOG_FILE"
