#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./test_hunyuan_image3_common.sh
. "$SCRIPT_DIR/test_hunyuan_image3_common.sh"

TASK_OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/hunyuan_image3_t2i_ar}"
LOG_FILE="$TASK_OUTPUT_DIR/logs/t2i_ar.log"

usage() {
  cat <<'USAGE'
Usage:
  bash test_hunyuan_image3_t2i_ar.sh

Optional environment variables:
  MODEL=/path/or/hf-id
  OUTPUT_DIR=/path/to/output
  PYTHON_BIN=python3
  PROMPT_T2I="A quiet snowy street at dusk with warm shop lights."

This script runs text2img against the AR-only deploy. Success means:
  - the AR-only stage initializes,
  - the request is processed,
  - and AR text output is produced without requiring the DiT stage.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

prepare_task_dir "$TASK_OUTPUT_DIR"
static_check_deploy_has_vit_dp "$AR_DEPLOY"

echo "[run] HunyuanImage3 text2img via AR-only deploy"
run_hunyuan_example "$LOG_FILE" \
  --model "$MODEL" \
  --modality text2img \
  --deploy-config "$AR_DEPLOY" \
  --output "$TASK_OUTPUT_DIR/output" \
  --prompts "$PROMPT_T2I"

assert_regex "Deploy config: .*hunyuan_image3_ar.yaml" "$LOG_FILE" \
  "text2img AR-only did not use the expected AR deploy."
assert_literal "Num stages: 1" "$LOG_FILE" \
  "text2img AR-only did not resolve to a single-stage deploy."
assert_literal "[Output] Text:" "$LOG_FILE" \
  "text2img AR-only finished without AR text output."

echo "[done] HunyuanImage3 text2img AR-only check passed."
echo "Log: $LOG_FILE"
