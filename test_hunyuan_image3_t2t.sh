#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./test_hunyuan_image3_common.sh
. "$SCRIPT_DIR/test_hunyuan_image3_common.sh"

TASK_OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/hunyuan_image3_t2t}"
LOG_FILE="$TASK_OUTPUT_DIR/logs/t2t.log"

usage() {
  cat <<'USAGE'
Usage:
  bash test_hunyuan_image3_t2t.sh

Optional environment variables:
  MODEL=/path/or/hf-id
  OUTPUT_DIR=/path/to/output
  PYTHON_BIN=python3
  PROMPT_T2T="Say hello in one sentence."
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

prepare_task_dir "$TASK_OUTPUT_DIR"

echo "[run] HunyuanImage3 text2text via AR-only deploy"
run_hunyuan_example "$LOG_FILE" \
  --model "$MODEL" \
  --modality text2text \
  --deploy-config "$AR_DEPLOY" \
  --output "$TASK_OUTPUT_DIR/output" \
  --prompts "$PROMPT_T2T"

assert_regex "Deploy config: .*hunyuan_image3_ar.yaml" "$LOG_FILE" \
  "text2text did not use the expected AR deploy."
assert_literal "Num stages: 1" "$LOG_FILE" \
  "text2text did not resolve to a single-stage deploy."
assert_literal "[Output] Text:" "$LOG_FILE" \
  "text2text finished without text output."

echo "[done] HunyuanImage3 text2text check passed."
echo "Log: $LOG_FILE"
