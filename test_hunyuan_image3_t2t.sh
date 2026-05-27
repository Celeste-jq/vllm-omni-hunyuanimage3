#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXAMPLE="${EXAMPLE:-examples/offline_inference/hunyuan_image3/end2end.py}"
AR_DEPLOY="${AR_DEPLOY:-vllm_omni/deploy/hunyuan_image3_ar.yaml}"
PROMPT_T2T="${PROMPT_T2T:-Say hello in one sentence.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_DIR:-$ROOT_DIR/test_outputs}}"

ensure_file_exists() {
  local path="$1"
  local label="${2:-file}"
  if [[ ! -f "$path" ]]; then
    echo "[error] Missing ${label}: $path" >&2
    exit 2
  fi
}

prepare_task_dir() {
  local path="$1"
  mkdir -p "$path/logs"
}

assert_regex() {
  local pattern="$1"
  local path="$2"
  local message="$3"
  if ! grep -Eq "$pattern" "$path"; then
    echo "[error] $message" >&2
    echo "[error] Check log/file: $path" >&2
    exit 1
  fi
}

assert_literal() {
  local text="$1"
  local path="$2"
  local message="$3"
  if ! grep -Fq "$text" "$path"; then
    echo "[error] $message" >&2
    echo "[error] Check log/file: $path" >&2
    exit 1
  fi
}

run_hunyuan_example() {
  local log_file="$1"
  shift
  ensure_file_exists "$EXAMPLE" "offline example"
  "$PYTHON_BIN" "$EXAMPLE" "$@" 2>&1 | tee "$log_file"
}

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
