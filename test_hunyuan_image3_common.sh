#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXAMPLE="${EXAMPLE:-examples/offline_inference/hunyuan_image3/end2end.py}"
AR_DEPLOY="${AR_DEPLOY:-vllm_omni/deploy/hunyuan_image3_ar.yaml}"
FULL_DEPLOY="${FULL_DEPLOY:-vllm_omni/deploy/hunyuan_image_3_moe.yaml}"
DIT_DEPLOY="${DIT_DEPLOY:-vllm_omni/deploy/hunyuan_image3_dit.yaml}"
SIGLIP="${SIGLIP:-vllm_omni/model_executor/models/hunyuan_image3/siglip2.py}"

IMAGE_PATH="${IMAGE_PATH:-}"
PROMPT_I2T="${PROMPT_I2T:-Describe this image in detail.}"
PROMPT_IT2I="${PROMPT_IT2I:-Make the scene snowy while preserving the main subject.}"
PROMPT_T2I="${PROMPT_T2I:-A quiet snowy street at dusk with warm shop lights.}"
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

require_image_path() {
  if [[ -z "$IMAGE_PATH" ]]; then
    echo "[error] IMAGE_PATH is required." >&2
    exit 2
  fi
  ensure_file_exists "$IMAGE_PATH" "input image"
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

static_check_deploy_has_vit_dp() {
  local path="$1"
  ensure_file_exists "$path" "deploy config"
  assert_literal "mm_encoder_tp_mode: data" "$path" "Missing mm_encoder_tp_mode: data in $path"
}

static_check_siglip_hook() {
  ensure_file_exists "$SIGLIP" "SigLIP source"
  assert_literal "is_vit_use_data_parallel" "$SIGLIP" "SigLIP2 is not wired to the vLLM ViT DP helper."
  assert_literal "disable_tp=use_data_parallel" "$SIGLIP" "SigLIP2 TP disable hook is missing."
  assert_literal "self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()" "$SIGLIP" \
    "SigLIP2 tp_size fallback logic is missing."
  assert_literal "HunyuanImage3 SigLIP2 init:" "$SIGLIP" "SigLIP2 init log is missing."
}

static_check_siglip_forward_stats_hook() {
  ensure_file_exists "$SIGLIP" "SigLIP source"
  assert_literal "HunyuanImage3 SigLIP2 forward local inputs:" "$SIGLIP" \
    "SigLIP2 forward local-input instrumentation is missing."
}

run_hunyuan_example() {
  local log_file="$1"
  shift
  ensure_file_exists "$EXAMPLE" "offline example"
  "$PYTHON_BIN" "$EXAMPLE" "$@" 2>&1 | tee "$log_file"
}
