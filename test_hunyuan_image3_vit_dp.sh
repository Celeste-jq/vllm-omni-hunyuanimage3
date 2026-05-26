#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
IMAGE_PATH="${IMAGE_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/test_outputs/hunyuan_image3_vit_dp}"
PROMPT_I2T="${PROMPT_I2T:-Describe this image in detail.}"
PROMPT_IT2I="${PROMPT_IT2I:-Make the scene snowy while preserving the main subject.}"
RUN_IT2I="${RUN_IT2I:-1}"
RUN_T2T="${RUN_T2T:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AR_DEPLOY="${AR_DEPLOY:-vllm_omni/deploy/hunyuan_image3_ar.yaml}"
FULL_DEPLOY="${FULL_DEPLOY:-vllm_omni/deploy/hunyuan_image_3_moe.yaml}"
DIT_DEPLOY="${DIT_DEPLOY:-vllm_omni/deploy/hunyuan_image3_dit.yaml}"
EXAMPLE="${EXAMPLE:-examples/offline_inference/hunyuan_image3/end2end.py}"
SIGLIP="${SIGLIP:-vllm_omni/model_executor/models/hunyuan_image3/siglip2.py}"

usage() {
  cat <<'USAGE'
Usage:
  IMAGE_PATH=/path/to/image.jpg bash test_hunyuan_image3_vit_dp.sh

Optional environment variables:
  MODEL=/path/or/hf-id                         Default: tencent/HunyuanImage-3.0-Instruct
  OUTPUT_DIR=/path/to/output                   Default: test_outputs/hunyuan_image3_vit_dp
  PROMPT_I2T="Describe this image in detail."
  PROMPT_IT2I="Make the scene snowy while preserving the main subject."
  RUN_IT2I=0                                   Skip img2img runtime test
  RUN_T2T=1                                    Also run text2text config-load smoke test
  PYTHON_BIN=python3

What this validates:
  1. AR-capable HunyuanImage3 deploy YAMLs enable existing vLLM ViT DP via mm_encoder_tp_mode: data.
  2. DiT-only deploy YAML does not accidentally set mm_encoder_tp_mode.
  3. The offline example defaults point to main-branch deploy YAMLs.
  4. HunyuanImage3 AR SigLIP2 code is wired to is_vit_use_data_parallel() and emits a runtime init log.
  5. Default img2text/img2img paths exercise AR-side ViT with deploy YAMLs under vllm_omni/deploy/.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$IMAGE_PATH" ]]; then
  echo "[error] IMAGE_PATH is required for ViT DP runtime coverage." >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "[error] IMAGE_PATH does not exist: $IMAGE_PATH" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/logs"

AR_DEPLOYS=(
  "$AR_DEPLOY"
  "$FULL_DEPLOY"
)

echo "[1/5] Static check: AR deploy configs enable existing ViT DP"
for cfg in "${AR_DEPLOYS[@]}"; do
  if ! grep -q "mm_encoder_tp_mode: data" "$cfg"; then
    echo "[error] Missing mm_encoder_tp_mode: data in $cfg" >&2
    exit 1
  fi
  echo "  ok: $cfg"
done

echo "[2/5] Static check: DiT-only deploy stays free of AR ViT DP settings"
if grep -q "mm_encoder_tp_mode" "$DIT_DEPLOY"; then
  echo "[error] Unexpected mm_encoder_tp_mode in DiT-only deploy $DIT_DEPLOY" >&2
  exit 1
fi
grep -q "hunyuan_image_3_moe.yaml" "$EXAMPLE"
grep -q "hunyuan_image3_ar.yaml" "$EXAMPLE"
echo "  ok: $DIT_DEPLOY"
echo "  ok: $EXAMPLE default deploy mapping"

echo "[3/5] Static check: HunyuanImage3 SigLIP2 uses vLLM ViT DP hook and emits init log"
grep -q "is_vit_use_data_parallel" "$SIGLIP"
grep -q "disable_tp=use_data_parallel" "$SIGLIP"
grep -q "self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()" "$SIGLIP"
grep -q "HunyuanImage3 SigLIP2 init:" "$SIGLIP"
echo "  ok: $SIGLIP"

echo "[4/5] Runtime: img2text with default AR deploy (AR ViT DP path)"
I2T_LOG="$OUTPUT_DIR/logs/i2t.log"
"$PYTHON_BIN" examples/offline_inference/hunyuan_image3/end2end.py \
  --model "$MODEL" \
  --modality img2text \
  --image-path "$IMAGE_PATH" \
  --output "$OUTPUT_DIR/i2t" \
  --prompts "$PROMPT_I2T" \
  2>&1 | tee "$I2T_LOG"

if ! grep -q "Deploy config: .*hunyuan_image3_ar.yaml" "$I2T_LOG"; then
  echo "[error] img2text did not select the expected default AR deploy. Check $I2T_LOG" >&2
  exit 1
fi
if ! grep -q "\[Output\] Text:" "$I2T_LOG"; then
  echo "[error] img2text completed without text output marker. Check $I2T_LOG" >&2
  exit 1
fi
if ! grep -q "HunyuanImage3 SigLIP2 init: use_data_parallel=True, vit_tp_size=1" "$I2T_LOG"; then
  echo "[error] img2text did not emit the expected ViT DP init log. Check $I2T_LOG" >&2
  exit 1
fi
echo "  ok: img2text output found"

if [[ "$RUN_IT2I" == "1" ]]; then
  echo "[5/5] Runtime: img2img with default AR+DiT deploy (AR ViT DP + diffusion path)"
  IT2I_LOG="$OUTPUT_DIR/logs/it2i.log"
  "$PYTHON_BIN" examples/offline_inference/hunyuan_image3/end2end.py \
    --model "$MODEL" \
    --modality img2img \
    --image-path "$IMAGE_PATH" \
    --output "$OUTPUT_DIR/it2i" \
    --prompts "$PROMPT_IT2I" \
    2>&1 | tee "$IT2I_LOG"

  if ! grep -q "Deploy config: .*hunyuan_image_3_moe.yaml" "$IT2I_LOG"; then
    echo "[error] img2img did not select the expected default AR+DiT deploy. Check $IT2I_LOG" >&2
    exit 1
  fi
  if ! grep -q "\[Output\] Saved image to" "$IT2I_LOG"; then
    echo "[error] img2img completed without saved image marker. Check $IT2I_LOG" >&2
    exit 1
  fi
  if ! grep -q "HunyuanImage3 SigLIP2 init: use_data_parallel=True, vit_tp_size=1" "$IT2I_LOG"; then
    echo "[error] img2img did not emit the expected ViT DP init log. Check $IT2I_LOG" >&2
    exit 1
  fi
  echo "  ok: img2img image output found"
else
  echo "[5/5] Runtime: img2img skipped because RUN_IT2I=$RUN_IT2I"
fi

if [[ "$RUN_T2T" == "1" ]]; then
  echo "[optional] Runtime: text2text default AR deploy smoke test"
  T2T_LOG="$OUTPUT_DIR/logs/t2t.log"
  "$PYTHON_BIN" examples/offline_inference/hunyuan_image3/end2end.py \
    --model "$MODEL" \
    --modality text2text \
    --output "$OUTPUT_DIR/t2t" \
    --prompts "Say hello in one sentence." \
    2>&1 | tee "$T2T_LOG"
fi

echo "[done] HunyuanImage3 AR ViT DP checks passed."
echo "Logs: $OUTPUT_DIR/logs"
