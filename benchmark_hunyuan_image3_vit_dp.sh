#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
IMAGE_PATH="${IMAGE_PATH:-}"
TASK="${TASK:-i2t}"
RUNS="${RUNS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXAMPLE="${EXAMPLE:-examples/offline_inference/hunyuan_image3/end2end.py}"
AR_DEPLOY="${AR_DEPLOY:-vllm_omni/deploy/hunyuan_image3_ar.yaml}"
FULL_DEPLOY="${FULL_DEPLOY:-vllm_omni/deploy/hunyuan_image_3_moe.yaml}"
PROMPT_I2T="${PROMPT_I2T:-Describe this image in detail.}"
PROMPT_IT2I="${PROMPT_IT2I:-Make the scene snowy while preserving the main subject.}"
STEPS="${STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
SEED="${SEED:-42}"

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/test_outputs/hunyuan_image3_vit_dp_metrics/${BRANCH}_${COMMIT}_${TASK}_${RUN_ID}}"
METRICS_CSV="${METRICS_CSV:-$OUTPUT_ROOT/metrics.csv}"

usage() {
  cat <<'USAGE'
Usage:
  IMAGE_PATH=/path/to/image.jpg MODEL=/path/to/model bash benchmark_hunyuan_image3_vit_dp.sh

Optional environment variables:
  TASK=i2t|it2i                 Default: i2t
  RUNS=3                        Default: 1
  OUTPUT_ROOT=/path/to/output   Default: test_outputs/hunyuan_image3_vit_dp_metrics/<branch>_<commit>_<task>_<time>
  METRICS_CSV=/path/file.csv    Default: $OUTPUT_ROOT/metrics.csv
  PYTHON_BIN=python3
  PROMPT_I2T="Describe this image in detail."
  PROMPT_IT2I="Make the scene snowy while preserving the main subject."
  STEPS=50                      Used by it2i
  GUIDANCE_SCALE=5.0            Used by it2i
  SEED=42                       Used by it2i

What it records:
  - total wall time per run
  - processed prompts latency from tqdm when available
  - whether text/image output markers appeared
  - ViT-DP init marker count
  - SigLIP2 forward local-input log count
  - AR ViT DP shard log count and local_count distribution

Use this on both old and new branches, then compare the generated metrics.csv files.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$TASK" != "i2t" && "$TASK" != "it2i" ]]; then
  echo "[error] TASK must be i2t or it2i, got: $TASK" >&2
  exit 2
fi

if [[ -z "$IMAGE_PATH" || ! -f "$IMAGE_PATH" ]]; then
  echo "[error] IMAGE_PATH is required and must exist: ${IMAGE_PATH:-<empty>}" >&2
  exit 2
fi

if [[ ! -f "$EXAMPLE" ]]; then
  echo "[error] Missing example script: $EXAMPLE" >&2
  exit 2
fi

case "$TASK" in
  i2t)
    MODALITY="img2text"
    DEPLOY_CONFIG="$AR_DEPLOY"
    PROMPT="$PROMPT_I2T"
    ;;
  it2i)
    MODALITY="img2img"
    DEPLOY_CONFIG="$FULL_DEPLOY"
    PROMPT="$PROMPT_IT2I"
    ;;
esac

if [[ ! -f "$DEPLOY_CONFIG" ]]; then
  echo "[error] Missing deploy config: $DEPLOY_CONFIG" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/outputs"

csv_escape() {
  local value="${1//$'\n'/ }"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

count_literal() {
  local text="$1"
  local path="$2"
  grep -F "$text" "$path" | wc -l | tr -d ' '
}

extract_local_counts() {
  local path="$1"
  grep -F "HunyuanImage3 AR ViT DP shard:" "$path" \
    | sed -n 's/.*local_count=\([0-9][0-9]*\).*/\1/p' \
    | paste -sd';' -
}

count_nonzero_local_counts() {
  local path="$1"
  grep -F "HunyuanImage3 AR ViT DP shard:" "$path" \
    | sed -n 's/.*local_count=\([0-9][0-9]*\).*/\1/p' \
    | awk '$1 > 0 { n += 1 } END { print n + 0 }'
}

extract_processed_prompts_line() {
  local path="$1"
  grep -F "Processed prompts:" "$path" | tail -1 | tr -d '\r' || true
}

extract_processed_prompts_seconds() {
  local line="$1"
  if [[ -z "$line" ]]; then
    return 0
  fi
  LINE="$line" awk '
    BEGIN {
      line = ENVIRON["LINE"]
      n = split(line, parts, "[")
      for (i = n; i >= 1; --i) {
        if (parts[i] ~ /[0-9.]+s\/it/) {
          if (match(parts[i], /[0-9.]+s\/it/)) {
            value = substr(parts[i], RSTART, RLENGTH)
            sub(/s\/it/, "", value)
            print value
            exit
          }
        }
        if (parts[i] ~ /[0-9]+:[0-9][0-9]</) {
          if (match(parts[i], /[0-9]+:[0-9][0-9]</)) {
            value = substr(parts[i], RSTART, RLENGTH - 1)
            split(value, mmss, ":")
            print (mmss[1] * 60 + mmss[2])
            exit
          }
        }
      }
    }
  '
}

write_header() {
  if [[ ! -f "$METRICS_CSV" ]]; then
    printf '%s\n' \
      "run,branch,commit,task,modality,deploy_config,elapsed_s,processed_prompts_s,output_text,output_image,vit_init_true_count,vit_forward_count,shard_log_count,shard_nonzero_count,shard_local_counts,processed_prompts_line,log_file" \
      > "$METRICS_CSV"
  fi
}

write_header

echo "[info] branch=$BRANCH commit=$COMMIT task=$TASK runs=$RUNS"
echo "[info] output_root=$OUTPUT_ROOT"
echo "[info] metrics=$METRICS_CSV"

for run in $(seq 1 "$RUNS"); do
  run_output_dir="$OUTPUT_ROOT/outputs/run_${run}"
  log_file="$OUTPUT_ROOT/logs/run_${run}.log"
  mkdir -p "$run_output_dir"

  echo "[run $run/$RUNS] modality=$MODALITY deploy=$DEPLOY_CONFIG"
  start_ts="$(date +%s)"

  cmd=(
    "$PYTHON_BIN" "$EXAMPLE"
    --model "$MODEL"
    --modality "$MODALITY"
    --deploy-config "$DEPLOY_CONFIG"
    --image-path "$IMAGE_PATH"
    --output "$run_output_dir"
    --prompts "$PROMPT"
  )

  if [[ "$TASK" == "it2i" ]]; then
    cmd+=(--steps "$STEPS" --guidance-scale "$GUIDANCE_SCALE" --seed "$SEED")
  fi

  "${cmd[@]}" 2>&1 | tee "$log_file"
  end_ts="$(date +%s)"
  elapsed_s="$((end_ts - start_ts))"

  output_text=0
  output_image=0
  if grep -Fq "[Output] Text:" "$log_file"; then
    output_text=1
  fi
  if grep -Fq "[Output] Saved image to" "$log_file"; then
    output_image=1
  fi

  vit_init_true_count="$(count_literal "HunyuanImage3 SigLIP2 init: use_data_parallel=True, vit_tp_size=1" "$log_file")"
  vit_forward_count="$(count_literal "HunyuanImage3 SigLIP2 forward local inputs:" "$log_file")"
  shard_log_count="$(count_literal "HunyuanImage3 AR ViT DP shard:" "$log_file")"
  shard_nonzero_count="$(count_nonzero_local_counts "$log_file")"
  shard_local_counts="$(extract_local_counts "$log_file")"
  processed_line="$(extract_processed_prompts_line "$log_file")"
  processed_prompts_s="$(extract_processed_prompts_seconds "$processed_line")"

  {
    printf '%s,' "$run"
    csv_escape "$BRANCH"; printf ','
    csv_escape "$COMMIT"; printf ','
    csv_escape "$TASK"; printf ','
    csv_escape "$MODALITY"; printf ','
    csv_escape "$DEPLOY_CONFIG"; printf ','
    printf '%s,%s,%s,%s,%s,%s,%s,%s,' \
      "$elapsed_s" \
      "$processed_prompts_s" \
      "$output_text" \
      "$output_image" \
      "$vit_init_true_count" \
      "$vit_forward_count" \
      "$shard_log_count" \
      "$shard_nonzero_count"
    csv_escape "$shard_local_counts"; printf ','
    csv_escape "$processed_line"; printf ','
    csv_escape "$log_file"
    printf '\n'
  } >> "$METRICS_CSV"

  echo "[metrics] run=$run elapsed_s=$elapsed_s processed_prompts_s=${processed_prompts_s:-<none>} output_text=$output_text output_image=$output_image vit_forward_count=$vit_forward_count shard_local_counts=${shard_local_counts:-<none>}"
done

echo "[done] Metrics written to $METRICS_CSV"
