#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_IT2I="${RUN_IT2I:-1}"
RUN_T2T="${RUN_T2T:-0}"

if [[ -n "${OUTPUT_DIR:-}" && -z "${OUTPUT_ROOT:-}" ]]; then
  export OUTPUT_ROOT="$OUTPUT_DIR"
  unset OUTPUT_DIR
fi

usage() {
  cat <<'USAGE'
Usage:
  IMAGE_PATH=/path/to/image.jpg bash test_hunyuan_image3_vit_dp.sh

Optional environment variables:
  OUTPUT_ROOT=/path/to/output-root             Default: test_outputs
  RUN_IT2I=0                                   Skip img2img
  RUN_T2T=1                                    Also run text2text smoke test

This wrapper keeps the old entrypoint but delegates to the task-specific scripts:
  - test_hunyuan_image3_i2t.sh
  - test_hunyuan_image3_it2i.sh
  - test_hunyuan_image3_t2t.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

echo "[1/3] Run img2text task script"
bash "$ROOT_DIR/test_hunyuan_image3_i2t.sh"

if [[ "$RUN_IT2I" == "1" ]]; then
  echo "[2/3] Run img2img task script"
  bash "$ROOT_DIR/test_hunyuan_image3_it2i.sh"
else
  echo "[2/3] Skip img2img because RUN_IT2I=$RUN_IT2I"
fi

if [[ "$RUN_T2T" == "1" ]]; then
  echo "[3/3] Run text2text task script"
  bash "$ROOT_DIR/test_hunyuan_image3_t2t.sh"
else
  echo "[3/3] Skip text2text because RUN_T2T=$RUN_T2T"
fi

echo "[done] Compatibility wrapper finished."
