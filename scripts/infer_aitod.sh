#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/infer_aitod.sh <image_path> [checkpoint] [output_dir]"
  exit 1
fi

image_path=$1
checkpoint=${2:-pt/FDQ_Det_AITODv2_best321.pth}
conf_thresh=${CONF_THRESH:-0.55}
nms_iou_thresh=${NMS_IOU_THRESH:-0.5}
output_dir=${3:-output/infer_aitod}

python inference.py \
  --dataset aitod \
  --input_image "${image_path}" \
  --model_weights "${checkpoint}" \
  --output_dir "${output_dir}" \
  --conf_thresh "${conf_thresh}" \
  --nms_iou_thresh "${nms_iou_thresh}" \
  --visualize
