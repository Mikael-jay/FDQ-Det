#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_visdrone.sh <visdrone_path> [checkpoint] [output_dir]"
  exit 1
fi

# bash scripts/eval_visdrone.sh ../VisDrone pt/FDQ_Det_VisDrone_best384.pth

visdrone_path=$1
checkpoint=${2:-pt/FDQ_Det_VisDrone_best384.pth}
output_dir=${3:-output/eval_visdrone}
use_high_freq=${USE_HIGH_FREQ_SUPPRESS:-True}
use_adv_training=${USE_ADV_TRAINING:-False}

python main.py \
  --output_dir "${output_dir}" \
  -c config/fdqdet_visdrone.py \
  --dataset_file visdrone \
  --coco_path "${visdrone_path}" \
  --eval \
  --resume "${checkpoint}" \
  --options \
    dn_scalar=100 embed_init_tgt=False \
    dn_label_coef=1.0 dn_bbox_coef=1.0 dn_box_noise_scale=1.0 \
    use_ema=False \
    use_high_freq_suppress="${use_high_freq}" \
    use_adv_training="${use_adv_training}" \
    use_dfl=False dfl_weight=0.0 dfl_gamma=0.0
