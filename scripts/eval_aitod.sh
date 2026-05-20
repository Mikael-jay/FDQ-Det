#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_aitod.sh <aitod_path> [checkpoint] [output_dir]"
  exit 1
fi

aitod_path=$1
checkpoint=${2:-pt/FDQ_Det_AITODv2_best321.pth}
output_dir=${3:-output/eval_aitod}
use_high_freq=${USE_HIGH_FREQ_SUPPRESS:-True}
use_adv_training=${USE_ADV_TRAINING:-False}

python main.py \
  --output_dir "${output_dir}" \
  -c config/fdqdet_aitod.py \
  --dataset_file aitod_v2 \
  --coco_path "${aitod_path}" \
  --eval \
  --resume "${checkpoint}" \
  --options \
    dn_scalar=100 embed_init_tgt=False \
    dn_label_coef=1.0 dn_bbox_coef=1.0 dn_box_noise_scale=1.0 \
    use_ema=False \
    use_high_freq_suppress="${use_high_freq}" \
    use_adv_training="${use_adv_training}" \
    use_dfl=False dfl_weight=0.0 dfl_gamma=0.0
