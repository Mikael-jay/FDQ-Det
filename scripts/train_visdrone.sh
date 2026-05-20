#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_visdrone.sh <visdrone_path> [pretrain_ckpt] [output_dir]"
  exit 1
fi

visdrone_path=$1
pretrain_ckpt=${2:-pt/pretrained_model.pth}
output_dir=${3:-logs/FDQ_Det_VisDrone_train}
nproc=${NPROC_PER_NODE:-2}
use_high_freq=${USE_HIGH_FREQ_SUPPRESS:-True}
use_adv_training=${USE_ADV_TRAINING:-False}

python -m torch.distributed.run --nproc_per_node="${nproc}" main.py \
  --output_dir "${output_dir}" \
  -c config/fdqdet_visdrone.py \
  --dataset_file visdrone \
  --coco_path "${visdrone_path}" \
  --pretrain_model_path "${pretrain_ckpt}" \
  --options \
    dn_scalar=100 \
    embed_init_tgt=False \
    dn_label_coef=1.0 dn_bbox_coef=1.0 dn_box_noise_scale=1.0 \
    use_ema=False \
    use_high_freq_suppress="${use_high_freq}" \
    use_adv_training="${use_adv_training}" \
    use_dfl=False dfl_weight=0.0 dfl_gamma=0.0
