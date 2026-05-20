_base_ = ['coco_transformer.py']

# Dataset (pass dataset_file / coco_path via CLI args to avoid argparse clash)
num_classes = 10

lr_backbone = 1e-05
lr = 0.0001
param_dict_type = 'default'
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 1
weight_decay = 0.0001
# epochs = 48
epochs = 48
lr_drop = 18
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = True
lr_drop_list = [13, 20, 40]
# lr_drop_list = [13]
val_epoch = [23, 38, 41, 44, 47]
# val_epoch = [23]

# Model
modelname = 'fdqdet'
frozen_weights = None
use_checkpoint = False
backbone = 'resnet50'
dilation = False
num_feature_levels = 5

position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [0, 1, 2, 3]
backbone_freeze_keywords = None
unic_layers = 0
pre_norm = False

hidden_dim = 256
nheads = 8
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
dropout = 0.0
enc_n_points = 4
dec_n_points = 4

ccm_params = [10, 100, 500]
ccm_cls_num = 4
dynamic_query_list = [300, 500, 900, 1500]
find_unused_parameters = True
num_queries = 900
query_dim = 4

# HF suppression
use_high_freq_suppress = True
high_freq_reduction = 4
high_freq_kernel_schedule = None

# Adversarial training
use_adv_training = True
adv_epsilon = 0.01
adv_loss_weight = 0.05
feature_adv_epsilon = None
query_adv_epsilon = None

num_patterns = 0
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0

cls_loss_coef = 1.0
mask_loss_coef = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25

matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
decoder_sa_type = 'sa'
nms_iou_threshold = -1

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

# DN
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = False
dn_labelbook_size = 91
match_unstable_error = True

# EMA
use_ema = False
ema_decay = 0.9997
ema_epoch = 0

use_detached_boxes_dec_out = False

# ============================================================
# 密度图监督参数（继承自 FDQ-Det AITOD 密度监督配置）
# ============================================================
use_density_supervision = True
density_sigma = 5
density_loss_type = 'smooth_l1'
density_pixel_weight = 0.1
density_smooth_weight = 0.0
density_integral_weight = 0.001
density_edge_weight = 0.0

density_ranking_weight = 0.1
density_feature_ranking_weight = 0.15
density_weight_support = 0.1
density_weight_distribution = 0.05

density_ranking_grid_size = 8
density_ranking_margin = 0.1
density_ranking_adaptive_margin = True

density_scale = 1.0
single_object_integral = 250.0
density_loss_coeff = 0.3

use_dfl = False
dfl_weight = 0.0
dfl_gamma = 0.0
