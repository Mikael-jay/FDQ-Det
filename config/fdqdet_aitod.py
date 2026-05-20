_base_ = ['coco_transformer.py']

num_classes = 9
lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 1
weight_decay = 0.0001
# epochs = 48
epochs = 24
lr_drop = 18  # 缩短到 24 epochs，提前第一次学习率衰减（原为 40）
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = True
# lr_drop_list = [13, 20, 40]
lr_drop_list = [13] # 学习率衰减点（缩短到 24 epochs，仅在第 13 epoch 衰减）
# val_epoch = [31, 47]
val_epoch = [23]    # 缩短到 24 epochs，调整验证点
# dataset_file='aitod_v2'

modelname = 'fdqdet'
frozen_weights = None  # segmentation-only branch uses this; keep None to avoid masks assertion
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

# hf suppression
use_high_freq_suppress = True
high_freq_reduction = 4
high_freq_kernel_schedule = None

# adv training
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
nms_iou_threshold = -1  # default not use nms

dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True

# for dn
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = False
dn_labelbook_size = 91
match_unstable_error = True

# for ema
use_ema = False
ema_decay = 0.9997
ema_epoch = 0

use_detached_boxes_dec_out = False

# ============================================================
# 密度图监督学习参数（改进版 v2 - 基于 Crowd Counting）
# ============================================================
# 是否使用密度图监督（True: 密度图监督  False: 分类监督 默认值）
use_density_supervision = True

# 高斯核标准差（用于实时密度图生成）
density_sigma = 5

# 改进的损失函数设计：平衡像素级和积分约束
# 关键改进：
# 1. 恢复 integral loss 以保持密度图本质约束（积分 = 数量）
# 2. 降低 pixel_weight 避免梯度规模灾难（66800 个像素的梯度会淹没检测损失）
# 3. 使用 Softplus 激活函数（crowd counting 主流，避免 Sigmoid 的梯度消失和上界限制）
# 4. 初始化最后一层 bias 为正值，避免模型停在全 0 解
# ============================================================
# 密度损失配置（引入 Ranking Loss 解决掩码化问题）
# ============================================================
# 核心思想：
# - Pixel Loss: 约束绝对值
# - Integral Loss: 约束总量
# - Ranking Loss: 约束相对关系（GT中密的区域，Pred中也应更亮）← 新增
#
# Ranking Loss 优势：
# 1. 强制模型学习密度梯度（解决掩码化/二值化）
# 2. 计算代价低（8×8 grid，64个patch）
# 3. 与pixel/integral互补（相对关系 vs 绝对值）
# ============================================================
density_loss_type = 'smooth_l1'  # 损失函数类型：'l2' (MSE), 'l1' (MAE), 'smooth_l1'
density_pixel_weight = 0.1  # 像素级损失权重
density_smooth_weight = 0.0  # 平滑性约束权重（暂不使用）
density_integral_weight = 0.001  # 积分约束权重（保证总量正确）
density_edge_weight = 0.0  # 边界约束权重（暂不使用）

# Ranking Loss 新增：多约束设计解决二维分布问题
density_ranking_weight = 0.1  # Ranking约束权重（强制相对密度关系）
density_feature_ranking_weight = 0.15  # 特征级Ranking约束权重（基于CCM特征）
density_weight_support = 0.1  # Support约束权重（GT为0的地方，预测必须≈0）
density_weight_distribution = 0.05  # 分布对齐权重（KL散度，使概率分布一致）

density_ranking_grid_size = 8  # Ranking grid大小（8×8=64个patch）
density_ranking_margin = 0.1  # Ranking margin
density_ranking_adaptive_margin = True  # 是否自适应margin（基于GT密度差异）

density_scale = 1.0  # 密度图尺度因子（GT被放大倍数，计数时还原）
single_object_integral = 250.0  # 单个目标的密度积分值（unit_mass=250 for sigma=5）
density_loss_coeff = 0.3  # 密度损失总权重（平衡密度监督与检测任务）

use_dfl = False  # 不使用 DFL
dfl_weight = 0.0
dfl_gamma = 0.0
