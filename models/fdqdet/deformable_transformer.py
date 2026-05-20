# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import random
import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util.misc import inverse_sigmoid
import util.misc as utils
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn
try:  # optional, only available when box attention ops are built
    from .ops.modules import MSDeformableBoxAttention  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without box attention
    MSDeformableBoxAttention = None  # noqa: N816
from .dn_components import prepare_for_cdn, dn_post_process

from .ccm import CategoricalCounting
try:
    from .ccm_improved import DensityMapper
except ImportError:
    DensityMapper = None
from ..loss_density import DensityMapLoss, count_objects_from_density
from ..loss_density_ranking import DensityMapLossWithRanking
try:
    from ..loss_density_with_dfl import DensityMapLossWithDFL
except ImportError:  # pragma: no cover - optional dependency
    DensityMapLossWithDFL = None
from .cgfe import CGFE, MultiScaleFeature
from .hf_suppressor import HighFrequencySuppressor
from .hf_suppressor_fixed import HighFrequencySuppressorFixed


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 # for deformable encoder-decoder
                 deformable_encoder=False,
                 deformable_decoder=False,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 # init query
                 learnable_tgt_init=False,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 # two stage
                 # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 two_stage_type='no',
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 # evo of anchors
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 # layer share
                 layer_share_type=None,
                 # for detach
                 rm_detach=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn'],
                 # for dn
                 embed_init_tgt=False,
                 use_detached_boxes_dec_out=False,
                 dynamic_query_list=None,
                 dynamic_query_margin=25,
                 ccm_cls_num=4,
                 use_high_freq_suppress=True,
                 use_high_freq_suppress_fixed=False,  # 新增：使用修复版HFS
                 high_freq_reduction=4,
                 high_freq_kernel_schedule=None,
                 use_adv_training=False,
                 adv_epsilon=1e-2,
                 adv_loss_weight=0.05,
                 feature_adv_epsilon=None,
                 query_adv_epsilon=None,
                 # 密度图监督学习参数
                 use_density_supervision=False,
                 density_loss_type='l2',
                 density_pixel_weight=1.0,
                 density_smooth_weight=0.1,
                 density_integral_weight=0.5,
                 density_edge_weight=0.0,
                 density_ranking_weight=0.0,
                 density_ranking_grid_size=8,
                 density_ranking_margin=0.1,
                 density_ranking_adaptive_margin=True,
                 single_object_integral=None,  # 可选，None时自动估计
                 density_scale=1.0,
                 use_dfl=False,
                 dfl_weight=1.0,
                 dfl_gamma=2.0,
                 ):

        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries  # number of query in decoder (default 300)
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.ccm_cls_num = ccm_cls_num
        self.use_high_freq_suppress = use_high_freq_suppress
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        assert layer_share_type in [None, 'encoder', 'decoder', 'both']
        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # choose encoder layer type
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(
                d_model, dim_feedforward, dropout, activation,
                num_feature_levels, nhead, enc_n_points,
                add_channel_attention=add_channel_attention,
                use_deformable_box_attn=use_deformable_box_attn,
                box_attn_type=box_attn_type
            )
        else:
            raise NotImplementedError   # FDQDet 强制使用可变形编码器
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None  # 编码器归一化层
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers,
            encoder_norm, d_model=d_model,
            num_queries=num_queries,
            deformable_encoder=deformable_encoder,
            enc_layer_share=enc_layer_share,
            two_stage_type=two_stage_type
        )

        # initialize custom components for FDQDet
        self.dynamic_query_list = dynamic_query_list    # 动态查询数量列表（根据CCM结果选择）
        self.dynamic_query_margin = dynamic_query_margin  # 在积分基础上增加的安全查询裕量
        self.use_density_supervision = use_density_supervision
        # 密度图积分值：可选参数，None时在forward中自动估计
        self.single_object_integral = single_object_integral
        self.density_scale = density_scale
        self._integral_sum = 0.0  # 用于在线估计积分值
        self._integral_count = 0  # 用于在线估计的样本数
        self.use_dfl = use_dfl
        self.dfl_weight = dfl_weight
        self.dfl_gamma = dfl_gamma
        
        # 根据配置选择CCM方案
        if use_density_supervision and DensityMapper is not None:
            # 使用新的密度图监督方案
            self.CCM = DensityMapper()
            self.ccm_mode = 'density'
            
            # 自动计算 single_object_integral（如果未指定）
            if self.single_object_integral is None:
                self.single_object_integral = self._estimate_single_object_integral(density_sigma=5)
                if utils.is_main_process():
                    print(f"[INFO] Auto-estimated single_object_integral = {self.single_object_integral:.6f}")
            
            # 选择损失函数：优先使用 Ranking Loss（若 ranking_weight > 0）
            if density_ranking_weight > 0:
                loss_cls = DensityMapLossWithRanking
                loss_kwargs = {
                    'loss_type': density_loss_type,
                    'weight_pixel': density_pixel_weight,
                    'weight_smooth': density_smooth_weight,
                    'weight_integral': density_integral_weight,
                    'weight_ranking': density_ranking_weight,
                    'ranking_grid_size': density_ranking_grid_size,
                    'ranking_margin': density_ranking_margin,
                    'adaptive_margin': density_ranking_adaptive_margin,
                    'density_scale': density_scale,
                }
                print(f"[INFO] Using DensityMapLossWithRanking (weight_ranking={density_ranking_weight})")
            else:
                loss_cls = DensityMapLossWithDFL if DensityMapLossWithDFL is not None else DensityMapLoss
                loss_kwargs = {
                    'loss_type': density_loss_type,
                    'weight_smooth': density_smooth_weight,
                    'weight_integral': density_integral_weight,
                    'weight_edge': density_edge_weight,
                    'density_scale': density_scale,
                }
                if loss_cls is DensityMapLossWithDFL:
                    loss_kwargs.update({
                        'weight_pixel': density_pixel_weight,
                        'use_dfl': use_dfl,
                        'dfl_weight': dfl_weight,
                        'dfl_gamma': dfl_gamma,
                    })
            self.density_loss = loss_cls(**loss_kwargs)
            print(f"[INFO] Using DensityMapper for CCM (Density Supervision Mode)")
        else:
            # 使用原始的分类方案
            self.CCM = CategoricalCounting(cls_num=self.ccm_cls_num)
            self.ccm_mode = 'classification'
            self.density_loss = None
            if use_density_supervision:
                print(f"[WARNING] use_density_supervision=True but DensityMapper not available, falling back to classification mode")
            else:
                print(f"[INFO] Using CategoricalCounting for CCM (Classification Mode)")
        self.use_high_freq_suppress_fixed = use_high_freq_suppress_fixed
        self.hf_suppressor = None
        if self.use_high_freq_suppress:
            # 根据配置选择原版或修复版HFS
            if use_high_freq_suppress_fixed:
                print(f"[INFO] Using Fixed HFS (sigmoid suppression) for training")
                self.hf_suppressor = HighFrequencySuppressorFixed(
                    channels=d_model,
                    reduction=high_freq_reduction,
                    num_feature_levels=self.num_feature_levels,
                    kernel_schedule=high_freq_kernel_schedule,
                )
            else:
                print(f"[INFO] Using Original HFS (tanh suppression)")
                self.hf_suppressor = HighFrequencySuppressor(
                    channels=d_model,
                    reduction=high_freq_reduction,
                    num_feature_levels=self.num_feature_levels,
                    kernel_schedule=high_freq_kernel_schedule,
                )
        self.CGFE = CGFE(   # 跨尺度特征融合模块（融合CCM特征与原始特征）
            gate_channels=256, reduction_ratio=16,
            num_feature_levels=self.num_feature_levels
        )
        self.multiscale = MultiScaleFeature(is_5_scale=True)    # 多尺度特征处理模块（将CCM特征升维到多尺度）
        self.use_adv_training = use_adv_training
        self.adv_epsilon = adv_epsilon
        self.adv_loss_weight = adv_loss_weight
        self.feature_adv_epsilon = feature_adv_epsilon if feature_adv_epsilon is not None else adv_epsilon
        self.query_adv_epsilon = query_adv_epsilon if query_adv_epsilon is not None else adv_epsilon

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(
                d_model, dim_feedforward,
                dropout, activation,
                num_feature_levels, nhead, dec_n_points,
                use_deformable_box_attn=use_deformable_box_attn,
                box_attn_type=box_attn_type,
                key_aware_type=key_aware_type,
                decoder_sa_type=decoder_sa_type,
                module_seq=module_seq
            )
        else:
            raise NotImplementedError   # FDQDet 强制使用可变形解码器
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model, query_dim=query_dim,
            modulate_hw_attn=modulate_hw_attn,
            num_feature_levels=num_feature_levels,
            deformable_decoder=deformable_decoder,
            decoder_query_perturber=decoder_query_perturber,
            dec_layer_number=dec_layer_number,
            rm_dec_query_scale=rm_dec_query_scale,
            dec_layer_share=dec_layer_share,
            use_detached_boxes_dec_out=use_detached_boxes_dec_out
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns    # 模式数量（两阶段候选查询生成用，默认0）

        self.query_redundancy = 1.15  # 查询裕量系数
        self.min_query_num = 300     # 最小查询数（防止计数值为0/过小）
        self.max_query_num = 1500    # 最大查询数（防止计数值异常大导致计算爆炸）
        self.query_step = 50         # 查询数取整步长（可选，让查询数是50的倍数，更稳定）

        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"     # DETR强制查询可学习
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)    # 查询嵌入层
            nn.init.normal_(self.tgt_embed.weight.data)     # 正态分布初始化查询
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed  # 两阶段模式嵌入数量
        self.two_stage_add_query_num = two_stage_add_query_num  # 两阶段新增查询数量
        self.two_stage_learn_wh = two_stage_learn_wh    # 两阶段是否学习参考点宽高
        assert two_stage_type in [
            'no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)

        # 标准两阶段的额外参数
        if two_stage_type == 'standard':
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)   # 编码器输出投影
            self.enc_output_norm = nn.LayerNorm(d_model)    # 编码器输出归一化

            # 两阶段的模式嵌入
            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(torch.Tensor(two_stage_pat_embed, d_model))
                nn.init.normal_(self.pat_embed_for_2stage)

            # 两阶段新增查询的嵌入
            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            # 两阶段是否学习参考点高宽
            if two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)    # 共享宽高嵌入（[1,2]）
            else:
                self.two_stage_wh_embedding = None

        # 单阶段的参考点初始化
        if two_stage_type == 'no':
            self.init_ref_points(num_queries)   # 初始化参考点（[num_queries,4]）

        # 编码器输出预测头（两阶段第一阶段用，后续由FDQDet主模型赋值）
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # evolution of anchors
        # 查询进化、自注意力移除、detach配置
        self.dec_layer_number = dec_layer_number    # 解码器查询数量进化配置
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[0] == num_queries * num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()    # 初始化参数（ Xavier 均匀初始化、可变形注意力参数重置）

        # 移除指定解码器层的自注意力（工程优化，减少计算）
        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            print("Removing the self-attn in {} decoder layers".format(rm_self_attn_layers))
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        # detach配置（控制部分模块输出是否detach，避免梯度问题）
        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        # 初始化所有可训练参数（Xavier均匀初始化，适合线性层/卷积层）
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 重置可变形注意力的参数（MSDeformAttn的特殊初始化）
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        # 多尺度层级嵌入的正态分布初始化
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)
        # 两阶段宽高嵌入的常数初始化（对应小尺寸参考点）
        if self.two_stage_learn_wh:
            nn.init.constant_(self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05)))

    def _estimate_single_object_integral(self, density_sigma=5, unit_mass=250.0, image_size=512):
        """
        从 DensityMapLoader 的参数自动估计 single_object_integral
        
        思路：
        1. 模拟在 image_size x image_size 图像中心放置单个bbox
        2. 使用 unit_mass 和 density_sigma 生成密度图
        3. 计算密度图的积分值
        4. 返回该值作为 single_object_integral
        
        Args:
            density_sigma: 高斯滤波标准差（与DensityMapLoader一致）
            unit_mass: 单个目标的初始脉冲大小（与DensityMapLoader.unit_mass一致）
            image_size: 用于估计的虚拟图像大小
            
        Returns:
            single_object_integral: float，单个目标的预期积分值
        """
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            print("[WARNING] scipy not available for single_object_integral estimation, using unit_mass as fallback")
            return float(unit_mass)
        
        # 创建虚拟图像中心的单个点脉冲
        density = np.zeros((image_size, image_size), dtype=np.float32)
        cx, cy = image_size // 2, image_size // 2
        density[cy, cx] = float(unit_mass)
        
        # 应用高斯平滑
        density = gaussian_filter(density, sigma=density_sigma)
        
        # 计算积分
        single_integral = float(density.sum())
        
        return single_integral


    def get_valid_ratio(self, mask):
        # 计算图像的有效区域比例（排除padding区域）
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)   # [bs, 2]
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        # 初始化参考点（单阶段用，两阶段参考点来自编码器筛选）
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)
        if self.random_refpoints_xy:
            # xy 坐标随机初始化（[0,1]均匀分布），转换为logit空间（sigmoid逆）
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False    # xy不学习

    def forward(self, srcs, masks, pos_embeds, dn_targets, args_dn,
                feature_perturbation: Optional[Tensor] = None,
                query_perturbation: Optional[Tensor] = None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + \
                    self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(
            lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=enc_refpoint_embed,  # bs, nq, 4
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

        counting_output, ccm_feature = self.CCM(memory, spatial_shapes)

        # 根据CCM模式选择目标数推理方式
        if self.ccm_mode == 'density':
            # 密度图模式：直接用低分辨率积分估计目标数，并加安全裕量
            # 首先计算每张图的积分（总质量）
            integral = counting_output.sum(dim=[2, 3])  # [bs, 1]
            if self.density_scale != 0:
                integral = integral / self.density_scale

            # 使用单目标积分值将积分转换为目标数（如果可用）
            # fallback: 若 single_object_integral 未配置，则保留原始积分作为估计（向后兼容）
            single_int = getattr(self, 'single_object_integral', None)
            if single_int is not None:
                try:
                    single_val = float(single_int)
                    if single_val > 0.0:
                        estimated_count = integral / single_val
                    else:
                        estimated_count = integral
                except Exception:
                    estimated_count = integral
            else:
                estimated_count = integral

            # 添加查询裕量，避免低估导致查询不足
            approx_count = estimated_count * self.query_redundancy + self.dynamic_query_margin
            approx_count = torch.clamp(approx_count, min=0.0)

            # 使用分位数而不是 max 来获取稳健的批次估计
            batch_count = approx_count.squeeze(1)
            robust_count = torch.quantile(batch_count, q=0.8).item()

            num_select = robust_count
            num_select = max(num_select, self.min_query_num)
            num_select = min(num_select, self.max_query_num)

            num_select = int(math.ceil(num_select / self.query_step) * self.query_step)
        else:
            # 分类模式：从分类输出选择
            _, predicted = torch.max(counting_output.data, 1)
            num_select = self.dynamic_query_list[max(predicted.tolist())]

        multi_ccm_feature = self.multiscale(ccm_feature)
        if self.hf_suppressor is not None and self.training:
            base_mask = (~masks[0]).float().unsqueeze(1)
            level_masks = [F.interpolate(base_mask, size=feat.shape[-2:], mode='nearest')
                           for feat in multi_ccm_feature]
            multi_ccm_feature = self.hf_suppressor(
                multi_ccm_feature, level_masks, apply=True)
        cgfe_out = self.CGFE(multi_ccm_feature, memory, spatial_shapes)
        memory = cgfe_out
        if feature_perturbation is not None:
            memory = memory + feature_perturbation

        tgt, refpoint_embed, attn_mask, dn_meta =\
            prepare_for_cdn(dn_args=(dn_targets, args_dn[0], args_dn[1], args_dn[2]),
                            training=args_dn[3], num_queries=num_select, num_classes=args_dn[4],
                            hidden_dim=args_dn[5], label_enc=args_dn[6])

        if self.two_stage_type == 'standard':
            # decide query hw
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None

            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, input_hw)
            output_memory = self.enc_output_norm(
                self.enc_output(output_memory))

            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                output_memory = output_memory.repeat(
                    1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(
                    1, self.two_stage_pat_embed, 1)

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat(
                    (output_proposals, refpoint_embed), dim=1)

            enc_outputs_class_unselected = self.enc_out_class_embed(
                output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(
                output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid

            topk = num_select
            topk_proposals = torch.topk(
                enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
            if self.embed_init_tgt:
                tgt_ = self.tgt_embed.weight[0:topk, None, :].repeat(
                    1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()
                if self.training and self.use_adv_training:
                    tgt_.requires_grad_(True)

            if refpoint_embed is not None:
                refpoint_embed = torch.cat(
                    [refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'no':
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(
                1, bs, 1).transpose(0, 1)  # nq, bs, d_model
            refpoint_embed_ = self.refpoint_embed.weight[:, None, :].repeat(
                1, bs, 1).transpose(0, 1)  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat(
                    [refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1)  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError(
                "unknown two_stage_type {}".format(self.two_stage_type))

        if query_perturbation is not None:
            tgt = tgt + query_perturbation.transpose(0, 1)

        #########################################################
        # Begin Decoder
        decoder_input = tgt.transpose(0, 1)
        hs, references = self.decoder(
            tgt=decoder_input,
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask)
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals

            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None

        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################
        adv_info = None
        if self.training and self.use_adv_training and feature_perturbation is None and query_perturbation is None:
            adv_info = {
                'encoder_memory': memory,
                'decoder_input': decoder_input,
            }

        return hs, references, hs_enc, ref_enc, init_box_proposal, dn_meta, counting_output, ccm_feature, num_select, adv_info
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None
class TransformerEncoder(nn.Module):

    def __init__(self,
                 encoder_layer, num_layers, norm=None, d_model=256,
                 num_queries=300,
                 deformable_encoder=False,
                 enc_layer_share=False, enc_layer_dropout_prob=None,
                 # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 two_stage_type='no'):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for prob in enc_layer_dropout_prob:
                assert 0.0 <= prob <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            proj_layer = nn.Linear(d_model, d_model)
            norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = nn.ModuleList([norm_layer])
                self.enc_proj = nn.ModuleList([proj_layer])
            else:
                self.enc_norm = nn.ModuleList([copy.deepcopy(norm_layer) for _ in range(num_layers - 1)])
                self.enc_proj = nn.ModuleList([copy.deepcopy(proj_layer) for _ in range(num_layers - 1)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(self,
                src: Tensor,
                pos: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                key_padding_mask: Tensor,
                ref_token_index: Optional[Tensor] = None,
                ref_token_coord: Optional[Tensor] = None):
        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None

        output = src
        if self.num_layers > 0 and self.deformable_encoder:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        else:
            reference_points = None

        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            gathered = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
            intermediate_output.append(gathered)
            intermediate_ref.append(ref_token_coord)

        for layer_id, layer in enumerate(self.layers):
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                rand_val = random.random()
                if rand_val < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                if self.deformable_encoder:
                    output = layer(src=output, pos=pos, reference_points=reference_points,
                                   spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                   key_padding_mask=key_padding_mask)
                else:
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(0, 1),
                                   key_padding_mask=key_padding_mask).transpose(0, 1)

            if ((layer_id == 0 and self.two_stage_type in ['enceachlayer', 'enclayer1'])
                    or self.two_stage_type == 'enceachlayer') and layer_id != self.num_layers - 1:
                output_memory, output_proposals = gen_encoder_output_proposals(output, key_padding_mask, spatial_shapes)
                output_memory = self.enc_norm[layer_id](self.enc_proj[layer_id](output_memory))

                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                ref_token_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
                ref_token_coord = torch.gather(output_proposals, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, 4))
                output = output_memory

            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                gathered = torch.gather(output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model))
                intermediate_output.append(gathered)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(intermediate_output)
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=False,
                 num_feature_levels=1,
                 deformable_decoder=False,
                 decoder_query_perturber=None,
                 dec_layer_number=None,
                 rm_dec_query_scale=False,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 use_detached_boxes_dec_out=False):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], f"query_dim should be 2/4 but {query_dim}"
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None if deformable_decoder else MLP(d_model, d_model, d_model, 2)

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError

        self.bbox_embed = None
        self.class_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder
        self.ref_anchor_head = MLP(d_model, d_model, 2, 2) if (not deformable_decoder and modulate_hw_attn) else None
        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for prob in dec_layer_dropout_prob:
                assert 0.0 <= prob <= 1.0

        self.rm_detach = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,
                level_start_index: Optional[Tensor] = None,
                spatial_shapes: Optional[Tensor] = None,
                valid_ratios: Optional[Tensor] = None):
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                else:
                    reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points)
                reference_points_input = None

            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder and self.query_pos_sine_scale is not None:
                query_sine_embed = query_sine_embed[..., :self.d_model] * self.query_pos_sine_scale(output)

            if not self.deformable_decoder and self.modulate_hw_attn:
                ref_hw_cond = self.ref_anchor_head(output).sigmoid()
                query_sine_embed[..., self.d_model // 2:] *= (ref_hw_cond[..., 0] / reference_points[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (ref_hw_cond[..., 1] / reference_points[..., 3]).unsqueeze(-1)

            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                rand_val = random.random()
                if rand_val < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                output = layer(
                    tgt=output,
                    tgt_query_pos=query_pos,
                    tgt_query_sine_embed=query_sine_embed,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_reference_points=reference_points_input,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,
                    memory_pos=pos,
                    self_attn_mask=tgt_mask,
                    cross_attn_mask=memory_mask)

            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                    nq_now = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](output)
                        topk_proposals = torch.topk(class_unselected.max(-1)[0], select_number, dim=0)[1]
                        new_reference_points = torch.gather(new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                else:
                    nq_now = select_number = 0

                reference_points = new_reference_points if (self.rm_detach and 'dec' in self.rm_detach) else new_reference_points.detach()
                ref_points.append(reference_points if self.use_detached_boxes_dec_out else new_reference_points)

            intermediate.append(self.norm(output))
            if self.dec_layer_number is not None and layer_id != self.num_layers - 1 and nq_now != select_number:
                output = torch.gather(output, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))

        intermediate_outputs = [itm.transpose(0, 1) for itm in intermediate]
        ref_points_outputs = [itm.transpose(0, 1) for itm in ref_points]
        return [intermediate_outputs, ref_points_outputs]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align'):
        super().__init__()
        if use_deformable_box_attn:
            self.self_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('dyrelu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        src = self.forward_ffn(src)
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='ca',
                 module_seq=['sa', 'ca', 'ffn']):
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']
        if use_deformable_box_attn:
            self.cross_attn = MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward_sa(self,
                   tgt: Optional[Tensor],
                   tgt_query_pos: Optional[Tensor] = None,
                   tgt_query_sine_embed: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   tgt_reference_points: Optional[Tensor] = None,
                   memory: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None,
                   memory_level_start_index: Optional[Tensor] = None,
                   memory_spatial_shapes: Optional[Tensor] = None,
                   memory_pos: Optional[Tensor] = None,
                   self_attn_mask: Optional[Tensor] = None,
                   cross_attn_mask: Optional[Tensor] = None):
        if self.self_attn is None:
            return tgt
        if self.decoder_sa_type == 'sa':
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        elif self.decoder_sa_type == 'ca_label':
            bs = tgt.shape[1]
            k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
            tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
        elif self.decoder_sa_type == 'ca_content':
            tgt2 = self.self_attn(
                self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                tgt_reference_points.transpose(0, 1).contiguous(),
                memory.transpose(0, 1),
                memory_spatial_shapes,
                memory_level_start_index,
                memory_key_padding_mask).transpose(0, 1)
        else:
            raise NotImplementedError(f"Unknown decoder_sa_type {self.decoder_sa_type}")

        tgt = tgt + self.dropout2(tgt2)
        return self.norm2(tgt)

    def forward_ca(self,
                   tgt: Optional[Tensor],
                   tgt_query_pos: Optional[Tensor] = None,
                   tgt_query_sine_embed: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   tgt_reference_points: Optional[Tensor] = None,
                   memory: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None,
                   memory_level_start_index: Optional[Tensor] = None,
                   memory_spatial_shapes: Optional[Tensor] = None,
                   memory_pos: Optional[Tensor] = None,
                   self_attn_mask: Optional[Tensor] = None,
                   cross_attn_mask: Optional[Tensor] = None):
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError(f"Unknown key_aware_type: {self.key_aware_type}")

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1),
            memory_spatial_shapes,
            memory_level_start_index,
            memory_key_padding_mask).transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        return self.norm1(tgt)

    def forward(self,
                tgt: Optional[Tensor],
                tgt_query_pos: Optional[Tensor] = None,
                tgt_query_sine_embed: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,
                memory: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,
                memory_spatial_shapes: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                self_attn_mask: Optional[Tensor] = None,
                cross_attn_mask: Optional[Tensor] = None):

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed,
                                      tgt_key_padding_mask, tgt_reference_points,
                                      memory, memory_key_padding_mask, memory_level_start_index,
                                      memory_spatial_shapes, memory_pos,
                                      self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed,
                                      tgt_key_padding_mask, tgt_reference_points,
                                      memory, memory_key_padding_mask, memory_level_start_index,
                                      memory_spatial_shapes, memory_pos,
                                      self_attn_mask, cross_attn_mask)
            else:
                raise ValueError(f"unknown funcname {funcname}")
        return tgt


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for _ in range(N)])
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise,
            w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out = False

    use_high_freq = getattr(args, 'use_high_freq_suppress', True)
    use_high_freq_fixed = getattr(args, 'use_high_freq_suppress_fixed', False)
    high_freq_reduction = getattr(args, 'high_freq_reduction', 4)
    high_freq_kernel_schedule = getattr(args, 'high_freq_kernel_schedule', None)
    use_adv_training = getattr(args, 'use_adv_training', False)
    adv_epsilon = getattr(args, 'adv_epsilon', 1e-2)
    adv_loss_weight = getattr(args, 'adv_loss_weight', 0.05)
    adv_feature_epsilon = getattr(args, 'adv_feature_epsilon', adv_epsilon)
    adv_query_epsilon = getattr(args, 'adv_query_epsilon', adv_epsilon)

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,

        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,

        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,

        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out,

        dynamic_query_list=args.dynamic_query_list,
        dynamic_query_margin=getattr(args, 'dynamic_query_margin', 50),
        ccm_cls_num=args.ccm_cls_num,
        use_high_freq_suppress=use_high_freq,
        use_high_freq_suppress_fixed=use_high_freq_fixed,
        high_freq_reduction=high_freq_reduction,
        high_freq_kernel_schedule=high_freq_kernel_schedule,
        use_adv_training=use_adv_training,
        adv_epsilon=adv_epsilon,
        adv_loss_weight=adv_loss_weight,
        feature_adv_epsilon=adv_feature_epsilon,
        query_adv_epsilon=adv_query_epsilon,

        # 密度图监督参数
        use_density_supervision=getattr(args, 'use_density_supervision', False),
        density_loss_type=getattr(args, 'density_loss_type', 'l2'),
        density_pixel_weight=getattr(args, 'density_pixel_weight', 1.0),
        density_smooth_weight=getattr(args, 'density_smooth_weight', 0.1),
        density_integral_weight=getattr(args, 'density_integral_weight', 0.5),
        density_edge_weight=getattr(args, 'density_edge_weight', 0.0),
        density_ranking_weight=getattr(args, 'density_ranking_weight', 0.0),
        density_ranking_grid_size=getattr(args, 'density_ranking_grid_size', 8),
        density_ranking_margin=getattr(args, 'density_ranking_margin', 0.1),
        density_ranking_adaptive_margin=getattr(args, 'density_ranking_adaptive_margin', True),
        single_object_integral=getattr(args, 'single_object_integral', None),
        density_scale=getattr(args, 'density_scale', 1.0),
        use_dfl=getattr(args, 'use_dfl', False),
        dfl_weight=getattr(args, 'dfl_weight', 1.0),
        dfl_gamma=getattr(args, 'dfl_gamma', 2.0),
    )
