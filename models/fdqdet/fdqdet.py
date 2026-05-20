# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn, dn_post_process


class FDQDet(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """

    def __init__(self,
                 backbone,  # 骨干网络
                 transformer,   # 可变形 Transformer (encoder + decoder)
                 num_classes,   # 最大类别数 + 1 (背景)
                 num_queries,   # object query 数量
                 aux_loss=False,    # 是否使用辅助损失 (解码器每层计算损失)
                 iter_update=False,  # 是否启用查询迭代更新
                 query_dim=2,       # 查询维度 (默认为 2: xy, 实际使用强制为 4: xyhw)
                 random_refpoints_xy=False,  # 是否随机初始化查询的 xy 坐标
                 # 是否固定查询的 hw (默认 -1: 学习每个查询的 hw； >0: 固定值； -2: 共享学习)
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,      # 使用的多尺度特征层数 (5)
                 nheads=8,                  # 注意力头数

                 # two stage
                 two_stage_type='no',               # 两阶段类型 ['no', 'standard']
                 two_stage_add_query_num=0,         # 两阶段中添加的查询数
                 dec_pred_class_embed_share=True,   # 解码器层间类别预测层是否共享
                 dec_pred_bbox_embed_share=True,    # 解码器层间边界框预测层是否共享
                 two_stage_class_embed_share=True,  # 两阶段类别预测层是否与解码器共享
                 two_stage_bbox_embed_share=True,   # 两阶段边界框预测层是否与解码器共享
                 # 解码器自注意力类型 ['sa', 'ca_label', 'ca_content']
                 decoder_sa_type='sa',
                 num_patterns=0,            # 位置编码模式数 (0: 不使用)

                 dn_number=100,             # dn 样本数量
                 dn_box_noise_scale=0.4,    # dn 边界框噪声比例
                 dn_label_noise_ratio=0.5,  # dn 标签噪声比例
                 dn_labelbook_size=100,     # dn 错误标签字典大小 (实际使用 91)
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw:   -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)  # 骨干网络输出的尺度数量 (5)
            input_proj_list = []

            # 处理骨干网络输出的每个尺度：1x1 降维 + GroupNorm
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]  # 当前尺度的通道数
                input_proj_list.append(nn.Sequential(
                    # 1x1卷积：通道数→hidden_dim
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    # 分组归一化：稳定训练，避免 BatchNorm 的 batch 依赖
                    nn.GroupNorm(32, hidden_dim),
                ))

            # 若需要更多尺度, 使用 3x3 步长为 2 的卷积下采样
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            # 用 ModuleList 包装，支持 GPU 并行
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss  # 是否启用辅助损失
        self.box_pred_damping = box_pred_damping = None  # 边界框预测阻尼（暂未使用）

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"  # FDQDet 强制启用迭代更新

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share

        # prepare class & box embed
        # 类别预测头：hidden_dim→num_classes（线性层）
        _class_embed = nn.Linear(hidden_dim, num_classes)
        # 边界框预测头：3 层 MLP，输入输出 hidden_dim，最终输出 4 维偏移
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # init the two embed layers
        prior_prob = 0.01  # 类别预测的先验概率（解决类别不平衡）
        bias_value = -math.log(
            (1 - prior_prob) / prior_prob)  # 使用逻辑斯蒂函数计算类别偏置：让初始预测背景的概率高
        _class_embed.bias.data = torch.ones(
            self.num_classes) * bias_value  # 类别预测头偏置初始化
        # 边界框预测头最后一层权重→0（初始偏移小）
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        # 边界框预测头最后一层偏置→0
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        # 根据是否共享预测头，构建解码器每层的预测头
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(
                transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(   # 若不共享则深拷贝
                _bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(
                transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(  # 若不共享则深拷贝
                _class_embed) for i in range(transformer.num_decoder_layers)]

        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        # 将预测头传给 Transformer 解码器（解码器每层需计算预测，用于迭代更新）
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num  # 0
        assert two_stage_type in [
            'no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(
                    _bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(
                    _class_embed)

            self.refpoint_embed = None  # 参考点嵌入（初始为None）
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        # 解码器自注意力类型
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            # 基于标签的交叉注意力：用类别标签引导注意力
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            # 自注意力（sa）或基于内容的交叉注意力（ca_content）：无需标签嵌入
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()  # 初始化输入投影层参数

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            # 卷积层权重：Xavier 均匀初始化（适合线性层/卷积层，保证前向和反向传播方差一致）
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            # 卷积层偏置：初始化为0（避免偏置对初始训练的干扰）
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        # 参考点嵌入：用 Embedding 生成 use_num_queries 个参考点（每个参考点维度: query_dim=4）
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

        if self.random_refpoints_xy:
            # 参考点 xy 坐标随机初始化（在[0,1]均匀分布）
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            # 转换为 sigmoid 的原空间（logit 空间）：因为最终输出要经过 sigmoid 归一化，原空间优化更稳定
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2])
            # xy 坐标不训练（固定初始位置）
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        # 模式>0：处理参考点宽高（hw）的初始化模式
        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy  # 固定宽高需配合随机xy
            # 宽高设为固定值（如fix_refpoints_hw=0.1，对应图像10%的宽高）
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        # 模式-1：每个参考点的宽高单独学习（requires_grad默认True）
        elif int(self.fix_refpoints_hw) == -1:
            pass
        # 模式-2：仅学习xy坐标（参考点维度改为2），宽高共享一个参数（hw_embed）
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError(
                'Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def forward(self, samples: NestedTensor, targets: List = None,
                feature_perturbation: torch.Tensor = None,
                query_perturbation: torch.Tensor = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []  # 存储处理后的特征张量（通道统一为hidden_dim）
        masks = []  # 存储处理后的特征掩码（与特征同分辨率）
        for l, feat in enumerate(features):
            src, mask = feat.decompose()  # 从NestedTensor中拆分特征张量和掩码
            srcs.append(self.input_proj[l](src))  # 输入通道映射：原通道数 → hidden_dim
            masks.append(mask)  # 收集掩码
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:  # 基于骨干网络最后一个尺度生成
                    src = self.input_proj[l](features[-1].tensors)
                else:  # 基于上一个生成的尺度生成（3x3卷积下采样）
                    src = self.input_proj[l](srcs[-1])
                # 生成该尺度的掩码（从原始图像掩码插值而来）
                m = samples.mask
                mask = F.interpolate(
                    m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # 生成该尺度的位置编码（复用骨干网络的位置编码逻辑）
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # 打包去噪训练相关参数（传给Transformer）
        args_dn = [self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale,
                   self.training, self.num_classes, self.hidden_dim, self.label_enc]

        # attn_mask !!!!!!!!!!!!!!!!!!!!!
        # 调用 Transformer 前向传播，输出关键结果
        hs, reference, hs_enc, ref_enc, init_box_proposal, dn_meta, counting_output, ccm_feature, num_select, adv_info = \
            self.transformer(
                srcs, masks, poss, targets, args_dn,
                feature_perturbation=feature_perturbation,
                query_perturbation=query_perturbation)
        """ hs: 解码器输出的查询向量, shape [num_dec_layers, batch_size, num_queries, hidden_dim]
            reference: 解码器每层的参考点, shape [num_dec_layers+1, batch_size, num_queries, 4] (+1 为初始参考点)
            hs_enc: 编码器输出的特征, shape [num_enc_layers, batch_size, num_enc_queries, hidden_dim]（两阶段用）
            ref_enc: 编码器输出的参考点, shape [num_enc_layers, batch_size, num_enc_queries, 4]（两阶段用）
            init_box_proposal: 两阶段第一阶段的候选边界框，用于匹配
            dn_meta: 去噪训练的元信息（如噪声查询索引），用于后处理分离噪声 / 正常预测
            counting_output: 预测的目标数量（可选）
            num_select: 两阶段筛选的候选查询数量
        """

        # In case num object=0: 若目标数量为0，添加微小值避免数值异常（如NaN）
        hs[0] += self.label_enc.weight[0, 0]*0.0

        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], self.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack([
            layer_cls_embed(layer_hs)
            for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
        ])

        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = dn_post_process(
                outputs_class,
                outputs_coord_list,
                dn_meta,
                self.aux_loss,
                self._set_aux_loss)

        # 取最后一层解码器的预测结果作为最终输出
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord_list)

        # for encoder output：若有编码器输出（两阶段），添加第一阶段的预测
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {
                'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {
                'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs：编码器中间层输出（若编码器层数>1）
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + \
                        inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b}for a, b in zip(enc_outputs_class, enc_outputs_coord)]

        # 补充去噪元信息、预测框数量、筛选的查询数
        out['dn_meta'] = dn_meta
        
        # 根据CCM模式输出相应的预测结果
        if hasattr(self.transformer, 'ccm_mode') and self.transformer.ccm_mode == 'density':
            # 密度图监督模式
            out['pred_density_map'] = counting_output  # [bs, 1, H, W]
            out['ccm_feature'] = ccm_feature
        else:
            # 分类监督模式（原始方案）
            out['pred_bbox_number'] = counting_output  # [bs, cls_num]
        
        out['num_select'] = num_select

        if adv_info is not None:
            out['adv_info'] = adv_info

        return out

    # 告知 JIT 无需编译此方法（兼容动态列表输出）
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
            box coordinate loss: L1 + GIoU
            classification loss: Focal loss
            auxiliary decoding losses (if aux_loss is activated)
            DN losses (if use_dn is activated)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss (0.25 in common)
        """
        super().__init__()
        self.num_classes = num_classes  # 类别数
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.losses = losses

    # 类别损失 (Sigmoid Focal Loss)
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """ Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # 预测类别 logits，shape [batch_size, num_queries, num_classes]
        src_logits = outputs['pred_logits']

        # 步骤1：根据匹配索引，获取 predictions 与 targets 的对应关系
        # idx = (batch_idx, src_idx)，对应匹配的预测索引
        idx = self._get_src_permutation_idx(indices)
        # 提取匹配的真实类别：将每个批次的真实目标类别按匹配关系拼接
        target_classes_o = torch.cat(
            # t 是单个批次的 targets，J 是该批次真实目标的匹配索引，t["labels"][J] 是匹配的真实类别
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # 步骤2：构建“目标类别矩阵”（默认背景类，匹配位置替换为真实类别）
        # 初始化：所有预测查询的类别默认设为 “背景类” （ID=self.num_classes）
        target_classes = torch.full(
            src_logits.shape[:2],   # 形状：[batch_size, num_queries]（每个查询对应一个类别）
            self.num_classes,       # 默认值：背景类ID
            dtype=torch.int64,
            device=src_logits.device
        )
        # 根据匹配索引 idx，将匹配位置替换为真实类别
        target_classes[idx] = target_classes_o

        # 步骤3：计算 Sigmoid Focal Loss
        # 将类别标签转换为 one-hot 编码，形状：[batch_size, num_queries, num_classes]
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,   # 内存布局 (strided / sparse_coo)
            device=src_logits.device
        )
        # scatter_：按索引填充 One-Hot（dim=2 是类别维度，target_classes.unsqueeze(-1) 是索引）
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # 截断最后一维（去掉临时扩展的维度），最终形状：[batch_size, num_queries, num_classes]
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        # 步骤4：计算 Sigmoid Focal Loss
        loss_ce = sigmoid_focal_loss(
            src_logits,             # 预测 logits
            target_classes_onehot,  # 目标 One-Hot 编码
            num_boxes,              # 平均目标数量（用于归一化，避免批次大小影响）
            alpha=self.focal_alpha,
            gamma=2
        ) * src_logits.shape[1]     # 乘以查询数量（补偿后续归一化）

        # 步骤5：构建损失字典
        losses = {'loss_ce': loss_ce}

        # 步骤6：计算分类误差（仅用于日志，不参与梯度传播）
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - \
                accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device

        # 步骤1：计算每个批次的真实目标数量
        tgt_lengths = torch.as_tensor(
            # 每个批次的真实目标数（len(v["labels"])是该批次目标数）
            [len(v["labels"]) for v in targets],
            device=device
        )

        # 步骤2：计算每个批次的预测目标数量
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)

        # 步骤3：计算L1损失（基数误差）
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """ Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
            The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        # 步骤1：提取匹配的预测边界框和真实边界框
        src_boxes = outputs['pred_boxes'][idx]
        # 拼接每个批次的匹配真实框：[num_matched, 4]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        # 步骤2：计算L1损失（坐标绝对误差）
        # [num_matched, 4]（不自动求和）
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        # 步骤3：构建损失字典，归一化L1损失（除以总目标数num_boxes）
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # 步骤4：计算GIoU损失（需先将cxcywh格式转为xyxy格式）
        # box_ops.box_cxcywh_to_xyxy：中心坐标+宽高 → 左上角+右下角坐标
        loss_giou = 1 - torch.diag(  # 取GIoU矩阵的对角线（匹配对的GIoU值）
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # 步骤5：计算xy（中心坐标）和hw（宽高）的单独损失（仅日志用，无梯度）
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """ Compute the losses related to the masks: the focal loss and the dice loss.
            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        # 步骤1：获取匹配的预测掩码和真实掩码索引
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # 步骤2：提取匹配的预测掩码和真实掩码
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]

        # 步骤3：处理真实掩码（转为NestedTensor，统一尺寸）
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # 步骤4：插值预测掩码，与真实掩码尺寸匹配（模型输出掩码尺寸可能与真实不同）
        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)

        # 步骤5：展平真实掩码（与预测掩码格式一致）
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)

        # 步骤6：计算掩码损失（Focal Loss + Dice Loss）
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices: 根据匹配关系生成预测的全局索引
        # 步骤1：生成批次索引（每个批次的样本对应其批次号i）
        batch_idx = torch.cat(  # 生成与 src 同形状的 i 索引向量
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        # 步骤2：生成预测查询的局部索引（每个批次内的匹配索引）
        src_idx = torch.cat([src for (src, _) in indices])
        # 步骤3：返回全局索引（批次索引+局部索引），可直接用于提取匹配的预测样本
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices (same as _get_src_permutation_idx)
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        # 步骤1：分离主输出（不含辅助输出aux_outputs），用于计算初始匹配
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != 'aux_outputs'
        }
        # outputs.values() 本身是一个 python 字典视图对象，不支持索引
        device = next(iter(outputs.values())).device

        # 步骤2：用匹配器计算主输出与真实目标的匹配关系（匈牙利匹配）
        indices = self.matcher(outputs_without_aux, targets)
        # 若需返回索引（可视化），保存初始匹配索引
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # 步骤3：计算平均目标数量（用于损失归一化，支持分布式训练）
        num_boxes = sum(len(t["labels"])
                        for t in targets)  # 在 CPU 中计算每个进程中的目标总数
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=device)  # 将计算结果转移到 GPU
        if is_dist_avail_and_initialized():  # 分布式训练：所有进程归约（求和）并同步 num_boxes 值
            torch.distributed.all_reduce(num_boxes)
        # 归一化系数：num_boxes / 进程数，最小为1（避免除以0），并重新转移到 CPU
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 步骤4：初始化损失字典
        losses = {}

        # 步骤5：处理去噪训练损失（仅训练时且 dn_meta 存在）
        dn_meta = outputs['dn_meta']
        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            # 准备去噪损失的参数（output_known_lbs_bboxes：去噪相关预测，single_pad：每组噪声padding尺寸）
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(
                dn_meta)
            # 构建去噪训练的正负样本索引（dn_pos_idx：噪声查询-真实目标匹配；dn_neg_idx：噪声查询-负样本）
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    # 真实目标索引：[len(targets[i]['labels'])] → 扩展为[scalar, len(...)]（scalar是噪声组数）
                    t = torch.range(
                        0, len(targets[i]['labels']) - 1).long().cuda()
                    # 在第 0 维增加一个维度，并复制 scalar 份
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    # 噪声查询索引：每组噪声的起始位置 + 真实目标索引的偏移
                    output_idx = (torch.tensor(range(scalar)) *
                                  single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))    # 正样本索引（噪声查询匹配真实目标）
                dn_neg_idx.append((output_idx + single_pad //
                                  2, tgt_idx))  # 负样本索引（噪声查询）

            # 计算去噪损失（调用get_loss，损失名加_dn后缀，如loss_ce→loss_ce_dn）
            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}  # 去噪损失不记录分类误差日志
                l_dict.update(self.get_loss(
                    loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar, **kwargs))
            l_dict = {k + f'_dn': v for k,
                      v in l_dict.items()}  # 加_dn后缀，区分普通损失
            losses.update(l_dict)
        else:   # 不启用去噪训练：添加空的去噪损失（避免KeyError）
            l_dict = {
                'loss_bbox_dn': torch.as_tensor(0.).to('cuda'),
                'loss_giou_dn': torch.as_tensor(0.).to('cuda'),
                'loss_ce_dn': torch.as_tensor(0.).to('cuda'),
                'loss_xy_dn': torch.as_tensor(0.).to('cuda'),
                'loss_hw_dn': torch.as_tensor(0.).to('cuda'),
                'cardinality_error_dn': torch.as_tensor(0.).to('cuda')
            }
            losses.update(l_dict)

        # 步骤6：计算主损失（主输出的损失，不含辅助输出）
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes))

        # 步骤7：计算辅助损失（aux_outputs，解码器中间层的输出，加速收敛）
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                # 分别计算每层辅助输出的匹配关系
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)

                # 计算每层辅助输出的损失
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    # 损失名加层号后缀，如loss_ce→loss_ce_0
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                # 计算每层辅助输出的去噪损失（仅训练时且 dn_meta 存在）
                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    # 辅助输出的去噪预测
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}
                        l_dict.update(self.get_loss(
                            loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar, **kwargs))
                    # 加_dn和层号后缀，如loss_ce→loss_ce_dn_0
                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:   # 不启用去噪：添加空的辅助去噪损失
                    l_dict = {
                        'loss_bbox_dn': torch.as_tensor(0.).to('cuda'),
                        'loss_giou_dn': torch.as_tensor(0.).to('cuda'),
                        'loss_ce_dn': torch.as_tensor(0.).to('cuda'),
                        'loss_xy_dn': torch.as_tensor(0.).to('cuda'),
                        'loss_hw_dn': torch.as_tensor(0.).to('cuda'),
                        'cardinality_error_dn': torch.as_tensor(0.).to('cuda')
                    }
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 步骤8：计算中间输出损失（interm_outputs，两阶段检测的第一阶段输出）
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(
                    loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # 步骤9：计算编码器输出损失（enc_outputs，编码器中间层的输出）
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 步骤10：返回结果（含匹配索引或仅损失）
        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list
        return losses

    def prep_for_dn(self, dn_meta):
        # 去噪相关的预测输出
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        # 噪声组数和每组的填充数
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0  # 每组的填充数必须能被组数整除
        single_pad = pad_size//num_dn_groups  # 每组噪声的padding尺寸

        return output_known_lbs_bboxes, single_pad, num_dn_groups

# new
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api """

    def __init__(self, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.nms_iou_threshold = nms_iou_threshold  # 默认 -1 不启用 NMS

    @torch.no_grad()
    def forward(self, outputs, target_sizes, target_num=300, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model (dict containing 'pred_logits', 'pred_boxes')
            target_sizes: tensor of dimension [batch_size, 2] containing the size of each images of the batch
                          - For evaluation, this must be the original image size (before any data augmentation)
                          - For visualization, this should be the image size after data augment, but before padding
            target_num: number of boxes to be selected (default 300). Should ideally match model's predicted num_select.
            not_to_xyxy: if True, do not convert box format from cxcywh to xyxy
            test: if True, xyxy converted to xywh format
        Returns:
            A list of dicts (one for each image) containing:
                "scores": Tensor of dim [num_boxes] containing the scores for each box
                "labels": Tensor of dim [num_boxes] containing the labels for each box
                "boxes": Tensor of dim [num_boxes, 4] containing the predicted boxes in (x1, y1, x2, y2) format,
                         with (0, 0) being the top-left corner of the image and (target_size[0 - 1, target_size[1])
                         being the bottom-right corner of the image
        """
        # out_logits=[batch_size, num_queries, num_classes]; out_bbox=[batch_size, num_queries, 4]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes), "Mismatched batch size between outputs and target_sizes"
        assert target_sizes.shape[1] == 2, "target_sizes must have shape [batch_size, 2] (h, w)"

        # CCM-style decode: sigmoid -> per-query max score/label -> sort & Top-K (no background channel)
        probs = torch.softmax(out_logits, dim=-1)   # [batch_size, num_queries, num_classes]
        max_scores, max_labels = probs.max(dim=-1)  # both [batch_size, num_queries]

        # Prefer model-predicted num_select if present
        num_select = target_num
        if isinstance(outputs, dict) and 'num_select' in outputs:
            try:
                num_select = int(outputs['num_select'])
            except Exception:
                num_select = target_num

        # Select the top-k queries based on their per-query maximum score
        topk_scores, topk_indices = torch.topk(max_scores, k=num_select, dim=1)  # [batch_size, num_select]

        # We'll convert boxes for ALL queries first, then gather selected ones for output
        # Start from normalized cxcywh boxes produced by the model
        all_boxes = out_bbox  # [batch_size, num_queries, 4]

        # --- Rest of the processing remains largely the same ---
        if not_to_xyxy:
            # Keep in cxcywh for both ALL and selected
            all_boxes_xy = all_boxes
        else:
            # Convert ALL normalized cxcywh to xyxy first
            all_boxes_xy = box_ops.box_cxcywh_to_xyxy(all_boxes)

        # For test mode, convert final format from xyxy->xywh (if applicable)
        if test:
            assert not not_to_xyxy, "not_to_xyxy should be False when test is True"

        # target_sizes is [batch_size, 2], unbind to get (img_h, img_w) for each image
        img_h, img_w = target_sizes.unbind(1) # img_h: [batch_size], img_w: [batch_size]

        # Create scale factors to convert normalized coords to absolute pixel coords
        # scale_fct shape: [batch_size, 4] -> (img_w, img_h, img_w, img_h)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        # Scale ALL boxes to absolute coordinates
        all_boxes_abs = all_boxes_xy * scale_fct[:, None, :]
        # Gather selected boxes (keeping the same coordinate system as all_boxes_abs)
        boxes_abs = torch.gather(all_boxes_abs, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))

        # --- Optional NMS ---
        if self.nms_iou_threshold > 0:
            # Apply NMS per image in the batch
            results = []
            for i in range(all_boxes_abs.shape[0]): # Iterate over batch dimension
                b = boxes_abs[i] # [num_select, 4]
                s = topk_scores[i] # [num_select]
                l = torch.gather(max_labels, 1, topk_indices)[i] # [num_select]
                # torchvision nms expects xyxy format and scores
                keep_indices = nms(b, s, self.nms_iou_threshold)
                kept = {
                    'scores': s[keep_indices],
                    'labels': l[keep_indices],
                    'boxes': b[keep_indices]
                }
                # Also include ALL pre-topk info for analysis parity
                kept['all_scores'] = max_scores[i]
                kept['all_labels'] = max_labels[i]
                kept['all_boxes'] = all_boxes_abs[i]
                kept['num_select_used'] = int(num_select)
                results.append(kept)
        else:
            # No NMS: Return the top-k selected results directly
            results = []
            gathered_labels = torch.gather(max_labels, 1, topk_indices)  # [B, num_select]
            for i in range(all_boxes_abs.shape[0]):
                item = {
                    'scores': topk_scores[i],
                    'labels': gathered_labels[i],
                    'boxes': boxes_abs[i],
                    # Provide ALL pre-topk info for downstream detailed analysis
                    'all_scores': max_scores[i],
                    'all_labels': max_labels[i],
                    'all_boxes': all_boxes_abs[i],
                    'num_select_used': int(num_select),
                }
                results.append(item)

        return results


# class PostProcess(nn.Module):
#     """ This module converts the model's output into the format expected by the coco api """

#     def __init__(self, nms_iou_threshold=-1) -> None:
#         super().__init__()
#         self.nms_iou_threshold = nms_iou_threshold  # 默认 -1 不启用 NMS

#     @torch.no_grad()
#     def forward(self, outputs, target_sizes, target_num=300, not_to_xyxy=False, test=False):
#         """ Perform the computation
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size, 2] containing the size of each images of the batch
#                           - For evaluation, this must be the original image size (before any data augmentation)
#                           - For visualization, this should be the image size after data augment, but before padding
#             target_num: number of boxes to be selected (default 300)
#             not_to_xyxy: if True, do not convert box format from cxcywh to xyxy
#             test: if True, xyxy converted to xywh format
#         Returns:
#             A list of dicts (one for each image) containing:
#                 "scores": Tensor of dim [num_boxes] containing the scores for each box
#                 "labels": Tensor of dim [num_boxes] containing the labels for each box
#                 "boxes": Tensor of dim [num_boxes, 4] containing the predicted boxes in (x1, y1, x2, y2) format,
#                          with (0, 0) being the top-left corner of the image and (target_size[0 - 1, target_size[1])
#                          being the bottom-right corner of the image
#         """

#         num_select = target_num
#         # out_logits=[batch_size, num_queries, num_classes+1]; out_bbox=[batch_size, num_queries, 4]
#         out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

#         assert len(out_logits) == len(target_sizes), "Mismatched batch size between outputs and target_sizes"
#         assert target_sizes.shape[1] == 2, "target_sizes must have shape [batch_size, 2] (h, w)"

#         prob = out_logits.sigmoid()

#         # Top-K筛选：从所有类别预测中选择置信度最高的num_select个类别
#         # topk_values=[batch_size, num_select]; topk_indexes=[batch_size, num_select]
#         topk_values, topk_indexes = torch.topk(
#             prob.view(out_logits.shape[0], -1), # 展平后的置信度：[batch_size, num_queries * (num_classes+1)]
#             num_select,     # 筛选数量：target_num（默认300）
#             dim=1           # 按批次内的元素维度筛选
#         )

#         # [batch_size, num_select]，筛选出的最高置信度
#         scores = topk_values
#         # 查询索引
#         topk_boxes = topk_indexes // out_logits.shape[2]
#         # 类别索引
#         labels = topk_indexes % out_logits.shape[2]

#         if not_to_xyxy:
#             boxes = out_bbox
#         else:
#             boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

#         if test:
#             assert not not_to_xyxy, "not_to_xyxy should be False when test is True"
#             # convert to xywh
#             boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]

#         # 根据查询索引从所有预测框中筛选出 topk_boxes 对应的边界框
#         # 输出形状：[batch_size, num_select, 4]（每个批次的 num_select 个目标，每个目标4个坐标）
#         boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

#         # target_sizes 是 [batch_size, 2]，每一行是 (img_h, img_w)，unbind(1) 按列拆分
#         img_h, img_w = target_sizes.unbind(1)

#         # 构建缩放因子，x坐标（x1, x2）需乘以图像宽度，y坐标（y1, y2）需乘以图像高度
#         # scale_fct 形状：[batch_size, 4]，每一行是 (img_w, img_h, img_w, img_h)
#         scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

#         # 将归一化坐标转为绝对坐标
#         # boxes: [batch_size, num_select, 4]
#         # scale_fct[:, None, :]: 扩展维度为 [batch_size, 1, 4]（与 boxes 的批次、坐标维度匹配）
#         boxes = boxes * scale_fct[:, None, :]

#         if self.nms_iou_threshold > 0:
#             # 对每个批次的目标单独执行 NMS
#             # 列表推导式：遍历每个批次的 boxes 和 scores，调用nms函数返回保留的索引
#             item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold)
#                             for b, s in zip(boxes, scores)]
#             # 根据 NMS 保留的索引，筛选各个批次的 scores、labels、boxes（这里i是当前批次筛选出来的索引的序列）
#             results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]}
#                        for s, l, b, i in zip(scores, labels, boxes, item_indices)]
#         else:
#             # 不启用 NMS：直接返回 Top-K 筛选后的结果（无去重）
#             results = [{'scores': s, 'labels': l, 'boxes': b}
#                        for s, l, b in zip(scores, labels, boxes)]

#         return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='fdqdet')
def build_fdqdet(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True

    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = FDQDet(
        backbone,  # 骨干网络（多尺度特征+位置编码）
        transformer,  # 可变形 Transformer（编码器+解码器）
        num_classes=num_classes,  # 最大类别 ID + 1（如 COCO 为 91）
        num_queries=args.num_queries,  # 解码器查询数量（如 300，每个查询预测一个目标）
        aux_loss=True,  # 是否使用“辅助损失”（解码器每层都计算损失，加速收敛）
        iter_update=True,  # 查询（Queries）迭代更新（解码过程中优化查询）
        query_dim=4,  # 查询向量的维度（4 对应边界框的 (x,y,w,h) 或 (x1,y1,x2,y2)）
        random_refpoints_xy=args.random_refpoints_xy,  # 参考点（RefPoints）的 XY 坐标是否随机初始化
        fix_refpoints_hw=args.fix_refpoints_hw,  # 参考点的高宽（HW）是否固定（不随特征尺度变化）
        num_feature_levels=args.num_feature_levels,  # 使用的特征尺度数量（如 3：layer2~layer4）
        nheads=args.nheads,  # Transformer 注意力头数量（如 8，多头注意力捕捉多维度信息）
        dec_pred_class_embed_share=dec_pred_class_embed_share,  # 类别预测头共享
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,  # 边界框预测头共享
        # 两阶段检测相关参数
        two_stage_type=args.two_stage_type,  # 两阶段类型（如 'query_selection'，先选候选查询再细化）
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,  # 两阶段边界框预测头是否共享
        two_stage_class_embed_share=args.two_stage_class_embed_share,  # 两阶段类别预测头是否共享
        decoder_sa_type=args.decoder_sa_type,  # 解码器自注意力类型（如 'deformable' 可变形注意力）
        num_patterns=args.num_patterns,  # 两阶段的“模式数量”（如候选查询的初始模式）
        # 去噪训练的噪声查询数量（use_dn=True 时生效, dn_number=100）
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,  # 边界框噪声尺度（对真实框添加噪声的幅度）
        dn_label_noise_ratio=args.dn_label_noise_ratio,  # 标签噪声比例（多少比例的标签被替换为错误类别）
        dn_labelbook_size=dn_labelbook_size,  # 去噪标签本大小
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # 1. 基础损失权重（类别、边界框、GIoU）
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef  # 单个添加、修改键值对
    clean_weight_dict_wo_dn = copy.deepcopy(
        weight_dict)    # 深拷贝：保存不含去噪损失的权重（用于两阶段中间损失）

    # 2. 去噪训练损失权重（use_dn=True 时添加）
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    # 3. 分割损失权重（args.masks=True 时添加）
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef  # 掩码损失
        weight_dict["loss_dice"] = args.dice_loss_coef  # Dice 损失
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    # 4. 辅助损失权重（aux_loss=True 时添加）
    # 解码器多层输出，每层都计算损失（除最后一层），加速收敛
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):    # 遍历（除最后一层）
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)  # 批量更新、合并键值对

    # 5. 两阶段中间损失权重（two_stage_type != 'no' 时添加）
    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        # 可选参数：是否排除中间损失的边界框损失
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        # 中间损失的系数：类别损失必加，边界框损失可选
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        # 中间损失的总系数（默认 1.0）
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        # 构建中间损失权重（添加 _interm 后缀）
        interm_weight_dict.update({
            k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k]
            for k, v in clean_weight_dict_wo_dn.items()  # 不含去噪损失
        })
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha,
                             losses=losses,
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(
        nms_iou_threshold=args.nms_iou_threshold)}    # 默认配置不启用 NMS

    # 针对分割任务添加额外的后处理
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
