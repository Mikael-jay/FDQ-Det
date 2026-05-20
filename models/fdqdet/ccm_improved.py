"""
改进的密度映射模块 - 直接生成密度图用于GT监督学习

替代原有的 CategoricalCounting 类

使用深度可分卷积 (Depthwise Separable Convolution) 的膨胀版本
以减少参数量同时保持表达能力
"""

import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class SEBlock(nn.Module):
    """轻量通道注意力模块, 增强DW卷积的通道融合能力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局池化捕捉通道语义
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.GELU(),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)  # 压缩空间维度，保留通道信息
        y = self.fc(y).view(b, c, 1, 1) # 生成通道注意力权重
        return x * y  # 加权增强关键通道的密度特征


class DWDilatedConv2d(nn.Module):
    """
    深度可分膨胀卷积 (Depthwise Separable Dilated Convolution)
    
    = Depthwise Conv (分组卷积) + Pointwise Conv (1x1卷积)
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        dilation: 膨胀率
        padding: 填充大小
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dilation=1, padding=1):
        super(DWDilatedConv2d, self).__init__()

        # Depthwise Conv: 逐通道卷积（每个输入通道单独卷积）
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=in_channels  # 关键：分组=输入通道数，实现depthwise
        )
        
        # Pointwise Conv: 1x1卷积（融合不同通道信息）
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1
        )

        self.se = SEBlock(out_channels)  # 通道注意力模块
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        return x


class DensityMapper(nn.Module):
    def __init__(self, input_channel=256, output_channel=1):
        super(DensityMapper, self).__init__()
        self.ccm_cfg = [512, 512, 512, 256, 256, 256]
        # self.ccm_cfg = [256, 256, 256, 256, 256, 256]
        # self.ccm_cfg = [256, 256, 256, 256]
        self.input_channel = input_channel  # 输入特征通道数（来自 transformer encoder，通常 256）
        self.in_channels = input_channel    # 内部工作通道数，保持与输入一致以兼容下游模块（CGFE/MultiScaleFeature）
        self.output_channel = output_channel

        # ===== Projection layer: stabilize input features =====
        # 保持输入/输出通道一致，投影到共享特征空间
        self.conv1 = nn.Conv2d(input_channel, self.in_channels, kernel_size=1)

        # ===== Shared stem: extract shared context =====
        # 这里直接用输入特征而不是密度特征作为主要输出（ccm_feature），
        # 密度分支只作为调制信号
        self.shared_stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.GELU()
        )

        self.structure_stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.GELU()
        )

        # ===== Density CCM: low-freq density branch =====
        # 密度估计分支，最终输出 256 通道 latent
        self.density_ccm = make_layers(
            self.ccm_cfg, in_channels = self.in_channels, d_rate=2)
        # 输出形状: [B, 256, H, W] (最后 ccm_cfg 元素是 256)

        # ===== Density head: density map generation =====
        # 从 density_ccm 输出的 256 通道 latent 生成单通道密度图
        # 注意：ccm_cfg 最后是 256，所以 density_head 输入应该是 256
        self.density_head = nn.Sequential(
            DWDilatedConv2d(self.ccm_cfg[-1], 128, kernel_size=3, dilation=1, padding=1),
            nn.GELU(),
            DWDilatedConv2d(128, 64, kernel_size=3, dilation=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, output_channel, kernel_size=1),
            nn.Softplus()
        )

        self._init_weights()

    def forward(self, features, spatial_shapes=None):
        # input shape: [B, HW, C] from transformer
        features = features.transpose(1, 2)  # -> [B, C, HW]
        bs, c, hw = features.shape

        # extract first scale features (highest resolution)
        h, w = spatial_shapes[0]
        v_feat = features[:, :, :h*w].reshape(bs, c, h, w)  # -> [B, C, H, W]
        # 期望：c == input_channel (typically 256)

        # ===== Shared feature extraction =====
        x = self.conv1(v_feat)          # [B, 256, H, W] -> [B, 256, H, W]
        x_sem = self.shared_stem(x)
        x_str = self.structure_stem(x)

        # ===== Density estimation branch =====
        density_latent = self.density_ccm(x_str)  # [B, 256, H, W] -> [B, 256, H, W]
        density_map = self.density_head(density_latent)  # [B, 256, H, W] -> [B, 1, H, W]

        # density_map: [B, 1, H, W], detached
        # D = density_map.detach()

        # 使用带梯度的密度图进行调制，但梯度比例较小以稳定训练（受控反传）
        density_gate_grad_scale = 0.1
        D = (
            density_map * density_gate_grad_scale +
            density_map.detach() * (1.0 - density_gate_grad_scale)
        )

        q_low   = 0.05
        q_high  = 0.95

        # --------- Percentile-based normalization --------
        p_low  = torch.quantile(D.flatten(1), q_low,  dim=1, keepdim=True).view(-1,1,1,1)
        p_high = torch.quantile(D.flatten(1), q_high, dim=1, keepdim=True).view(-1,1,1,1)

        D_norm = (D - p_low) / (p_high - p_low + 1e-6)

        D_norm = torch.clamp(D_norm, 0.0, 2.0)

        # -------- thresholds (critical) --------
        T_low  =  0.3    # 背景上界（可调：-0.3 ~ -0.8）
        T_high =  1.2    # 前景下界（可调：0.8 ~ 1.5）

        # -------- modulation strength --------
        alpha = 0.3      # 最大抑制比例（<=30%）
        beta  = 0.4      # 放大斜率
        gamma = 0.25     # 结构特征补偿强度

        # -------- Asymmetric & bounded gate --------
        gate = torch.ones_like(D_norm)

        neg_mask = D_norm < T_low
        pos_mask = D_norm > T_high

        # 抑制：最多 -alpha
        gate[neg_mask] = 1.0 - alpha

        # 增强：最多 +beta
        gate[pos_mask] = 1.0 + beta * torch.clamp(
            (D_norm[pos_mask] - T_high) / (2.0 - T_high),
            min=0.0,
            max=1.0
        )

        # -------- residual modulation --------
        x_sem = x_sem / (x_sem.abs().mean(dim=[2,3], keepdim=True) + 1e-6)
        x_str = x_str / (x_str.abs().mean(dim=[2,3], keepdim=True) + 1e-6)
        ccm_feature = x_sem * gate + gamma * x_str * (gate > 1.0).float()

        return density_map, ccm_feature

    def _init_weights(self):
        """
        自动初始化权重（Xavier initialization）
        适配 GELU 激活和密度图低值分布特性：
        1. Conv2d 权重：Xavier uniform（平衡前/反向梯度方差）
        2. Conv2d 偏置：0 初始化（贴合密度图均值 ≈ 0.0001）
        3. BatchNorm2d：weight=1, bias=0（标准做法）
        4. DW 卷积内部层：递归初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, DWDilatedConv2d):
                self._init_weights_for_submodule(m.depthwise)
                self._init_weights_for_submodule(m.pointwise)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        print(f"[INFO] DensityMapper initialized: in_channels={self.in_channels}, out_channels={self.output_channel}")

    def _init_weights_for_submodule(self, submodule):
        """Helper: initialize depthwise separable conv submodules."""
        if isinstance(submodule, nn.Conv2d):
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.constant_(submodule.bias, 0.0)

    def _load_state_dict_adaptive(self, state_dict, strict=True):
        """
        自适应加载权重：处理通道不匹配（e.g., 512->256）。
        如果 checkpoint 中的权重形状比当前模型大，进行切片。
        """
        model_state = self.state_dict()
        adapted_state = {}
        mismatches = []

        for key, ckpt_tensor in state_dict.items():
            if key not in model_state:
                adapted_state[key] = ckpt_tensor
                continue

            model_tensor = model_state[key]
            if ckpt_tensor.shape == model_tensor.shape:
                adapted_state[key] = ckpt_tensor
            else:
                # 尝试通过切片适配（假设前部分是有效的）
                if all(c >= m for c, m in zip(ckpt_tensor.shape, model_tensor.shape)):
                    # 切片：按每个维度取前 model_shape 个元素
                    slices = tuple(slice(0, m) for m in model_tensor.shape)
                    adapted_state[key] = ckpt_tensor[slices]
                    mismatches.append(f'{key}: {ckpt_tensor.shape} -> {model_tensor.shape} (sliced)')
                else:
                    mismatches.append(f'{key}: {ckpt_tensor.shape} vs {model_tensor.shape} (incompatible, skipped)')

        if mismatches and not strict:
            print(f'[WARNING] Adaptive loading with {len(mismatches)} shape mismatches:')
            for m in mismatches[:5]:  # 只打印前 5 个
                print(f'  {m}')
            if len(mismatches) > 5:
                print(f'  ... and {len(mismatches) - 5} more')

        self.load_state_dict(adapted_state, strict=False)
        print(f'[INFO] Loaded {len(adapted_state)} parameters (adaptive mode)')


def make_dw_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
    """
    构建深度可分膨胀卷积层序列
    
    Args:
        cfg: 通道数配置列表，如 [512, 512, 512, 256, 256, 256]
        in_channels: 输入通道数
        batch_norm: 是否使用批标准化
        d_rate: 膨胀率 (dilation)
        
    Returns:
        nn.Sequential: 卷积层序列
    """
    layers = []
    for v in cfg:
        # 使用DW Dilated卷积: Depthwise + Pointwise
        dw_conv = DWDilatedConv2d(in_channels, v, kernel_size=3,
                                   dilation=d_rate, padding=d_rate)
        if batch_norm:
            layers += [dw_conv, nn.BatchNorm2d(v), nn.GELU()]
        else:
            layers += [dw_conv, nn.GELU()]
        in_channels = v
    return nn.Sequential(*layers)


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
    """
    构建膨胀卷积层序列（保留以实现向后兼容）
    
    Args:
        cfg: 通道数配置列表，如 [512, 512, 512, 256, 256, 256]
        in_channels: 输入通道数
        batch_norm: 是否使用批标准化
        d_rate: 膨胀率（dilation）
        
    Returns:
        nn.Sequential: 卷积层序列
    """
    layers = []
    for v in cfg:
        # 卷积层：3x3 kernel
        # padding = dilation（保持特征图尺寸）
        # dilation = d_rate（扩大感受野）
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                           padding=d_rate, dilation=d_rate)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.GELU()]
        else:
            layers += [conv2d, nn.GELU()]
        in_channels = v
    return nn.Sequential(*layers)


# 向后兼容性：保留原类名的别名
CategoricalCounting = DensityMapper