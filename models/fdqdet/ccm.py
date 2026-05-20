import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F


class CategoricalCounting(nn.Module):
    def __init__(self, cls_num=4):
        super(CategoricalCounting, self).__init__()
        self.ccm_cfg = [512, 512, 512, 256, 256, 256]   # 卷积层通道数配置
        self.in_channels = 512  # 输入的特征通道数
        # 1x1卷积：用于将输入的256通道特征升维到512通道（与CCM的输入通道匹配）
        self.conv1 = nn.Conv2d(256, self.in_channels, kernel_size=1)
        self.ccm = make_layers( # 构建CCM主网络（多层卷积，带膨胀率）
            self.ccm_cfg, in_channels=self.in_channels, d_rate=2)
        # 全局平均池化：将特征图压缩为1x1（保留通道维度）
        self.output = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 线性层：将512维特征映射到4个数量级（cls_num）
        self.linear = nn.Linear(256, cls_num)

    def forward(self, features, spatial_shapes=None):
        # 调整特征维度（假设输入features来自Transformer，形状为[bs, num_queries, hidden_dim]）
        features = features.transpose(1, 2) # 维度交换-> [bs, hidden_dim, num_queries]
        bs, c, hw = features.shape

        # 提取空间特征（v_feat）并重塑为特征图
        # spatial_shapes[0]：第一个特征层的高和宽（如[H, W]）
        h, w = spatial_shapes[0][0], spatial_shapes[0][1]
        # 从features中取出前h*w个查询（对应空间特征），重塑为2D特征图 [bs, 256, H, W]
        v_feat = features[:, :, 0:h*w].view(bs, 256, h, w)

        # 通过1x1卷积升维，然后通过CCM网络处理，最后通过全局池化和线性层输出类别计数
        x = self.conv1(v_feat)  # 1x1卷积升维：[bs, 256, H, W] → [bs, 512, H, W]
        x = self.ccm(x)         # 多层膨胀卷积：[bs, 512, H, W] → [bs, 256, H, W]（最终通道数由ccm_cfg决定）
        out = self.output(x)    # 全局平均池化：[bs, 256, H, W] → [bs, 256, 1, 1]
        out = out.squeeze(3).squeeze(2) # 移除空间维度：[bs, 256]
        out = self.linear(out)  # 线性映射：[bs, 256] → [bs, cls_num]（每个类别预测数量）

        return out, x


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=1):
    layers = []
    for v in cfg:
        # 卷积层：3x3 kernel，padding=膨胀率（保持特征图尺寸），dilation=膨胀率（扩大感受野）
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                           padding=d_rate, dilation=d_rate)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
        else:
            layers += [conv2d, nn.ReLU(inplace=False)]
        in_channels = v
    return nn.Sequential(*layers)
