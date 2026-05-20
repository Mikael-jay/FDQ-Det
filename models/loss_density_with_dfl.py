"""
改进的密度图监督损失函数 - 支持 DFL (Distributed Focal Loss)

新增功能：
1. DFL (Distributed Focal Loss) 支持计数监督
2. OTA (Optimal Transport Assignment) 策略
3. 自适应权重调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DistributedFocalLoss(nn.Module):
    """
    分布焦点损失 (Distributed Focal Loss)
    
    用于计数监督（从密度积分推断目标数）
    将计数问题建模为分布估计问题
    
    原理：
    - 预测：离散分布 (0, 1, 2, ..., max_count)
    - GT: 离散分布（one-hot编码的目标数）
    - 优势：
      * 相比MSE，对异常值更鲁棒
      * 学习分布而非点估计
      * 自适应焦点权重
    """
    
    def __init__(self, max_count=300, alpha=2.0, gamma=2.0, reduction='mean'):
        """
        Args:
            max_count: 最大目标数（分布范围）
            alpha: DFL中的缩放因子
            gamma: 焦点参数（控制困难样本权重）
            reduction: 'mean' 或 'sum'
        """
        super().__init__()
        self.max_count = max_count
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # 创建目标数bins [0, 1, 2, ..., max_count]
        self.register_buffer('count_bins', torch.arange(max_count + 1, dtype=torch.float32))
    
    def forward(self, pred_density_integral, gt_count, pred_density=None):
        """
        计算DFL损失
        
        Args:
            pred_density_integral: [bs] 预测密度图的积分值
            gt_count: [bs] GT目标数（整数）
            pred_density: [bs, 1, H, W] 可选，预测密度图用于计算置信度
            
        Returns:
            loss: 标量张量
        """
        bs = gt_count.shape[0]
        
        # 将积分值转换为分布
        # 使用softmax使其成为概率分布
        device = pred_density_integral.device
        count_bins = self.count_bins.to(device)
        
        # 计算预测分布：将积分映射到分布
        # 使用高斯核函数：P(k) ∝ exp(-(k - integral)^2 / 2σ^2)
        sigma = max(1.0, pred_density_integral.std().item())
        pred_dist = torch.exp(
            -((count_bins.unsqueeze(0) - pred_density_integral.unsqueeze(1)) ** 2) / (2 * sigma ** 2)
        )  # [bs, max_count+1]
        
        # 归一化为概率分布
        pred_dist = F.softmax(pred_dist / self.alpha, dim=1)  # [bs, max_count+1]
        
        # GT分布：one-hot编码
        gt_count_clipped = torch.clamp(gt_count, min=0, max=self.max_count)
        gt_dist = F.one_hot(gt_count_clipped.long(), num_classes=self.max_count + 1).float()  # [bs, max_count+1]
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(
            (pred_dist + 1e-8).log(), 
            gt_count_clipped.long(),
            reduction='none'
        )  # [bs]
        
        # 焦点加权：困难样本权重更高
        # 计算预测目标数在分布中的概率
        pred_count_probs = pred_dist[torch.arange(bs), gt_count_clipped.long()]
        focal_weight = (1 - pred_count_probs) ** self.gamma  # [bs]
        
        # 加权损失
        weighted_loss = ce_loss * focal_weight
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class DensityMapLossWithDFL(nn.Module):
    """
    改进的密度图损失函数
    
    支持两种计数监督方式：
    1. 积分约束 (Integral Constraint) - 原始方式
    2. DFL (Distributed Focal Loss) - 新增方式
    
    Loss = pixel_loss + w_integral * integral_loss + w_smooth * smooth_loss + w_dfl * dfl_loss
    """
    
    def __init__(self, loss_type='l2', weight_pixel=1.0, weight_smooth=0.1, weight_integral=0.5, 
                 weight_edge=0.0, use_dfl=False, dfl_weight=1.0, dfl_gamma=2.0, integral_lambda_low=5.0, integral_lambda_high=0.5,
                 pixel_over_weight=0.5, pixel_under_weight=1.0):
        """
        Args:
            loss_type: 像素级损失类型 ('l2', 'l1', 'smooth_l1')
            weight_pixel: 像素级损失权重（默认1.0）
            weight_smooth: 平滑性约束权重
            weight_integral: 积分约束权重
            weight_edge: 边界约束权重
            use_dfl: 是否使用DFL进行计数监督
            dfl_weight: DFL损失权重
            dfl_gamma: DFL焦点参数
        """
        super().__init__()
        self.loss_type = loss_type
        self.weight_pixel = weight_pixel
        self.weight_smooth = weight_smooth
        self.weight_integral = weight_integral
        self.weight_edge = weight_edge
        self.use_dfl = use_dfl
        self.dfl_weight = dfl_weight
        # asymmetric integral loss weights
        self.integral_lambda_low = integral_lambda_low
        self.integral_lambda_high = integral_lambda_high
        # asymmetric pixel loss weights
        self.pixel_over_weight = pixel_over_weight
        self.pixel_under_weight = pixel_under_weight
        
        # 像素级损失函数
        if loss_type == 'l2':
            self.pixel_loss_fn = nn.MSELoss(reduction='mean')
        elif loss_type == 'l1':
            self.pixel_loss_fn = nn.L1Loss(reduction='mean')
        elif loss_type == 'smooth_l1':
            self.pixel_loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # DFL损失函数（可选）
        if use_dfl:
            self.dfl_loss_fn = DistributedFocalLoss(max_count=300, gamma=dfl_gamma)
        else:
            self.dfl_loss_fn = None
    
    def forward(self, pred_density, gt_density, gt_count=None, mask=None):
        """
        前向传播
        
        Args:
            pred_density: [bs, 1, H, W] 预测的密度图
            gt_density: [bs, 1, H, W] GT密度图
            gt_count: [bs] GT目标数（使用DFL时需要提供）
            mask: [bs, 1, H, W] 可选的有效区域掩码
            
        Returns:
            loss: 标量张量，总损失
            loss_dict: 损失分量字典
        """
        # 应用掩码（如果提供）
        if mask is not None:
            pred_density = pred_density * mask
            gt_density = gt_density * mask
        
        # 1. 像素级回归损失（asymmetric：over-predict 更便宜）
        diff = pred_density - gt_density
        abs_diff = diff.abs()
        pixel_loss = torch.mean(
            torch.where(
                diff >= 0,
                self.pixel_over_weight * abs_diff,
                self.pixel_under_weight * abs_diff,
            )
        )
        
        # 2. 积分约束损失（确保总和相近）
        pred_integral = pred_density.sum(dim=[2, 3], keepdim=True)  # [bs, 1, 1, 1]
        gt_integral = gt_density.sum(dim=[2, 3], keepdim=True)      # [bs, 1, 1, 1]
        
        if self.use_dfl and self.dfl_loss_fn is not None and gt_count is not None:
            # DFL方式：将积分作为分布的输入
            pred_integral_flat = pred_integral.squeeze()  # [bs]
            dfl_loss = self.dfl_loss_fn(pred_integral_flat, gt_count, pred_density)
            integral_loss = dfl_loss
        else:
            # 传统MSE方式
            # Asymmetric integral loss (apply on squeezed integrals)
            pred_int = pred_integral.squeeze()
            gt_int = gt_integral.squeeze()
            diff = pred_int - gt_int
            pos = F.relu(diff)
            neg = F.relu(-diff)
            integral_loss = self.integral_lambda_low * neg.mean() + self.integral_lambda_high * pos.mean()
        
        # 3. 平滑性约束（总变差正则化）
        smooth_loss = self._compute_smoothness(pred_density)
        
        # 4. 边界约束（可选）
        edge_loss = 0.0
        if self.weight_edge > 0:
            edge_loss = self._compute_edge_constraint(pred_density, gt_density)
        
        # 加权组合
        total_loss = (
            self.weight_pixel * pixel_loss + 
            self.weight_integral * integral_loss + 
            self.weight_smooth * smooth_loss +
            self.weight_edge * edge_loss
        )
        
        # 构建损失字典（用于日志记录）
        loss_dict = {
            'pixel_loss': pixel_loss.item(),
            'integral_loss': integral_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'total_loss': total_loss.item()
        }
        
        if self.weight_edge > 0:
            loss_dict['edge_loss'] = edge_loss.item()
        
        if self.use_dfl and self.dfl_loss_fn is not None:
            loss_dict['use_dfl'] = True
        
        return total_loss, loss_dict
    
    def _compute_smoothness(self, density_map):
        """
        计算总变差（TV）正则化，鼓励空间平滑
        
        TV = mean(|∇x| + |∇y|)
        
        Args:
            density_map: [bs, c, H, W]
            
        Returns:
            smooth_loss: 标量张量
        """
        # 计算梯度（差分）
        dx = torch.abs(density_map[:, :, :-1, :] - density_map[:, :, 1:, :])
        dy = torch.abs(density_map[:, :, :, :-1] - density_map[:, :, :, 1:])
        
        # 总变差
        tv_loss = (dx.mean() + dy.mean()) / 2
        return tv_loss
    
    def _compute_edge_constraint(self, pred_density, gt_density):
        """
        边界约束：确保预测密度的高值位置接近GT的高值位置
        
        使用余弦相似度约束两个密度图的空间分布
        
        Args:
            pred_density: [bs, 1, H, W]
            gt_density: [bs, 1, H, W]
            
        Returns:
            edge_loss: 标量张量
        """
        bs, c, h, w = pred_density.shape
        pred_flat = pred_density.view(bs, -1)  # [bs, h*w]
        gt_flat = gt_density.view(bs, -1)      # [bs, h*w]
        
        # 通过阈值化获取密度高的区域
        pred_binary = (pred_flat > pred_flat.mean(dim=1, keepdim=True)).float()
        gt_binary = (gt_flat > gt_flat.mean(dim=1, keepdim=True)).float()
        
        # 交并比（IoU）损失
        intersection = (pred_binary * gt_binary).sum(dim=1)
        union = (pred_binary + gt_binary).sum(dim=1) - intersection
        iou_loss = 1 - (intersection / (union + 1e-6)).mean()
        
        return iou_loss


class AdaptiveWeightingLoss(nn.Module):
    """
    自适应加权策略
    
    根据训练进度动态调整损失权重
    早期：重视像素级回归
    中期：逐步加入积分约束
    后期：均衡所有约束
    """
    
    def __init__(self, base_loss_fn, warmup_epochs=5, total_epochs=48):
        """
        Args:
            base_loss_fn: DensityMapLossWithDFL 实例
            warmup_epochs: 预热epoch数
            total_epochs: 总epoch数
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    def forward(self, pred_density, gt_density, gt_count=None, mask=None):
        """
        前向传播 - 使用自适应权重
        """
        # 计算权重衰减因子
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：逐步增加积分损失权重
            alpha = self.current_epoch / max(1, self.warmup_epochs)
        else:
            # 常规阶段：保持稳定
            alpha = 1.0
        
        # 临时调整权重
        original_integral_weight = self.base_loss_fn.weight_integral
        self.base_loss_fn.weight_integral = original_integral_weight * alpha
        
        # 计算损失
        loss, loss_dict = self.base_loss_fn(
            pred_density, gt_density, gt_count=gt_count, mask=mask
        )
        
        # 恢复权重
        self.base_loss_fn.weight_integral = original_integral_weight
        
        # 添加权重信息到损失字典
        loss_dict['weight_alpha'] = alpha
        
        return loss, loss_dict


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    # 测试代码
    bs, c, h, w = 2, 1, 32, 32
    
    # 示例1：传统积分约束
    print("="*60)
    print("示例1: 传统积分约束损失")
    print("="*60)
    
    loss_fn = DensityMapLossWithDFL(
        loss_type='l2',
        weight_smooth=0.1,
        weight_integral=0.5,
        use_dfl=False  # 关闭DFL
    )
    
    pred = torch.randn(bs, c, h, w).abs()
    gt = torch.randn(bs, c, h, w).abs()
    gt_count = torch.tensor([5, 10])
    
    loss, loss_dict = loss_fn(pred, gt, gt_count=gt_count)
    print(f"总损失: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # 示例2：DFL计数监督
    print("\n" + "="*60)
    print("示例2: DFL计数监督损失")
    print("="*60)
    
    loss_fn_dfl = DensityMapLossWithDFL(
        loss_type='l2',
        weight_smooth=0.1,
        weight_integral=0.5,
        use_dfl=True,  # 启用DFL
        dfl_weight=1.0,
        dfl_gamma=2.0
    )
    
    loss_dfl, loss_dict_dfl = loss_fn_dfl(pred, gt, gt_count=gt_count)
    print(f"总损失: {loss_dfl.item():.4f}")
    for k, v in loss_dict_dfl.items():
        print(f"  {k}: {v:.4f}")
    
    # 示例3：自适应加权
    print("\n" + "="*60)
    print("示例3: 自适应加权")
    print("="*60)
    
    adaptive_loss = AdaptiveWeightingLoss(loss_fn_dfl, warmup_epochs=5, total_epochs=48)
    
    for epoch in [0, 2, 5, 10, 48]:
        adaptive_loss.set_epoch(epoch)
        loss, loss_dict = adaptive_loss(pred, gt, gt_count=gt_count)
        print(f"Epoch {epoch}: 权重α={loss_dict.get('weight_alpha', 1.0):.2f}, 损失={loss.item():.4f}")
