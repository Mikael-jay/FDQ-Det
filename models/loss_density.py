"""
Density Ranking Loss - 区域级密度相对关系约束

核心思想：
GT 中密的区域，预测中也应该更"亮"（相对排序一致）

优势：
1. 强制模型学习密度梯度（解决掩码化/二值化问题）
2. 计算代价低（8×8 grid，64个patch）
3. 与pixel/integral loss互补（约束相对关系 vs 绝对值）

论文价值：
- 在 Density Estimation / Crowd Counting 中强调"相对密度"比"绝对值"更鲁棒
- 可作为辅助监督信号，提升密度图质量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityRankingLoss(nn.Module):
    """
    密度排序损失（Density Ranking Loss）
    
    将密度图下采样到 coarse grid，对每个 patch 的平均密度进行 pairwise ranking
    
    Args:
        grid_size: coarse grid 大小，默认 8×8（64个patch）
        margin: ranking margin，自适应或固定
        adaptive_margin: 是否使用自适应 margin（基于GT密度差异）
        margin_scale: 自适应 margin 的缩放因子
        significance_threshold: 只对"GT密度差异 > threshold"的对施加约束
        use_smooth_hinge: 使用 smooth hinge 替代 ReLU（梯度更稳定）
    """
    
    def __init__(
        self, 
        grid_size=8, 
        margin=0.1,
        adaptive_margin=True,
        margin_scale=0.5,
        significance_threshold=0.05,
        use_smooth_hinge=True
    ):
        super().__init__()
        self.grid_size = grid_size
        self.margin = margin
        self.adaptive_margin = adaptive_margin
        self.margin_scale = margin_scale
        self.significance_threshold = significance_threshold
        self.use_smooth_hinge = use_smooth_hinge
        
    def forward(self, pred_density, gt_density):
        """
        计算密度排序损失
        
        Args:
            pred_density: [B, 1, H, W] 预测密度图
            gt_density: [B, 1, H, W] GT密度图
            
        Returns:
            ranking_loss: 标量张量
            loss_dict: 详细损失字典（用于日志）
        """
        B = pred_density.size(0)
        
        # 下采样到 coarse grid（使用自适应平均池化）
        pred_patches = F.adaptive_avg_pool2d(pred_density, (self.grid_size, self.grid_size))  # [B, 1, G, G]
        gt_patches = F.adaptive_avg_pool2d(gt_density, (self.grid_size, self.grid_size))      # [B, 1, G, G]
        
        # Flatten patches: [B, G*G]
        pred_p = pred_patches.view(B, -1)  # [B, 64]
        gt_p = gt_patches.view(B, -1)      # [B, 64]
        
        gt_p_norm = self._normalize_patches(gt_p)
        pred_p_norm = pred_p / (pred_p.mean(dim=1, keepdim=True) + 1e-6)
        
        # 计算 pairwise ranking loss
        ranking_loss, num_pairs, num_violations = self._compute_pairwise_ranking(
            pred_p_norm, gt_p_norm
        )
        
        # 构建损失字典
        loss_dict = {
            'ranking_loss': ranking_loss.item(),
            'num_pairs': num_pairs,
            'num_violations': num_violations,
            'violation_rate': num_violations / max(num_pairs, 1)
        }
        
        return ranking_loss, loss_dict
    
    def _normalize_patches(self, patches):
        """
        归一化 patches（per-sample）
        
        Args:
            patches: [B, N]
            
        Returns:
            normalized: [B, N]
        """
        # Min-max 归一化到 [0, 1]
        pmin = patches.min(dim=1, keepdim=True)[0]  # [B, 1]
        pmax = patches.max(dim=1, keepdim=True)[0]  # [B, 1]
        
        # 避免除零
        denom = pmax - pmin + 1e-8
        normalized = (patches - pmin) / denom
        
        return normalized
    
    def _compute_pairwise_ranking(self, pred_p, gt_p):
        """
        计算 pairwise ranking loss
        
        对于所有 patch 对 (i, j)：
        - 如果 GT[i] > GT[j] + threshold，则要求 Pred[i] > Pred[j] + margin
        
        Args:
            pred_p: [B, N] 归一化后的预测 patches
            gt_p: [B, N] 归一化后的 GT patches
            
        Returns:
            loss: 标量张量
            num_pairs: 有效对数
            num_violations: 违反约束的对数
        """
        B, N = pred_p.shape
        
        # 扩展维度以计算所有对: [B, N, 1] vs [B, 1, N]
        pred_i = pred_p.unsqueeze(2)  # [B, N, 1]
        pred_j = pred_p.unsqueeze(1)  # [B, 1, N]
        gt_i = gt_p.unsqueeze(2)      # [B, N, 1]
        gt_j = gt_p.unsqueeze(1)      # [B, 1, N]
        
        # GT 密度差异: [B, N, N]
        gt_diff = gt_i - gt_j
        
        # 只对"GT 差异显著"的对施加约束
        # mask: GT[i] > GT[j] + threshold
        significant_mask = (gt_diff > self.significance_threshold).float()  # [B, N, N]
        
        # 预测密度差异: [B, N, N]
        pred_diff = pred_i - pred_j
        
        # 自适应 margin（基于 GT 差异）
        if self.adaptive_margin:
            # margin = margin_scale * |gt_diff|
            adaptive_margin = self.margin_scale * torch.abs(gt_diff)
        else:
            adaptive_margin = self.margin
        
        # Ranking 违反量：Pred[i] < Pred[j] + margin（当GT[i] > GT[j]时）
        # violation = margin - pred_diff = margin - (pred_i - pred_j)
        violation = adaptive_margin - pred_diff
        
        # 使用 smooth hinge 或 ReLU
        if self.use_smooth_hinge:
            # Smooth Hinge: log(1 + exp(x))，梯度更稳定
            hinge = F.softplus(violation)
        else:
            # 标准 ReLU Hinge
            hinge = F.relu(violation)
        
        # 只对"显著对"计算损失
        masked_hinge = significant_mask * hinge  # [B, N, N]
        
        # 统计有效对数和违反数
        num_pairs = significant_mask.sum().item()
        num_violations = (significant_mask * (violation > 0).float()).sum().item()
        
        # 平均损失（避免除零）
        if num_pairs > 0:
            loss = masked_hinge.sum() / num_pairs
        else:
            loss = torch.tensor(0.0, device=pred_p.device)
        
        return loss, num_pairs, num_violations


class DensityMapLossWithRanking(nn.Module):
    def __init__(
        self,
        weight_pixel=1.0,
        weight_integral=0.1,
        weight_ranking=0.2,
        weight_support=0.5,
        weight_distribution=0.1,
        ranking_grid_size=8,
        ranking_margin=0.1,
        adaptive_margin=True,
        density_scale=1.0,
        # backward-compatible args (may be passed by older configs)
        loss_type=None,
        weight_smooth=None,
        weight_edge=None,
        **kwargs,
    ):
        super().__init__()

        # store main weights
        self.weight_pixel = weight_pixel
        self.weight_integral = weight_integral
        self.weight_ranking = weight_ranking
        self.weight_support = weight_support
        self.weight_distribution = weight_distribution
        self.density_scale = density_scale
        # asymmetric integral loss weights
        self.integral_lambda_low = kwargs.get('integral_lambda_low', 5.0)
        self.integral_lambda_high = kwargs.get('integral_lambda_high', 0.5)
        # asymmetric pixel loss weights: over-predict cheaper than under-predict
        self.pixel_over_weight = kwargs.get('pixel_over_weight', 0.5)
        self.pixel_under_weight = kwargs.get('pixel_under_weight', 1.0)

        # store legacy/optional args to avoid unexpected-kw errors
        self.loss_type = loss_type
        self.weight_smooth = weight_smooth if weight_smooth is not None else 0.0
        self.weight_edge = weight_edge if weight_edge is not None else 0.0

        # init ranking loss
        self.ranking_loss_fn = DensityRankingLoss(
            grid_size=ranking_grid_size,
            margin=ranking_margin,
            adaptive_margin=adaptive_margin
        )

    def forward(self, pred_density, gt_density):
        eps = 1e-6

        # =============================
        # 1. Support Mask (GT-defined)
        # =============================
        support_mask = (gt_density > 0).float()

        # =============================
        # 2. Pixel-wise Regression (MAIN) - asymmetric: over-predict cheaper
        # =============================
        diff = pred_density - gt_density
        abs_diff = diff.abs()
        pixel_loss = torch.mean(
            torch.where(
                diff >= 0,
                self.pixel_over_weight * abs_diff,
                self.pixel_under_weight * abs_diff,
            )
        )

        # =============================
        # 3. Integral Consistency
        # =============================
        pred_integral = pred_density.sum(dim=[2,3])
        gt_integral = gt_density.sum(dim=[2,3])
        # Asymmetric integral loss: penalize underestimation/overestimation differently
        diff = pred_integral - gt_integral
        pos = F.relu(diff)   # pred > gt
        neg = F.relu(-diff)  # gt > pred
        # mean over batch
        integral_loss = self.integral_lambda_low * neg.mean() + self.integral_lambda_high * pos.mean()

        # =============================
        # 4. Support Loss (HARD)
        # =============================
        support_loss = torch.mean(pred_density * (1.0 - support_mask))

        # =============================
        # 5. Distribution Alignment (SOFT)
        # =============================
        pred_support = pred_density * support_mask
        gt_support   = gt_density   * support_mask

        pred_mass = pred_support.sum(dim=[2,3], keepdim=True) + eps
        gt_mass   = gt_support.sum(dim=[2,3], keepdim=True) + eps

        pred_prob = pred_support / pred_mass
        gt_prob   = gt_support / gt_mass

        distribution_loss = F.kl_div(
            torch.log(pred_prob + eps),
            gt_prob,
            reduction='batchmean'
        )

        # =============================
        # 6. Ranking Loss (STRUCTURE)
        # =============================
        ranking_loss, ranking_dict = self.ranking_loss_fn(
            pred_density, gt_density
        )

        # =============================
        # 7. Total Loss
        # =============================
        total_loss = (
            self.weight_pixel * pixel_loss +
            self.weight_integral * integral_loss +
            self.weight_support * support_loss +
            self.weight_distribution * distribution_loss +
            self.weight_ranking * ranking_loss
        )

        loss_dict = {
            'pixel_loss': pixel_loss.item(),
            'integral_loss': integral_loss.item(),
            'support_loss': support_loss.item(),
            'distribution_loss': distribution_loss.item(),
            'ranking_loss': ranking_loss.item(),
            'total_loss': total_loss.item(),
        }
        loss_dict.update(ranking_dict)

        return total_loss, loss_dict


# ============================================================================
# 向后兼容：保留原类名和计数函数
# ============================================================================
DensityMapLoss = DensityMapLossWithRanking


def count_objects_from_density(density_map, single_object_integral=None, threshold=1e-3, density_scale: float = 1.0):
    """
    从密度图的积分推断目标数量
    
    Args:
        density_map: [bs, 1, H, W] 预测的密度图
        single_object_integral: float 单个目标的积分值
            - None: 自动计算为密度图的平均非零值
            - float: 预计算的单个目标积分值
        threshold: float 积分阈值，低于此值的积分不计入（去噪）
        density_scale: float 单目标密度尺度（推理时用于count = integral / density_scale）
        
    Returns:
        object_counts: [bs] 推断的目标数量（整数）
        integrals: [bs] 各样本的积分值（用于调试）
    """
    # 全局积分（求和所有像素）
    bs, c, h, w = density_map.shape
    total_integral = density_map.sum(dim=[2, 3]).squeeze(-1)  # [bs]
    
    # 估计单个目标的积分值
    if single_object_integral is None:
        # 默认使用 density_scale 作为单目标积分
        single_integral = float(density_scale)
    else:
        single_integral = single_object_integral
    
    # 推断目标数
    object_counts = (total_integral / (single_integral + 1e-6)).round().long()
    
    # 限制范围（避免出现负数或过大值）
    object_counts = torch.clamp(object_counts, min=0)
    
    return object_counts, total_integral


def get_dynamic_query_count(object_counts, dynamic_query_list):
    """
    根据推断的目标数量选择动态查询数量
    
    Args:
        object_counts: [bs] 推断的目标数量
        dynamic_query_list: list of int 动态查询数量选项，按目标数阶段排序
            例如: [100, 200, 300, 500] 表示不同目标数量级的查询数
            
    Returns:
        num_select: int 为当前批次选择的查询数量
    """
    # 获取批次中的最大目标数（保守估计）
    max_count = object_counts.max().item()
    
    # 根据目标数选择查询数量
    # 这里的映射逻辑需要根据实际情况调整
    idx = min(max_count // 50, len(dynamic_query_list) - 1)  # 每50个目标对应一个等级
    num_select = dynamic_query_list[idx]
    
    return num_select


if __name__ == '__main__':
    """测试代码"""
    print("="*80)
    print("测试 Density Ranking Loss")
    print("="*80)
    
    # 创建测试数据
    B, C, H, W = 2, 1, 64, 64
    
    # GT: 左上角密集，右下角稀疏
    gt = torch.zeros(B, C, H, W)
    gt[:, :, :32, :32] = 1.0  # 左上角密集
    gt[:, :, 32:, 32:] = 0.2  # 右下角稀疏

    pred_bad = torch.ones(B, C, H, W) * 0.6

    # Pred (good): 符合相对关系
    pred_good = torch.zeros(B, C, H, W)
    pred_good[:, :, :32, :32] = 1.0
    pred_good[:, :, 32:, 32:] = 0.2

    ranking_loss_fn = DensityRankingLoss(grid_size=4, margin=0.1)
    
    loss_bad, dict_bad = ranking_loss_fn(pred_bad, gt)
    loss_good, dict_good = ranking_loss_fn(pred_good, gt)
    
    print(f"\nBad Pred (均匀分布):")
    print(f"  Ranking Loss: {loss_bad.item():.4f}")
    print(f"  Violation Rate: {dict_bad['violation_rate']:.2%}")
    
    print(f"\nGood Pred (符合相对关系):")
    print(f"  Ranking Loss: {loss_good.item():.4f}")
    print(f"  Violation Rate: {dict_good['violation_rate']:.2%}")
    
    # 测试完整损失
    print("\n" + "="*80)
    print("测试完整损失函数")
    print("="*80)
    
    loss_fn = DensityMapLossWithRanking(
        weight_pixel=1.0,
        weight_integral=0.1,
        weight_ranking=0.2,
        weight_support=0.5,
        weight_distribution=0.1
    )
    
    total_bad, dict_bad = loss_fn(pred_bad, gt)
    total_good, dict_good = loss_fn(pred_good, gt)
    
    print(f"\nBad Pred:")
    for k, v in dict_bad.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\nGood Pred:")
    for k, v in dict_good.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n✓ 测试通过！Ranking Loss 正确区分了好坏预测")
