#!/usr/bin/env python3
"""
Fixed High Frequency Suppressor

主要改进：
1. 使用sigmoid而非tanh，确保抑制权重∈(0,1)
2. 改为相乘形式: output = content * (1 - suppression_weight)
3. 保证数学上的抑制行为
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighFrequencySuppressorFixed(nn.Module):
    """Fixed version of High Frequency Suppressor that truly suppresses."""

    def __init__(self, channels: int = 256, reduction: int = 4,
                 num_feature_levels: int = 4,
                 kernel_schedule: Optional[List[int]] = None) -> None:
        super().__init__()
        assert channels % reduction == 0
        reduced = channels // reduction

        if kernel_schedule is None:
            kernel_schedule = [7, 5, 5, 3, 3]
        self.kernel_schedule = kernel_schedule[:num_feature_levels]

        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
            for _ in range(num_feature_levels)
        ])
        self.depthwise_filters = nn.ModuleList([
            nn.Conv2d(reduced, reduced, kernel_size=k, padding=k // 2,
                      groups=reduced, bias=False)
            for k in self.kernel_schedule
        ])
        self.expand_layers = nn.ModuleList([
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
            for _ in range(num_feature_levels)
        ])
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(32, channels) for _ in range(num_feature_levels)
        ])

    def forward(self, features: List[torch.Tensor], masks: List[Optional[torch.Tensor]],
                apply: bool = True) -> List[torch.Tensor]:
        if not apply:
            return features

        processed = []
        for feat, mask, reduce, depthwise, expand, norm, kernel_size in zip(
            features, masks, self.reduce_layers, self.depthwise_filters,
            self.expand_layers, self.norm_layers, self.kernel_schedule):
            
            # Prepare valid mask
            if mask is None:
                valid = torch.ones((feat.size(0), 1, feat.size(2), feat.size(3)),
                                   device=feat.device, dtype=feat.dtype)
            else:
                valid = mask.to(feat.dtype)
                if valid.dim() == 3:
                    valid = valid.unsqueeze(1)

            # Separate padded and content regions
            padded = feat * (1.0 - valid)
            content = feat * valid

            # Estimate high-frequency residual
            low_pass = F.avg_pool2d(content, kernel_size=kernel_size, stride=1,
                                    padding=kernel_size // 2)
            high_residual = content - low_pass

            # Learn suppression weight
            reduced = reduce(high_residual)
            filtered = depthwise(F.relu(reduced))
            restored = expand(filtered)
            restored = norm(restored)

            # KEY FIX: Use sigmoid to ensure suppression weight ∈ (0, 1)
            # Then multiply: output = content * (1 - weight)
            # This guarantees suppression, not enhancement
            suppression_weight = torch.sigmoid(restored)
            suppressed_content = content * (1.0 - suppression_weight)

            # Combine with padded region
            suppressed = suppressed_content * valid + padded
            processed.append(suppressed)

        return processed


def convert_old_hfs_to_fixed(old_hfs_state_dict):
    """Convert old HFS checkpoint to fixed version (weights compatible)."""
    # Weights are compatible - only forward logic changed
    return old_hfs_state_dict


if __name__ == '__main__':
    print("Testing Fixed HFS...")
    hfs = HighFrequencySuppressorFixed(channels=256, reduction=4, num_feature_levels=4)
    hfs.eval()

    # Test with random features
    features = [torch.randn(1, 256, 8, 8) * 0.1 for _ in range(4)]
    masks = [torch.ones(1, 1, 8, 8) for _ in range(4)]  # All valid

    print("\nBefore:")
    for i, f in enumerate(features):
        print(f"  Level {i}: norm={f.norm():.4f}")

    with torch.no_grad():
        features_after = hfs(features, masks, apply=True)

    print("\nAfter:")
    for i, (fb, fa) in enumerate(zip(features, features_after)):
        ratio = fa.norm() / fb.norm()
        change = (ratio - 1) * 100
        status = "✓ SUPPRESSED" if ratio < 1.0 else "✗ ENHANCED"
        print(f"  Level {i}: norm={fa.norm():.4f}, ratio={ratio:.4f} ({change:+.2f}%) {status}")
