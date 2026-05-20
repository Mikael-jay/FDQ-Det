from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_kernel_schedule(num_levels: int) -> List[int]:
    base = [7, 5, 5, 3, 3]
    if num_levels <= len(base):
        return base[:num_levels]
    extra = [3] * (num_levels - len(base))
    return base + extra


class HighFrequencySuppressor(nn.Module):
    """Light-weight high frequency suppressor for multi-scale features.

    The module performs a depthwise low-pass filtering per feature level and
    subtracts learnable responses from the original activations to attenuate
    high-frequency noise. Padding locations are preserved through the provided
    masks so that suppression is only applied on valid regions.
    """

    def __init__(self, channels: int = 256, reduction: int = 4,
                 num_feature_levels: int = 4,
                 kernel_schedule: Optional[List[int]] = None) -> None:
        super().__init__()
        assert channels % reduction == 0, "channels must be divisible by reduction"
        reduced = channels // reduction

        if kernel_schedule is None:
            kernel_schedule = _default_kernel_schedule(num_feature_levels)
        assert len(kernel_schedule) >= num_feature_levels, "kernel schedule too short"

        self.kernel_schedule = kernel_schedule

        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
            for _ in range(num_feature_levels)
        ])
        self.depthwise_filters = nn.ModuleList([
            nn.Conv2d(
                reduced,
                reduced,
                kernel_size=k,
                padding=k // 2,
                groups=reduced,
                bias=False,
            )
            for k in kernel_schedule[:num_feature_levels]
        ])
        self.expand_layers = nn.ModuleList([
            nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
            for _ in range(num_feature_levels)
        ])
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(32, channels) for _ in range(num_feature_levels)
        ])

    def forward(self, features: List[torch.Tensor], masks: List[Optional[torch.Tensor]], apply: bool = True
                ) -> List[torch.Tensor]:
        if not apply:
            return features

        processed = []
        for feat, mask, reduce, depthwise, expand, norm, kernel_size in zip(
            features, masks, self.reduce_layers, self.depthwise_filters,
            self.expand_layers, self.norm_layers, self.kernel_schedule):
            if mask is None:
                valid = torch.ones((feat.size(0), 1, feat.size(2), feat.size(3)), device=feat.device, dtype=feat.dtype)
            else:
                valid = mask.to(feat.dtype)
                if valid.dim() == 3:
                    valid = valid.unsqueeze(1)

            # Keep padded region intact
            padded = feat * (1.0 - valid)
            content = feat * valid

            # Estimate the high-frequency residual on valid positions
            low_pass = F.avg_pool2d(content, kernel_size=kernel_size, stride=1,
                                    padding=kernel_size // 2)
            high_residual = content - low_pass

            reduced = reduce(high_residual)
            filtered = depthwise(F.relu(reduced, inplace=False))
            restored = expand(filtered)
            restored = norm(restored)

            suppressed_content = content - torch.tanh(restored)
            suppressed = suppressed_content * valid + padded
            processed.append(suppressed)
        return processed
