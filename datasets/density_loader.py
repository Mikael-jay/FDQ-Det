"""
GT密度图加载和生成工具

在数据加载阶段实时生成GT密度图，无需预计算
"""

import numpy as np
import torch


class DensityMapLoader:
    """
    GT密度图加载器
    
    在数据加载阶段实时根据边界框生成密度图
    - 无需预生成和保存大量.npy文件
    - 无需计算single_object_integral
    - 直接从标注生成，保证一致性
    """
    
    def __init__(self, sigma=3):
        """
        Args:
            sigma: 高斯核标准差
        """
        self.sigma = sigma
        # per-object impulse magnitude placed at bbox center before gaussian smoothing
        # Historically this was 1.0; experiments show larger unit_mass (e.g. 250) produces
        # value ranges closer to model outputs for sigma=5.
        self.unit_mass = 90.0
        self.scale = 1.0
        
        try:
            from scipy.ndimage import gaussian_filter
            self.gaussian_filter = gaussian_filter
        except ImportError:
            print("[WARNING] scipy not installed. Install with: pip install scipy")
            self.gaussian_filter = None
    
    def generate(self, boxes, image_shape):
        """
        根据边界框生成GT密度图
        
        Args:
            boxes: [N, 4] numpy/tensor，格式为 (x_min, y_min, x_max, y_max)
            image_shape: (H, W) 图像尺寸
            
        Returns:
            density_map: [H, W] float32 numpy数组，非负密度值
        """
        if self.gaussian_filter is None:
            print("[ERROR] gaussian_filter not available, returning zeros")
            h, w = image_shape
            return np.zeros((h, w), dtype=np.float32)
        
        h, w = image_shape
        
        # 初始化密度图
        density = np.zeros((h, w), dtype=np.float32)
        
        # 将boxes转换为numpy数组
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        if len(boxes) == 0:
            # 如果没有目标，返回零密度图
            return density
        
        # 在边界框中心放置高斯响应
        for bbox in boxes:
            x_min, y_min, x_max, y_max = bbox
            # 计算边界框中心
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)
            
            # 确保在图像范围内
            if 0 <= cx < w and 0 <= cy < h:
                density[cy, cx] += float(self.unit_mass)
    
        # 应用高斯滤波平滑
        density = self.gaussian_filter(density, sigma=self.sigma)
        density *= self.scale
        
        return density
