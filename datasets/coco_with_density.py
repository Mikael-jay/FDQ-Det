"""
COCO数据集的密度图包装器

在数据加载阶段实时生成GT密度图
"""

import os
import torch
from pathlib import Path
from PIL import Image
from .coco import CocoDetection
from .density_loader import DensityMapLoader


class CocoDetectionWithDensity(CocoDetection):
    """
    扩展CocoDetection以支持GT密度图生成
    
    在数据加载阶段根据边界框实时生成密度图
    无需预生成和保存大量.npy文件
    """
    
    def __init__(self, img_folder, ann_file, transforms, return_masks, 
                 aux_target_hacks=None, 
                 use_density_supervision=False,
                 density_sigma=5,
                 dataset_file=None,
                 num_classes=None):
        """
        Args:
            img_folder: 图像文件夹
            ann_file: 标注文件
            transforms: 数据增强
            return_masks: 是否返回masks
            aux_target_hacks: 辅助hooks
            use_density_supervision: 是否使用密度图监督
            density_sigma: 高斯核标准差
        """
        super().__init__(img_folder, ann_file, transforms, return_masks, aux_target_hacks, dataset_file=dataset_file, num_classes=num_classes)
        
        self.use_density_supervision = use_density_supervision
        self.density_loader = None
        
        if use_density_supervision:
            self.density_loader = DensityMapLoader(sigma=density_sigma)
            print(f"[INFO] DensityMapLoader initialized:")
            print(f"       sigma={density_sigma}")
            print(f"       mode=real-time generation")
    
    def __getitem__(self, idx):
        """获取数据项，包括GT密度图（如果启用）"""
        # 直接调用父类的基础 __getitem__，获取 PIL Image 和原始 target
        coco = self.coco
        img_id = self.ids[idx]
        coco_image = coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, coco_image['file_name'])
        img = Image.open(path).convert('RGB')
        
        # 获取标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        # 构建target字典（格式同CocoDetection.__getitem__）
        target = {'image_id': img_id, 'annotations': annotations}
        
        # 应用prepare转换（将COCO格式转换为检测格式）
        # prepare 会返回处理后的 (img, target)
        img, target = self.prepare(img, target)
        
        # Remap labels to 0-based for VisDrone (categories are 1..10 in JSON)
        if self.dataset_file == 'visdrone' and 'labels' in target:
            target['labels'] = target['labels'] - 1
        
        # Sanitize labels: remove boxes with out-of-range labels to avoid indexing errors
        if getattr(self, 'num_classes', None) is not None and 'labels' in target:
            labels = target['labels']
            if labels.numel() > 0:
                valid = (labels >= 0) & (labels < int(self.num_classes))
                if valid.all() is False:
                    idx = valid.nonzero(as_tuple=False).squeeze(1)
                    if idx.numel() == 0:
                        # no valid boxes, set empty tensors
                        target['boxes'] = torch.zeros((0, 4), dtype=target['boxes'].dtype)
                        target['labels'] = torch.zeros((0,), dtype=target['labels'].dtype)
                        if 'area' in target:
                            target['area'] = torch.zeros((0,), dtype=target['area'].dtype)
                        if 'iscrowd' in target:
                            target['iscrowd'] = torch.zeros((0,), dtype=target['iscrowd'].dtype)
                    else:
                        for key in ['boxes', 'labels', 'area', 'iscrowd']:
                            if key in target:
                                target[key] = target[key][idx]
        
        # 如果启用密度图监督，在transforms之前生成密度图
        if self.use_density_supervision and self.density_loader is not None:
            boxes = target.get('boxes', None)
            if boxes is not None and len(boxes) > 0:
                # 获取图像大小
                h, w = img.height, img.width
                
                # 实时生成密度图
                density = self.density_loader.generate(boxes, (h, w))
                
                if density is not None:
                    # 添加到target [1, H, W]
                    target['gt_density_map'] = torch.from_numpy(density).float().unsqueeze(0)
                else:
                    # 生成失败，使用零张量
                    target['gt_density_map'] = torch.zeros(1, h, w, dtype=torch.float32)
            else:
                # 没有目标，生成零密度图
                target['gt_density_map'] = torch.zeros(1, img.height, img.width, dtype=torch.float32)
        
        # 最后应用transforms（使用父类中的 _transforms）
        if getattr(self, "_transforms", None) is not None:
            img, target = self._transforms(img, target)
        
        return img, target


def build_coco_with_density(image_set, args):
    """
    构建支持密度图的COCO数据集
    
    与原始build()兼容，但添加了density参数支持
    """
    from .coco import get_aux_target_hacks_list, make_coco_transforms
    from .coco import preparing_dataset
    
    root = Path(args.coco_path)
    
    if args.dataset_file == 'aitod_v2':
        PATHS = {
            "train": (root / "train", root / "annotations" / 'aitodv2_train.json'),
            "trainval": (root / "images/trainval", root / "annotations" / 'aitodv2_trainval.json'),
            "val": (root / "valid", root / "annotations" / 'aitodv2_val.json'),
            "eval_debug": (root / "valid", root / "annotations" / 'aitodv2_val.json'),
            "test": (root / "test", root / "annotations" / 'aitodv2_test.json'),
        }
    elif args.dataset_file == 'visdrone':
        PATHS = {
            "train": (root / "VisDrone2019-DET-train" / "images", root / "annotations_coco" / 'VisDrone2019-DET_train_coco.json'),
            "trainval": (root / "VisDrone2019-DET-train" / "images", root / "annotations_coco" / 'VisDrone2019-DET_train_coco.json'),
            "val": (root / "VisDrone2019-DET-val" / "images", root / "annotations_coco" / 'VisDrone2019-DET_val_coco.json'),
            "eval_debug": (root / "VisDrone2019-DET-val" / "images", root / "annotations_coco" / 'VisDrone2019-DET_val_coco.json'),
            "test": (root / "VisDrone2019-DET-val" / "images", root / "annotations_coco" / 'VisDrone2019-DET_val_coco.json'),
        }
    else:
        raise ValueError(f"Unsupported dataset_file: {args.dataset_file}")

    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]

    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    
    # 获取密度图相关参数
    use_density_supervision = getattr(args, 'use_density_supervision', False)
    density_sigma = getattr(args, 'density_sigma', 5)
    
    dataset = CocoDetectionWithDensity(
        img_folder, 
        ann_file,
        transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args),
        return_masks=args.masks,
        aux_target_hacks=aux_target_hacks_list,
        use_density_supervision=use_density_supervision,
        density_sigma=density_sigma,
        dataset_file=args.dataset_file,
        num_classes=getattr(args, 'num_classes', None),
    )

    return dataset
