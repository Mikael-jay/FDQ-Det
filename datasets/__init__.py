# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return None


def build_dataset(image_set, args):
    use_density_supervision = getattr(args, "use_density_supervision", False)

    if args.dataset_file in ["aitod_v2", "visdrone"]:
        if use_density_supervision:
            from .coco_with_density import build_coco_with_density
            return build_coco_with_density(image_set, args)
        return build_coco(image_set, args)

    if args.dataset_file == "coco":
        return build_coco(image_set, args)

    if args.dataset_file == "coco_panoptic":
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)

    raise ValueError(f"dataset {args.dataset_file} is not supported")
