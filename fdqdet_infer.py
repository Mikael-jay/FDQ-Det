import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from torchvision.transforms import functional as F

from models.fdqdet import build_fdqdet
from util.misc import NestedTensor
from util.slconfig import SLConfig


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DEFAULTS = {
    "aitod": {
        "config": PROJECT_ROOT / "config" / "fdqdet_aitod.py",
        "checkpoint": PROJECT_ROOT / "pt" / "FDQ_Det_AITODv2_best321.pth",
    },
    "aitod_v2": {
        "config": PROJECT_ROOT / "config" / "fdqdet_aitod.py",
        "checkpoint": PROJECT_ROOT / "pt" / "FDQ_Det_AITODv2_best321.pth",
    },
    "visdrone": {
        "config": PROJECT_ROOT / "config" / "fdqdet_visdrone.py",
        "checkpoint": PROJECT_ROOT / "pt" / "FDQ_Det_VisDrone_best384.pth",
    },
}


class ModelCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, str], Tuple[torch.nn.Module, object]] = {}

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, config_path: Path, checkpoint_path: Path, device: str) -> Tuple[torch.nn.Module, object]:
        config_path = config_path.resolve()
        checkpoint_path = checkpoint_path.resolve()
        key = (str(config_path), str(checkpoint_path), device)
        if key not in self._cache:
            self._cache[key] = load_model(config_path, checkpoint_path, device)
        return self._cache[key]


MODEL_CACHE = ModelCache()


def resolve_dataset_defaults(dataset: str) -> Dict[str, Path]:
    key = dataset.lower()
    if key not in DATASET_DEFAULTS:
        choices = ", ".join(sorted(DATASET_DEFAULTS))
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {choices}")
    return DATASET_DEFAULTS[key]


def resolve_config_checkpoint(
    dataset: str,
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> Tuple[Path, Path]:
    defaults = resolve_dataset_defaults(dataset)
    config_path = Path(config) if config else defaults["config"]
    checkpoint_path = Path(checkpoint) if checkpoint else defaults["checkpoint"]
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    return config_path, checkpoint_path


def ensure_checkpoint_allowed(checkpoint_path: Path, allow_external: bool = False) -> None:
    checkpoint_path = checkpoint_path.resolve()
    pt_dir = (PROJECT_ROOT / "pt").resolve()
    if allow_external:
        return
    if os.path.commonpath([str(pt_dir), str(checkpoint_path)]) != str(pt_dir):
        raise PermissionError(
            "External checkpoints are disabled. Put weights under pt/ or start "
            "the API with --allow-external-checkpoints."
        )


def load_model(config_path: Path, checkpoint_path: Path, device: str) -> Tuple[torch.nn.Module, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = SLConfig.fromfile(str(config_path))
    cfg.device = device
    model, _, postprocessors = build_fdqdet(cfg)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Checkpoint must be a dict containing a 'model' key: {checkpoint_path}")

    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model, postprocessors


def preprocess_image(image: Image.Image, device: str) -> NestedTensor:
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    mask = torch.zeros(
        (1, image_tensor.shape[-2], image_tensor.shape[-1]),
        dtype=torch.bool,
        device=device,
    )
    return NestedTensor(image_tensor, mask)


def filter_detections(
    result: Dict[str, torch.Tensor],
    conf_thresh: float,
    nms_iou_thresh: Optional[float] = 0.5,
) -> List[Dict[str, object]]:
    boxes = result["boxes"].detach().cpu()
    labels = result["labels"].detach().cpu()
    scores = result["scores"].detach().cpu()

    keep = scores >= conf_thresh
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if nms_iou_thresh is not None and nms_iou_thresh > 0 and boxes.numel() > 0:
        # Apply NMS per class so overlapping boxes from different classes are not suppressed.
        keep_indices = []
        for label in labels.unique():
            class_indices = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
            selected = nms(boxes[class_indices], scores[class_indices], float(nms_iou_thresh))
            keep_indices.append(class_indices[selected])
        keep = torch.cat(keep_indices) if keep_indices else torch.empty(0, dtype=torch.long)
        keep = keep[torch.argsort(scores[keep], descending=True)]
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
    else:
        order = torch.argsort(scores, descending=True)
        boxes = boxes[order]
        labels = labels[order]
        scores = scores[order]

    detections: List[Dict[str, object]] = []
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        detections.append(
            {
                "label": int(label),
                "score": float(score),
                "box": [float(v) for v in box],
            }
        )
    return detections


def draw_detections(image: Image.Image, detections: List[Dict[str, object]]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return canvas


def predict_image(
    image: Image.Image,
    dataset: str = "aitod",
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
    conf_thresh: float = 0.55,
    nms_iou_thresh: Optional[float] = 0.5,
    device: str = "cuda",
    use_cache: bool = True,
) -> Dict[str, object]:
    config_path, checkpoint_path = resolve_config_checkpoint(dataset, config, checkpoint)
    rgb_image = image.convert("RGB")
    inputs = preprocess_image(rgb_image, device)

    if use_cache:
        model, postprocessors = MODEL_CACHE.get(config_path, checkpoint_path, device)
    else:
        model, postprocessors = load_model(config_path.resolve(), checkpoint_path.resolve(), device)

    start = time.time()
    with torch.no_grad():
        outputs = model(inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_time = time.time() - start

    image_size = torch.tensor([[rgb_image.height, rgb_image.width]], device=device)
    results = postprocessors["bbox"](outputs, image_size)
    detections = filter_detections(results[0], conf_thresh, nms_iou_thresh)

    return {
        "image": {"width": rgb_image.width, "height": rgb_image.height},
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "inference_time": inference_time,
        "conf_thresh": conf_thresh,
        "nms_iou_thresh": nms_iou_thresh,
        "detections": detections,
    }


def predict_image_file(
    image_path: str,
    dataset: str = "aitod",
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
    conf_thresh: float = 0.55,
    nms_iou_thresh: Optional[float] = 0.5,
    device: str = "cuda",
    output_dir: Optional[str] = None,
    visualize: bool = False,
) -> Dict[str, object]:
    image = Image.open(image_path).convert("RGB")
    result = predict_image(
        image=image,
        dataset=dataset,
        config=config,
        checkpoint=checkpoint,
        conf_thresh=conf_thresh,
        nms_iou_thresh=nms_iou_thresh,
        device=device,
        use_cache=False,
    )

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        json_path = out_dir / f"{stem}_result.json"
        import json

        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["result_path"] = str(json_path)

        if visualize:
            vis_path = out_dir / f"{stem}_visualization.jpg"
            draw_detections(image, result["detections"]).save(vis_path)
            result["visualization_path"] = str(vis_path)

    return result


def image_digest(image_bytes: bytes) -> str:
    return hashlib.sha1(image_bytes).hexdigest()[:12]
