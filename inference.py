import argparse
import json
from pathlib import Path
from typing import Iterable, List

from fdqdet_infer import predict_image_file, resolve_config_checkpoint

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FDQ-Det inference")
    parser.add_argument("--dataset", default="aitod", choices=["aitod", "aitod_v2", "visdrone"])
    parser.add_argument("--config_file", "--config", dest="config", default=None)
    parser.add_argument("--model_weights", "--checkpoint", dest="checkpoint", default=None)
    parser.add_argument("--input_image", required=True, help="Path to an image file or an image directory")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--conf_thresh", type=float, default=0.55)
    parser.add_argument("--nms_iou_thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    return parser.parse_args()


def iter_image_files(input_path: Path, recursive: bool = True) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Input file is not a supported image: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pattern: Iterable[Path]
    pattern = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(
        path for path in pattern
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No supported images found under: {input_path}")
    return images


def result_output_dir(base_output_dir: Path, input_root: Path, image_path: Path, batch_mode: bool) -> Path:
    if not batch_mode:
        return base_output_dir
    parent = image_path.parent
    try:
        relative_parent = parent.relative_to(input_root)
    except ValueError:
        relative_parent = Path()
    return base_output_dir / relative_parent


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_image)
    image_paths = iter_image_files(input_path, recursive=args.recursive)
    batch_mode = input_path.is_dir()
    input_root = input_path if batch_mode else input_path.parent

    config_path, checkpoint_path = resolve_config_checkpoint(
        args.dataset,
        config=args.config,
        checkpoint=args.checkpoint,
    )

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path}")
        out_dir = result_output_dir(Path(args.output_dir), input_root, image_path, batch_mode)
        result = predict_image_file(
            image_path=str(image_path),
            dataset=args.dataset,
            config=str(config_path),
            checkpoint=str(checkpoint_path),
            conf_thresh=args.conf_thresh,
            nms_iou_thresh=args.nms_iou_thresh,
            device=args.device,
            output_dir=str(out_dir),
            visualize=args.visualize,
        )
        result["input_path"] = str(image_path)
        results.append(result)

    if batch_mode:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "input_dir": str(input_path),
            "num_images": len(image_paths),
            "results": results,
        }
        summary_path = output_dir / "batch_results.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"summary_path": str(summary_path), "num_images": len(image_paths)}, indent=2))
    else:
        print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
