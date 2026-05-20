import argparse
import io
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image


from fdqdet_infer import (
    MODEL_CACHE,
    PROJECT_ROOT,
    draw_detections,
    ensure_checkpoint_allowed,
    image_digest,
    predict_image,
    resolve_config_checkpoint,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def create_app(device: str, allow_external_checkpoints: bool = False) -> FastAPI:
    app = FastAPI(title="FDQ-Det Inference API")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "device": device,
            "cached_models": len(MODEL_CACHE),
        }

    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        checkpoint: Optional[str] = Form(default=None),
        config: Optional[str] = Form(default=None),
        dataset: str = Form(default="aitod"),
        conf_thresh: float = Form(default=0.55),
        nms_iou_thresh: float = Form(default=0.5),
        visualize: bool = Form(default=False),
    ) -> dict:
        try:
            config_path, checkpoint_path = resolve_config_checkpoint(dataset, config, checkpoint)
            ensure_checkpoint_allowed(checkpoint_path, allow_external=allow_external_checkpoints)
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = predict_image(
                image=pil_image,
                dataset=dataset,
                config=str(config_path),
                checkpoint=str(checkpoint_path),
                conf_thresh=conf_thresh,
                nms_iou_thresh=nms_iou_thresh,
                device=device,
                use_cache=True,
            )
            if visualize:
                out_dir = PROJECT_ROOT / "output" / "api"
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(image.filename or "image").stem
                vis_path = out_dir / f"{stem}_{image_digest(image_bytes)}.jpg"
                draw_detections(pil_image, result["detections"]).save(vis_path)
                result["visualization_path"] = str(vis_path)
            return result
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/predict_batch")
    async def predict_batch(
        images: list[UploadFile] = File(...),
        checkpoint: Optional[str] = Form(default=None),
        config: Optional[str] = Form(default=None),
        dataset: str = Form(default="aitod"),
        conf_thresh: float = Form(default=0.55),
        nms_iou_thresh: float = Form(default=0.5),
        visualize: bool = Form(default=False),
    ) -> dict:
        """Process multiple images in one request."""
        try:
            config_path, checkpoint_path = resolve_config_checkpoint(dataset, config, checkpoint)
            ensure_checkpoint_allowed(checkpoint_path, allow_external=allow_external_checkpoints)
            
            results = []
            for idx, image_file in enumerate(images, start=1):
                try:
                    image_bytes = await image_file.read()
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    result = predict_image(
                        image=pil_image,
                        dataset=dataset,
                        config=str(config_path),
                        checkpoint=str(checkpoint_path),
                        conf_thresh=conf_thresh,
                        nms_iou_thresh=nms_iou_thresh,
                        device=device,
                        use_cache=True,
                    )
                    result["filename"] = image_file.filename or f"image_{idx}"
                    
                    if visualize:
                        out_dir = PROJECT_ROOT / "output" / "api"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        stem = Path(image_file.filename or f"image_{idx}").stem
                        vis_path = out_dir / f"{stem}_{image_digest(image_bytes)}.jpg"
                        draw_detections(pil_image, result["detections"]).save(vis_path)
                        result["visualization_path"] = str(vis_path)
                    
                    results.append(result)
                except Exception as e:
                    results.append({
                        "filename": image_file.filename or f"image_{idx}",
                        "error": str(e),
                    })
            
            return {
                "num_images": len(images),
                "results": results,
            }
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FDQ-Det API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-external-checkpoints", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        create_app(args.device, args.allow_external_checkpoints),
        host=args.host,
        port=args.port,
    )
