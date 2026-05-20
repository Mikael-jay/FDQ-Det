import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"
DEFAULT_REGISTRY = APP_ROOT / "model_registry.json"


class ModelRegistry:
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path

    def load(self) -> List[Dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise ValueError("model_registry.json field 'models' must be a list")
        return models


def draw_detections_locally(image_bytes: bytes, detections: List[Dict[str, Any]]) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for det in detections:
        box = det.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95)
    return out.getvalue()


def resolve_model_selection(
    models: List[Dict[str, Any]],
    model_id: Optional[str],
    custom_checkpoint: Optional[str],
    custom_config: Optional[str],
) -> Tuple[Optional[str], Optional[str], str]:
    if custom_checkpoint:
        return custom_config, custom_checkpoint, "custom"

    if not model_id:
        raise ValueError("Missing model_id or custom_checkpoint")

    selected = next((m for m in models if m.get("id") == model_id), None)
    if selected is None:
        raise ValueError(f"Unknown model_id: {model_id}")

    return selected.get("config"), selected.get("checkpoint"), str(selected.get("id"))


def resolve_selected_dataset(models: List[Dict[str, Any]], model_id: Optional[str], dataset: Optional[str]) -> str:
    selected_dataset = dataset
    if not selected_dataset and model_id:
        selected = next((m for m in models if m.get("id") == model_id), {})
        selected_dataset = selected.get("dataset")
    return selected_dataset or "aitod"


async def call_upstream_predict(
    client: httpx.AsyncClient,
    api_base_url: str,
    image_name: str,
    image_bytes: bytes,
    image_content_type: Optional[str],
    dataset: str,
    conf_thresh: float,
    nms_iou_thresh: float,
    checkpoint: Optional[str],
    config: Optional[str],
) -> Dict[str, Any]:
    files = {
        "image": (image_name, image_bytes, image_content_type or "image/jpeg")
    }
    data = {
        "dataset": dataset,
        "conf_thresh": str(conf_thresh),
        "nms_iou_thresh": str(nms_iou_thresh),
        "visualize": "false",
    }
    if checkpoint:
        data["checkpoint"] = checkpoint
    if config:
        data["config"] = config

    response = await client.post(f"{api_base_url}/predict", files=files, data=data)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


def create_app(api_base_url: str, registry_path: Path) -> FastAPI:
    app = FastAPI(title="HDU-Det Demo UI")
    registry = ModelRegistry(registry_path)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{api_base_url}/health")
            resp.raise_for_status()
            upstream = resp.json()
            return {
                "status": "ok",
                "upstream": upstream,
                "api_base_url": api_base_url,
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Upstream API unavailable: {exc}") from exc

    @app.get("/api/models")
    def get_models() -> Dict[str, Any]:
        try:
            return {"models": registry.load()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/predict")
    async def predict(
        image: UploadFile = File(...),
        model_id: Optional[str] = Form(default=None),
        custom_checkpoint: Optional[str] = Form(default=None),
        custom_config: Optional[str] = Form(default=None),
        dataset: Optional[str] = Form(default=None),
        conf_thresh: float = Form(default=0.55),
        nms_iou_thresh: float = Form(default=0.5),
    ) -> Dict[str, Any]:
        try:
            image_bytes = await image.read()
            if not image_bytes:
                raise ValueError("Uploaded image is empty")

            models = registry.load()
            config, checkpoint, selected_model_id = resolve_model_selection(
                models=models,
                model_id=model_id,
                custom_checkpoint=custom_checkpoint,
                custom_config=custom_config,
            )
            selected_dataset = resolve_selected_dataset(models=models, model_id=model_id, dataset=dataset)

            async with httpx.AsyncClient(timeout=120.0) as client:
                result = await call_upstream_predict(
                    client=client,
                    api_base_url=api_base_url,
                    image_name=image.filename or "image.jpg",
                    image_bytes=image_bytes,
                    image_content_type=image.content_type,
                    dataset=selected_dataset,
                    conf_thresh=conf_thresh,
                    nms_iou_thresh=nms_iou_thresh,
                    checkpoint=checkpoint,
                    config=config,
                )
            rendered = draw_detections_locally(image_bytes, result.get("detections", []))
            rendered_base64 = base64.b64encode(rendered).decode("ascii")

            return {
                "model_id": selected_model_id,
                "dataset": selected_dataset,
                "result": result,
                "rendered_image_base64": rendered_base64,
            }
        except HTTPException:
            raise
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/predict_batch")
    async def predict_batch(
        images: List[UploadFile] = File(...),
        model_id: Optional[str] = Form(default=None),
        custom_checkpoint: Optional[str] = Form(default=None),
        custom_config: Optional[str] = Form(default=None),
        dataset: Optional[str] = Form(default=None),
        conf_thresh: float = Form(default=0.55),
        nms_iou_thresh: float = Form(default=0.5),
    ) -> Dict[str, Any]:
        try:
            if not images:
                raise ValueError("No images uploaded")

            models = registry.load()
            config, checkpoint, selected_model_id = resolve_model_selection(
                models=models,
                model_id=model_id,
                custom_checkpoint=custom_checkpoint,
                custom_config=custom_config,
            )
            selected_dataset = resolve_selected_dataset(models=models, model_id=model_id, dataset=dataset)

            results: List[Dict[str, Any]] = []
            async with httpx.AsyncClient(timeout=240.0) as client:
                for index, image in enumerate(images):
                    filename = image.filename or f"image_{index + 1}.jpg"
                    try:
                        image_bytes = await image.read()
                        if not image_bytes:
                            raise ValueError("Uploaded image is empty")

                        result = await call_upstream_predict(
                            client=client,
                            api_base_url=api_base_url,
                            image_name=filename,
                            image_bytes=image_bytes,
                            image_content_type=image.content_type,
                            dataset=selected_dataset,
                            conf_thresh=conf_thresh,
                            nms_iou_thresh=nms_iou_thresh,
                            checkpoint=checkpoint,
                            config=config,
                        )
                        rendered = draw_detections_locally(image_bytes, result.get("detections", []))
                        rendered_base64 = base64.b64encode(rendered).decode("ascii")
                        results.append(
                            {
                                "filename": filename,
                                "ok": True,
                                "result": result,
                                "rendered_image_base64": rendered_base64,
                            }
                        )
                    except Exception as item_exc:
                        results.append(
                            {
                                "filename": filename,
                                "ok": False,
                                "error": str(item_exc),
                            }
                        )

            success_count = sum(1 for item in results if item.get("ok"))
            return {
                "model_id": selected_model_id,
                "dataset": selected_dataset,
                "total": len(results),
                "success": success_count,
                "failed": len(results) - success_count,
                "results": results,
            }
        except HTTPException:
            raise
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HDU-Det standalone demo UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(api_base_url=args.api_base_url.rstrip("/"), registry_path=Path(args.registry))
    uvicorn.run(app, host=args.host, port=args.port)
