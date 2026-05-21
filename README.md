# FDQ-Det

FDQ-Det is a density-aware tiny object detection project for **AITODv2** and **VisDrone2019**. The repository provides the cleaned training, evaluation, command-line inference, HTTP inference API, and a lightweight web demo UI.

This repository is a cleaned release of the FDQ-Det implementation. Public-facing names and model registry entries use `FDQ-Det` / `fdqdet`.

## Repository Layout

```text
config/                 Model configs for AITOD v2 and VisDrone
models/fdqdet/          FDQ-Det model, transformer, CUDA op source, matcher, backbone
datasets/               AITOD/VisDrone COCO loaders and extension dataset utilities
scripts/                Train, eval, and inference shell entrypoints
demo_api_ui/            Standalone browser demo that proxies to the FastAPI service
main.py                 Training and evaluation entrypoint
inference.py            CLI inference for one image or a folder of images
fdqdet_infer.py         Shared inference core used by CLI and API
api.py                  FastAPI inference service
requirements.txt        Main project dependencies
```

## Weights

Put the weight files under `pt/` with these names when you want to reproduce local inference or evaluation:

```text
pt/FDQ_Det_AITODv2_best321.pth
pt/FDQ_Det_VisDrone_best384.pth
pt/pretrained_model.pth
```

Click the link below to get pre-trained and well-trained model weights: [FDQ-Det Pretrained Weight](https://github.com/Mikael-jay/FDQ-Det/releases/tag/weight-v1.0)

## Environment

The project has been used with Python 3.9, PyTorch, torchvision, CUDA, and a compiled multi-scale deformable attention op. Install PyTorch/torchvision for your CUDA version first, then install the remaining dependencies:

```bash
conda create -n fdqdet python=3.9 --y
conda activate fdqdet
bash install.sh
```

Build the CUDA extension:

```bash
cd models/fdqdet/ops
python setup.py build install
# unit test (should see all checking is True)
python test.py
cd ../../..
```

If the build fails, check that `CUDA_HOME`, PyTorch, torchvision, GCC, and your CUDA toolkit are compatible.

## Dataset Layout

### AITOD v2

`config/fdqdet_aitod.py` expects `--dataset_file aitod_v2` and a COCO-style layout under `<aitod_path>`:

```text
<aitod_path>/
  images/trainval/
  test/
  annotations/aitodv2_trainval.json
  annotations/aitodv2_test.json
```

### VisDrone

`config/fdqdet_visdrone.py` expects `--dataset_file visdrone` and converted COCO annotations:

```text
<visdrone_path>/
  VisDrone2019-DET-train/images/
  VisDrone2019-DET-val/images/
  annotations_coco/VisDrone2019-DET_train_coco.json
  annotations_coco/VisDrone2019-DET_val_coco.json
```

## Training

```bash
bash scripts/train_aitod.sh <aitod_path> [pretrain_ckpt] [output_dir]

bash scripts/train_visdrone.sh <visdrone_path> [pretrain_ckpt] [output_dir]
```

Defaults:

- `pretrain_ckpt`: `pt/pretrained_model.pth`
- `NPROC_PER_NODE`: `2`
- `USE_HIGH_FREQ_SUPPRESS`: `True`
- `USE_ADV_TRAINING`: `False`

Example:

```bash
NPROC_PER_NODE=2 bash scripts/train_aitod.sh /data/AI-TOD pt/pretrained_model.pth logs/fdqdet_aitod
```

## Evaluation

```bash
bash scripts/eval_aitod.sh <aitod_path> [checkpoint] [output_dir]

bash scripts/eval_visdrone.sh <visdrone_path> [checkpoint] [output_dir]
```

Examples:

```bash
bash scripts/eval_aitod.sh /data/AI-TOD pt/FDQ_Det_AITODv2_best321.pth output/eval_aitod

bash scripts/eval_visdrone.sh /data/VisDrone pt/FDQ_Det_VisDrone_best384.pth output/eval_visdrone
```

## CLI Inference

`inference.py` accepts either one image file or a directory. Directory inputs are scanned recursively by default and save one JSON file per image plus `batch_results.json` under `--output_dir`.

```bash
python inference.py \
  --dataset aitod \
  --input_image <image_path_or_dir> \
  --model_weights pt/FDQ_Det_AITODv2_best321.pth \
  --output_dir output/aitod_infer \
  --conf_thresh 0.6 \
  --nms_iou_thresh 0.5 \
  --visualize
```

```bash
python inference.py \
  --dataset visdrone \
  --input_image <image_path_or_dir> \
  --model_weights pt/FDQ_Det_VisDrone_best384.pth \
  --output_dir output/visdrone_infer \
  --conf_thresh 0.35 \
  --nms_iou_thresh 0.5 \
  --visualize
```

Supported image extensions:

```text
.jpg .jpeg .png .bmp .tif .tiff .webp
```

Use `--no-recursive` to scan only the top level of a directory. Visualizations draw bounding boxes only; labels and confidence text are omitted to avoid covering small objects.

## Demo Web UI

The demo UI is a small standalone FastAPI app that calls the inference API and renders the returned boxes in a browser.

Start the inference API first:

```bash
python api.py --device cuda --host 0.0.0.0 --port 8000
```

Then start the demo app:

```bash
cd demo_api_ui
python app.py --api-base-url http://127.0.0.1:8000 --host 0.0.0.0 --port 8081
```

Open:

```text
http://127.0.0.1:8081
```

### Advanced Options

In the demo UI, you can:
- **Device Selection**: Choose `cpu`, `cuda`, or `auto` (default startup device) from the Device dropdown in Advanced Options. This allows overriding the API's startup device on a per-request basis.
- **Custom Checkpoint/Config**: Specify paths to custom model files for inference.
- **Confidence Threshold & IoU Threshold**: Adjust detection filtering parameters.

Edit `demo_api_ui/model_registry.json` to add more models or change default thresholds.

## Extending To Other Datasets

AITOD v2 and VisDrone are the maintained release paths. The repository also keeps COCO and panoptic utilities for future extension:

- `datasets/coco.py`
- `datasets/coco_panoptic.py`
- `datasets/panoptic_eval.py`
- `datasets/sltransform.py`
- `datasets/random_crop.py`

For a new COCO-style dataset, add a config file, update dataset path mapping in the dataset builder, and set the correct `num_classes` and annotation paths.

## License

This project is released under the Apache License 2.0. See `LICENSE` for details.

## Acknowledgements

This project builds on ideas and components from DETR-family detectors, Deformable DETR/DINO-style implementations, COCO evaluation tooling, and small-object detection research. Please also follow the licenses of upstream dependencies and datasets.
