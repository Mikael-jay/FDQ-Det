# FDQ-Det Standalone Demo UI

This demo app is independent from the training and inference scripts and only depends on the existing HTTP API service.

## Features

- Image picker button
- Model weight selector (from registry)
- Start recognition button
- Two preview windows: original image and detection result
- Extensible model registry for future models

## Structure

- `app.py`: demo backend + API proxy
- `model_registry.json`: model list and defaults
- `static/index.html`: UI layout
- `static/styles.css`: UI style
- `static/app.js`: page logic

## Run

1. Start FDQ-Det API first:

```bash
python api.py --device cuda --host 0.0.0.0 --port 8000
```

2. In another terminal, start demo app:

```bash
cd demo_api_ui

pip install -r requirements.txt

python app.py --api-base-url http://127.0.0.1:8000 --host 0.0.0.0 --port 8081
```

3. Open browser:

```text
http://127.0.0.1:8081
```

## Extend Models

Edit `model_registry.json` and append new model entries under `models`.
