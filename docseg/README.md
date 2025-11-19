## `docseg` — Document Region Segmenter

`docseg` wraps a YOLOv8-seg model to detect fixed layout regions on forms
(`header`, `notes_block`, `billing_codes`, etc.), and exposes a tiny API for
training and inference.

### Structure

- **`config.py`**: small helpers / registries for mapping a `doc_type` to a
  weights file and class names (used by inference).
- **`inference.py`**: main runtime API:
  - `Region` / `InferenceResult` dataclasses
  - `SegmenterInference` for batched document segmentation and visualization
- **`training.py`**: thin wrapper around `ultralytics.YOLO` for training /
  finetuning segmentation models (`SegmenterTrain`).
- **`utils/coco2yoloseg.py`**: script utilities for converting COCO-style
  annotations to YOLO-seg format.

### Dependencies

- Python 3.10+
- `ultralytics` (YOLOv8)
- `opencv-python`
- `numpy`

Install from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Model weights and data

- Place / configure your YOLO segmentation weights (e.g. `best.pt`) in a path
  referenced by your `DocTypeRegistry` (see `config.py`).
- Typical training data layout (see `data/dataset.yaml`):
  - `data/images/train`, `data/images/val`
  - `data/labels/train`, `data/labels/val`
  - `data/class_names.txt` for index → label mapping

### Inference usage

Minimal example using `SegmenterInference`:

```python
from docseg.config import DocTypeRegistry
from docseg.inference import SegmenterInference

registry = DocTypeRegistry(
    default_weights="yolov8n-seg.pt",           # or path to your trained weights
    default_class_names_path="data/class_names.txt",
)

segmenter = SegmenterInference(registry, imgsz=1024, conf=0.3)

result = segmenter.run(
    image_path="input/1.jpeg",
    doc_type="er_form",                         # logical name used in the registry
    out_overlay="out/overlay.png",
    out_crops_dir="out/crops",
)

for r in result.regions:
    print(r.label, r.score, r.bbox)
```

Key behaviors:

- Lazily loads YOLO weights per `doc_type` and caches models in memory.
- Returns box coordinates in pixel space and (optionally) RLE-encoded masks.
- Can save an overlay image and cropped regions for downstream OCR.

### Training / finetuning

Use `SegmenterTrain` to train a new model or finetune an existing one:

```python
from docseg.training import SegmenterTrain

trainer = SegmenterTrain()

# Train from a base YOLOv8-seg checkpoint
run_dir = trainer.train_new(
    data_yaml="data/dataset.yaml",
    model_arch="yolov8n-seg.pt",
    imgsz=1024,
    epochs=200,
    batch=8,
)

print("Training run saved under:", run_dir)
```

For finetuning a previous run:

```python
run_dir = trainer.finetune_existing(
    weights_path="runs/segment/forms-3cls7/weights/best.pt",
    data_yaml="data/dataset.yaml",
    imgsz=1024,
)
```

The `runs/segment/...` directory (ignored by git) will contain checkpoints and
metrics. You can update your `DocTypeRegistry` to point at the `best.pt` from
your desired run.
