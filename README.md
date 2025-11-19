# docseg — Document Region Segmenter (YOLO-seg)

A clean Python module for training and inference of region segmentation on fixed-layout documents.

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

See `main.py` for a minimal example.

## Chandra Structure Detector (remote service)

The pipeline can call a remote OpenAI-compatible microservice that serves the Chandra model to detect layout regions:
`patient_header` (top printed header), `notes_block` (bottom free-text area), and `orders_block` (circled codes).

TODO:

- Add parallel processing for OCR
- Add local OCR model usage. Ensure parallel processing works.
- Wrap in small fastapi server
- Clean up saved files. No need to save for production.
- add dockerfile and docker-compose file
