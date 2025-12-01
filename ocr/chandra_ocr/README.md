## `chandra_ocr` — Layout-aware OCR with prompts and constraints

`chandra_ocr` is a small library for taking cropped document regions (e.g.,
from `docseg`) and running layout-aware OCR / form decoding with:

- Pluggable backends (`openai`, local LLMs, etc.)
- Per-segment prompts (YAML)
- Optional dictionary / constraint lookups (JSON)
- Validators and normalizers for structured fields like MSP billing codes and PHNs.

### Structure

- **`ocr/backends/`**:
  - `base.py`: `BaseOCRBackend` interface (callable on a batch of inputs).
  - `chandra.py`: backend that talks to a Chandra multimodal model server.
  - `tesseract.py`: classic OCR backend (where supported).
- **`ocr/predictor.py`**:
  - `PromptRegistry`: loads YAML templates from `ocr/prompts`.
  - `ConstraintStore`: loads JSON constraint lists from `ocr/constraints`.
  - `ChandraOCRPredictor`: main high-level predictor that:
    - accepts a list of `Segment` objects
    - builds prompts + metadata for each region
    - calls a backend
    - validates and normalizes results into `OCRResult` objects.
- **`ocr/prompts/`**:
  - `patient_info.yaml`, `msp_code.yaml`, etc. for per-segment instructions.
- **`ocr/constraints/msp_code.json`**:
  - canonical code list and aliases for MSP billing codes (used by validators).
- **`ocr/types.py`**:
  - dataclasses / pydantic-style types (`Segment`, `OCRResult`, `Candidate`, `DecodeMode`).
- **`ocr/validators.py` / `ocr/normalizers.py`**:
  - simple helpers for cleaning and validating decoded fields.
- **`ocr/finetune.py`**:
  - optional tools for adapting models (if applicable).

### Dependencies

- `pydantic` (for some types)
- `pyyaml` (for YAML prompts; falls back to raw text if missing)
- Backend-specific dependencies (e.g., `openai`, `httpx`, etc.)

Install from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Model / backend configuration

`ChandraOCRPredictor` itself does **not** own a model; it relies on a backend
implementing `BaseOCRBackend`. The typical setup is:

- A remote multimodal server (e.g., llama.cpp or OpenAI-compatible) running a
  model like Chandra-OCR.
- A backend class that:
  - accepts a batch of dict inputs: `{"image": ..., "prompt": ..., "system": ..., "meta": ...}`
  - returns a list of dicts with keys like `"text"`, `"confidence"`, `"candidates"`, `"latency_ms"`, `"meta"`.

Model paths and API URLs are configured in the backend itself (see
`ocr/backends/chandra.py`) and often through environment variables (e.g.,
`CHANDRA_ENDPOINT`, `OPENAI_API_KEY`, etc.).

### Using `ChandraOCRPredictor`

Typical usage with a custom backend:

```python
from chandra_ocr.ocr.backends.chandra import ChandraBackend
from chandra_ocr.ocr.predictor import ChandraOCRPredictor
from chandra_ocr.ocr.types import Segment

backend = ChandraBackend(
    api_base="http://localhost:8000/v1",
    api_key="sk-your-key",
    model="chandra-ocr",
)

predictor = ChandraOCRPredictor(
    backend=backend,
    prompt_dir="../ocr/prompts",       # defaults; can be made absolute
    constraint_dir="../ocr/constraints",
)

segments = [
    Segment(
        segment_id="header-0",
        segment_type="patient_info",
        bbox=(0, 0, 100, 100),
        crop_path="out/crops/02_header.png",
    ),
    Segment(
        segment_id="codes-0",
        segment_type="msp_code",
        bbox=(0, 100, 200, 200),
        crop_path="out/crops/00_billing_codes.png",
    ),
]

results = predictor.predict(segments, context=None, top_k=3)
for r in results:
    print(r.segment_type, r.normalized_value, r.confidence)
```

Key behaviors:

- Resolves `prompt_dir` / `constraint_dir` relative to the module path, so you
  can call it from any working directory.
- Per-segment validators and normalizers are automatically selected based on
  `segment_type` (e.g., MSP code vs free text).
- Returns structured `OCRResult` objects with warnings and errors lists.

### Extending and customizing

- **New segment types**:
  - Add a prompt YAML in `ocr/prompts/{segment_type}.yaml`.
  - Optionally add a constraint JSON in `ocr/constraints/{segment_type}.json`.
  - Add a case in `_validator_for` in `ocr/predictor.py` if you need special
    validation.
- **New backend**:
  - Implement the `BaseOCRBackend` interface in `ocr/backends/your_backend.py`.
  - Make it callable on a list of `inputs` and return structured outputs.

# Flow Chandra OCR Module

This module provides a **swappable OCR backend** design with a production-focused `ChandraOCRPredictor` and a skeleton `ChandraOCRFineTuner`. It mirrors the structure you used for your segmenter: clean classes, no CLI, and simple integration points.

## Key Features

- **Backend abstraction**: `BaseOCRBackend` with pluggable backends (`ChandraBackend`, `TesseractBackend` stub).
- **Prompt/Constraint Registry**: YAML prompts per `segment_type` and JSON constraints (whitelists) for dictionary snapping.
- **Per-field validators/normalizers**: Regex/shape checks (e.g., MSP code), normalization (uppercasing, O→0 correction).
- **Batch inference**: Provide a list of `Segment` objects, get structured `OCRResult` with top-k candidates and telemetry.
- **Fine-tuning scaffold**: Dataset prep, train stub (replace with real loop), evaluate stub, export manifest.

## Install (dev)

```
pip install pillow pyyaml
```

(Real backends will add their own deps.)

## Layout

```
flow_chandra_ocr/
  ocr/
    backends/
      base.py
      chandra.py
      tesseract.py
    constraints/
      msp_code.json
    prompts/
      msp_code.yaml
      patient_name.yaml
    predictor.py
    finetune.py
    types.py
    validators.py
    normalizers.py
    utils.py
  README.md
```

## Inference Example

```python
from flow_chandra_ocr.ocr import ChandraOCRPredictor
from flow_chandra_ocr.ocr.backends import ChandraBackend
from flow_chandra_ocr.ocr.types import Segment, DecodeMode

backend = ChandraBackend(device="cuda", precision="fp16")  # replace with real model
predictor = ChandraOCRPredictor(
    backend=backend,
    prompt_dir="flow_chandra_ocr/ocr/prompts",
    constraint_dir="flow_chandra_ocr/ocr/constraints",
    decode_prefs={"msp_code": DecodeMode.DICT}
)

segments = [
    Segment(segment_id="s1", page_id="p1", segment_type="msp_code", bbox=(0,0,100,40), crop_path="/path/to/crop.png"),
]

ctx = {"msp_code": {"candidates": ["A001","A002","B123","C456","C457"]}}
results = predictor.predict(segments, context=ctx, top_k=3)
```

## Notes

- Full-page ROI fallback is intentionally **omitted** per requirements.
- Swap in other backends by implementing `BaseOCRBackend`.
- Replace `ChandraBackend.infer_batch` with a real model call.

## Auto-pick prompts & system prompts

- The predictor now **auto-loads the user prompt and system prompt** per `segment_type` from `ocr/prompts/{segment_type}.yaml`.
- Add `system: |` and `template: |` keys to each YAML; both are optional. If `system` is absent, the backend default is used.

## Models

- Download and place the gguf chandra model and mmproj model in the `models` folder

run llama.cpp server with the chandra model:

```bash
llama-server -m "/Users/ricky/Desktop/Flow Health App/flow-ocr/ocr/chandra_ocr/models/Chandra-OCR-Q4_K_M.gguf" --mmproj "/Users/ricky/Desktop/Flow Health App/flow-ocr/ocr/chandra_ocr/models/mmproj-F32.gguf" --ctx-size 8192 -ngl 999
```