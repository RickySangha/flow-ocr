## `scanify` — Stage 0 document scanner

`scanify` converts raw phone photos of documents into clean, deskewed,
optionally binarized "scanned" pages suitable for downstream segmentation
(`docseg`) and OCR (`chandra_ocr`).

### Structure

- **`scanify.py`**:
  - `ScanifyConfig`: dataclass of all processing options (geometry, binarization,
    debug saving, etc.).
  - `ScanifyResult`: output container with:
    - `image_bw`: final B/W or grayscale image (`np.ndarray`)
    - `warped_color`: optional color view after warping (if `keep_color_debug`)
    - `debug`: dictionary of intermediate visualizations / metadata.
  - `Scanify`: main pipeline class with:
    - `process(img_or_path, save_stem_override=None)` — single image
    - `process_batch(items, out_dir=None, stem_fn=None)` — batch processing
    - many internal helpers for perspective detection, skew estimation, etc.

### Dependencies

- `opencv-python` (`cv2`)
- `numpy`

Install from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### How it works (high level)

For each input image:

1. **Resize** to a configurable long-side (default 2400px).
2. **Page detection**:
   - Edge-based quad detection; optional fallback heuristic if edges fail.
   - Perspective warp to unskew the page rectangle.
3. **Deskew**:
   - Smart skew estimation around a seed angle; rotates to straighten text lines.
4. **Content crop**:
   - Optional cropping to the main content region with a small margin.
5. **Illumination + readability** (optional):
   - Background flattening
   - CLAHE + unsharp mask for light contrast enhancement.
6. **Binarization** (optional):
   - Sauvola / adaptive / Otsu thresholding, then despeckle + unsharp.
7. **Debug trail**:
   - If enabled, saves intermediate images into a `stage0_debug/` folder.

### Usage

Single-image example:

```python
from scanify.scanify import Scanify, ScanifyConfig

cfg = ScanifyConfig(
    target_long_side_px=2400,
    do_perspective=True,
    do_auto_deskew=True,
    do_binarize=True,
    save_debug=True,
    save_dir="stage0_debug",
)

scanner = Scanify(cfg)
res = scanner.process("input/1.jpeg")

print("Output shape:", res.image_bw.shape)
```

Batch example:

```python
from glob import glob
from scanify.scanify import Scanify

scanner = Scanify()
paths = sorted(glob("input/*.jpeg"))
results = scanner.process_batch(paths, out_dir="stage0_debug")
```

### Where outputs go

- The **processed B/W images** are returned in `ScanifyResult.image_bw`.
- If `save_debug=True` and `save_dir` is set, intermediate steps and a montage
  are saved under `stage0_debug/{index_or_stem}/...` (this folder is ignored by git).
