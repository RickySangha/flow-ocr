"""
Simple end-to-end pipeline:

Scanify (stage 0) -> DocSeg (segmentation) -> Chandra OCR (LLM backend).

Usage (from project root):

    python main.py input/1.jpeg

You must have:
- YOLO weights trained for your form type (see docseg_test.py).
- A running Chandra-compatible OpenAI-style API (see chandra_inference_test.py).
"""

import argparse
import os
import time
from typing import Dict

import cv2 as cv

from scanify import Scanify, ScanifyConfig
from docseg import SegmenterInference, DocTypeRegistry, DocTypeConfig
from chandra_ocr.ocr import ChandraOCRPredictor, ChandraBackend, Segment
from utils import logger


DOC_TYPE_NAME = "forms-3cls7"

# Map docseg class labels -> OCR segment types / prompt names
SEGMENT_TYPE_MAP: Dict[str, str] = {
    "header": "patient_info",
    "billing_codes": "msp_code",
    # "notes" stays "notes" and will use a generic free-text prompt (if any)
}


def build_stage0() -> Scanify:
    """Configure Scanify to produce a color-warped page for DocSeg."""
    cfg = ScanifyConfig(
        return_color=True,
        save_every_step=False,
    )
    return Scanify(cfg)


def build_segmenter() -> SegmenterInference:
    """Create a DocSeg segmenter with the trained form weights."""
    registry = DocTypeRegistry(
        {
            DOC_TYPE_NAME: DocTypeConfig(
                weights_path="runs/segment/forms-3cls7/weights/best.pt",
                class_names=["header", "notes", "billing_codes"],
            )
        }
    )
    return SegmenterInference(registry, imgsz=1280, conf=0.05)


def build_ocr() -> ChandraOCRPredictor:
    """Create the Chandra OCR predictor backed by a remote LLM."""
    base_url = os.getenv("CHANDRA_BASE_URL", "http://127.0.0.1:8080/v1")
    api_key = os.getenv("CHANDRA_API_KEY", "your_api_key")
    model = os.getenv("CHANDRA_MODEL", "chandra")

    backend = ChandraBackend(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=1000,
    )
    return ChandraOCRPredictor(
        backend=backend,
    )


def run_pipeline(image_path: str) -> None:
    logger.stage(f"Processing image: {image_path}")
    pipeline_start = time.time()

    # ---------- Stage 0: Scanify ----------
    logger.stage("Stage 0: Scanify (Preprocessing)")
    stage0_start = time.time()

    with logger.timer("Building Scanify processor"):
        stage0 = build_stage0()

    with logger.timer("Running Scanify preprocessing"):
        scan_res = stage0.process(image_path)

    if scan_res.image_color is None:
        logger.error(
            "Scanify did not produce a color image. "
            "Ensure return_color=True (and do_binarize=False if you want un-thresholded color)."
        )
        raise RuntimeError(
            "Scanify did not produce a color image. "
            "Ensure return_color=True (and do_binarize=False if you want un-thresholded color)."
        )

    image_bgr = scan_res.image_color
    stage0_elapsed = time.time() - stage0_start
    logger.time("Stage 0: Scanify", stage0_elapsed)

    # Optional: save the preprocessed image for inspection
    os.makedirs("out", exist_ok=True)
    preproc_path = os.path.join("out", "stage0_preprocessed.png")
    cv.imwrite(preproc_path, image_bgr)
    logger.info(f"Saved preprocessed image to {preproc_path}")

    # ---------- Stage 1: DocSeg ----------
    logger.stage("Stage 1: DocSeg (Segmentation)")
    stage1_start = time.time()

    with logger.timer("Building segmenter"):
        segmenter = build_segmenter()

    with logger.timer("Running document segmentation"):
        seg_res = segmenter.predict(
            image_bgr,
            doc_type=DOC_TYPE_NAME,
            required_labels=("header", "notes"),
            return_crops=True,
            crops_dir="out/crops",
            save_overlay_path="out/overlay.png",
            per_label_nms_iou=0.5,
            max_per_label=1,
        )

    stage1_elapsed = time.time() - stage1_start
    logger.time("Stage 1: DocSeg", stage1_elapsed)

    logger.info(f"Segmentation abstain: {seg_res.abstain}, reasons: {seg_res.reasons}")
    for r in seg_res.regions:
        logger.info(f"  - {r.label} (score: {r.score:.3f}, bbox: {r.bbox})")

    if not seg_res.regions:
        logger.warning("No regions detected; skipping OCR.")
        return

    # ---------- Stage 2: Chandra OCR ----------
    logger.stage("Stage 2: Chandra OCR")
    stage2_start = time.time()

    with logger.timer("Building OCR predictor"):
        ocr = build_ocr()

    with logger.timer("Preparing segments"):
        segments = []
        crops_dir = seg_res.crops_dir or "out/crops"
        for idx, r in enumerate(seg_res.regions):
            seg_type = SEGMENT_TYPE_MAP.get(r.label, r.label)
            crop_path = os.path.join(crops_dir, f"{idx:02d}_{r.label}.png")

            segments.append(
                Segment(
                    segment_id=f"{seg_type}_{idx}",
                    page_id="page_0",
                    segment_type=seg_type,
                    bbox=r.bbox,
                    crop_path=crop_path,
                )
            )
        logger.info(f"Prepared {len(segments)} segments for OCR")

    with logger.timer("Running OCR on segments"):
        ocr_results = ocr.predict(segments, context=None, top_k=3)

    stage2_elapsed = time.time() - stage2_start
    logger.time("Stage 2: Chandra OCR", stage2_elapsed)

    logger.stage("OCR Results")
    for res in ocr_results:
        logger.info(
            f"[{res.segment_type}] {res.normalized_value!r} "
            f"(conf={res.confidence:.2f}, warnings={res.warnings})"
        )

    total_elapsed = time.time() - pipeline_start
    logger.stage(f"Pipeline completed in {total_elapsed:.2f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Scanify -> DocSeg -> Chandra OCR on a single image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="input/1.jpeg",
        help="Path to the input form image (default: input/1.jpeg)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.image)
