"""
Simple end-to-end pipeline:

Scanify (stage 0) -> DocSeg (segmentation) -> Chandra OCR (LLM backend) -> LLM Extractor (Structured Data).

Usage (from project root):

    python main.py input/1.jpeg

You must have:
- YOLO weights trained for your form type (see docseg_test.py).
- A running Chandra-compatible OpenAI-style API (see chandra_inference_test.py).
- OPENAI_API_KEY set for the final extraction step.
"""

import argparse
import os
import time
from typing import Dict, List

import cv2 as cv
import numpy as np

from ocr_modules.scanify import Scanify, ScanifyConfig
from ocr_modules.docseg import SegmenterInference, DocTypeRegistry, DocTypeConfig
from ocr_modules.ocr import ChandraOCRPredictor, ChandraBackend, Segment
from ocr_modules.llm_extractor import LLMExtractor, Document
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
        save_debug=False,
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
    return SegmenterInference(registry, imgsz=1280, conf=0.01)


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


def build_extractor() -> LLMExtractor:
    """Create the LLM Extractor for structured data."""
    return LLMExtractor()


def _crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> bytes:
    """Crop image and return PNG bytes."""
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    success, encoded_image = cv.imencode(".png", crop)
    if not success:
        raise ValueError("Failed to encode cropped image")
    return encoded_image.tobytes()


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
            return_crops=False,  # Disable saving crops to disk
            save_overlay_path=None, # Disable saving overlay
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
        for idx, r in enumerate(seg_res.regions):
            seg_type = SEGMENT_TYPE_MAP.get(r.label, r.label)
            
            # Crop in memory
            image_bytes = _crop_image(image_bgr, r.bbox)

            segments.append(
                Segment(
                    segment_id=f"{seg_type}_{idx}",
                    page_id="page_0",
                    segment_type=seg_type,
                    bbox=r.bbox,
                    image_bytes=image_bytes,
                )
            )
        logger.info(f"Prepared {len(segments)} segments for OCR")

    with logger.timer("Running OCR on segments"):
        ocr_results = ocr.predict(segments, context=None, top_k=3)

    stage2_elapsed = time.time() - stage2_start
    logger.time("Stage 2: Chandra OCR", stage2_elapsed)

    # ---------- Stage 3: LLM Extraction ----------
    logger.stage("Stage 3: LLM Extraction")
    stage3_start = time.time()

    # Construct OCR text block
    ocr_text_parts = []
    for res in ocr_results:
        text = res.normalized_value or res.raw_text
        ocr_text_parts.append(f"{res.segment_type}: '{text}'")
    
    full_ocr_text = "\n\n".join(ocr_text_parts)
    logger.info("Constructed OCR Text for Extraction:")
    print(full_ocr_text)

    with logger.timer("Building LLM Extractor"):
        extractor = build_extractor()

    with logger.timer("Running Extraction"):
        doc: Document = extractor.extract(
            document_id=os.path.basename(image_path),
            ocr_text=full_ocr_text,
        )

    stage3_elapsed = time.time() - stage3_start
    logger.time("Stage 3: LLM Extraction", stage3_elapsed)

    total_elapsed = time.time() - pipeline_start
    logger.stage(f"Pipeline completed in {total_elapsed:.2f}s")

    print("\n" + "="*40)
    print("FINAL EXTRACTED DOCUMENT")
    print("="*40)
    print(doc.model_dump_json(indent=2))
    print("="*40 + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Scanify -> DocSeg -> Chandra OCR -> LLM Extraction on a single image."
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
