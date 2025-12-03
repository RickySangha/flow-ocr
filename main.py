import os
import uuid
import shutil
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from google.cloud import storage
import cv2
import numpy as np

# --- Module Imports ---
from ocr_modules.scanify import Scanify, ScanifyConfig
from ocr_modules.docseg import SegmenterInference, DocTypeRegistry, DocTypeConfig
from ocr_modules.ocr import ChandraOCRPredictor, ChandraBackend, Segment
# from ocr_modules.llm_extractor import LLMExtractor, Document # Assuming this exists or will be mocked for now
from utils import logger

# --- Configuration ---
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "flow-ocr-bucket") # Replace with actual default or env var
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# --- Global Models ---
models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models on startup and clean up on shutdown.
    """
    logger.info("Loading models...")
    
    # 1. Scanify (CPU-bound, lightweight init)
    scanify_config = ScanifyConfig(
        return_color=True,
        do_binarize=False, # We might want binarized for OCR, but color for display/debug
        return_bw=True     # Get grayscale/bw for OCR if needed
    )
    models["scanify"] = Scanify(config=scanify_config)
    logger.info("Scanify loaded.")

    # 2. DocSeg (GPU-bound)
    # Ensure registry is populated. Assuming config files are in a standard location.
    # You might need to point this to the correct config directory.
    registry = DocTypeRegistry() 
    # For now, we might need to manually register if not auto-loaded, 
    # but assuming DocTypeRegistry handles defaults or we load from a known path.
    # If specific weights are needed:
    # registry.register("medical_form", "path/to/weights.pt", ["header", "field"])
    
    # Initialize Segmenter
    # Note: SegmenterInference checks for ultralytics and loads YOLO.
    # It lazy-loads models in `predict` usually, but we can pre-warm if we know the doc types.
    models["docseg"] = SegmenterInference(registry=registry)
    logger.info("DocSeg initialized.")

    # 3. OCR (GPU-bound)
    # Initialize backend (e.g., Chandra/Paddle/Tesseract)
    # Using ChandraBackend as per import
    ocr_backend = ChandraBackend(use_gpu=True) # Ensure GPU is enabled
    models["ocr"] = ChandraOCRPredictor(backend=ocr_backend)
    logger.info("OCR initialized.")

    # 4. LLM Extractor (Optional/API-based)
    # models["llm"] = LLMExtractor(...)
    
    yield
    
    logger.info("Shutting down and cleaning up...")
    models.clear()

app = FastAPI(lifespan=lifespan)
storage_client = storage.Client()

# --- Data Models ---

class OcrRequest(BaseModel):
    gcs_path: str        # e.g. "uploads/doc_123/page_1.png"
    job_id: str
    doc_type: str = "generic" # To select DocSeg model
    extra_metadata: Optional[Dict[str, Any]] = None

class OcrResponse(BaseModel):
    job_id: str
    status: str
    results: Dict[str, Any]
    latency_ms: float

# --- Helper Functions ---

def download_from_gcs(gcs_path: str, local_dest: str):
    """Downloads a file from GCS to a local destination."""
    try:
        # If gcs_path starts with gs://, parse it
        if gcs_path.startswith("gs://"):
            parts = gcs_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1]
            bucket = storage_client.bucket(bucket_name)
        else:
            bucket = storage_client.bucket(BUCKET_NAME)
            blob_name = gcs_path
            
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_dest)
        logger.info(f"Downloaded {gcs_path} to {local_dest}")
    except Exception as e:
        logger.error(f"Failed to download {gcs_path}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Failed to cleanup {path}: {e}")

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.post("/ocr", response_model=OcrResponse)
async def ocr_endpoint(req: OcrRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    job_id = req.job_id
    
    # Create temp file path
    tmp_filename = f"{job_id}_{uuid.uuid4()}.png"
    local_path = os.path.join("/tmp", tmp_filename)
    
    try:
        # 1. Download
        download_from_gcs(req.gcs_path, local_path)
        
        # 2. Scanify (Preprocessing)
        logger.info(f"[{job_id}] Starting Scanify...")
        scanify: Scanify = models["scanify"]
        # process returns ScanifyResult(image_bw, image_color, debug)
        scan_result = scanify.process(local_path)
        
        # Use color image for segmentation (YOLO usually trained on color)
        # Use BW image for OCR if needed, or color depending on OCR backend.
        # Chandra/Paddle often works fine with color.
        processed_img = scan_result.image_color if scan_result.image_color is not None else scan_result.image_bw
        
        if processed_img is None:
             raise HTTPException(status_code=500, detail="Scanify failed to produce an image")

        # 3. DocSeg (Segmentation)
        logger.info(f"[{job_id}] Starting DocSeg...")
        docseg: SegmenterInference = models["docseg"]
        
        # Predict regions
        # Note: docseg.predict expects BGR image (OpenCV format), which Scanify returns.
        seg_result = docseg.predict(
            image_bgr=processed_img,
            doc_type=req.doc_type,
            required_labels=None # Don't fail if specific labels missing, just return what's found
        )
        
        if seg_result.abstain:
             logger.warning(f"[{job_id}] DocSeg abstained: {seg_result.reasons}")
             # Proceed with empty regions or fail? Let's proceed but note it.
        
        # 4. OCR (Text Recognition)
        logger.info(f"[{job_id}] Starting OCR...")
        ocr_predictor: ChandraOCRPredictor = models["ocr"]
        
        segments_to_ocr = []
        for region in seg_result.regions:
            # Crop the region from the processed image
            x1, y1, x2, y2 = region.bbox
            crop = processed_img[y1:y2, x1:x2]
            
            # Create Segment object
            seg = Segment(
                segment_id=f"{region.label}_{uuid.uuid4().hex[:6]}",
                page_id="page_1",
                segment_type=region.label,
                crop_path=None, # We pass bytes/array directly if supported, or need to save?
                # ChandraOCRPredictor expects `image_bytes` or `crop_path`. 
                # Let's encode to bytes for simplicity if backend supports it, 
                # or modify Predictor to accept numpy arrays.
                # Looking at predictor.py: `img = s.image_bytes if s.image_bytes is not None else s.crop_path`
                # And backend usually takes list of dicts with 'image'.
                # If backend handles numpy, great. If not, encode.
                # Assuming backend handles numpy or we encode. Let's encode to be safe/standard.
                image_bytes=cv2.imencode('.png', crop)[1].tobytes(),
                bbox=region.bbox
            )
            segments_to_ocr.append(seg)
            
        ocr_results = []
        if segments_to_ocr:
            ocr_results = ocr_predictor.predict(segments_to_ocr)
        
        # Format results
        formatted_results = {
            "regions": [
                {
                    "label": r.label,
                    "bbox": r.bbox,
                    "score": r.score
                } for r in seg_result.regions
            ],
            "ocr": [
                {
                    "label": res.segment_type,
                    "text": res.raw_text,
                    "confidence": res.confidence,
                    "normalized": res.normalized_value
                } for res in ocr_results
            ]
        }

        # 5. LLM Extraction (Optional - Placeholder)
        # if models.get("llm"):
        #     formatted_results["structured"] = models["llm"].extract(formatted_results["ocr"])

        latency = (time.time() - start_time) * 1000
        logger.info(f"[{job_id}] Completed in {latency:.2f}ms")
        
        return OcrResponse(
            job_id=job_id,
            status="success",
            results=formatted_results,
            latency_ms=latency
        )

    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        background_tasks.add_task(cleanup_file, local_path)