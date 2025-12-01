
from __future__ import annotations
from typing import Dict, List
from .base import BaseOCRBackend, OCRBatch

class TesseractBackend(BaseOCRBackend):
    name: str = "tesseract"
    version: str = "0.1.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_inputs(self, segments: List[Dict]) -> OCRBatch:
        images, prompts, systems, meta = [], [], [], []
        for s in segments:
            images.append(s['image'])
            prompts.append(s.get('prompt', ''))
            systems.append(s.get('system', ''))
            meta.append(s.get('meta', {}))
        return OCRBatch(images=images, prompts=prompts, systems=systems, meta=meta)

    def infer_batch(self, batch: OCRBatch, top_k: int = 3) -> List[Dict]:
        outputs: List[Dict] = []
        for img, prompt, meta in zip(batch.images, batch.prompts, batch.meta):
            txt = "TESSERACT_DUMMY"
            conf = 0.6
            outputs.append({
                "text": txt,
                "confidence": conf,
                "candidates": [(txt, conf)],
                "meta": meta,
            })
        return outputs
