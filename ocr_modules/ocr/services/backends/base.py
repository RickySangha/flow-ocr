
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import time

@dataclass
class OCRBatch:
    images: List[Any]
    prompts: List[str]
    systems: List[str]
    meta: List[Dict]

class BaseOCRBackend:
    name: str = "base"
    version: str = "0.0.0"

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def prepare_inputs(self, segments: List[Dict]) -> OCRBatch:
        raise NotImplementedError

    def infer_batch(self, batch: OCRBatch, top_k: int = 3) -> List[Dict]:
        raise NotImplementedError

    def __call__(self, segments: List[Dict], top_k: int = 3) -> List[Dict]:
        t0 = time.time()
        batch = self.prepare_inputs(segments)
        outputs = self.infer_batch(batch, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000.0
        for o in outputs:
            o.setdefault("latency_ms", latency_ms)
        return outputs
