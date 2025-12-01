from __future__ import annotations
import json, os
from typing import Any, Dict, List, Optional
from ..models import Segment, OCRResult, Candidate
from .validators import validate_msp_code, validate_phn, always_true
from .normalizers import normalize_msp_code, normalize_whitespace
from .backends.base import BaseOCRBackend
from utils import logger

# from .utils import best_fuzzy_match

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


class PromptRegistry:
    def __init__(self, root_dir: str):
        # Resolve relative root_dir against this module's directory to avoid CWD issues
        base_dir = os.path.dirname(__file__)
        if not os.path.isabs(root_dir):
            root_dir = os.path.normpath(os.path.join(base_dir, root_dir))
        self.root = root_dir

    def load_for(
        self, segment_type: str, dynamic: Optional[Dict[str, Any]] = None
    ) -> str:
        path = os.path.join(self.root, f"{segment_type}.yaml")
        logger.info(f"Loading prompt for {segment_type} from {path}")
        if not os.path.exists(path):
            return ""
        if yaml is None:
            with open(path, "r") as f:
                return f.read()
        data = yaml.safe_load(open(path, "r"))
        template: str = data.get("template", "")
        dynamic = dynamic or {}
        for k, v in dynamic.items():
            if isinstance(v, (list, dict)):
                v = json.dumps(v)
            template = template.replace(f"{{{{{k}}}}}", str(v))
        return template

    def load_system_for(self, segment_type: str) -> str:
        path = os.path.join(self.root, f"{segment_type}.yaml")
        if not os.path.exists(path):
            return ""
        if yaml is None:
            return ""
        data = yaml.safe_load(open(path, "r"))
        return data.get("system", "")


class ConstraintStore:
    def __init__(self, root_dir: str):
        base_dir = os.path.dirname(__file__)
        if not os.path.isabs(root_dir):
            root_dir = os.path.normpath(os.path.join(base_dir, root_dir))
        self.root = root_dir

    def list_for(self, segment_type: str) -> List[str]:
        path = os.path.join(self.root, f"{segment_type}.json")
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            data = json.load(f)
        items = data.get("items", [])
        aliases = data.get("aliases", {})
        all_items = list(dict.fromkeys(items + list(aliases.keys())))
        return all_items


class ChandraOCRPredictor:
    def __init__(
        self,
        backend: BaseOCRBackend,
        prompt_dir: str = "../config/prompts",
        constraint_dir: str = "constraints",
        min_confidence: float = 0.6,
    ):
        self.backend = backend
        self.prompts = PromptRegistry(prompt_dir)
        self.constraints = ConstraintStore(constraint_dir)
        self.min_conf = min_confidence

    def _validator_for(self, segment_type: str):
        st = segment_type.lower()
        if st == "msp_code":
            return validate_msp_code, normalize_msp_code
        elif st in ("phn", "health_number"):
            return validate_phn, normalize_whitespace
        else:
            return always_true, normalize_whitespace

    def predict(
        self,
        segments: List[Segment],
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[OCRResult]:
        inputs = []
        for s in segments:
            img = s.image_bytes if s.image_bytes is not None else s.crop_path
            dynamic_ctx = (context or {}).get(s.segment_type, {})
            prompt = self.prompts.load_for(s.segment_type, dynamic=dynamic_ctx)
            inputs.append(
                {
                    "image": img,
                    "prompt": prompt,
                    "system": self.prompts.load_system_for(s.segment_type),
                    "meta": {
                        "segment_id": s.segment_id,
                        "bbox": s.bbox,
                        "segment_type": s.segment_type,
                    },
                }
            )
            logger.info(f"Running OCR for segment {s.segment_type} ({s.segment_id})")

        raw_outputs = self.backend(inputs, top_k=top_k)

        results: List[OCRResult] = []
        for inp, out in zip(inputs, raw_outputs):
            seg_meta = out.get("meta", {})
            segment_type = seg_meta.get("segment_type", "")
            validator, normalizer = self._validator_for(segment_type)

            raw_text = out.get("text", "").strip()
            conf = float(out.get("confidence", 0.0))
            candidates = [
                Candidate(text=c[0], score=float(c[1]))
                for c in out.get("candidates", [])
            ]
            warnings, errors = [], []

            normalized = normalizer(raw_text)

            if not validator(normalized):
                warnings.append("Validation failed for normalized value")

            results.append(
                OCRResult(
                    segment_id=seg_meta.get("segment_id", ""),
                    segment_type=segment_type,
                    raw_text=raw_text,
                    normalized_value=normalized,
                    confidence=conf,
                    candidates=candidates[:top_k],
                    bbox=seg_meta.get("bbox", (0, 0, 0, 0)),
                    latency_ms=float(out.get("latency_ms", 0.0)),
                    model_version=f"{self.backend.name}:{getattr(self.backend, 'version', 'unknown')}",
                    warnings=warnings,
                    errors=errors,
                )
            )
        return results
