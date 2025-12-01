from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Segment:
    segment_id: str
    page_id: str
    segment_type: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    crop_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    text: str
    score: float


@dataclass
class OCRResult:
    segment_id: str
    segment_type: str
    raw_text: str
    normalized_value: Optional[str]
    confidence: float
    candidates: List[Candidate]
    bbox: Tuple[int, int, int, int]
    latency_ms: float
    model_version: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
