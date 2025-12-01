from __future__ import annotations
import io, base64, json
from typing import Any, Dict, List, Optional
from .base import BaseOCRBackend, OCRBatch

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

from PIL import Image
import logging

logger = logging.getLogger(__name__)


def _to_png_bytes(img) -> bytes:
    if isinstance(img, (bytes, bytearray)):
        try:
            # Validate it's an image, re-encode to PNG for consistency
            pil = Image.open(io.BytesIO(img)).convert("RGB")
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            # If already some other bytes, just return
            return bytes(img)
    elif isinstance(img, str):
        pil = Image.open(img).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    else:
        # Assume PIL Image or numpy array-like that PIL can handle
        if not isinstance(img, Image.Image):
            pil = Image.fromarray(img)  # type: ignore
        else:
            pil = img
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()


def _to_data_uri(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        # remove first and last fence line
        lines = s.splitlines()
        if len(lines) >= 2:
            # drop first and last lines
            core = "\n".join(lines[1:-1]).strip()
            return core
    return s


def _normalize_output_text(content: Optional[str]) -> str:
    if not content:
        return ""
    s = _strip_code_fences(content).strip()
    # try JSON with common keys
    try:
        obj = None
        # Some models may wrap JSON in backticks or have trailing commas
        obj = json.loads(s)
        if isinstance(obj, dict):
            for k in ["text", "value", "output", "answer"]:
                if k in obj and isinstance(obj[k], str):
                    return obj[k].strip()
        if isinstance(obj, list) and obj and isinstance(obj[0], str):
            return obj[0].strip()
    except Exception:
        pass
    # fallback: return raw string
    return s


class ChandraBackend(BaseOCRBackend):
    """
    Calls a remote OpenAI-compatible vision chat endpoint to perform OCR
    using the provided system + user prompts and a single image per item.
    """

    name: str = "chandra"
    version: str = "0.2.0"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
            or "You are a precise OCR assistant. Return only the requested field text.",
            **kwargs,
        )
        if OpenAI is None:
            raise RuntimeError(
                "openai package is required for ChandraBackend. pip install openai"
            )
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._remote_model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = self.cfg["system_prompt"]

    def prepare_inputs(self, segments: List[Dict]) -> OCRBatch:
        images, prompts, systems, meta = [], [], [], []
        for s in segments:
            img = s["image"]
            png_bytes = _to_png_bytes(img)
            data_uri = _to_data_uri(png_bytes)
            images.append(data_uri)
            prompts.append(s.get("prompt", ""))
            systems.append(s.get("system", ""))
            meta.append(s.get("meta", {}))
        return OCRBatch(images=images, prompts=prompts, systems=systems, meta=meta)

    def infer_batch(self, batch: OCRBatch, top_k: int = 3) -> List[Dict]:
        outputs: List[Dict] = []
        # Remote API typically processes one image per request; do simple loop
        for data_uri, prompt, system_override, meta in zip(
            batch.images, batch.prompts, batch.systems, batch.meta
        ):
            user_instr = (
                prompt or "Read the field text from the image. Return only the text."
            )
            try:
                out = self._client.chat.completions.create(
                    model=self._remote_model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": (system_override or self._system_prompt),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_instr},
                                {"type": "image_url", "image_url": {"url": data_uri}},
                            ],
                        },
                    ],
                )
                raw = (out.choices[0].message.content or "").strip()
                text = _normalize_output_text(raw)
                conf = 0.0  # remote API doesn't return calibrated probs; leave 0 and let predictor calibrate/snapping
                outputs.append(
                    {
                        "text": text,
                        "confidence": conf,
                        "candidates": [(text, conf)],
                        "meta": meta,
                    }
                )
            except Exception as e:
                logger.error(f"Error inferring batch: {e}")
                outputs.append(
                    {
                        "text": "",
                        "confidence": 0.0,
                        "candidates": [],
                        "meta": meta,
                        "error": str(e),
                    }
                )
        return outputs
