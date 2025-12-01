"""
LLMExtractor - Step 1: structured data extraction from OCR text.

Uses OpenAI's `responses.parse` API to directly return a Pydantic `Document`
instance with type-checked parsed output.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

from ..config.prompts import EXTRACTION_SYSTEM_PROMPT
from ..models import Document
import dotenv

dotenv.load_dotenv()


class LLMExtractor:
    """Wraps a small LLM (e.g. gpt-4.1-mini) to convert noisy OCR text into a `Document`.

    This step is strictly about extraction/normalization. It does *not* do
    guideline-based code reasoning. That is reserved for Step 2.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano-2025-08-07",
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set and no api_key was provided.")

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or EXTRACTION_SYSTEM_PROMPT
        self.temperature = temperature

    def extract(
        self,
        document_id: str,
        ocr_text: str,
        existing_billing: Optional[List[str]] = None,
        existing_diag: Optional[List[str]] = None,
        billing_code_db: Optional[Dict[str, str]] = None,
        diagnosis_code_db: Optional[Dict[str, str]] = None,
    ) -> Document:
        """Run the extraction model on a block of OCR text.

        Args:
            document_id: Unique identifier for this document.
            ocr_text: Raw OCR text from your pipeline (can be noisy).
            existing_billing: Optional list of billing codes from earlier stages.
            existing_diag: Optional list of diagnosis codes from earlier stages.
            billing_code_db: Optional dict of {code: description} for billing codes.
            diagnosis_code_db: Optional dict of {code: description} for diagnosis codes.

        Returns:
            A `Document` instance following the Pydantic schema in models.py.
        """
        existing_billing = existing_billing or []
        existing_diag = existing_diag or []
        billing_code_db = billing_code_db or {}
        diagnosis_code_db = diagnosis_code_db or {}

        user_payload = {
            "document_id": document_id,
            "ocr_text": ocr_text,
            "existing_codes": {
                "billing": existing_billing,
                "diagnosis": existing_diag,
            },
            "billing_code_db": billing_code_db,
            "diagnosis_code_db": diagnosis_code_db,
        }

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            text_format=Document,
            reasoning={ "effort": "low" },
        )

        return response.output_parsed
