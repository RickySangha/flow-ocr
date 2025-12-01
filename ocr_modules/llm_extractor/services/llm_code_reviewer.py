"""
LLMCodeReviewer - Step 2: code review and recommendation (RAG-ready).

Takes a `Document` from LLMExtractor (Step 1) plus small billing/diagnosis code DBs,
and returns an updated `Document` with:
- `codes` refined (descriptions & codes adjusted)
- `code_review` populated

In the future, you can replace the internals of this class with an agent + RAG
while keeping the same function signature and Document schema.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from openai import OpenAI

from ..config.prompts import CODE_REVIEW_SYSTEM_PROMPT
from ..models import Document


class LLMCodeReviewer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set and no api_key was provided.")

        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt or CODE_REVIEW_SYSTEM_PROMPT
        self.temperature = temperature

    def review(
        self,
        document: Document,
        billing_code_db: Dict[str, str],
        diagnosis_code_db: Dict[str, str],
    ) -> Document:
        """Run the code review model on an extracted Document.

        Args:
            document: Structured document from LLMExtractor (Step 1).
            billing_code_db: Dict of {code: description} for billing codes.
            diagnosis_code_db: Dict of {code: description} for diagnosis codes.

        Returns:
            Updated `Document` with `codes` and `code_review` populated.
        """
        payload = {
            "billing_code_db": billing_code_db,
            "diagnosis_code_db": diagnosis_code_db,
            "document": document.dict(),
        }

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": payload},
            ],
            response_format=Document,
            temperature=self.temperature,
        )

        return response.output_parsed
