"""
flow_ocr_llm

LLM-based utilities for your OCR -> billing pipeline, using OpenAI's structured
response API (`responses.parse`) and Pydantic models.

Main entrypoints:

    from flow_ocr_llm import (
        LLMExtractor,
        LLMCodeReviewer,
        Document,
        CodeReview,
        BillingCode,
        DiagnosisCode,
    )
"""
from .models import (
    Vitals,
    BillingCode,
    DiagnosisCode,
    Patient,
    Encounter,
    Clinical,
    CodeReviewIssue,
    CodeReview,
    Document,
    FamilyDoctor,
    CodeSet,
)
from .services.llm_extractor import LLMExtractor
from .services.llm_code_reviewer import LLMCodeReviewer

__all__ = [
    "Vitals",
    "BillingCode",
    "DiagnosisCode",
    "Patient",
    "Encounter",
    "Clinical",
    "CodeReviewIssue",
    "CodeReview",
    "Document",
    "FamilyDoctor",
    "CodeSet",
    "LLMExtractor",
    "LLMCodeReviewer",
]
