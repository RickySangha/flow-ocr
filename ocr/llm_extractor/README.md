# flow_ocr_llm (structured responses)

Self-contained Python package for LLM-based OCR -> billing extraction and review,
using OpenAI's `responses.parse` + Pydantic models.

Folder layout:

- `models.py`         : Pydantic models (Document, Patient, Encounter, Clinical, etc.)
- `config/prompts.py` : System prompts for extraction + code review.
- `services/`         : LLMExtractor (Step 1) and LLMCodeReviewer (Step 2).
- `examples/`         : Runnable demo script.
