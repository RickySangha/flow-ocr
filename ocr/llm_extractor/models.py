from typing import List, Optional
from pydantic import BaseModel, Field


class Vitals(BaseModel):
    temperature_c: Optional[float] = Field(default=None)
    heart_rate: Optional[int] = Field(default=None)
    bp_systolic: Optional[int] = Field(default=None)
    bp_diastolic: Optional[int] = Field(default=None)
    resp_rate: Optional[int] = Field(default=None)
    spo2: Optional[int] = Field(default=None)
    gcs: Optional[int] = Field(default=None)


class BillingCode(BaseModel):
    code: str
    # description: Optional[str] = None
    # source: str = Field(
    #     description='Where this code came from, e.g. "from_text", "provided_existing", "llm_adjusted".'
    # )
    # confidence: float = Field(
    #     ge=0.0,
    #     le=1.0,
    #     description="Model confidence that this code belongs on the case (0–1).",
    # )


class DiagnosisCode(BaseModel):
    code: str
    # description: Optional[str] = None
    # source: str = Field(
    #     description='Where this code came from, e.g. "from_text", "provided_existing", "llm_adjusted".'
    # )
    # confidence: float = Field(
    #     ge=0.0,
    #     le=1.0,
    #     description="Model confidence that this code belongs on the case (0–1).",
    # )


class Patient(BaseModel):
    full_name: str
    first_name: str
    last_name: str
    dob: Optional[str] = Field(
        default=None,
        description='ISO date string "YYYY-MM-DD" if available.',
    )
    sex: Optional[str] = Field(default=None, description='Sex marker, e.g. "M", "F", "X".')
    phn: Optional[str] = Field(default=None, description="Patient health number if available.")


class Encounter(BaseModel):
    arrival_datetime: Optional[str] = Field(
        default=None,
        description='ISO datetime "YYYY-MM-DDTHH:MM:SS" if available.',
    )
    triage_datetime: Optional[str] = Field(
        default=None,
        description='ISO datetime "YYYY-MM-DDTHH:MM:SS" if available.',
    )
    ctas: Optional[int] = Field(default=None, description="Canadian Triage and Acuity Scale (1–5).")
    location: Optional[str] = None
    visit_type: Optional[str] = None
    mrn: Optional[str] = None


class FamilyDoctor(BaseModel):
    name: Optional[str] = None

class Clinical(BaseModel):
    chief_complaint: Optional[str] = None
    history_of_present_illness: Optional[str] = None
    allergies: Optional[str] = None
    vitals: Vitals = Field(default_factory=Vitals)
    normalized_notes: str = Field(description="Normalize the raw doctors notes. Remove irrelevant info.")
    raw_text: str = Field(
        description="The full OCR text used for extraction (for traceability)."
    )


class CodeReviewIssue(BaseModel):
    type: str = Field(
        description='Issue type, e.g. "missing_code", "potential_mismatch", "duplicate", "unclear_support".'
    )
    severity: str = Field(description='Severity level, e.g. "low", "medium", "high".')
    message: str = Field(description="Human-readable description of the issue.")
    existing_code: Optional[str] = Field(default=None)
    suggested_code: Optional[str] = Field(default=None)
    rationale: str = Field(description="Short explanation of why this is an issue.")


class CodeSet(BaseModel):
    billing: List[BillingCode] = Field(default_factory=list)
    diagnosis: List[DiagnosisCode] = Field(default_factory=list)


class CodeReview(BaseModel):
    issues: List[CodeReviewIssue] = Field(default_factory=list)
    final_recommended_code_set: CodeSet = Field(
        default_factory=CodeSet,
        description='Final recommended codes.',
    )
    notes: str = Field(
        default="",
        description="Brief summary of reasoning behind the final recommended code set.",
    )


class Document(BaseModel):
    """End-to-end document schema.

    Step 1 (extraction):
        - Fills everything except `code_review`.
        - Codes are simply extracted/normalized; no guideline reasoning yet.

    Step 2 (code review):
        - Starts from a Document instance.
        - Updates `codes` and populates `code_review`.

    This shared schema makes it easy to swap in a future RAG-based agent for
    Step 2 without changing your pipeline wiring.
    """

    document_id: str
    patient: Patient
    family_doctor: FamilyDoctor
    encounter: Encounter
    clinical: Clinical
    codes: CodeSet = Field(
        default_factory=CodeSet,
        description='Extracted billing and diagnosis codes.',
    )
    code_review: Optional[CodeReview] = Field(
        default=None,
        description="Populated in Step 2 after code review. Optional for Step 1.",
    )
