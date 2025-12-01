"""
Example: run extraction + code review end to end using structured responses.

Usage:

    pip install openai pydantic
    export OPENAI_API_KEY="sk-..."
    python -m llm_extractor.examples.run_extraction_and_review_structured
"""

from __future__ import annotations

import os
from pathlib import Path

from ocr_modules.llm_extractor import LLMExtractor, LLMCodeReviewer, Document


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Please set OPENAI_API_KEY before running this example.")

    ocr_text = (
"""
notes: 'PRI. DATE/TIME: 23 Oct 2025 1912 CTAS: 3 ALERTS: CCI: Temp 37.4 HR 100 BP 122/75 RR 16 O2 Sat 100 GCS 15 Glucometer Weight-Kg Location in Dept. PHYSICIAN NOTES: Allergies: No Known Drug Allergies Sore throat x 3 days. earing mcdonalds in SP. cough pneumonia. 015 min was Cx: familiar hypertrophy, w/o. 27 mm dia. C/S 31-52 yrs year. Rev: 2000s, cc: recurrent. cold cough. 1823 1822 2270 27 mm 2230 DISCHARGE TIME: REFERRALS Fax Chart'

billing_codes: '1823, 01822, 2270'

patient_info: 'SIDHU, SUNEHA KAUR ARRIVAL: 23oct2025 TIME: 1910 REG.CAT: P.EM (N) (604)591-8008 ADM: 23oct2025 TIME: 1916 LOCATION: PA.E DOB: 23Jun2007 F 18 GP: Amir, Mohammad F NOTIFY: SIDHU, HARDEEP REL: MOTHER PHONE: (778)896-2478 REASON FOR VISIT: SORE THROAT CHIEF COMPLAINT: Difficulty swallowing/dysphag COMMENT: OTHER NAME: ADDRESS: 6318 128 ST SURREY BC V3X 188 PHONE: (778)791-8724 OTHER: PHN: 9804325494 INS: MSP 9804325494 ACCIDENT INFO: TRI. DATE/TIME: 23oct2025 1912 | CTAS: 3 | ALERTS: | CCI: Temp HR BP RR O2 Sat GCS Glucometer Weight-Kg Location in D 37.4 100 122/75 16 100 15 PHYSICIAN NOTES allergies: V PHYSICIAN ORDERS Time:'
"""
    )

    billing_code_db = {
        "01823": "Emergency department reassessment",
        "01822": "Emergency visit, level 2",
    }
    diagnosis_code_db = {
        "J02.9": "Acute pharyngitis, unspecified",
        "J06.9": "Acute upper respiratory infection, unspecified",
    }

    extractor = LLMExtractor()
    # reviewer = LLMCodeReviewer()

    doc: Document = extractor.extract(
        document_id="example-doc-1",
        ocr_text=ocr_text,
        existing_billing=["1823", "01822", "2270"],
        existing_diag=[],
        billing_code_db=billing_code_db,
        diagnosis_code_db=diagnosis_code_db,
    )

    print(doc)

    # reviewed_doc: Document = reviewer.review(
    #     document=doc,
    #     billing_code_db=billing_code_db,
    #     diagnosis_code_db=diagnosis_code_db,
    # )

    # out_path = Path("extracted_and_reviewed_example_structured.json")
    # out_path.write_text(reviewed_doc.model_dump_json(indent=2))
    # print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
