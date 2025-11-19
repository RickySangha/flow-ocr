from chandra_ocr.ocr import ChandraOCRPredictor
from chandra_ocr.ocr.backends import ChandraBackend
from chandra_ocr.ocr.types import Segment, DecodeMode

backend = ChandraBackend(
    base_url="http://127.0.0.1:8080/v1",
    api_key="your_api_key",
    model="chandra",
    max_tokens=1000,
)
predictor = ChandraOCRPredictor(
    backend=backend,
)

segments = [
    Segment(
        segment_id="s1",
        page_id="p1",
        segment_type="msp_code",
        bbox=(0, 0, 100, 40),
        crop_path="out/crops/00_billing_codes.png",
    ),
]

ctx = None
res = predictor.predict(segments, context=ctx, top_k=3)
for r in res:
    print(r)
