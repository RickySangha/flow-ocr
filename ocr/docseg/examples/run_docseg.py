import os
import cv2 as cv
from ocr.docseg import SegmenterInference, DocTypeRegistry, DocTypeConfig

# Registry for your form type (class order must match dataset.yaml)
registry = DocTypeRegistry(
    {
        "forms-3cls7": DocTypeConfig(
            weights_path="runs/segment/forms-3cls7/weights/best.pt",
            class_names=["header", "notes", "billing_codes"],
        )
    }
)

seg = SegmenterInference(registry, imgsz=1280, conf=0.05)

# Preprocessed image from Stage-0
img = cv.imread("input/2.jpeg")

res = seg.predict(
    img,
    doc_type="forms-3cls7",
    required_labels=("header", "notes"),
    include_labels=None,
    per_label_nms_iou=0.5,
    max_per_label=1,
    # Paired prediction artifacts:
    save_pred_base_dir="pred",  # root folder -> pred/
    save_pred_split="train",  # -> pred/image/train & pred/label/train
    save_pred_labels_stem="form_0001",  # pred/image/train/form_0001.png + pred/label/train/form_0001.txt
)

print("abstain:", res.abstain, "reasons:", res.reasons)
print("overlay:", res.overlay_path)  # pred/image/train/form_0001.png
print("labels dir:", res.labels_dir)  # pred/label/train
for r in res.regions:
    print(r.label, r.score, r.bbox)
