
from typing import Dict, Any, List, Optional
import os, json, pathlib, sys
import numpy as np
import cv2 as cv

def _rle_to_mask(rle: Dict[str, Any], h: int, w: int) -> np.ndarray:
    if isinstance(rle.get("counts"), list):
        counts = rle["counts"]
        mask = np.zeros(h*w, dtype=np.uint8)
        val = 0; idx = 0
        for c in counts:
            if idx + c > h*w: c = max(0, h*w - idx)
            if val == 1: mask[idx:idx+c] = 1
            idx += c; val ^= 1
        return mask.reshape((h,w), order="F")
    raise RuntimeError("Compressed RLE encountered (counts is string). Export uncompressed or use pycocotools.")

def _write_line(fh, cls_id: int, pts: np.ndarray, w: int, h: int):
    parts = [str(cls_id)]
    for (x,y) in pts:
        parts += [f\"{x/float(w):.6f}\", f\"{y/float(h):.6f}\"]
    fh.write(\" \".join(parts) + \"\\n\")

def _biggest_contour(mask: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv.contourArea)
    if cv.contourArea(cnt) < min_area: return None
    cnt = cnt.squeeze(1)
    eps = 0.002 * cv.arcLength(cnt, True)
    cnt = cv.approxPolyDP(cnt, eps, True).squeeze(1)
    return cnt

def convert_coco_to_yolo(
    coco_json_path: str,
    images_root: str,
    out_labels_root: str,
    class_order: List[str],
    split_by_image_subdirs: bool = False,
    min_area: float = 50.0
) -> int:
    with open(coco_json_path, \"r\") as f:
        coco = json.load(f)

    name2idx = {n.strip().lower(): i for i, n in enumerate(class_order)}
    cat_mapping = {}
    for cat in coco.get(\"categories\", []):
        nm = cat.get(\"name\",\"\" ).strip().lower()
        if nm in name2idx:
            cat_mapping[int(cat[\"id\"])] = int(name2idx[nm])

    id2img = {int(img[\"id\"]): img for img in coco.get(\"images\",[])}
    anns_by_img = {}
    for ann in coco.get(\"annotations\", []):
        anns_by_img.setdefault(int(ann[\"image_id\"]), []).append(ann)

    total_lines = 0
    os.makedirs(out_labels_root, exist_ok=True)

    for img_id, img in id2img.items():
        file_name = img[\"file_name\"]; w, h = int(img[\"width\"]), int(img[\"height\"])
        anns = anns_by_img.get(img_id, [])

        out_dir = out_labels_root
        if split_by_image_subdirs:
            parts = pathlib.Path(file_name).parts
            if \"train\" in parts: out_dir = os.path.join(out_dir, \"train\")
            elif \"val\" in parts or \"valid\" in parts: out_dir = os.path.join(out_dir, \"val\")
            os.makedirs(out_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(file_name))[0]
        out_path = os.path.join(out_dir, stem + \".txt\")
        with open(out_path, \"w\") as fh:
            for ann in anns:
                cat_id = int(ann[\"category_id\"])
                if cat_id not in cat_mapping: continue
                cls_id = cat_mapping[cat_id]
                seg = ann.get(\"segmentation\")
                if seg is None: continue
                if isinstance(seg, list):
                    for poly in seg:
                        pts = np.asarray(poly, dtype=float).reshape(-1,2)
                        if pts.shape[0] < 3: continue
                        area = cv.contourArea(pts.astype(np.float32))
                        if area < min_area: continue
                        _write_line(fh, cls_id, pts, w, h); total_lines += 1
                elif isinstance(seg, dict) and \"counts\" in seg and \"size\" in seg:
                    try:
                        mask = _rle_to_mask(seg, h, w)
                    except Exception as e:
                        print(f\"[WARN] {file_name}: {e}\", file=sys.stderr); continue
                    cnt = _biggest_contour(mask, min_area)
                    if cnt is None: continue
                    _write_line(fh, cls_id, cnt, w, h); total_lines += 1

        if not os.path.exists(out_path):
            open(out_path, \"w\").close()

    return total_lines
