from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence, Tuple
import os
import json
import numpy as np
import cv2 as cv

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from ..config.config import DocTypeRegistry


@dataclass
class Region:
    label: str
    score: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask_rle: Optional[str] = None


@dataclass
class InferenceResult:
    regions: List[Region]
    abstain: bool
    reasons: List[str]
    overlay_path: Optional[str] = None
    crops_dir: Optional[str] = None
    labels_dir: Optional[str] = None


class SegmenterInference:
    def __init__(
        self,
        registry: DocTypeRegistry,
        imgsz: int = 1024,
        conf: float = 0.30,
    ):
        """
        Initialize a reusable document segmenter (YOLO-seg based).

        This class loads and caches YOLO segmentation models per **document type** so
        you can run fast, repeated inference across multiple fixed-layout forms.
        It expects **preprocessed** images (Stage-0 already applied) and returns
        per-region boxes/masks plus convenience outputs (overlay, crops).

        Args:
            registry: A DocTypeRegistry mapping `doc_type -> (weights_path, class_names)`.
            imgsz:    Inference image size passed to YOLO (e.g., 1280 for sharper masks).
            conf:     Confidence threshold for YOLO predictions. With few-shot data
                      you may temporarily run lower (e.g., 0.01–0.10) and rely on
                      per-label NMS to keep only sensible regions.

        Raises:
            RuntimeError: If `ultralytics` is not installed.
        """
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. `pip install ultralytics`"
            )
        self.registry = registry
        self.imgsz = imgsz
        self.conf = conf
        self._models: Dict[str, Any] = {}

    # ---------- helpers ----------
    @staticmethod
    def _rle(mask: np.ndarray) -> str:
        pixels = mask.flatten(order="C")
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return " ".join(str(int(x)) for x in runs)

    @staticmethod
    def _mask_to_box(
        mask: np.ndarray, pad: int = 0
    ) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            return None
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        return max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad

    @staticmethod
    def _save_overlay(
        img_bgr: np.ndarray, regions: Sequence["Region"], out_path: str
    ) -> str:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        colors = {}
        vis = img_bgr.copy()
        overlay = vis.copy()
        rng = np.random.default_rng(7)
        for r in regions:
            if r.label not in colors:
                colors[r.label] = tuple(int(x) for x in rng.integers(60, 255, size=3))
            color = colors[r.label]
            x1, y1, x2, y2 = r.bbox
            cv.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

        vis = cv.addWeighted(overlay, 0.25, vis, 0.75, 0)
        for r in regions:
            x1, y1, x2, y2 = r.bbox
            color = colors[r.label]
            cv.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv.putText(
                vis,
                f"{r.label} ({r.score:.2f})",
                (x1, max(20, y1 - 6)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv.LINE_AA,
            )
        ok = cv.imwrite(out_path, vis)
        if not ok:
            raise IOError(f"Failed to write overlay to {out_path}")
        return out_path

    @staticmethod
    def _save_crops(
        img_bgr: np.ndarray, regions: Sequence["Region"], out_dir: str
    ) -> str:
        os.makedirs(out_dir, exist_ok=True)
        for i, r in enumerate(regions):
            x1, y1, x2, y2 = r.bbox
            crop = img_bgr[y1:y2, x1:x2].copy()
            cv.imwrite(os.path.join(out_dir, f"{i:02d}_{r.label}.png"), crop)
        return out_dir

    @staticmethod
    def _mask_to_polygon_line(mask: np.ndarray, cls_id: int) -> Optional[str]:
        h, w = mask.shape
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv.contourArea).squeeze(1)
        eps = 0.002 * cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, eps, True).squeeze(1)
        parts = [str(cls_id)]
        for x, y in cnt:
            parts += [f"{x/float(w):.6f}", f"{y/float(h):.6f}"]
        return " ".join(parts)

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
        inter = iw * ih
        area_a = max(0, (ax2 - ax1 + 1) * (ay2 - ay1 + 1))
        area_b = max(0, (bx2 - bx1 + 1) * (by2 - by1 + 1))
        denom = area_a + area_b - inter
        return float(inter) / float(denom) if denom > 0 else 0.0

    @staticmethod
    def _nms_per_label(entries, iou_thresh=0.5, max_keep=1):
        """
        entries: list of dicts with keys ['label', 'score', 'bbox', 'mask_bin', 'cls_id']
        Returns the kept entries after per-label NMS, capped by max_keep per label.
        """
        if not entries:
            return []

        def area(b):
            x1, y1, x2, y2 = b
            return max(0, (x2 - x1 + 1) * (y2 - y1 + 1))

        entries = sorted(
            entries, key=lambda e: (e["score"], area(e["bbox"])), reverse=True
        )
        kept = []
        for e in entries:
            suppress = False
            for k in kept:
                if SegmenterInference._iou(e["bbox"], k["bbox"]) >= iou_thresh:
                    suppress = True
                    break
            if not suppress:
                kept.append(e)
            if len(kept) >= max_keep:
                break
        return kept

    def _ensure_model(self, doc_type: str):
        if doc_type in self._models:
            return self._models[doc_type]
        weights = self.registry.get_weights(doc_type)
        if not weights:
            raise ValueError(f"No weights registered for doc_type='{doc_type}'.")
        model = YOLO(weights)
        self._models[doc_type] = model
        return model

    # ---------- public ----------
    def predict(
        self,
        image_bgr: np.ndarray,
        doc_type: str,
        required_labels: Optional[Sequence[str]] = ("header", "notes"),
        include_labels: Optional[Sequence[str]] = None,
        return_crops: bool = False,
        crops_dir: Optional[str] = None,
        save_overlay_path: Optional[str] = None,
        save_pred_labels_dir: Optional[str] = None,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        *,
        per_label_nms_iou: float = 0.5,
        max_per_label: int = 1,
        save_pred_labels_stem: Optional[str] = None,
        # NEW paired-output controls:
        save_pred_base_dir: Optional[str] = None,  # e.g., "pred"
        save_pred_split: str = "train",  # "train" or "val"
    ) -> InferenceResult:
        """
        Run segmentation on a single preprocessed document image.

        Args:
            image_bgr:         Preprocessed **BGR** image (H×W×3, uint8) from Stage-0.
            doc_type:          Key into `DocTypeRegistry` selecting trained weights & classes.
            required_labels:   Labels that must be present; else `abstain=True`.
            include_labels:    Optional whitelist of labels to keep.
            return_crops:      If True, write per-region crops to `crops_dir`.
            crops_dir:         Output dir for crops (required if return_crops=True).
            save_overlay_path: If provided, write overlay visualization to this path.
            save_pred_labels_dir:
                               Legacy single-folder mode: write labels to this folder
                               as `<stem>.txt` (use `save_pred_labels_stem`).
            conf:              Per-call confidence override; else uses init’s `conf`.
            imgsz:             Per-call size override; else uses init’s `imgsz`.
            per_label_nms_iou: IoU threshold for per-label NMS. Lower = more aggressive.
            max_per_label:     Keep at most this many detections **per class** after NMS.
            save_pred_labels_stem:
                               Filename stem for prediction labels (and overlay when
                               using paired-output mode).
            save_pred_base_dir:
                               If set, writes **paired outputs**:
                                 - `pred/image/<split>/<stem>.png` (overlay)
                                 - `pred/label/<split>/<stem>.txt` (YOLO-seg polygons)
                               where `pred` = `save_pred_base_dir`, `<split>` = `save_pred_split`.
            save_pred_split:   Subdir name under `image/` and `label/` (e.g., "train" or "val").

        Returns:
            InferenceResult with regions, abstain/reasons, and any saved paths.
        """
        classes = self.registry.get_classes(doc_type)
        if not classes:
            raise ValueError(f"No class_names registered for doc_type='{doc_type}'.")

        model = self._ensure_model(doc_type)
        th = self.conf if conf is None else conf
        imsz = self.imgsz if imgsz is None else imgsz

        base = image_bgr
        H, W = base.shape[:2]
        res = model.predict(base[:, :, ::-1], imgsz=imsz, conf=th, verbose=False)[0]

        # Collect all raw entries first (we'll NMS per label)
        raw_entries = []
        if res.masks is not None and res.boxes is not None:
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i].item())
                if not (0 <= cls_id < len(classes)):
                    continue
                label = classes[cls_id]
                if include_labels and (label not in include_labels):
                    continue
                score = float(res.boxes.conf[i].item())
                mask = res.masks.data[i].cpu().numpy()
                mask = cv.resize(mask, (W, H), interpolation=cv.INTER_LINEAR)
                mask_bin = (mask > 0.5).astype(np.uint8)
                box = self._mask_to_box(mask_bin, pad=0)
                if box is None:
                    continue
                raw_entries.append(
                    {
                        "label": label,
                        "cls_id": cls_id,
                        "score": score,
                        "bbox": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                        "mask_bin": mask_bin,
                    }
                )

        # Per-label NMS + top-k
        kept_entries = []
        if raw_entries:
            by_label: Dict[str, List[dict]] = {}
            for e in raw_entries:
                by_label.setdefault(e["label"], []).append(e)
            for lbl, group in by_label.items():
                kept_entries.extend(
                    self._nms_per_label(
                        group, iou_thresh=per_label_nms_iou, max_keep=max_per_label
                    )
                )

        # Build Regions from kept_entries
        out_regions: List[Region] = []
        for e in kept_entries:
            out_regions.append(
                Region(
                    label=e["label"],
                    score=e["score"],
                    bbox=e["bbox"],
                    mask_rle=self._rle(e["mask_bin"]),
                )
            )

        # Abstention
        labels_present = {r.label for r in out_regions}
        abstain = False
        reasons: List[str] = []
        if required_labels:
            missing = [l for l in required_labels if l not in labels_present]
            if missing:
                abstain = True
                reasons.append("missing:" + ",".join(missing))

        # --- resolve paired pred paths (overlay + label) if base_dir is provided ---
        overlay_path = save_overlay_path
        labels_dir = save_pred_labels_dir
        txt_path = None

        if save_pred_base_dir:
            stem = save_pred_labels_stem or "pred"
            img_dir = os.path.join(save_pred_base_dir, "image", save_pred_split)
            lbl_dir = os.path.join(save_pred_base_dir, "label", save_pred_split)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            overlay_path = os.path.join(img_dir, f"{stem}.png")
            labels_dir = lbl_dir
            txt_path = os.path.join(lbl_dir, f"{stem}.txt")

        # Optional outputs
        if overlay_path:
            overlay_path = self._save_overlay(base, out_regions, overlay_path)

        crops_out = None
        if return_crops:
            if not crops_dir:
                raise ValueError("return_crops=True requires `crops_dir`.")
            crops_out = self._save_crops(base, out_regions, crops_dir)

        # Save labels (paired mode or legacy single-folder mode)
        if save_pred_base_dir:
            H, W = base.shape[:2]
            lines = []
            for e in kept_entries:
                line = self._mask_to_polygon_line(e["mask_bin"], e["cls_id"])
                if line is None:
                    x1, y1, x2, y2 = e["bbox"]
                    parts = [
                        str(e["cls_id"]),
                        f"{x1/float(W):.6f}",
                        f"{y1/float(H):.6f}",
                        f"{x2/float(W):.6f}",
                        f"{y1/float(H):.6f}",
                        f"{x2/float(W):.6f}",
                        f"{y2/float(H):.6f}",
                        f"{x1/float(W):.6f}",
                        f"{y2/float(H):.6f}",
                    ]
                    line = " ".join(parts)
                lines.append(line)
            with open(txt_path, "w") as fh:
                for ln in lines:
                    fh.write(ln + "\n")

        elif save_pred_labels_dir:
            os.makedirs(save_pred_labels_dir, exist_ok=True)
            stem = save_pred_labels_stem or "pred"
            txt_path = os.path.join(save_pred_labels_dir, f"{stem}.txt")
            H, W = base.shape[:2]
            lines = []
            for e in kept_entries:
                line = self._mask_to_polygon_line(e["mask_bin"], e["cls_id"])
                if line is None:
                    x1, y1, x2, y2 = e["bbox"]
                    parts = [
                        str(e["cls_id"]),
                        f"{x1/float(W):.6f}",
                        f"{y1/float(H):.6f}",
                        f"{x2/float(W):.6f}",
                        f"{y1/float(H):.6f}",
                        f"{x2/float(W):.6f}",
                        f"{y2/float(H):.6f}",
                        f"{x1/float(W):.6f}",
                        f"{y2/float(H):.6f}",
                    ]
                    line = " ".join(parts)
                lines.append(line)
            with open(txt_path, "w") as fh:
                for ln in lines:
                    fh.write(ln + "\n")
            labels_dir = save_pred_labels_dir

        return InferenceResult(
            regions=out_regions,
            abstain=abstain,
            reasons=reasons,
            overlay_path=overlay_path,
            crops_dir=crops_out,
            labels_dir=labels_dir,
        )
