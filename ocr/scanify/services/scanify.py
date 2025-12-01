from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union, Dict

import cv2
import numpy as np

# Import logger from parent directory (optional)
try:
    import sys
    from pathlib import Path

    # Add parent directory to path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from utils import logger
except ImportError:
    # Fallback if utils not available
    logger = None


# =========================
# Public API
# =========================


@dataclass
class ScanifyConfig:
    # geometry / prep
    target_long_side_px: int = 2400
    do_perspective: bool = True
    strong_page_detect: bool = True
    min_page_fill: float = 0.25
    do_auto_deskew: bool = True
    # outputs and binarization
    # if return_color is True, provide warped_color for downstream color models
    # if return_bw is True, provide a grayscale/binarized image as image_bw
    return_color: bool = True
    return_bw: bool = False
    do_binarize: bool = False
    # crop to content after deskew
    do_content_crop: bool = True
    content_crop_margin_frac: float = 0.02  # fraction of min(h,w)

    # illumination
    background_flatten: bool = False  # colored forms: better off by default
    # light readability boost on color (before binarization or grayscale output)
    color_clahe: bool = True
    color_unsharp_strength: float = 0.25

    # binarization
    bw_method: str = "sauvola"  # 'sauvola' | 'adaptive' | 'otsu'
    sauvola_window: int = 61
    sauvola_k: float = 0.18
    adaptive_block_size: int = 31
    adaptive_C: int = 10
    # deskew fine search (around seed angle)
    fine_search_range_deg: float = 6.0
    fine_search_step_deg: float = 0.25

    # cleanup & finish
    despeckle_min_area: int = 160
    unsharp_strength: float = 0.2
    output_format: str = "png"
    jpg_quality: int = 92
    keep_color_debug: bool = False

    # saving trail
    save_debug: bool = False
    save_dir: Optional[str] = None
    save_every_step: bool = True
    save_stem: Optional[str] = None


@dataclass
class ScanifyResult:
    image_bw: Optional[np.ndarray]
    image_color: Optional[np.ndarray]
    debug: Dict[str, object]


class Scanify:
    """
    Stage 0: Convert phone photos of documents into crisp, B/W “scanned” pages.
    """

    # ------------------- Public -------------------

    def __init__(self, config: Optional[ScanifyConfig] = None):
        self.cfg = config or ScanifyConfig()

    def process(
        self,
        img_or_path: Union[str, np.ndarray],
        *,
        save_stem_override: Optional[str] = None,
    ) -> ScanifyResult:
        color = self._load_to_bgr(img_or_path)
        debug: Dict[str, object] = {}
        save_ctx = self._prepare_save_ctx(img_or_path, save_stem_override)

        # Resize early
        if logger:
            logger.step("Resizing image")
        color = (
            self._resize_long_side(color, self.cfg.target_long_side_px)
            if self.cfg.target_long_side_px > 0
            else color
        )
        debug["initial_shape"] = color.shape[:2]
        self._maybe_save(color, "00_input_resized", save_ctx)

        # Page detection (edges; fallback)
        warped = color.copy()
        quad, edges_vis = (None, None)
        if self.cfg.do_perspective:
            if logger:
                logger.step("Detecting document page boundaries")
            quad, edges_vis = self._find_document_quad_with_edges(color)
            if edges_vis is not None:
                self._maybe_save(edges_vis, "01_edges", save_ctx)

            if quad is None and self.cfg.strong_page_detect:
                if logger:
                    logger.step("Using fallback page detection")
                quad = self._fallback_page_quad(color)
                if quad is not None:
                    overlay = color.copy()
                    cv2.polylines(
                        overlay,
                        [quad.astype(np.int32)],
                        True,
                        (255, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    self._maybe_save(overlay, "02_contour_overlay_fallback", save_ctx)

            if quad is not None:
                overlay = color.copy()
                cv2.polylines(
                    overlay, [quad.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA
                )
                self._maybe_save(overlay, "02_contour_overlay", save_ctx)
                if logger:
                    logger.step("Applying perspective warp")
                warped = self._four_point_warp(color, quad)
                self._maybe_save(warped, "03_warped", save_ctx)

        # Optional illumination
        if self.cfg.background_flatten:
            if logger:
                logger.step("Applying illumination correction")
            warped = self._illumination_correction(warped)
            self._maybe_save(warped, "04_illum_flattened", save_ctx)

        # SMART DESKEW (replaces old simple Hough)
        if self.cfg.do_auto_deskew:
            if logger:
                logger.step("Estimating and correcting skew angle")
            angle, details = self._estimate_skew_angle_smart(warped, quad)
            debug["deskew_angle_deg"] = angle
            debug["deskew_debug"] = details
            if abs(angle) > 0.1:
                warped = self._rotate_bound(warped, -angle)
                if logger:
                    logger.info(f"Corrected skew: {angle:.2f}°")
            self._maybe_save(warped, "05_deskewed", save_ctx)

        # Optional content crop (tighten to writing region with a small margin)
        if self.cfg.do_content_crop:
            if logger:
                logger.step("Auto-cropping to content")
            cropped, crop_dbg = self._auto_content_crop(
                warped, self.cfg.content_crop_margin_frac
            )
            if cropped is not None:
                warped = cropped
                debug["content_crop"] = crop_dbg
                self._maybe_save(warped, "05b_cropped", save_ctx)

        # Optional readability boost on color
        if self.cfg.color_clahe or self.cfg.color_unsharp_strength > 0:
            if logger:
                logger.step("Enhancing color readability")
            warped = self._enhance_color_readability(
                warped, self.cfg.color_clahe, self.cfg.color_unsharp_strength
            )
            self._maybe_save(warped, "05c_color_enhanced", save_ctx)

        # Outputs: produce only what is requested
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        bw: Optional[np.ndarray] = None
        if self.cfg.do_binarize:
            # Produce a binarized B/W image if binarization is requested
            method = self.cfg.bw_method.lower()
            if method == "sauvola":
                bw = self._sauvola(gray, self.cfg.sauvola_window, self.cfg.sauvola_k)
            elif method == "adaptive":
                bw = self._adaptive_binarize(
                    gray, self.cfg.adaptive_block_size, self.cfg.adaptive_C
                )
            elif method == "otsu":
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                raise ValueError(f"Unknown bw_method: {self.cfg.bw_method}")
            self._maybe_save(bw, "06_binarized_raw", save_ctx)
            if self.cfg.despeckle_min_area > 0:
                bw = self._despeckle(bw, self.cfg.despeckle_min_area)
            if self.cfg.unsharp_strength > 0:
                bw = self._unsharp_on_binary(bw, amount=self.cfg.unsharp_strength)
            self._maybe_save(bw, "07_binarized_final", save_ctx)
        elif self.cfg.return_bw:
            # Return grayscale as B/W output when requested (no thresholding)
            bw = gray.copy()
            self._maybe_save(bw, "06_gray_output", save_ctx)

        # Montage
        image_color = (
            warped if (self.cfg.return_color or self.cfg.keep_color_debug) else None
        )
        if self.cfg.keep_color_debug:
            montage = self._make_montage(
                [
                    color,
                    warped,
                    cv2.cvtColor((bw if bw is not None else gray), cv2.COLOR_GRAY2BGR),
                ],
                [
                    "input",
                    "warped",
                    "binarized" if self.cfg.do_binarize else "grayscale",
                ],
            )
            debug["montage"] = montage
            self._maybe_save(montage, "08_montage", save_ctx)

        return ScanifyResult(image_bw=bw, image_color=image_color, debug=debug)

    def process_batch(
        self,
        items: Iterable[Union[str, np.ndarray]],
        out_dir: Optional[str] = None,
        stem_fn: Optional[callable] = None,
    ) -> List[ScanifyResult]:
        results: List[ScanifyResult] = []
        img_dir = dbg_dir = None
        if out_dir:
            img_dir = os.path.join(out_dir, "images")
            dbg_dir = os.path.join(out_dir, "debug")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(dbg_dir, exist_ok=True)

        for idx, item in enumerate(items):
            stem = (
                self._default_stem(item, idx) if stem_fn is None else stem_fn(item, idx)
            )
            res = self.process(item, save_stem_override=stem)
            results.append(res)

            if out_dir:
                # Save B/W output if present
                if res.image_bw is not None:
                    img_path = os.path.join(
                        img_dir, f"{stem}.{self.cfg.output_format.lower()}"
                    )
                    self._save_bw(res.image_bw, img_path)
                if (
                    self.cfg.keep_color_debug
                    and "montage" in res.debug
                    and isinstance(res.debug["montage"], np.ndarray)
                ):
                    dbg_path = os.path.join(dbg_dir, f"{stem}__debug.png")
                    cv2.imwrite(dbg_path, res.debug["montage"])
        return results

    # ------------------- Internals -------------------

    @staticmethod
    def _load_to_bgr(img_or_path: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(img_or_path, str):
            img = cv2.imdecode(
                np.fromfile(img_or_path, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                raise ValueError(f"Failed to read image: {img_or_path}")
            return img
        arr = np.asarray(img_or_path)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            if Scanify._looks_like_rgb(arr):
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr.copy()
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        raise ValueError("Unsupported array shape for image.")

    @staticmethod
    def _looks_like_rgb(arr: np.ndarray) -> bool:
        h, w, _ = arr.shape
        sample = arr[h // 2, w // 2].astype(np.int32)
        return int(sample[0]) < int(sample[2])

    @staticmethod
    def _resize_long_side(img: np.ndarray, target: int) -> np.ndarray:
        h, w = img.shape[:2]
        if target <= 0 or max(h, w) <= target:
            return img
        scale = target / float(max(h, w))
        new_size = (int(round(w * scale)), int(round(h * scale)))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # ---- Page detection: edges + fallback ----

    @staticmethod
    def _find_document_quad_with_edges(
        img_bgr: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        img_area = h * w
        best_quad = None
        best_area = 0

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = abs(cv2.contourArea(approx))
                if area > 0.15 * img_area and area > best_area:
                    best_area = area
                    best_quad = approx.reshape(-1, 2).astype(np.float32)

        if best_quad is None:
            return None, edges_vis
        return Scanify._order_quad_points(best_quad), edges_vis

    def _fallback_page_quad(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        Lb = cv2.GaussianBlur(L, (0, 0), 3.0)
        Lnorm = cv2.normalize(Lb, None, 0, 255, cv2.NORM_MINMAX)
        th = cv2.threshold(Lnorm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        if num <= 1:
            return None
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        area = stats[idx, cv2.CC_STAT_AREA]
        h, w = L.shape
        if area < self.cfg.min_page_fill * h * w:
            return None

        mask = (labels == idx).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        return self._order_quad_points(box)

    @staticmethod
    def _order_quad_points(pts: np.ndarray) -> np.ndarray:
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def _four_point_warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        (tl, tr, br, bl) = quad
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxW = int(round(max(widthA, widthB)))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxH = int(round(max(heightA, heightB)))

        dst = np.array(
            [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(quad, dst)
        return cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_CUBIC)

    # ---- Illumination / deskew ----

    @staticmethod
    def _illumination_correction(img_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        k = max(25, (min(L.shape[:2]) // 20) | 1)
        background = cv2.morphologyEx(
            L, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
        L2 = cv2.normalize(cv2.subtract(L, background), None, 0, 255, cv2.NORM_MINMAX)
        L2 = cv2.equalizeHist(L2)
        lab2 = cv2.merge([L2, A, B])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    def _estimate_skew_angle_smart(
        self, img_bgr: np.ndarray, quad: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Robust skew estimator:
          1) If quad is available, coarse angle from TL->TR edge.
          2) HoughLinesP on thick text/lines for fine angle (median of long near-horizontal lines).
          3) Fallback: minAreaRect on text mask.
          4) Fine search: sweep +/- range around best seed maximizing row-variance.
        Returns (angle_deg, debug_dict). Positive angle means image leans clockwise.
        """
        dbg: Dict[str, float] = {}

        # 1) coarse from quad
        coarse = 0.0
        if quad is not None:
            tl, tr, _, _ = quad
            v = tr - tl
            coarse = np.degrees(np.arctan2(v[1], v[0]))
            # normalize around 0 (horizontal)
            while coarse <= -90:
                coarse += 180
            while coarse > 90:
                coarse -= 180
            if coarse > 45:
                coarse -= 90
            if coarse < -45:
                coarse += 90
            dbg["coarse_from_quad"] = float(coarse)

        # Build a text mask once
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # light denoise + normalize
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        # Otsu on inverted tends to highlight strokes on yellow paper
        inv = cv2.bitwise_not(gray)
        th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # strengthen long strokes
        kx = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kx, iterations=1)

        # 2) Hough lines (probabilistic) on edges from text mask
        edges = cv2.Canny(th, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=90, minLineLength=80, maxLineGap=10
        )

        def angle_norm(a):
            a = np.degrees(a)
            while a <= -90:
                a += 180
            while a > 90:
                a -= 180
            if a > 45:
                a -= 90
            if a < -45:
                a += 90
            return a

        hough_angle = None
        if lines is not None and len(lines) > 0:
            angs = []
            lens = []
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx, dy = (x2 - x1), (y2 - y1)
                length = np.hypot(dx, dy)
                if length < 70:
                    continue
                ang = angle_norm(np.arctan2(dy, dx))
                # prefer near-horizontal lines within +/- 30 deg
                if abs(ang) <= 30:
                    angs.append(ang)
                    lens.append(length)
            if angs:
                # length-weighted median
                order = np.argsort(angs)
                angs_sorted = np.array(angs)[order]
                lens_sorted = np.array(lens)[order]
                cum = np.cumsum(lens_sorted)
                target = cum[-1] / 2
                idx = int(np.searchsorted(cum, target))
                hough_angle = float(angs_sorted[idx])
                dbg["hough_linesp_angle"] = hough_angle

        # 3) Fallback: minAreaRect on text pixels
        mar_angle = None
        if hough_angle is None:
            ys, xs = np.where(th > 0)
            if xs.size > 500:
                pts = np.column_stack([xs, ys]).astype(np.float32)
                rect = cv2.minAreaRect(pts)
                a = rect[2]  # OpenCV gives angle in [-90, 0)
                # convert to our normalized range
                if a < -45:
                    a = a + 90
                mar_angle = float(a)
                dbg["minarearect_angle"] = mar_angle

        # choose best available with tie-breaking
        if hough_angle is not None:
            seed = hough_angle
        elif mar_angle is not None:
            seed = mar_angle
        else:
            seed = coarse  # may still be 0.0

        # 4) Fine search around seed to lock in the best alignment of text rows
        fine_angle = self._fine_angle_search(
            th, seed, self.cfg.fine_search_range_deg, self.cfg.fine_search_step_deg
        )
        dbg["fine_search_seed"] = float(seed)
        dbg["fine_search_final"] = float(fine_angle)

        final = float(fine_angle)
        dbg["final_angle"] = final
        return final, dbg

    @staticmethod
    def _fine_angle_search(
        text_mask: np.ndarray, seed_deg: float, range_deg: float, step_deg: float
    ) -> float:
        """
        Sweep small angles around seed on a downscaled text mask and pick the angle
        that maximizes row-wise variance (i.e., makes text lines most horizontal).
        """
        h, w = text_mask.shape[:2]
        scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
        if scale != 1.0:
            small = cv2.resize(
                text_mask,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            small = text_mask

        # ensure binary {0,255} with white text on black background
        small = (small > 0).astype(np.uint8) * 255

        def rotate(img, ang_deg):
            (hh, ww) = img.shape[:2]
            center = (ww / 2.0, hh / 2.0)
            M = cv2.getRotationMatrix2D(center, ang_deg, 1.0)
            return cv2.warpAffine(
                img, M, (ww, hh), flags=cv2.INTER_NEAREST, borderValue=0
            )

        best_ang = seed_deg
        best_score = -1.0
        start = seed_deg - range_deg
        end = seed_deg + range_deg + 1e-6
        ang = start
        while ang <= end:
            rot = rotate(small, -ang)  # rotate opposite to estimate
            rowsum = rot.sum(axis=1).astype(np.float64)
            # normalize to [0,1] and compute variance
            rowsum /= 255.0 * rot.shape[1] + 1e-6
            score = rowsum.var()
            if score > best_score:
                best_score = score
                best_ang = ang
            ang += step_deg
        return float(best_ang)

    @staticmethod
    def _rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
        (h, w) = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        return cv2.warpAffine(
            image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    # ---- Content crop + readability ----
    @staticmethod
    def _auto_content_crop(
        img_bgr: np.ndarray, margin_frac: float
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """
        Find a tight bounding box around text/lines on the page and crop with margin.
        Keeps a conservative fallback to avoid over-cropping.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # strengthen strokes so the text block is contiguous
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k1, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k2, iterations=1)

        nz = cv2.findNonZero(th)
        if nz is None:
            return None, {"used": 0.0}
        x, y, ww, hh = cv2.boundingRect(nz)
        # add margin
        m = int(round(min(h, w) * max(0.0, margin_frac)))
        x0 = max(0, x - m)
        y0 = max(0, y - m)
        x1 = min(w, x + ww + m)
        y1 = min(h, y + hh + m)

        # if crop is almost the same as original (>95%), skip to avoid needless resize
        crop_area = (x1 - x0) * (y1 - y0)
        if crop_area > 0.95 * h * w:
            return None, {"used": 0.0}
        cropped = img_bgr[y0:y1, x0:x1].copy()
        return cropped, {
            "used": 1.0,
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
        }

    @staticmethod
    def _enhance_color_readability(
        img_bgr: np.ndarray, use_clahe: bool, unsharp_amount: float
    ) -> np.ndarray:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            L = clahe.apply(L)
        if unsharp_amount > 0:
            blur = cv2.GaussianBlur(L, (0, 0), sigmaX=1.0)
            L = cv2.addWeighted(L, 1 + unsharp_amount, blur, -unsharp_amount, 0)
        lab2 = cv2.merge([L, A, B])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # ---- Binarization / cleanup ----

    @staticmethod
    def _adaptive_binarize(gray: np.ndarray, block_size: int, C: int) -> np.ndarray:
        block = block_size if block_size % 2 == 1 else block_size + 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C
        )

    @staticmethod
    def _sauvola(gray: np.ndarray, win: int, k: float) -> np.ndarray:
        win = win if win % 2 == 1 else win + 1
        gray_f = gray.astype(np.float32)
        mean = cv2.boxFilter(
            gray_f, ddepth=cv2.CV_32F, ksize=(win, win), normalize=True
        )
        sqmean = cv2.boxFilter(
            gray_f * gray_f, ddepth=cv2.CV_32F, ksize=(win, win), normalize=True
        )
        var = np.maximum(sqmean - mean * mean, 0.0)
        std = cv2.sqrt(var)
        R = 128.0
        thresh = (mean * (1 + k * ((std / R) - 1))).astype(np.float32)
        return (gray_f > thresh).astype(np.uint8) * 255

    @staticmethod
    def _despeckle(bw: np.ndarray, min_area: int) -> np.ndarray:
        def filter_cc(thresh_img: np.ndarray, white_foreground: bool) -> np.ndarray:
            img = thresh_img.copy()
            fg = (
                (img == 255).astype(np.uint8)
                if white_foreground
                else (img == 0).astype(np.uint8)
            )
            num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
            for i in range(1, num):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:
                    mask = labels == i
                    img[mask] = 0 if white_foreground else 255
            return img

        bw = filter_cc(bw, True)
        bw = filter_cc(bw, False)
        return bw

    @staticmethod
    def _unsharp_on_binary(bw: np.ndarray, amount: float = 0.6) -> np.ndarray:
        gray = bw.astype(np.uint8)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
        _, out = cv2.threshold(sharp, 127, 255, cv2.THRESH_BINARY)
        return out

    def _save_bw(self, bw: np.ndarray, path: str) -> None:
        fmt = self.cfg.output_format.lower()
        if fmt == "png":
            cv2.imwrite(path, bw, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        elif fmt in ("jpg", "jpeg"):
            cv2.imwrite(path, bw, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpg_quality])
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    @staticmethod
    def _default_stem(item: Union[str, np.ndarray], idx: int) -> str:
        if isinstance(item, str):
            base = os.path.basename(item)
            stem, _ = os.path.splitext(base)
            return stem
        return f"item_{idx:04d}"

    @staticmethod
    def _make_montage(
        images: List[np.ndarray], titles: Optional[List[str]] = None, pad: int = 6
    ) -> np.ndarray:
        heights = [im.shape[0] for im in images]
        max_h = max(heights)
        padded = []
        for im in images:
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            scale = max_h / im.shape[0]
            im2 = cv2.resize(
                im,
                (int(round(im.shape[1] * scale)), max_h),
                interpolation=cv2.INTER_CUBIC,
            )
            padded.append(im2)
        sep = np.ones((max_h, pad, 3), dtype=np.uint8) * 240
        row = []
        for i, im in enumerate(padded):
            row.append(im)
            if i != len(padded) - 1:
                row.append(sep.copy())
        montage = np.concatenate(row, axis=1)
        if titles:
            band_h = 30
            band = np.ones((band_h, montage.shape[1], 3), dtype=np.uint8) * 245
            x = 6
            for t in titles:
                cv2.putText(
                    band,
                    t,
                    (x, int(band_h * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (30, 30, 30),
                    1,
                    cv2.LINE_AA,
                )
                x += montage.shape[1] // len(titles)
            montage = np.concatenate([band, montage], axis=0)
        return montage

    # -------- Debug save helpers --------

    def _prepare_save_ctx(
        self, img_or_path: Union[str, np.ndarray], save_stem_override: Optional[str]
    ):
        if not self.cfg.save_debug:
            return None
        root = self.cfg.save_dir or "stage0_debug"
        os.makedirs(root, exist_ok=True)
        stem = (
            save_stem_override
            or self.cfg.save_stem
            or self._default_stem(img_or_path, 0)
        )
        folder = os.path.join(root, stem)
        os.makedirs(folder, exist_ok=True)
        return {"folder": folder, "stem": stem, "save_all": self.cfg.save_every_step}

    def _maybe_save(
        self, img: np.ndarray, tag: str, save_ctx: Optional[Dict[str, str]]
    ):
        if save_ctx is None:
            return
        to_save = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        path = os.path.join(save_ctx["folder"], f"{tag}.png")
        cv2.imwrite(path, to_save)
