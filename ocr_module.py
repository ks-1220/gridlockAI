"""
ocr_module.py — License Plate Extraction via PaddleOCR
=======================================================
Extracts and validates license plate text from vehicle detections.
Operates on cropped vehicle regions (NOT full images) for speed and
accuracy.  Includes preprocessing (CLAHE + adaptive thresholding),
Indian plate format validation, and graceful error handling.

Author: Team Gridlock | Flipkart Grid 6.0
"""

from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils import (
    Detection,
    BoundingBox,
    crop_region,
    OCR_LANGUAGE,
    OCR_USE_GPU,
    OCR_DET_DB_THRESH,
    OCR_DROP_SCORE,
    CROP_PADDING,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

# Vehicle class names eligible for plate extraction
_VEHICLE_CLASSES: frozenset[str] = frozenset(
    {"car", "motorcycle", "bus", "truck", "bicycle"}
)

# Indian license plate regex — e.g. MH 12 AB 1234
_INDIAN_PLATE_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4}$"
)

# Loose plate regex — any mix of ≥ 4 alpha-numeric chars
_LOOSE_PLATE_PATTERN: re.Pattern[str] = re.compile(
    r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9 ]{4,}$"
)

# Bottom fraction of the vehicle crop to focus on for plates
_PLATE_BOTTOM_FRACTION: float = 0.45

# Minimum crop dimension (px) below which OCR is unreliable
_MIN_CROP_DIM: int = 20


# ──────────────────────────────────────────────
#  LicensePlateOCR
# ──────────────────────────────────────────────

class LicensePlateOCR:
    """PaddleOCR-based license plate reader.

    Workflow
    --------
    1. Receive a full image + vehicle bounding box.
    2. Crop the bottom portion of the vehicle region (plates sit low).
    3. Pre-process (grayscale → CLAHE → adaptive threshold).
    4. Run PaddleOCR on the crop.
    5. Filter / validate candidate strings against Indian plate format.
    6. Return the best plate string, or ``None``.
    """

    def __init__(self) -> None:
        """Initialise the PaddleOCR engine with project-wide settings."""
        logger.info("Initialising PaddleOCR engine …")
        t_start = time.perf_counter()

        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]

            self.ocr = PaddleOCR(
                use_textline_orientation=True,
                lang=OCR_LANGUAGE,
                device="gpu" if OCR_USE_GPU else "cpu",
                text_det_thresh=OCR_DET_DB_THRESH,
                text_rec_score_thresh=OCR_DROP_SCORE,
            )
        except ImportError:
            logger.error(
                "PaddleOCR is not installed.  "
                "Run:  pip install paddlepaddle paddleocr"
            )
            raise

        elapsed = time.perf_counter() - t_start
        logger.info("PaddleOCR ready in %.2f s", elapsed)

    # ── public API ────────────────────────────

    def extract_plate(
        self,
        image: np.ndarray,
        vehicle_bbox: BoundingBox,
    ) -> Optional[str]:
        """Extract a license plate string from one vehicle region.

        Parameters
        ----------
        image:
            Full scene image (BGR, ``np.uint8``).
        vehicle_bbox:
            Bounding box of the detected vehicle in *image*.

        Returns
        -------
        str or None
            Cleaned plate text, or ``None`` if nothing valid was found.
        """
        t_start = time.perf_counter()

        try:
            # 1.  Crop vehicle region with extra bottom padding
            bottom_padded_bbox = self._bottom_focus_bbox(
                vehicle_bbox, image.shape[:2]
            )
            vehicle_crop = crop_region(
                image, bottom_padded_bbox, padding=CROP_PADDING
            )

            if (
                vehicle_crop.size == 0
                or vehicle_crop.shape[0] < _MIN_CROP_DIM
                or vehicle_crop.shape[1] < _MIN_CROP_DIM
            ):
                logger.debug(
                    "Crop too small (%s) — skipping OCR.", vehicle_crop.shape
                )
                return None

            # 2.  Pre-process for cleaner OCR
            processed = self._preprocess_crop(vehicle_crop)

            # 3.  Run PaddleOCR on the crop (NOT full image)
            results = self.ocr.ocr(processed)

            if not results or not results[0]:
                logger.debug("PaddleOCR returned no results for crop.")
                return None

            # 4.  Collect candidate texts
            best_plate: Optional[str] = None
            best_score: float = 0.0

            for line in results[0]:
                # Each line: [bbox_points, (text, confidence)]
                raw_text: str = line[1][0]
                confidence: float = float(line[1][1])

                cleaned = self._clean_plate_text(raw_text)
                if not cleaned:
                    continue

                if self._is_valid_plate(cleaned) and confidence > best_score:
                    best_plate = cleaned
                    best_score = confidence

            elapsed = time.perf_counter() - t_start
            if best_plate:
                logger.info(
                    "Plate detected: '%s'  (conf=%.2f, %.3f s)",
                    best_plate,
                    best_score,
                    elapsed,
                )
            else:
                logger.debug("No valid plate found (%.3f s).", elapsed)

            return best_plate

        except Exception:
            logger.exception("OCR failed for bbox %s", vehicle_bbox)
            return None

    def extract_all_plates(
        self,
        image: np.ndarray,
        detections: List[Detection],
    ) -> Dict[str, str]:
        """Run plate extraction on every *vehicle* detection.

        Parameters
        ----------
        image:
            Full scene image (BGR, ``np.uint8``).
        detections:
            List of detections from the object-detector stage.

        Returns
        -------
        dict
            Mapping ``"<detection_index>"`` → ``"<plate_text>"`` for
            every vehicle where a plate was successfully read.
        """
        t_start = time.perf_counter()
        plates: Dict[str, str] = {}

        for idx, det in enumerate(detections):
            # Skip non-vehicle classes (persons, traffic lights, etc.)
            if det.class_name not in _VEHICLE_CLASSES:
                continue

            plate = self.extract_plate(image, det.bbox)
            if plate is not None:
                plates[str(idx)] = plate

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Extracted %d plate(s) from %d detection(s) in %.3f s",
            len(plates),
            len(detections),
            elapsed,
        )
        return plates

    # ── preprocessing ─────────────────────────

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Enhance a vehicle crop for more reliable OCR.

        Pipeline
        --------
        1. Convert BGR → grayscale.
        2. CLAHE contrast enhancement (clip-limit 2.0, tile 8×8).
        3. Adaptive Gaussian thresholding for binarisation.
        4. Convert back to 3-channel (PaddleOCR expects BGR/RGB).

        Parameters
        ----------
        crop:
            Raw BGR crop of the vehicle / plate area.

        Returns
        -------
        np.ndarray
            Pre-processed 3-channel image ready for OCR.
        """
        # Grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # PaddleOCR expects a 3-channel image
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── text cleaning & validation ────────────

    @staticmethod
    def _clean_plate_text(raw_text: str) -> str:
        """Sanitise raw OCR output into a canonical plate string.

        Steps
        -----
        1. Strip non-alphanumeric characters (except spaces).
        2. Collapse multiple spaces.
        3. Upper-case.

        Parameters
        ----------
        raw_text:
            Raw text string returned by PaddleOCR.

        Returns
        -------
        str
            Cleaned, upper-cased text (may be empty).
        """
        # Keep only alphanumeric + space
        cleaned = re.sub(r"[^A-Za-z0-9 ]", "", raw_text)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned.upper()

    @staticmethod
    def _is_valid_plate(text: str) -> bool:
        """Check whether *text* looks like a real license plate.

        Validation rules
        ----------------
        * At least 4 characters long.
        * Contains both letters **and** digits.
        * Optionally matches the Indian format ``XX 00 XX 0000``.
          If the strict pattern matches we return ``True`` immediately;
          otherwise we fall back to the loose pattern.

        Parameters
        ----------
        text:
            Cleaned, upper-cased candidate string.

        Returns
        -------
        bool
        """
        if len(text) < 4:
            return False

        has_alpha = any(ch.isalpha() for ch in text)
        has_digit = any(ch.isdigit() for ch in text)
        if not (has_alpha and has_digit):
            return False

        # Prefer strict Indian format
        if _INDIAN_PLATE_PATTERN.match(text):
            return True

        # Fall back to loose alphanumeric pattern
        return bool(_LOOSE_PLATE_PATTERN.match(text))

    # ── geometry helpers ──────────────────────

    @staticmethod
    def _bottom_focus_bbox(
        bbox: BoundingBox,
        image_hw: tuple[int, int],
    ) -> BoundingBox:
        """Return a bbox covering the bottom portion of a vehicle.

        License plates are typically in the lower 40-45 % of the
        vehicle bounding box.  We keep a bit of horizontal padding
        and extend the vertical range downward.

        Parameters
        ----------
        bbox:
            Original vehicle bounding box.
        image_hw:
            ``(height, width)`` of the source image (for clamping).

        Returns
        -------
        BoundingBox
            A new, tighter box focused on the plate region.
        """
        img_h, img_w = image_hw
        cut_y = bbox.y1 + int(bbox.height * (1 - _PLATE_BOTTOM_FRACTION))

        return BoundingBox(
            x1=max(0, bbox.x1 - CROP_PADDING),
            y1=max(0, cut_y),
            x2=min(img_w, bbox.x2 + CROP_PADDING),
            y2=min(img_h, bbox.y2 + CROP_PADDING),
        )


# ──────────────────────────────────────────────
#  Factory (Streamlit caching friendly)
# ──────────────────────────────────────────────

def get_ocr() -> LicensePlateOCR:
    """Create and return a :class:`LicensePlateOCR` instance.

    Intended to be wrapped by ``@st.cache_resource`` in the Streamlit
    app layer so the heavy PaddleOCR model is loaded only once per
    session.

    Returns
    -------
    LicensePlateOCR
    """
    return LicensePlateOCR()
