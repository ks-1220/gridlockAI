"""
annotator.py — Image Annotation Engine
=======================================
Draws bounding boxes, violation badges, license-plate labels, and an
informational overlay panel onto frames for the Traffic Violation
Detection System.

All drawing is performed on a **copy** of the source image; the original
numpy array is never mutated.

Author: Team Gridlock | Flipkart Grid 6.0
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    ViolationRecord,
    Detection,
    BoundingBox,
    ViolationType,
    Severity,
    VIOLATION_COLORS_BGR,
    SEVERITY_COLORS,
    VIOLATION_SEVERITY,
    format_confidence,
    get_severity_emoji,
)

# ──────────────────────────────────────────────
#  Module-level drawing constants
# ──────────────────────────────────────────────

_FONT: int = cv2.FONT_HERSHEY_SIMPLEX
_LINE_TYPE: int = cv2.LINE_AA

# Default (muted) color for non-violated detections (BGR)
_MUTED_COLOR: Tuple[int, int, int] = (160, 160, 160)

# Info-panel styling
_PANEL_BG_COLOR: Tuple[int, int, int] = (30, 25, 20)   # Near-black BGR
_PANEL_ALPHA: float = 0.72
_PANEL_TEXT_COLOR: Tuple[int, int, int] = (230, 233, 239)
_PANEL_ACCENT_COLOR: Tuple[int, int, int] = (247, 143, 79)  # Accent blue in BGR

# Severity indicator colors (BGR, derived from hex in SEVERITY_COLORS)
_SEVERITY_BGR: Dict[Severity, Tuple[int, int, int]] = {
    Severity.CRITICAL: (68, 68, 239),    # #ef4444
    Severity.MODERATE: (22, 115, 249),   # #f97316
    Severity.MINOR:    (8, 179, 234),    # #eab308
}

# Severity priority for comparison (lower index = higher priority)
_SEVERITY_PRIORITY: List[Severity] = [
    Severity.CRITICAL,
    Severity.MODERATE,
    Severity.MINOR,
]


class ImageAnnotator:
    """Render production-quality annotations on traffic camera frames.

    Provides methods to overlay bounding boxes for every detected object,
    highlight violations with colour-coded badges, attach license-plate
    labels, and composite a translucent summary panel at the top of the
    image.

    Parameters
    ----------
    font_scale : float
        Base font scale used for label text (default ``0.5``).
    thickness : int
        Base stroke thickness for boxes and text (default ``2``).

    Example
    -------
    >>> annotator = ImageAnnotator(font_scale=0.6, thickness=2)
    >>> result = annotator.annotate(frame, detections, violations)
    >>> cv2.imwrite("annotated.jpg", result)
    """

    # ── Construction ───────────────────────────

    def __init__(self, font_scale: float = 0.5, thickness: int = 2) -> None:
        """Initialise drawing parameters.

        Args:
            font_scale: Relative scale factor for cv2.putText.
            thickness:  Stroke width (px) for rectangles and text.
        """
        self.font_scale: float = font_scale
        self.thickness: int = thickness

    # ── Public API ─────────────────────────────

    def annotate(
        self,
        image: np.ndarray,
        detections: List[Detection],
        violations: List[ViolationRecord],
    ) -> np.ndarray:
        """Annotate an image with detections, violation badges, and an info panel.

        The original *image* is **never** mutated; all drawing is done on
        an internal copy.

        Args:
            image:      Source BGR image (H × W × 3, ``uint8``).
            detections: All objects found by the detector (e.g. RT-DETR).
            violations: Confirmed violations produced by the rule engine.

        Returns:
            A new ``np.ndarray`` with all annotations composited.
        """
        canvas: np.ndarray = image.copy()

        # Build a lookup: bbox-tuple → ViolationRecord for O(1) matching
        violation_map: Dict[Tuple[int, int, int, int], ViolationRecord] = {
            v.bbox.as_tuple(): v for v in violations
        }

        # 1. Draw detection boxes (muted style for non-violated objects)
        for det in detections:
            bbox_key = det.bbox.as_tuple()
            if bbox_key not in violation_map:
                self._draw_detection_box(canvas, det)

        # 2. Overlay violation badges (thicker, coloured)
        for violation in violations:
            self._draw_violation_box(canvas, violation)
            self._draw_license_plate_label(canvas, violation)

        # 3. Info panel at the top of the frame
        if violations:
            self._draw_info_panel(canvas, violations)

        return canvas

    # ── Private: Detection drawing ─────────────

    def _draw_detection_box(
        self, image: np.ndarray, detection: Detection
    ) -> None:
        """Draw a muted bounding box with a class-name label.

        Non-violated objects receive a subtle gray box so the viewer can
        still see everything the detector found without visual clutter.

        Args:
            image:     Canvas to draw on (modified in-place).
            detection: The detection whose box should be rendered.
        """
        bb: BoundingBox = detection.bbox
        pt1: Tuple[int, int] = (bb.x1, bb.y1)
        pt2: Tuple[int, int] = (bb.x2, bb.y2)

        # Thin gray rectangle
        cv2.rectangle(image, pt1, pt2, _MUTED_COLOR, thickness=1, lineType=_LINE_TYPE)

        # Small label: "car 92.3%"
        label: str = f"{detection.class_name} {format_confidence(detection.confidence)}"
        self._put_text_with_background(
            image,
            text=label,
            position=(bb.x1, bb.y1 - 4),
            font_scale=self.font_scale * 0.75,
            color=(255, 255, 255),
            bg_color=_MUTED_COLOR,
            padding=3,
        )

    # ── Private: Violation drawing ─────────────

    def _draw_violation_box(
        self, image: np.ndarray, violation: ViolationRecord
    ) -> None:
        """Draw a prominent, colour-coded violation bounding box.

        The box colour matches the violation type from
        ``VIOLATION_COLORS_BGR``.  A filled label strip shows the
        violation name and confidence.  A small severity-indicator dot is
        placed in the top-right corner of the box.

        Args:
            image:     Canvas to draw on (modified in-place).
            violation: The violation record to visualise.
        """
        bb: BoundingBox = violation.bbox
        pt1: Tuple[int, int] = (bb.x1, bb.y1)
        pt2: Tuple[int, int] = (bb.x2, bb.y2)

        vtype: ViolationType = violation.violation_type
        color: Tuple[int, int, int] = VIOLATION_COLORS_BGR.get(vtype, (0, 0, 255))

        # Thick coloured rectangle
        box_thickness: int = max(self.thickness + 1, 3)
        cv2.rectangle(image, pt1, pt2, color, thickness=box_thickness, lineType=_LINE_TYPE)

        # ── Label strip ───────────────────────
        label: str = f"{vtype.value}  {format_confidence(violation.confidence)}"
        label_font_scale: float = self.font_scale * 0.9
        (tw, th), baseline = cv2.getTextSize(label, _FONT, label_font_scale, 1)

        label_pad: int = 6
        label_h: int = th + baseline + label_pad * 2
        label_w: int = tw + label_pad * 2

        # Position label above the box; if too close to the top edge,
        # push it inside the box instead.
        if bb.y1 - label_h < 0:
            label_y1: int = bb.y1
        else:
            label_y1 = bb.y1 - label_h

        label_y2: int = label_y1 + label_h
        label_x2: int = min(bb.x1 + label_w, image.shape[1])

        # Filled label background
        self._draw_rounded_rect(
            image,
            pt1=(bb.x1, label_y1),
            pt2=(label_x2, label_y2),
            color=color,
            radius=4,
            thickness=-1,
        )

        # White text on the label
        text_origin: Tuple[int, int] = (bb.x1 + label_pad, label_y2 - label_pad - baseline)
        cv2.putText(
            image,
            label,
            text_origin,
            _FONT,
            label_font_scale,
            (255, 255, 255),
            1,
            _LINE_TYPE,
        )

        # ── Severity indicator dot (top-right corner) ──
        sev_color: Tuple[int, int, int] = _SEVERITY_BGR.get(
            violation.severity, (255, 255, 255)
        )
        dot_radius: int = max(6, int(bb.width * 0.025))
        dot_center: Tuple[int, int] = (bb.x2 - dot_radius - 4, bb.y1 + dot_radius + 4)
        cv2.circle(image, dot_center, dot_radius, sev_color, thickness=-1, lineType=_LINE_TYPE)
        cv2.circle(image, dot_center, dot_radius, (255, 255, 255), thickness=1, lineType=_LINE_TYPE)

    # ── Private: License-plate label ───────────

    def _draw_license_plate_label(
        self, image: np.ndarray, violation: ViolationRecord
    ) -> None:
        """Draw the license-plate string beneath the bounding box.

        Renders white text on a dark semi-transparent strip below the
        detection box.  If no plate was captured the method returns
        immediately.

        Args:
            image:     Canvas to draw on (modified in-place).
            violation: The violation record (may or may not have a plate).
        """
        if not violation.license_plate:
            return

        bb: BoundingBox = violation.bbox
        plate_text: str = f"  {violation.license_plate}  "
        plate_font_scale: float = self.font_scale * 0.85

        (tw, th), baseline = cv2.getTextSize(plate_text, _FONT, plate_font_scale, 1)
        pad: int = 4

        strip_x1: int = bb.x1
        strip_y1: int = bb.y2 + 2
        strip_x2: int = bb.x1 + tw + pad * 2
        strip_y2: int = strip_y1 + th + baseline + pad * 2

        # Clamp to image bounds
        img_h, img_w = image.shape[:2]
        strip_x2 = min(strip_x2, img_w)
        strip_y2 = min(strip_y2, img_h)

        # Dark background strip
        cv2.rectangle(
            image,
            (strip_x1, strip_y1),
            (strip_x2, strip_y2),
            (40, 35, 30),
            thickness=-1,
            lineType=_LINE_TYPE,
        )

        # White plate text
        text_origin: Tuple[int, int] = (strip_x1 + pad, strip_y2 - pad - baseline)
        cv2.putText(
            image,
            plate_text,
            text_origin,
            _FONT,
            plate_font_scale,
            (255, 255, 255),
            1,
            _LINE_TYPE,
        )

    # ── Private: Info panel ────────────────────

    def _draw_info_panel(
        self, image: np.ndarray, violations: List[ViolationRecord]
    ) -> None:
        """Composite a semi-transparent summary panel at the top of the image.

        Shows total violation count, the highest severity level found,
        and the timestamp of the most recent violation.

        Args:
            image:      Canvas to draw on (modified in-place).
            violations: All violations to summarise.
        """
        img_h, img_w = image.shape[:2]
        panel_height: int = 50
        panel_region: np.ndarray = image[0:panel_height, 0:img_w]

        # Create a solid-colour overlay and blend
        overlay: np.ndarray = np.full_like(panel_region, _PANEL_BG_COLOR, dtype=np.uint8)
        cv2.addWeighted(overlay, _PANEL_ALPHA, panel_region, 1.0 - _PANEL_ALPHA, 0, panel_region)

        # ── Determine highest severity ──
        highest: Severity = Severity.MINOR
        for sev in _SEVERITY_PRIORITY:
            if any(v.severity == sev for v in violations):
                highest = sev
                break

        # ── Gather summary text pieces ──
        total: int = len(violations)
        timestamp: str = violations[0].timestamp if violations else ""

        # Left section: violation count
        count_text: str = f"VIOLATIONS: {total}"
        cv2.putText(
            image,
            count_text,
            (12, 32),
            _FONT,
            self.font_scale * 1.0,
            _PANEL_ACCENT_COLOR,
            1,
            _LINE_TYPE,
        )

        # Centre section: severity badge
        sev_label: str = f"Severity: {highest.value}"
        sev_color: Tuple[int, int, int] = _SEVERITY_BGR.get(highest, (255, 255, 255))
        (sw, _), _ = cv2.getTextSize(sev_label, _FONT, self.font_scale * 0.9, 1)
        sev_x: int = (img_w - sw) // 2
        cv2.putText(
            image,
            sev_label,
            (sev_x, 32),
            _FONT,
            self.font_scale * 0.9,
            sev_color,
            1,
            _LINE_TYPE,
        )

        # Right section: timestamp
        if timestamp:
            ts_scale: float = self.font_scale * 0.75
            (tsw, _), _ = cv2.getTextSize(timestamp, _FONT, ts_scale, 1)
            cv2.putText(
                image,
                timestamp,
                (img_w - tsw - 12, 32),
                _FONT,
                ts_scale,
                _PANEL_TEXT_COLOR,
                1,
                _LINE_TYPE,
            )

        # Thin accent line at panel bottom
        cv2.line(image, (0, panel_height), (img_w, panel_height), _PANEL_ACCENT_COLOR, 1, _LINE_TYPE)

    # ── Private: Rounded rectangle helper ──────

    def _draw_rounded_rect(
        self,
        image: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        radius: int = 8,
        thickness: int = -1,
    ) -> None:
        """Draw a rectangle with rounded corners.

        Falls back to a plain rectangle when the corner radius exceeds
        half the shortest side.

        Args:
            image:     Canvas to draw on.
            pt1:       Top-left corner ``(x, y)``.
            pt2:       Bottom-right corner ``(x, y)``.
            color:     BGR colour tuple.
            radius:    Corner rounding radius in pixels.
            thickness: ``-1`` for filled, or positive for stroke width.
        """
        x1, y1 = pt1
        x2, y2 = pt2

        # Sanity: ensure pt1 < pt2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        w: int = x2 - x1
        h: int = y2 - y1

        # Clamp radius so it fits
        radius = min(radius, w // 2, h // 2)
        if radius <= 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, _LINE_TYPE)
            return

        # Inner rectangles (cross shape)
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness, _LINE_TYPE)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness, _LINE_TYPE)

        # Four corner circles
        corners: List[Tuple[int, int]] = [
            (x1 + radius, y1 + radius),   # top-left
            (x2 - radius, y1 + radius),   # top-right
            (x1 + radius, y2 - radius),   # bottom-left
            (x2 - radius, y2 - radius),   # bottom-right
        ]
        for center in corners:
            cv2.circle(image, center, radius, color, thickness, _LINE_TYPE)

    # ── Private: Text with background ──────────

    def _put_text_with_background(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float,
        color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int],
        padding: int = 5,
    ) -> None:
        """Render text with a filled background rectangle for readability.

        The background rectangle is auto-sized to fit the rendered text
        with the specified padding on every side.

        Args:
            image:      Canvas to draw on.
            text:       The string to render.
            position:   ``(x, y)`` of the text baseline origin.  The
                        background extends *above* this point.
            font_scale: Font scale factor.
            color:      Text colour (BGR).
            bg_color:   Background fill colour (BGR).
            padding:    Pixels of padding around the text.
        """
        (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, 1)

        x, y = position

        # Background rectangle: extends upward from baseline
        bg_x1: int = x - padding
        bg_y1: int = y - th - padding
        bg_x2: int = x + tw + padding
        bg_y2: int = y + baseline + padding

        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1, _LINE_TYPE)
        cv2.putText(image, text, (x, y), _FONT, font_scale, color, 1, _LINE_TYPE)
