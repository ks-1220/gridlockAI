"""
utils.py — Configuration, Constants & Helpers
==============================================
Central configuration hub for the Traffic Violation Detection System.
Defines violation types, severity mappings, color schemes, CLIP prompts,
detection thresholds, and shared utility functions.

Author: Team Gridlock | Flipkart Grid 6.0
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
#  Enums & Constants
# ──────────────────────────────────────────────

class ViolationType(str, Enum):
    """Enumeration of all detectable traffic violations."""
    HELMET = "No Helmet"
    SEATBELT = "No Seatbelt"
    TRIPLE_RIDING = "Triple Riding"
    WRONG_SIDE = "Wrong-Side Driving"
    RED_LIGHT = "Red Light Violation"
    STOP_LINE = "Stop Line Violation"
    ILLEGAL_PARKING = "Illegal Parking"


class Severity(str, Enum):
    """Violation severity levels for UI color-coding and prioritization."""
    CRITICAL = "Critical"
    MODERATE = "Moderate"
    MINOR = "Minor"


class VehicleType(str, Enum):
    """Supported vehicle categories from RT-DETR detections."""
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    BICYCLE = "bicycle"
    PERSON = "person"
    TRAFFIC_LIGHT = "traffic light"


# ──────────────────────────────────────────────
#  Violation ↔ Severity Mapping
# ──────────────────────────────────────────────

VIOLATION_SEVERITY: Dict[ViolationType, Severity] = {
    ViolationType.HELMET:          Severity.CRITICAL,
    ViolationType.SEATBELT:        Severity.CRITICAL,
    ViolationType.TRIPLE_RIDING:   Severity.CRITICAL,
    ViolationType.WRONG_SIDE:      Severity.CRITICAL,
    ViolationType.RED_LIGHT:       Severity.CRITICAL,
    ViolationType.STOP_LINE:       Severity.MODERATE,
    ViolationType.ILLEGAL_PARKING: Severity.MINOR,
}

# ──────────────────────────────────────────────
#  UI Color Palette (Dark Theme)
# ──────────────────────────────────────────────

# Dashboard background & surface colors
COLORS = {
    "bg_primary":    "#0f1117",
    "bg_secondary":  "#1a1c23",
    "bg_card":       "#21232d",
    "bg_hover":      "#2a2d3a",
    "text_primary":  "#e6e9ef",
    "text_secondary":"#8b8fa3",
    "accent_blue":   "#4f8ff7",
    "accent_purple": "#8b5cf6",
    "border":        "#2d3040",
}

# Severity → badge color mapping for violation cards
SEVERITY_COLORS: Dict[Severity, str] = {
    Severity.CRITICAL: "#ef4444",   # Red
    Severity.MODERATE: "#f97316",   # Orange
    Severity.MINOR:    "#eab308",   # Yellow
}

# Per-violation-type colors for charts & annotations
VIOLATION_COLORS: Dict[ViolationType, str] = {
    ViolationType.HELMET:          "#ef4444",
    ViolationType.SEATBELT:        "#f97316",
    ViolationType.TRIPLE_RIDING:   "#ec4899",
    ViolationType.WRONG_SIDE:      "#8b5cf6",
    ViolationType.RED_LIGHT:       "#dc2626",
    ViolationType.STOP_LINE:       "#eab308",
    ViolationType.ILLEGAL_PARKING: "#22c55e",
}

# BGR equivalents for OpenCV annotation drawing
VIOLATION_COLORS_BGR: Dict[ViolationType, Tuple[int, int, int]] = {
    vtype: tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
    for vtype, hex_color in VIOLATION_COLORS.items()
}

# ──────────────────────────────────────────────
#  CLIP Prompt Engineering
# ──────────────────────────────────────────────

# Each violation maps to (positive_prompts, negative_prompts)
# CLIP classifies by comparing similarity to both sets.
CLIP_PROMPTS: Dict[ViolationType, Tuple[List[str], List[str]]] = {
    ViolationType.HELMET: (
        [
            "a person riding a motorcycle without wearing a helmet",
            "a motorcyclist with no helmet on their head",
            "a rider on a two-wheeler without head protection",
            "bare-headed person on a motorbike",
        ],
        [
            "a person wearing a helmet while riding a motorcycle",
            "a motorcyclist with a helmet on their head",
            "a rider on a two-wheeler wearing protective headgear",
        ],
    ),
    ViolationType.SEATBELT: (
        [
            "a car driver not wearing a seatbelt",
            "a person driving without a seatbelt",
            "a driver without seatbelt in a car",
            "an unbelted driver behind the steering wheel",
        ],
        [
            "a car driver wearing a seatbelt",
            "a person driving with a seatbelt fastened",
            "a driver properly buckled up in a car",
        ],
    ),
    ViolationType.TRIPLE_RIDING: (
        [
            "three people riding on one motorcycle",
            "three persons on a single motorbike",
            "a motorcycle carrying three riders",
            "triple riding on a two-wheeler",
        ],
        [
            "one person riding a motorcycle",
            "two people on a motorcycle",
            "a single rider on a motorbike",
        ],
    ),
    ViolationType.WRONG_SIDE: (
        [
            "a vehicle driving on the wrong side of the road",
            "a car going against traffic flow",
            "a vehicle moving in the opposite direction on a road",
        ],
        [
            "a vehicle driving normally on the correct side",
            "a car following the traffic direction",
            "vehicles moving in the proper lane direction",
        ],
    ),
    ViolationType.RED_LIGHT: (
        [
            "a vehicle running a red traffic light",
            "a car crossing an intersection with a red signal",
            "a vehicle not stopping at a red light",
        ],
        [
            "a vehicle stopped at a red traffic light",
            "a car waiting at a red signal",
            "vehicles obeying the red traffic light",
        ],
    ),
    ViolationType.STOP_LINE: (
        [
            "a vehicle crossing the stop line at a traffic signal",
            "a car past the stop line at an intersection",
            "a vehicle over the zebra crossing line",
        ],
        [
            "a vehicle stopped before the stop line",
            "a car properly behind the stop line",
            "a vehicle waiting behind the zebra crossing",
        ],
    ),
    ViolationType.ILLEGAL_PARKING: (
        [
            "a vehicle parked illegally on the road",
            "a car parked in a no-parking zone",
            "an improperly parked vehicle blocking traffic",
            "a stationary vehicle parked on the roadside illegally",
        ],
        [
            "a vehicle moving on the road",
            "a car driving normally",
            "a properly parked vehicle in a parking lot",
        ],
    ),
}

# ──────────────────────────────────────────────
#  Detection & Model Configuration
# ──────────────────────────────────────────────

# RT-DETR model configuration
RTDETR_MODEL: str = "rtdetr-l"   # Lighter variant for CPU inference
RTDETR_CONFIDENCE: float = 0.35  # Minimum detection confidence
RTDETR_IOU_THRESHOLD: float = 0.45

# COCO class IDs relevant to traffic analysis
COCO_VEHICLE_CLASSES: Dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
}

# Mapping COCO class names to VehicleType enum
COCO_TO_VEHICLE: Dict[str, VehicleType] = {
    "person":        VehicleType.PERSON,
    "bicycle":       VehicleType.BICYCLE,
    "car":           VehicleType.CAR,
    "motorcycle":    VehicleType.MOTORCYCLE,
    "bus":           VehicleType.BUS,
    "truck":         VehicleType.TRUCK,
    "traffic light": VehicleType.TRAFFIC_LIGHT,
}

# Two-wheeler and four-wheeler groupings for rule engine
TWO_WHEELERS = {VehicleType.MOTORCYCLE, VehicleType.BICYCLE}
FOUR_WHEELERS = {VehicleType.CAR, VehicleType.BUS, VehicleType.TRUCK}

# CLIP model configuration
CLIP_MODEL: str = "ViT-B/32"   # Balanced speed/accuracy for CPU
CLIP_THRESHOLD: float = 0.55   # Minimum CLIP confidence for violation

# PaddleOCR configuration
OCR_LANGUAGE: str = "en"
OCR_USE_GPU: bool = False
OCR_DET_DB_THRESH: float = 0.3
OCR_DROP_SCORE: float = 0.5

# Image processing limits
MAX_IMAGE_SIZE: int = 1280      # Max dimension before resize
CROP_PADDING: int = 15          # Pixels padding around crops

# Inference timeout (seconds)
INFERENCE_TIMEOUT: int = 10


# ──────────────────────────────────────────────
#  Data Classes
# ──────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union with another box."""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: "BoundingBox", threshold: float = 0.6) -> bool:
        """Check if this box substantially contains another box."""
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        return (inter / other.area) >= threshold if other.area > 0 else False


@dataclass
class Detection:
    """Single object detection from RT-DETR."""
    bbox: BoundingBox
    class_name: str
    class_id: int
    confidence: float
    vehicle_type: Optional[VehicleType] = None
    track_id: Optional[str] = None

    def __post_init__(self):
        self.vehicle_type = COCO_TO_VEHICLE.get(self.class_name)


@dataclass
class ViolationRecord:
    """Complete violation record for a single detected infraction."""
    violation_id: str
    vehicle_id: str
    violation_type: ViolationType
    severity: Severity
    confidence: float
    detection_confidence: float
    clip_confidence: float
    bbox: BoundingBox
    license_plate: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    metadata: Dict[str, Any] = field(default_factory=dict)
    vehicle_crop: Optional[np.ndarray] = None


    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/CSV export."""
        return {
            "violation_id": self.violation_id,
            "vehicle_id": self.vehicle_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "clip_confidence": round(self.clip_confidence, 4),
            "license_plate": self.license_plate or "N/A",
            "bbox": self.bbox.as_tuple(),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisResult:
    """Aggregated result from a full image analysis pipeline run."""
    image_id: str
    violations: List[ViolationRecord]
    detections: List[Detection]
    annotated_image: Optional[np.ndarray] = None
    processing_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def has_violations(self) -> bool:
        return self.violation_count > 0

    def violations_by_type(self) -> Dict[ViolationType, int]:
        """Count violations grouped by type."""
        counts: Dict[ViolationType, int] = {}
        for v in self.violations:
            counts[v.violation_type] = counts.get(v.violation_type, 0) + 1
        return counts

    def violations_by_severity(self) -> Dict[Severity, int]:
        """Count violations grouped by severity."""
        counts: Dict[Severity, int] = {}
        for v in self.violations:
            counts[v.severity] = counts.get(v.severity, 0) + 1
        return counts

    def max_severity(self) -> Optional[Severity]:
        """Return the highest severity found across all violations."""
        if not self.violations:
            return None
        severity_order = [Severity.CRITICAL, Severity.MODERATE, Severity.MINOR]
        for sev in severity_order:
            if any(v.severity == sev for v in self.violations):
                return sev
        return None


# ──────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────

def generate_id(prefix: str = "VEH") -> str:
    """Generate a short unique ID for vehicles/violations."""
    hash_input = f"{prefix}_{time.time_ns()}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8].upper()
    return f"{prefix}-{short_hash}"


def resize_image_if_needed(
    image: np.ndarray,
    max_size: int = MAX_IMAGE_SIZE
) -> np.ndarray:
    """Resize image proportionally if largest dimension exceeds max_size."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    from cv2 import resize, INTER_AREA
    return resize(image, (new_w, new_h), interpolation=INTER_AREA)


def crop_region(
    image: np.ndarray,
    bbox: BoundingBox,
    padding: int = CROP_PADDING
) -> np.ndarray:
    """Crop a region from an image with optional padding."""
    h, w = image.shape[:2]
    x1 = max(0, bbox.x1 - padding)
    y1 = max(0, bbox.y1 - padding)
    x2 = min(w, bbox.x2 + padding)
    y2 = min(h, bbox.y2 + padding)
    return image[y1:y2, x1:x2]


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV-compatible numpy array (BGR)."""
    rgb = np.array(pil_image.convert("RGB"))
    return rgb[:, :, ::-1].copy()  # RGB → BGR


def numpy_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR numpy array to PIL Image."""
    rgb = cv_image[:, :, ::-1]  # BGR → RGB
    return Image.fromarray(rgb)


def image_to_bytes(image: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode numpy image to bytes for download."""
    import cv2
    success, encoded = cv2.imencode(fmt, image)
    if not success:
        raise ValueError(f"Failed to encode image to {fmt}")
    return encoded.tobytes()


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute IoU between two bounding boxes (convenience wrapper)."""
    return box1.iou(box2)


def format_confidence(conf: float) -> str:
    """Format confidence score as percentage string."""
    return f"{conf * 100:.1f}%"


def get_severity_emoji(severity: Severity) -> str:
    """Return emoji indicator for severity level."""
    return {
        Severity.CRITICAL: "🔴",
        Severity.MODERATE: "🟠",
        Severity.MINOR:    "🟡",
    }.get(severity, "⚪")


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))
