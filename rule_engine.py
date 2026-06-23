"""
rule_engine.py — Traffic Violation Rule Engine
================================================
Central decision-making module that combines RT-DETR detection results with
CLIP zero-shot classification scores to determine traffic violations.

Each violation type has a dedicated checker method guarded by vehicle-class
constraints (e.g., helmet checks only on two-wheelers). A combined confidence
score — ``clip_score × detection_score`` — is compared against a configurable
threshold to decide whether a violation is flagged.

Architecture
------------
    RT-DETR Detections ──┐
                         ├──► RuleEngine.evaluate() ──► List[ViolationRecord]
    CLIP Scores ─────────┘

Author: Team Gridlock | Flipkart Grid 6.0
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from utils import (
    BoundingBox,
    CLIP_THRESHOLD,
    Detection,
    FOUR_WHEELERS,
    Severity,
    TWO_WHEELERS,
    VehicleType,
    VIOLATION_SEVERITY,
    ViolationRecord,
    ViolationType,
    clamp,
    format_confidence,
    generate_id,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Internal Constants
# ──────────────────────────────────────────────

# Minimum IoU between a person bbox and a vehicle bbox to consider them
# "associated" (e.g., rider on a motorcycle).
_PERSON_VEHICLE_OVERLAP_THRESHOLD: float = 0.15

# Maximum pixel distance (center-to-center) for a traffic light to be
# considered relevant to a vehicle.
_TRAFFIC_LIGHT_PROXIMITY_PX: int = 300

# Severity priority used for deterministic sorting (lower = more severe).
_SEVERITY_PRIORITY: Dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.MODERATE: 1,
    Severity.MINOR: 2,
}

# Violation types that legally require a license plate for enforcement.
_LICENSE_PLATE_REQUIRED_VIOLATIONS = frozenset({
    ViolationType.RED_LIGHT,
    ViolationType.WRONG_SIDE,
})

# Person-count threshold above which triple-riding is suspected independent
# of CLIP (serves as a strong prior).
_TRIPLE_RIDING_PERSON_THRESHOLD: int = 3


class RuleEngine:
    """Evaluate detected objects against traffic-rule violation criteria.

    The engine fuses two complementary signals:

    1. **Detection confidence** — how certain the object detector (RT-DETR)
       is that the bounding box contains a vehicle / person / traffic-light.
    2. **CLIP classification score** — how likely a zero-shot CLIP model
       considers the cropped region to depict a specific violation.

    A combined score ``clip_score × detection_score`` (clamped to [0, 1]) is
    compared against ``confidence_threshold`` to decide whether a violation
    record is emitted.

    Parameters
    ----------
    confidence_threshold : float, optional
        Minimum combined confidence to flag a violation (default ``0.5``).
    """

    # ── Construction ──────────────────────────

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        """Initialise the rule engine with a combined-confidence threshold.

        Parameters
        ----------
        confidence_threshold : float
            Minimum ``clip_conf × detection_conf`` score required to emit a
            violation.  Values outside (0, 1] are clamped automatically.
        """
        self.confidence_threshold: float = clamp(confidence_threshold, 0.01, 1.0)

        # Mapping from ViolationType → handler callable.  Each handler receives
        # a ``Detection``, a CLIP-scores dict for that detection, and returns
        # an optional ``ViolationRecord``.  Methods that need extra context
        # (e.g., nearby persons) are wrapped at call-time in ``evaluate()``.
        self._violation_handlers: Dict[
            ViolationType,
            Callable[..., Optional[ViolationRecord]],
        ] = {
            ViolationType.HELMET: self._check_helmet,
            ViolationType.SEATBELT: self._check_seatbelt,
            ViolationType.TRIPLE_RIDING: self._check_triple_riding,
            ViolationType.WRONG_SIDE: self._check_wrong_side,
            ViolationType.RED_LIGHT: self._check_red_light,
            ViolationType.STOP_LINE: self._check_stop_line,
            ViolationType.ILLEGAL_PARKING: self._check_illegal_parking,
        }

        logger.info(
            "RuleEngine initialised — threshold=%.2f, handlers=%d",
            self.confidence_threshold,
            len(self._violation_handlers),
        )

    # ── Public API ────────────────────────────

    def evaluate(
        self,
        detections: List[Detection],
        clip_results: Dict[str, Dict[ViolationType, float]],
        ocr_results: Dict[str, str],
        active_violations: Optional[List[ViolationType]] = None,
        image: Optional[np.ndarray] = None,
    ) -> List[ViolationRecord]:
        """Run every applicable violation check on a set of detections.

        Parameters
        ----------
        detections : List[Detection]
            All object detections from a single frame / image.
        clip_results : Dict[str, Dict[ViolationType, float]]
            CLIP scores keyed by a detection identifier (stringified index).
            Each value maps ``ViolationType`` → score in [0, 1].
        ocr_results : Dict[str, str]
            OCR-extracted license-plate text keyed by the same detection
            identifier.
        active_violations : Optional[List[ViolationType]]
            If provided, only these violation types are checked (UI filter).
        image : Optional[np.ndarray]
            The raw image/frame array, used to extract visual crop evidence.

        Returns
        -------
        List[ViolationRecord]
            Aggregated, de-duplicated violation records sorted by severity.
        """
        if not detections:
            logger.debug("evaluate() called with empty detections list.")
            return []

        # Determine which violations to check.
        checks: List[ViolationType] = (
            list(active_violations) if active_violations else list(ViolationType)
        )

        raw_violations: List[ViolationRecord] = []

        for idx, detection in enumerate(detections):
            det_key = str(idx)
            clip_scores: Dict[ViolationType, float] = clip_results.get(det_key, {})

            # Skip non-vehicle detections (persons, traffic lights are
            # context — not violation subjects themselves).
            if detection.vehicle_type in {
                VehicleType.PERSON,
                VehicleType.TRAFFIC_LIGHT,
                None,
            }:
                continue

            # Pre-compute contextual data once per vehicle.
            nearby_persons = self._count_persons_near_vehicle(detection, detections)
            nearby_lights = self._find_nearby_traffic_lights(detection, detections)

            # Assign a stable vehicle-level ID for grouping (reuse track ID if available).
            vehicle_id = detection.track_id if detection.track_id else generate_id("VEH")

            # Resolve license plate (if OCR matched).
            license_plate: Optional[str] = ocr_results.get(det_key)

            # Prepare vehicle crop if image is provided
            vehicle_crop = None
            if image is not None:
                from utils import crop_region
                try:
                    vehicle_crop = crop_region(image, detection.bbox, padding=10)
                except Exception:
                    pass

            for vtype in checks:
                try:
                    record = self._dispatch_check(
                        vtype=vtype,
                        detection=detection,
                        clip_scores=clip_scores,
                        nearby_persons=nearby_persons,
                        nearby_lights=nearby_lights,
                        image_width=self._estimate_image_width(detections),
                    )
                except Exception:
                    logger.exception(
                        "Unhandled error in %s check for detection %d",
                        vtype.value,
                        idx,
                    )
                    continue

                if record is not None:
                    # Attach shared vehicle metadata.
                    record.vehicle_id = vehicle_id
                    record.license_plate = license_plate
                    record.vehicle_crop = vehicle_crop

                    # Mark enforcement-critical violations.
                    if vtype in _LICENSE_PLATE_REQUIRED_VIOLATIONS:
                        record.metadata["license_plate_required"] = True

                    raw_violations.append(record)
                    logger.info(
                        "Violation detected — type=%s conf=%s vehicle=%s",
                        vtype.value,
                        format_confidence(record.confidence),
                        vehicle_id,
                    )

        return self._aggregate_violations(raw_violations)

    # ── Violation Checkers ────────────────────

    def _check_helmet(
        self,
        detection: Detection,
        clip_scores: Dict[ViolationType, float],
    ) -> Optional[ViolationRecord]:
        """Check for missing helmet on a two-wheeler rider.

        Parameters
        ----------
        detection : Detection
            The vehicle detection to evaluate.
        clip_scores : Dict[ViolationType, float]
            CLIP classification scores for this detection crop.

        Returns
        -------
        Optional[ViolationRecord]
            A violation record if the combined confidence exceeds the
            threshold; ``None`` otherwise.
        """
        if detection.vehicle_type not in TWO_WHEELERS:
            return None

        clip_conf = clip_scores.get(ViolationType.HELMET, 0.0)
        if clip_conf < CLIP_THRESHOLD:
            return None

        combined = self._compute_combined_confidence(detection.confidence, clip_conf)
        if combined < self.confidence_threshold:
            return None

        return self._build_record(
            vtype=ViolationType.HELMET,
            detection=detection,
            clip_conf=clip_conf,
            combined_conf=combined,
        )

    def _check_seatbelt(
        self,
        detection: Detection,
        clip_scores: Dict[ViolationType, float],
    ) -> Optional[ViolationRecord]:
        """Check for missing seatbelt in a four-wheeler.

        Parameters
        ----------
        detection : Detection
            The vehicle detection to evaluate.
        clip_scores : Dict[ViolationType, float]
            CLIP classification scores for this detection crop.

        Returns
        -------
        Optional[ViolationRecord]
            A violation record if the combined confidence exceeds the
            threshold; ``None`` otherwise.
        """
        if detection.vehicle_type not in FOUR_WHEELERS:
            return None

        clip_conf = clip_scores.get(ViolationType.SEATBELT, 0.0)
        if clip_conf < CLIP_THRESHOLD:
            return None

        combined = self._compute_combined_confidence(detection.confidence, clip_conf)
        if combined < self.confidence_threshold:
            return None

        return self._build_record(
            vtype=ViolationType.SEATBELT,
            detection=detection,
            clip_conf=clip_conf,
            combined_conf=combined,
        )

    def _check_triple_riding(
        self,
        detection: Detection,
        clip_scores: Dict[ViolationType, float],
        nearby_persons: int = 0,
    ) -> Optional[ViolationRecord]:
        """Check for three-or-more riders on a motorcycle.

        Combines the CLIP visual signal with a spatial person-count heuristic:
        if the number of persons whose bounding boxes overlap with the
        motorcycle exceeds ``_TRIPLE_RIDING_PERSON_THRESHOLD``, the CLIP
        threshold is relaxed to provide a softer prior.

        Parameters
        ----------
        detection : Detection
            The vehicle detection to evaluate.
        clip_scores : Dict[ViolationType, float]
            CLIP classification scores for this detection crop.
        nearby_persons : int
            Count of ``person`` detections overlapping the vehicle bbox.

        Returns
        -------
        Optional[ViolationRecord]
            A violation record if the combined confidence exceeds the
            threshold; ``None`` otherwise.
        """
        if detection.vehicle_type not in TWO_WHEELERS:
            return None

        clip_conf = clip_scores.get(ViolationType.TRIPLE_RIDING, 0.0)

        # Allow a relaxed CLIP gate when the spatial heuristic strongly
        # supports triple-riding (≥ 3 persons on/near the vehicle).
        effective_clip_threshold = (
            CLIP_THRESHOLD * 0.7
            if nearby_persons >= _TRIPLE_RIDING_PERSON_THRESHOLD
            else CLIP_THRESHOLD
        )

        if clip_conf < effective_clip_threshold:
            return None

        # Boost combined confidence when person count independently confirms.
        person_boost = min(nearby_persons / 3.0, 1.0) * 0.15 if nearby_persons >= 3 else 0.0
        combined = self._compute_combined_confidence(detection.confidence, clip_conf)
        combined = clamp(combined + person_boost)

        if combined < self.confidence_threshold:
            return None

        record = self._build_record(
            vtype=ViolationType.TRIPLE_RIDING,
            detection=detection,
            clip_conf=clip_conf,
            combined_conf=combined,
        )
        record.metadata["nearby_persons"] = nearby_persons
        return record

    def _check_wrong_side(
        self,
        detection: Detection,
        clip_scores: Dict[ViolationType, float],
        image_width: int = 1280,
    ) -> Optional[ViolationRecord]:
        """Check for wrong-side driving using position heuristic + CLIP.

        Heuristic: if the vehicle's centre x-coordinate lies in the
        left-most 35 % of the frame (suggesting it is on the oncoming
        lane in a standard left-hand-traffic setup), the CLIP threshold
        is relaxed slightly.

        Parameters
        ----------
        detection : Detection
            The vehicle detection to evaluate.
        clip_scores : Dict[ViolationType, float]
            CLIP classification scores for this detection crop.
        image_width : int
            Width of the source image in pixels (used for the positional
            heuristic).

        Returns
        -------
        Optional[ViolationRecord]
            A violation record if the combined confidence exceeds the
            threshold; ``None`` otherwise.
        """
        if detection.vehicle_type in {VehicleType.PERSON, VehicleType.TRAFFIC_LIGHT, None}:
            return None

        clip_conf = clip_scores.get(ViolationType.WRONG_SIDE, 0.0)

        # Positional heuristic — vehicle in the left third of the frame.
        center_x, _ = detection.bbox.center
        on_wrong_side_heuristic = center_x < (image_width * 0.35)

        effective_clip_threshold = (
            CLIP_THRESHOLD * 0.85
            if on_wrong_side_heuristic
            else CLIP_THRESHOLD
        )

        if clip_conf < effective_clip_threshold:
            return None

        combined = self._compute_combined_confidence(detection.confidence, clip_conf)

        # Slight boost when the spatial heuristic agrees.
        if on_wrong_side_heuristic:
            combined = clamp(combined + 0.05)

        if combined < self.confidence_threshold:
            return None

        record = self._build_record(
            vtype=ViolationType.WRONG_SIDE,
            detection=detection,
            clip_conf=clip_conf,
            combined_conf=combined,
        )
        record.metadata["license_plate_required"] = True
        record.metadata["position_heuristic"] = on_wrong_side_heuristic
        record.metadata["center_x_ratio"] = round(center_x / max(image_width, 1), 3)
        return record

    def _check_red_light(
        self,
        detection: Detection,
        clip_scores: Dict[ViolationType, float],
        traffic_lights: Optional[List[Detection]] = None,
    ) -> Optional[ViolationRecord]:
        """Check for red-light violation given nearby traffic-light context.

        A red-light violation is only flagged when **at least one traffic
        light detection** exists in close proximity to the vehicle.  This
        prevents false positives on open roads with no signals.

        Parameters
        ----------
        detection : Detection
            The vehicle detection to evaluate.
        clip_scores : Dict[ViolationType, float]
            CLIP classification scores for this detection crop.
        traffic_lights : Optional[List[Detection]]
            Traffic-light detections in the vicinity of this vehicle.

        Returns
        -------
        Optional[ViolationRecord]
            A violation record if the combined confidence exceeds the
            threshold; ``None`` otherwise.
        """
        if detection.vehicle_type in {VehicleType.PERSON, VehicleType.TRAFFIC_LIGHT, None}:
            return None

        # Require at least one nearby traffic light as physical evidence.
        if not traffic_lights:
            return None

        clip_conf = clip_scores.get(ViolationType.RED_LIGHT, 0.0)
        if clip_conf < CLIP_THRESHOLD:
            return None

        combined = self._compute_combined_confidence(detection.confidence, clip_conf)
        if combined < self.confidence_threshold:
            return None

        record = self._build_record(
            vtype=ViolationType.RED_LIGHT,
            detection=detection,
            clip_conf=clip_conf,
            combined_conf=combined,
        )
        record.metadata["license_plate_required"] = True
        record.metadata["nearby_traffic_lights"] = len(traffic_lights)
        return record

    def _check_stop_line(
        self,
        detection: Detection,
        clip_scores: Dict[ViolationType, float],
    ) -> Optional[ViolationRecord]:
        """Check for stop-line / zebra-crossing encroachment.

        This is a zone-based check: the CLIP model decides whether the
        vehicle crop appears to be past the stop line.  No additional
        spatial heuristic is applied here because stop-line positions
        vary wildly across intersections.

        Parameters
        ----------
        detection : Detection
            The vehicle detection to evaluate.
        clip_scores : Dict[ViolationType, float]
            CLIP classification scores for this detection crop.

        Returns
        -------
        Optional[ViolationRecord]
            A violation record if the combined confidence exceeds the
            threshold; ``None`` otherwise.
        """
        if detection.vehicle_type in {VehicleType.PERSON, VehicleType.TRAFFIC_LIGHT, None}:
            return None

        clip_conf = clip_scores.get(ViolationType.STOP_LINE, 0.0)
        if clip_conf < CLIP_THRESHOLD:
            return None

        combined = self._compute_combined_confidence(detection.confidence, clip_conf)
        if combined < self.confidence_threshold:
            return None

        return self._build_record(
            vtype=ViolationType.STOP_LINE,
            detection=detection,
            clip_conf=clip_conf,
            combined_conf=combined,
        )

    def _check_illegal_parking(
        self,
        detection: Detection,
        clip_scores: Dict[ViolationType, float],
    ) -> Optional[ViolationRecord]:
        """Check for illegal parking using static-vehicle heuristic + CLIP.

        A low bounding-box aspect ratio (wide but not tall) combined with
        a high CLIP score is used as a soft indicator that the vehicle is
        stationary and parked.

        Parameters
        ----------
        detection : Detection
            The vehicle detection to evaluate.
        clip_scores : Dict[ViolationType, float]
            CLIP classification scores for this detection crop.

        Returns
        -------
        Optional[ViolationRecord]
            A violation record if the combined confidence exceeds the
            threshold; ``None`` otherwise.
        """
        if detection.vehicle_type in {VehicleType.PERSON, VehicleType.TRAFFIC_LIGHT, None}:
            return None

        clip_conf = clip_scores.get(ViolationType.ILLEGAL_PARKING, 0.0)

        # Static-vehicle heuristic: wider-than-tall boxes are more likely
        # to be parked (vehicles viewed from the side).
        aspect_ratio = (
            detection.bbox.width / max(detection.bbox.height, 1)
        )
        is_likely_static = aspect_ratio > 1.3

        effective_clip_threshold = (
            CLIP_THRESHOLD * 0.9 if is_likely_static else CLIP_THRESHOLD
        )

        if clip_conf < effective_clip_threshold:
            return None

        combined = self._compute_combined_confidence(detection.confidence, clip_conf)
        if combined < self.confidence_threshold:
            return None

        record = self._build_record(
            vtype=ViolationType.ILLEGAL_PARKING,
            detection=detection,
            clip_conf=clip_conf,
            combined_conf=combined,
        )
        record.metadata["aspect_ratio"] = round(aspect_ratio, 3)
        record.metadata["static_heuristic"] = is_likely_static
        return record

    # ── Dispatch Helper ───────────────────────

    def _dispatch_check(
        self,
        vtype: ViolationType,
        detection: Detection,
        clip_scores: Dict[ViolationType, float],
        nearby_persons: int,
        nearby_lights: List[Detection],
        image_width: int,
    ) -> Optional[ViolationRecord]:
        """Route a violation type to its specific checker with extra context.

        This method exists so that ``evaluate()`` doesn't need to know the
        unique signatures of each checker — extra arguments are passed only
        to methods that require them.

        Parameters
        ----------
        vtype : ViolationType
            The violation type to check.
        detection : Detection
            The vehicle detection to evaluate.
        clip_scores : Dict[ViolationType, float]
            CLIP classification scores for this detection crop.
        nearby_persons : int
            Count of persons overlapping the vehicle bbox.
        nearby_lights : List[Detection]
            Traffic-light detections near the vehicle.
        image_width : int
            Source image width in pixels.

        Returns
        -------
        Optional[ViolationRecord]
            The result of the specific violation checker.
        """
        if vtype == ViolationType.TRIPLE_RIDING:
            return self._check_triple_riding(detection, clip_scores, nearby_persons)
        if vtype == ViolationType.WRONG_SIDE:
            return self._check_wrong_side(detection, clip_scores, image_width)
        if vtype == ViolationType.RED_LIGHT:
            return self._check_red_light(detection, clip_scores, nearby_lights)

        # Standard two-argument checkers.
        handler = self._violation_handlers.get(vtype)
        if handler is None:
            logger.warning("No handler registered for violation type: %s", vtype.value)
            return None
        return handler(detection, clip_scores)

    # ── Confidence Fusion ─────────────────────

    @staticmethod
    def _compute_combined_confidence(
        detection_conf: float,
        clip_conf: float,
    ) -> float:
        """Fuse detection and CLIP confidences into a single score.

        The combined score is the product of both confidences, clamped to
        the [0, 1] interval.  This naturally penalises cases where either
        signal is weak.

        Parameters
        ----------
        detection_conf : float
            Object-detector confidence in [0, 1].
        clip_conf : float
            CLIP zero-shot classification confidence in [0, 1].

        Returns
        -------
        float
            Combined confidence score in [0, 1].
        """
        return clamp(clip_conf * detection_conf)

    # ── Spatial Heuristics ────────────────────

    @staticmethod
    def _count_persons_near_vehicle(
        vehicle: Detection,
        all_detections: List[Detection],
    ) -> int:
        """Count person detections whose bboxes significantly overlap the vehicle.

        A person is considered "near" (e.g., riding on) a vehicle if the
        intersection of their bounding box with the vehicle's bounding box
        divided by the person's bounding-box area exceeds
        ``_PERSON_VEHICLE_OVERLAP_THRESHOLD``.

        Parameters
        ----------
        vehicle : Detection
            The vehicle detection whose vicinity to scan.
        all_detections : List[Detection]
            Full list of detections in the frame.

        Returns
        -------
        int
            Number of person detections overlapping the vehicle bbox.
        """
        count = 0
        for det in all_detections:
            if det.vehicle_type != VehicleType.PERSON:
                continue

            # Compute intersection area / person area.
            xi1 = max(vehicle.bbox.x1, det.bbox.x1)
            yi1 = max(vehicle.bbox.y1, det.bbox.y1)
            xi2 = min(vehicle.bbox.x2, det.bbox.x2)
            yi2 = min(vehicle.bbox.y2, det.bbox.y2)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            person_area = det.bbox.area

            if person_area > 0 and (inter_area / person_area) >= _PERSON_VEHICLE_OVERLAP_THRESHOLD:
                count += 1

        return count

    @staticmethod
    def _find_nearby_traffic_lights(
        vehicle: Detection,
        all_detections: List[Detection],
    ) -> List[Detection]:
        """Find traffic-light detections within proximity of a vehicle.

        Proximity is measured as the Euclidean distance between the centre
        points of the two bounding boxes, thresholded by
        ``_TRAFFIC_LIGHT_PROXIMITY_PX``.

        Parameters
        ----------
        vehicle : Detection
            The vehicle detection to search around.
        all_detections : List[Detection]
            Full list of detections in the frame.

        Returns
        -------
        List[Detection]
            Traffic-light detections deemed close to the vehicle.
        """
        vx, vy = vehicle.bbox.center
        nearby: List[Detection] = []

        for det in all_detections:
            if det.vehicle_type != VehicleType.TRAFFIC_LIGHT:
                continue

            tx, ty = det.bbox.center
            distance = ((vx - tx) ** 2 + (vy - ty) ** 2) ** 0.5

            if distance <= _TRAFFIC_LIGHT_PROXIMITY_PX:
                nearby.append(det)

        return nearby

    # ── Aggregation & Sorting ─────────────────

    @staticmethod
    def _aggregate_violations(
        violations: List[ViolationRecord],
    ) -> List[ViolationRecord]:
        """Group violations by vehicle, sort by severity (critical first).

        Within each vehicle group, violations are sorted so the most
        severe appears first.  Groups themselves are also ordered by the
        maximum severity encountered in the group.

        Parameters
        ----------
        violations : List[ViolationRecord]
            Raw, unordered violation records.

        Returns
        -------
        List[ViolationRecord]
            De-duplicated and severity-sorted violation records.
        """
        if not violations:
            return []

        # Group by vehicle_id.
        groups: Dict[str, List[ViolationRecord]] = defaultdict(list)
        for v in violations:
            groups[v.vehicle_id].append(v)

        # Sort within each group by severity priority (critical → minor).
        for records in groups.values():
            records.sort(
                key=lambda r: _SEVERITY_PRIORITY.get(r.severity, 99),
            )

        # Order groups by the best (lowest-value) severity in each group.
        sorted_groups = sorted(
            groups.values(),
            key=lambda recs: _SEVERITY_PRIORITY.get(recs[0].severity, 99),
        )

        # Flatten back into a single list.
        return [record for group in sorted_groups for record in group]

    # ── Internal Helpers ──────────────────────

    @staticmethod
    def _build_record(
        vtype: ViolationType,
        detection: Detection,
        clip_conf: float,
        combined_conf: float,
    ) -> ViolationRecord:
        """Construct a ``ViolationRecord`` with standard fields populated.

        Parameters
        ----------
        vtype : ViolationType
            The violation being recorded.
        detection : Detection
            Source detection for bbox and detection confidence.
        clip_conf : float
            CLIP classification confidence.
        combined_conf : float
            Fused confidence score.

        Returns
        -------
        ViolationRecord
            Fully populated violation record (vehicle_id and license_plate
            are set to placeholders and should be overwritten by the
            caller).
        """
        return ViolationRecord(
            violation_id=generate_id("VIO"),
            vehicle_id="",  # Overwritten by evaluate()
            violation_type=vtype,
            severity=VIOLATION_SEVERITY.get(vtype, Severity.MINOR),
            confidence=combined_conf,
            detection_confidence=detection.confidence,
            clip_confidence=clip_conf,
            bbox=detection.bbox,
            metadata={
                "vehicle_class": (
                    detection.vehicle_type.value
                    if detection.vehicle_type
                    else detection.class_name
                ),
            },
        )

    @staticmethod
    def _estimate_image_width(detections: List[Detection]) -> int:
        """Estimate image width from the rightmost detection edge.

        When the original image dimensions are not passed, this provides
        a reasonable fallback by finding the largest ``x2`` coordinate
        across all bounding boxes.

        Parameters
        ----------
        detections : List[Detection]
            All detections in the current frame.

        Returns
        -------
        int
            Estimated image width (defaults to 1280 if no detections).
        """
        if not detections:
            return 1280
        return max(det.bbox.x2 for det in detections)
