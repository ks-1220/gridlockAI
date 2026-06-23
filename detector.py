"""
detector.py — RT-DETR Object Detection Wrapper
================================================
Wraps the Ultralytics RT-DETR model for real-time traffic
object detection. Filters results to traffic-relevant COCO
classes and returns structured Detection dataclass instances.

Architecture Note:
    RT-DETR (Real-Time Detection Transformer) combines the accuracy
    of DETR-style transformer decoders with efficient hybrid encoder
    design, making it suitable for CPU-bound deployment scenarios.

Author: Team Gridlock | Flipkart Grid 6.0
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from utils import (
    Detection,
    BoundingBox,
    VehicleType,
    COCO_VEHICLE_CLASSES,
    COCO_TO_VEHICLE,
    RTDETR_MODEL,
    RTDETR_CONFIDENCE,
    RTDETR_IOU_THRESHOLD,
    MAX_IMAGE_SIZE,
    resize_image_if_needed,
)

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    RT-DETR based vehicle and traffic object detector.

    Loads the RT-DETR-L model from Ultralytics and provides
    high-level methods for detecting, filtering, and categorizing
    traffic-relevant objects (vehicles, persons, traffic lights).

    Attributes:
        model: The loaded RT-DETR model instance.
        is_loaded: Whether the model has been successfully loaded.
        model_name: Name/path of the model variant in use.
    """

    def __init__(self) -> None:
        """Initialize detector with model reference (lazy-loaded)."""
        self.model = None
        self.is_loaded: bool = False
        self.model_name: str = RTDETR_MODEL

    def load_model(self) -> None:
        """
        Load the RT-DETR model weights.

        This method should be called once and its result cached
        (via @st.cache_resource in the Streamlit app layer).

        Raises:
            RuntimeError: If model loading fails (e.g., download error).
        """
        try:
            logger.info("Loading RT-DETR model: %s", self.model_name)
            start = time.perf_counter()

            from ultralytics import RTDETR
            self.model = RTDETR(self.model_name)
            self.is_loaded = True

            elapsed = time.perf_counter() - start
            logger.info("RT-DETR model loaded in %.2fs", elapsed)

        except Exception as e:
            logger.error("Failed to load RT-DETR model: %s", e)
            self.is_loaded = False
            raise RuntimeError(
                f"Could not load RT-DETR model '{self.model_name}'. "
                f"Ensure 'ultralytics' is installed and network is available. "
                f"Error: {e}"
            ) from e

    def detect(
        self,
        image: np.ndarray,
        confidence: float = RTDETR_CONFIDENCE,
    ) -> List[Detection]:
        """
        Run object detection on an image.

        Args:
            image: Input image as a BGR numpy array (OpenCV format).
            confidence: Minimum confidence threshold for detections.

        Returns:
            List of Detection objects for traffic-relevant classes.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If the input image is invalid.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first or use get_detector()."
            )

        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        # Resize to limit inference time on CPU
        processed = resize_image_if_needed(image, MAX_IMAGE_SIZE)

        logger.info(
            "Running RT-DETR inference (conf=%.2f, iou=%.2f) on %s image",
            confidence, RTDETR_IOU_THRESHOLD, processed.shape[:2],
        )
        start = time.perf_counter()

        try:
            results = self.model(
                processed,
                conf=confidence,
                iou=RTDETR_IOU_THRESHOLD,
                verbose=False,
            )
        except Exception as e:
            logger.error("RT-DETR inference failed: %s", e)
            return []

        elapsed = time.perf_counter() - start
        logger.info("RT-DETR inference completed in %.3fs", elapsed)

        # If image was resized, compute scale factors to map boxes
        # back to original coordinates
        orig_h, orig_w = image.shape[:2]
        proc_h, proc_w = processed.shape[:2]
        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())

                # Filter: only traffic-relevant COCO classes
                if class_id not in COCO_VEHICLE_CLASSES:
                    continue

                class_name = COCO_VEHICLE_CLASSES[class_id]
                conf = float(boxes.conf[i].item())

                # Extract and scale bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                bbox = BoundingBox(
                    x1=int(x1 * scale_x),
                    y1=int(y1 * scale_y),
                    x2=int(x2 * scale_x),
                    y2=int(y2 * scale_y),
                )

                detection = Detection(
                    bbox=bbox,
                    class_name=class_name,
                    class_id=class_id,
                    confidence=conf,
                )
                detections.append(detection)

        logger.info(
            "Detected %d traffic-relevant objects out of %d total",
            len(detections),
            sum(len(r.boxes) for r in results if r.boxes is not None),
        )

        return detections

    def filter_by_class(
        self,
        detections: List[Detection],
        classes: List[str],
    ) -> List[Detection]:
        """
        Filter detections to include only specified class names.

        Args:
            detections: List of Detection objects to filter.
            classes: List of class names to keep (e.g., ["car", "motorcycle"]).

        Returns:
            Filtered list of Detection objects.
        """
        class_set = set(classes)
        return [d for d in detections if d.class_name in class_set]

    def get_vehicles(self, detections: List[Detection]) -> List[Detection]:
        """
        Return only vehicle detections (car, motorcycle, bus, truck, bicycle).

        Args:
            detections: List of all detections.

        Returns:
            Filtered list containing only vehicle-class detections.
        """
        vehicle_classes = {"car", "motorcycle", "bus", "truck", "bicycle"}
        return self.filter_by_class(detections, list(vehicle_classes))

    def get_persons(self, detections: List[Detection]) -> List[Detection]:
        """
        Return only person detections.

        Args:
            detections: List of all detections.

        Returns:
            Filtered list containing only person detections.
        """
        return self.filter_by_class(detections, ["person"])

    def get_traffic_lights(self, detections: List[Detection]) -> List[Detection]:
        """
        Return only traffic light detections.

        Args:
            detections: List of all detections.

        Returns:
            Filtered list containing only traffic light detections.
        """
        return self.filter_by_class(detections, ["traffic light"])


def get_detector() -> VehicleDetector:
    """
    Factory function for creating and loading a VehicleDetector.

    Intended to be wrapped with @st.cache_resource in the app layer
    so the model is loaded exactly once per Streamlit session.

    Returns:
        A fully initialized VehicleDetector with model loaded.
    """
    detector = VehicleDetector()
    detector.load_model()
    return detector
