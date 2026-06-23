"""
classifier.py — CLIP Zero-Shot Violation Classifier
=====================================================
Uses OpenAI CLIP to classify cropped vehicle regions against
engineered positive/negative prompt pairs for each violation type.

Vehicle-type–aware routing ensures only semantically relevant
violations are scored (helmets → two-wheelers, seatbelts → cars, etc.).

Author: Team Gridlock | Flipkart Grid 6.0
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import clip
import numpy as np
import torch
from PIL import Image

from utils import (
    BoundingBox,
    CLIP_MODEL,
    CLIP_PROMPTS,
    CLIP_THRESHOLD,
    CROP_PADDING,
    Detection,
    ViolationType,
    crop_region,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Vehicle ↔ Violation Applicability Rules
# ──────────────────────────────────────────────

# Which violations make sense for each vehicle category.
_TWO_WHEELER_VIOLATIONS: Set[ViolationType] = {
    ViolationType.HELMET,
    ViolationType.TRIPLE_RIDING,
}
_FOUR_WHEELER_VIOLATIONS: Set[ViolationType] = {
    ViolationType.SEATBELT,
}
_UNIVERSAL_VIOLATIONS: Set[ViolationType] = {
    ViolationType.WRONG_SIDE,
    ViolationType.RED_LIGHT,
    ViolationType.STOP_LINE,
    ViolationType.ILLEGAL_PARKING,
}

_VEHICLE_VIOLATION_MAP: Dict[str, Set[ViolationType]] = {
    "motorcycle": _TWO_WHEELER_VIOLATIONS | _UNIVERSAL_VIOLATIONS,
    "bicycle":    _TWO_WHEELER_VIOLATIONS | _UNIVERSAL_VIOLATIONS,
    "car":        _FOUR_WHEELER_VIOLATIONS | _UNIVERSAL_VIOLATIONS,
    "bus":        _FOUR_WHEELER_VIOLATIONS | _UNIVERSAL_VIOLATIONS,
    "truck":      _FOUR_WHEELER_VIOLATIONS | _UNIVERSAL_VIOLATIONS,
}


# ──────────────────────────────────────────────
#  Classifier
# ──────────────────────────────────────────────

class ViolationClassifier:
    """CLIP-based zero-shot traffic violation classifier.

    Encodes cropped vehicle regions and compares them against curated
    positive (violation present) and negative (no violation) text
    prompts.  The softmax-normalised similarity to the positive set
    serves as the violation confidence score.

    Attributes:
        device: Torch device used for inference (`cuda` or `cpu`).
        model:  Loaded CLIP vision-language model.
        preprocess: CLIP image preprocessing transform.
    """

    def __init__(self) -> None:
        """Load CLIP model and select compute device."""
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CLIP model '%s' on %s …", CLIP_MODEL, self.device)

        self.model, self.preprocess = clip.load(CLIP_MODEL, device=self.device)
        self.model.eval()

        logger.info(
            "CLIP model loaded  |  threshold=%.2f  |  device=%s",
            CLIP_THRESHOLD,
            self.device,
        )

    # ── public API ────────────────────────────

    @torch.no_grad()
    def classify_region(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        violation_types: List[ViolationType],
    ) -> Dict[ViolationType, float]:
        """Score a single cropped region for multiple violation types.

        Args:
            image: Full frame as a BGR numpy array (H×W×3).
            bbox: Bounding box delineating the region of interest.
            violation_types: Which violations to evaluate.

        Returns:
            Mapping of each requested ``ViolationType`` to a confidence
            score in [0, 1].  Higher values indicate stronger evidence
            that the violation is present.
        """
        if not violation_types:
            return {}

        # 1.  Crop & encode the image region once.
        crop = crop_region(image, bbox, padding=CROP_PADDING)
        if crop.size == 0:
            logger.warning("Empty crop for bbox %s — returning zeros.", bbox)
            return {vt: 0.0 for vt in violation_types}

        image_features = self._encode_image(crop)

        # 2.  Score each violation type independently.
        scores: Dict[ViolationType, float] = {}
        for vtype in violation_types:
            prompts_pair = CLIP_PROMPTS.get(vtype)
            if prompts_pair is None:
                logger.warning("No CLIP prompts defined for %s", vtype)
                scores[vtype] = 0.0
                continue

            positive_prompts, negative_prompts = prompts_pair
            all_prompts = positive_prompts + negative_prompts
            num_positive = len(positive_prompts)

            text_features = self._encode_text(all_prompts)
            probabilities = self._compute_similarity(image_features, text_features)

            # Aggregate positive prompt probabilities as violation confidence.
            violation_confidence = float(probabilities[:num_positive].sum())
            scores[vtype] = round(violation_confidence, 4)

        return scores

    @torch.no_grad()
    def classify_all_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        active_violations: Optional[List[ViolationType]] = None,
    ) -> Dict[str, Dict[ViolationType, float]]:
        """Classify every detection in a frame for applicable violations.

        For each detection the set of candidate violations is determined
        by the vehicle type (two-wheeler → helmet / triple-riding,
        four-wheeler → seatbelt, any vehicle → universal violations).
        An optional *active_violations* filter further restricts the set.

        Args:
            image: Full frame as a BGR numpy array.
            detections: Object detections produced by the detector.
            active_violations: If provided, only these violation types
                will be evaluated (intersected with vehicle-level rules).

        Returns:
            ``Dict[str, Dict[ViolationType, float]]`` keyed by the
            detection index (as a string).  Each value maps applicable
            violation types to their confidence scores.
        """
        active_set: Optional[Set[ViolationType]] = (
            set(active_violations) if active_violations else None
        )

        results: Dict[str, Dict[ViolationType, float]] = {}

        for idx, det in enumerate(detections):
            # Determine applicable violations for this vehicle type.
            applicable = self._applicable_violations(det, active_set)
            if not applicable:
                logger.debug(
                    "Detection %d (%s) — no applicable violations, skipping.",
                    idx,
                    det.class_name,
                )
                continue

            logger.debug(
                "Detection %d (%s, conf=%.2f) — checking %s",
                idx,
                det.class_name,
                det.confidence,
                [v.value for v in applicable],
            )

            scores = self.classify_region(image, det.bbox, applicable)
            results[str(idx)] = scores

        logger.info(
            "Classified %d / %d detections.", len(results), len(detections)
        )
        return results

    # ── private helpers ───────────────────────

    @torch.no_grad()
    def _encode_image(self, image_crop: np.ndarray) -> torch.Tensor:
        """Encode a BGR numpy crop into normalised CLIP image features.

        Args:
            image_crop: Cropped region as a BGR numpy array (H×W×3).

        Returns:
            L2-normalised image feature tensor of shape ``(1, D)``.
        """
        # BGR → RGB → PIL
        rgb_array = image_crop[:, :, ::-1]
        pil_image = Image.fromarray(rgb_array.astype(np.uint8))

        # Apply CLIP preprocessing and move to device.
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # Forward pass + L2 normalisation.
        features = self.model.encode_image(image_input)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Tokenise and encode a list of text prompts with CLIP.

        Args:
            prompts: Natural-language descriptions to encode.

        Returns:
            L2-normalised text feature tensor of shape ``(N, D)``
            where *N* is the number of prompts.
        """
        tokens = clip.tokenize(prompts, truncate=True).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def _compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> np.ndarray:
        """Compute softmax probabilities over cosine similarities.

        Args:
            image_features: Normalised image features ``(1, D)``.
            text_features: Normalised text features ``(N, D)``.

        Returns:
            1-D numpy array of length *N* with softmax probabilities
            summing to 1.
        """
        # CLIP logit scale (learned temperature parameter).
        logit_scale = self.model.logit_scale.exp()

        # Cosine similarity → scaled logits → probabilities.
        logits = (logit_scale * image_features @ text_features.T).squeeze(0)
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        return probabilities

    @staticmethod
    def _applicable_violations(
        detection: Detection,
        active_filter: Optional[Set[ViolationType]] = None,
    ) -> List[ViolationType]:
        """Return the list of violations applicable to a detection.

        Combines vehicle-type rules with an optional user filter.

        Args:
            detection: A single object detection.
            active_filter: If provided, only violations in this set are
                considered.

        Returns:
            Sorted list of applicable ``ViolationType`` values.
        """
        candidate_set = _VEHICLE_VIOLATION_MAP.get(detection.class_name)
        if candidate_set is None:
            # Non-vehicle detection (e.g. "person", "traffic light").
            return []

        if active_filter is not None:
            candidate_set = candidate_set & active_filter

        # Return in a deterministic order for reproducibility.
        return sorted(candidate_set, key=lambda v: v.value)


# ──────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────

def get_classifier() -> ViolationClassifier:
    """Factory function for Streamlit ``@st.cache_resource`` caching.

    Returns:
        A fully initialised ``ViolationClassifier`` instance ready for
        inference.
    """
    return ViolationClassifier()
