"""
tracker.py — Lightweight IoU-Based Object Tracker
=================================================
Assigns persistent IDs to vehicle detections across video frames
using Intersection over Union (IoU) mapping. Helps avoid duplicate
ticketing for the same vehicle in consecutive frames.
"""

from typing import List, Dict, Tuple, Optional
from utils import BoundingBox, Detection, generate_id, VehicleType, TWO_WHEELERS, FOUR_WHEELERS

class Track:
    """Represents a tracked vehicle or object across multiple video frames."""
    def __init__(self, track_id: str, detection: Detection):
        self.track_id: str = track_id
        self.bbox: BoundingBox = detection.bbox
        self.class_name: str = detection.class_name
        self.class_id: int = detection.class_id
        self.vehicle_type: Optional[VehicleType] = detection.vehicle_type
        self.inactive_frames: int = 0
        self.violations_flagged: set = set()
        self.license_plate: Optional[str] = None
        self.best_plate_confidence: float = 0.0

    def update(self, detection: Detection):
        """Update track coordinates with a matched detection."""
        self.bbox = detection.bbox
        self.inactive_frames = 0


class IoUTracker:
    """Performs Intersection-over-Union tracking across sequential frames."""
    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 10):
        self.iou_threshold: float = iou_threshold
        self.max_lost_frames: int = max_lost_frames
        self.tracks: Dict[str, Track] = {}

    def update(self, detections: List[Detection]) -> List[Tuple[str, Detection]]:
        """
        Updates active tracks with new detections.
        Returns a list of tuples containing (track_id, detection) for all matched detections.
        """
        # Filter for trackable objects (vehicles: cars, bikes, trucks, etc.)
        trackable_detections = [
            d for d in detections
            if d.vehicle_type in TWO_WHEELERS or d.vehicle_type in FOUR_WHEELERS
        ]

        matched_detections: List[Tuple[str, Detection]] = []
        unmatched_detections = list(trackable_detections)

        # 1. Try to match existing tracks with new detections using IoU
        track_ids = list(self.tracks.keys())
        if track_ids and unmatched_detections:
            # Compute IoU matrix
            iou_matrix = []
            for tid in track_ids:
                track = self.tracks[tid]
                row = []
                for det in unmatched_detections:
                    row.append(track.bbox.iou(det.bbox))
                iou_matrix.append(row)

            # Greedy assignment
            while True:
                max_iou = -1.0
                best_track_idx = -1
                best_det_idx = -1

                for t_idx, row in enumerate(iou_matrix):
                    for d_idx, val in enumerate(row):
                        if val > max_iou:
                            max_iou = val
                            best_track_idx = t_idx
                            best_det_idx = d_idx

                if max_iou < self.iou_threshold:
                    break

                # Assign track to detection
                tid = track_ids[best_track_idx]
                det = unmatched_detections[best_det_idx]
                
                self.tracks[tid].update(det)
                matched_detections.append((tid, det))

                # Zero out row/col to prevent re-matching
                for idx in range(len(unmatched_detections)):
                    iou_matrix[best_track_idx][idx] = -1.0
                for idx in range(len(track_ids)):
                    iou_matrix[idx][best_det_idx] = -1.0

                # Remove from unmatched
                unmatched_detections[best_det_idx] = None

            unmatched_detections = [d for d in unmatched_detections if d is not None]

        # 2. Create new tracks for unmatched detections
        for det in unmatched_detections:
            new_id = generate_id(prefix="VEH")
            new_track = Track(new_id, det)
            self.tracks[new_id] = new_track
            matched_detections.append((new_id, det))

        # 3. Increment inactive frames for tracks that were not matched
        matched_tids = {tid for tid, _ in matched_detections}
        lost_tids = []
        for tid, track in self.tracks.items():
            if tid not in matched_tids:
                track.inactive_frames += 1
                if track.inactive_frames > self.max_lost_frames:
                    lost_tids.append(tid)

        # 4. Remove lost tracks
        for tid in lost_tids:
            del self.tracks[tid]

        return matched_detections
