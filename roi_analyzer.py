"""
=============================================================================
roi_analyzer.py — Point-in-Polygon Analysis for Track Safety
=============================================================================

Determines whether detected objects are ON the track (inside the ROI)
or OFF the track (outside the ROI).

Also assesses proximity (distance) for objects on the track:
  - DANGER: On track and close (y2 > threshold)
  - WARNING: On track but far (y2 <= threshold)

Uses OpenCV's cv2.pointPolygonTest() which implements the ray-casting
algorithm.
"""

import cv2
import numpy as np
from typing import List, Tuple
from yolo_detector import Detection
import config as cfg


def classify_detections(
    detections: List[Detection],
    roi_polygon: np.ndarray,
    frame_width: int,
    frame_height: int
) -> Tuple[List[Detection], List[Detection], List[Detection]]:
    """
    Classify each detection.
    Any object whose bounding box overlaps with the track ROI is DANGER.
    """
    danger_dets = []
    warning_dets = []
    safe_dets = []

    if roi_polygon is None or len(detections) == 0:
        return [], [], detections

    # Create a binary mask of the track ROI
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon.astype(np.int32)], 1)

    for det in detections:
        # Get bounding box coordinates bounded by frame
        x1 = max(0, det.x1)
        y1 = max(0, det.y1)
        x2 = min(frame_width, det.x2)
        y2 = min(frame_height, det.y2)
        
        # Calculate overlap (sum of 1s in the bounding box area of the mask)
        overlap = mask[y1:y2, x1:x2].sum()

        if overlap > 0:
            # Bounding box touches the track ROI
            danger_dets.append(det)
        else:
            safe_dets.append(det)

    return danger_dets, warning_dets, safe_dets


def get_track_status(danger_count: int, warning_count: int) -> Tuple[str, Tuple[int, int, int]]:
    """
    Determine the overall track status message and color.
    """
    if danger_count > 0:
        return f"🚨 DANGER — {danger_count} OBSTACLES ON TRACK", (0, 0, 255)
    elif warning_count > 0:
        return f"⚠ WARNING — OBSTACLE AHEAD", (0, 200, 255)
    else:
        return "TRACK CLEAR", (0, 200, 0)
