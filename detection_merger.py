"""
=============================================================================
detection_merger.py — Cross-Model Detection Merging with NMS
=============================================================================

Merges detections from both YOLO models and applies cross-model
Non-Maximum Suppression to remove duplicate/overlapping boxes.

NMS Algorithm:
  1. Sort all detections by confidence (descending)
  2. Pick highest-confidence detection, add to keep list
  3. Compute IoU of this box with all remaining boxes
  4. Remove any box with IoU > threshold (it's a duplicate)
  5. Repeat until no boxes remain

IoU (Intersection over Union):
  IoU = Area(Intersection) / Area(Union)
  Where Union = Area(A) + Area(B) - Area(Intersection)
"""

import numpy as np
from typing import List
from yolo_detector import Detection
import config as cfg


def compute_iou(det_a: Detection, det_b: Detection) -> float:
    """
    Compute Intersection over Union between two detections.

    Given two boxes A = (ax1, ay1, ax2, ay2) and B = (bx1, by1, bx2, by2):
      intersection_x = max(0, min(ax2, bx2) - max(ax1, bx1))
      intersection_y = max(0, min(ay2, by2) - max(ay1, by1))
      intersection = intersection_x * intersection_y
      union = area_A + area_B - intersection
      IoU = intersection / union
    """
    # Intersection rectangle
    ix1 = max(det_a.x1, det_b.x1)
    iy1 = max(det_a.y1, det_b.y1)
    ix2 = min(det_a.x2, det_b.x2)
    iy2 = min(det_a.y2, det_b.y2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    union = det_a.area + det_b.area - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def merge_detections(
    fallen_tree_dets: List[Detection],
    coco_dets: List[Detection],
) -> List[Detection]:
    """
    Merge detections from both models and apply cross-model NMS.

    Priority: If two boxes overlap significantly, the higher-confidence
    detection is kept. The custom fallen tree model's detections are
    slightly prioritized (confidence boost) since it's specifically
    trained for this domain.

    Returns: Deduplicated list of Detection objects.
    """
    # Combine all detections
    all_dets = list(fallen_tree_dets) + list(coco_dets)

    if not all_dets:
        return []

    # Sort by confidence (descending)
    all_dets.sort(key=lambda d: d.confidence, reverse=True)

    # Greedy NMS
    keep = []
    suppressed = set()

    for i, det_i in enumerate(all_dets):
        if i in suppressed:
            continue

        keep.append(det_i)

        for j in range(i + 1, len(all_dets)):
            if j in suppressed:
                continue

            iou = compute_iou(det_i, all_dets[j])
            if iou > cfg.CROSS_MODEL_NMS_IOU_THRESHOLD:
                suppressed.add(j)

    return keep
