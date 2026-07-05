"""
=============================================================================
yolo_detector.py — YOLO Model Wrapper for Dual-Model Detection
=============================================================================

Wraps ultralytics YOLO for:
  - Model A: Custom Fallen Tree detector
  - Model B: COCO pretrained detector (filtered to classes of interest)

Each model returns a list of Detection namedtuples.
"""

import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List
import config as cfg


@dataclass
class Detection:
    """Single detection result."""
    x1: int       # Bounding box top-left x
    y1: int       # Bounding box top-left y
    x2: int       # Bounding box bottom-right x
    y2: int       # Bounding box bottom-right y
    confidence: float
    class_name: str
    source: str   # "fallen_tree" or "coco"

    @property
    def center(self):
        """Center point of bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class YOLODetector:
    """Manages both YOLO models and provides unified detection interface."""

    def __init__(self):
        print("=" * 60)
        print("Loading YOLO models...")
        print("=" * 60)

        # Load custom Fallen Tree model
        print(f"  [Model A] Fallen Tree: {cfg.FALLEN_TREE_MODEL_PATH}")
        self.fallen_tree_model = YOLO(cfg.FALLEN_TREE_MODEL_PATH)
        print("  ✅ Fallen Tree model loaded.")

        # Load COCO pretrained model
        print(f"  [Model B] COCO:        {cfg.COCO_MODEL_PATH}")
        self.coco_model = YOLO(cfg.COCO_MODEL_PATH)
        print("  ✅ COCO model loaded.")
        print("=" * 60)

    def run_fallen_tree_model(self, frame) -> List[Detection]:
        """
        Run the custom Fallen Tree detection model.
        Returns list of Detection objects.
        """
        results = self.fallen_tree_model.predict(
            source=frame,
            conf=cfg.YOLO_CONFIDENCE_THRESHOLD,
            iou=cfg.YOLO_NMS_IOU_THRESHOLD,
            device=cfg.YOLO_DEVICE,
            verbose=False,
            max_det=cfg.YOLO_MAX_DETECTIONS,
            imgsz=cfg.YOLO_IMG_SIZE,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_name=cls_name,
                    source="fallen_tree"
                ))

        return detections

    def run_coco_model(self, frame) -> List[Detection]:
        """
        Run the COCO pretrained model, filtering to classes of interest.
        Returns list of Detection objects.
        """
        results = self.coco_model.predict(
            source=frame,
            conf=cfg.YOLO_CONFIDENCE_THRESHOLD,
            iou=cfg.YOLO_NMS_IOU_THRESHOLD,
            device=cfg.YOLO_DEVICE,
            verbose=False,
            max_det=cfg.YOLO_MAX_DETECTIONS,
            imgsz=cfg.YOLO_IMG_SIZE,
            classes=list(cfg.COCO_CLASSES_OF_INTEREST.keys()),
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                # Use our custom name mapping
                cls_name = cfg.COCO_CLASSES_OF_INTEREST.get(cls_id, f"Object_{cls_id}")

                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_name=cls_name,
                    source="coco"
                ))

        return detections
