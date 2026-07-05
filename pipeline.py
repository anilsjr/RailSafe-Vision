"""
=============================================================================
pipeline.py — Main Frame Processing Pipeline
=============================================================================

Orchestrates the complete per-frame processing:
  1. Track detection (OpenCV) — every N frames for performance
  2. YOLO detection (both models)
  3. Detection merging (NMS)
  4. ROI analysis (point-in-polygon)
  5. Visualization

This is the central module that ties all components together.
"""

import time
import numpy as np
from track_detector import TrackDetector
from yolo_detector import YOLODetector
from detection_merger import merge_detections
from roi_analyzer import classify_detections, get_track_status
import visualizer as viz
import config as cfg


class InferencePipeline:
    """Complete inference pipeline for railway track obstacle detection."""

    def __init__(self, frame_width, frame_height, total_frames):
        self.W = frame_width
        self.H = frame_height
        self.total_frames = total_frames

        # Initialize modules
        self.track_detector = TrackDetector(frame_width, frame_height)
        self.yolo_detector = YOLODetector()

        # State
        self._current_roi = None
        self._current_left = None
        self._current_right = None
        self._current_vp = None
        self._frame_count = 0

        # FPS tracking
        self._fps_buffer = []
        self._fps_display = 0.0

    def process_frame(self, frame):
        """
        Process a single video frame through the complete pipeline.

        Args:
            frame: BGR numpy array from video capture.

        Returns:
            annotated_frame: The frame with all visualizations drawn.
        """
        t_start = time.time()
        self._frame_count += 1

        # ===================================================================
        # STEP 1: Track Detection (OpenCV)
        # Run every N frames for performance; reuse last ROI otherwise.
        # ===================================================================
        if self._frame_count % cfg.TRACK_DETECTION_INTERVAL == 1 or self._current_roi is None:
            roi, left, right, vp, edges = self.track_detector.detect(frame)
            self._current_roi = roi
            self._current_left = left
            self._current_right = right
            self._current_vp = vp

        # ===================================================================
        # STEP 2: YOLO Detection (Both Models)
        # ===================================================================
        fallen_tree_dets = self.yolo_detector.run_fallen_tree_model(frame)
        coco_dets = self.yolo_detector.run_coco_model(frame)

        # ===================================================================
        # STEP 3: Merge Detections (Cross-Model NMS)
        # ===================================================================
        all_detections = merge_detections(fallen_tree_dets, coco_dets)

        # ===================================================================
        # STEP 4: ROI Analysis (Bounding Box Overlap)
        # ===================================================================
        danger_dets, warning_dets, safe_dets = classify_detections(all_detections, self._current_roi, self.W, self.H)
        status_text, status_color = get_track_status(len(danger_dets), len(warning_dets))

        # ===================================================================
        # STEP 5: Visualization
        # ===================================================================
        annotated = frame.copy()

        # Draw track ROI (semi-transparent overlay)
        annotated = viz.draw_roi(annotated, self._current_roi)

        # Draw rail lines
        annotated = viz.draw_rail_lines(annotated, self._current_left, self._current_right)

        # Draw vanishing point
        annotated = viz.draw_vanishing_point(annotated, self._current_vp)

        # Draw detections (boxes, labels, confidence)
        annotated = viz.draw_detections(annotated, danger_dets, warning_dets, safe_dets)

        # Draw status bar (top center)
        annotated = viz.draw_status_bar(annotated, status_text, status_color)

        # Draw info overlay (FPS, frame number)
        fps = self._compute_fps(time.time() - t_start)
        annotated = viz.draw_info_overlay(
            annotated, fps, self._frame_count, self.total_frames,
            len(danger_dets) + len(warning_dets), len(safe_dets)
        )

        # Draw alert effects if obstacles on track
        annotated = viz.draw_alerts(annotated, danger_dets, warning_dets)

        # ===================================================================
        # Logging
        # ===================================================================
        if self._frame_count % cfg.LOG_INTERVAL == 0:
            print(
                f"  Frame {self._frame_count:>5}/{self.total_frames}  |  "
                f"FPS: {fps:>5.1f}  |  "
                f"Dets: {len(all_detections):>2} (Danger: {len(danger_dets)}, Warning: {len(warning_dets)}, Safe: {len(safe_dets)})  |  "
                f"{status_text}"
            )

        return annotated

    def _compute_fps(self, elapsed):
        """Compute smoothed FPS using a rolling buffer."""
        if elapsed > 0:
            self._fps_buffer.append(1.0 / elapsed)
        if len(self._fps_buffer) > 30:
            self._fps_buffer.pop(0)
        if self._fps_buffer:
            self._fps_display = np.mean(self._fps_buffer)
        return self._fps_display
