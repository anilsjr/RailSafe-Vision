"""
=============================================================================
visualizer.py — Professional Inference Video Visualization
=============================================================================

Handles all drawing operations to produce a clean, industrial-looking
inference output:
  - Semi-transparent track ROI overlay
  - Rail lines
  - Vanishing point marker
  - Bounding boxes (red=danger, green=safe)
  - Labels with confidence
  - Status bar
  - FPS & frame counter
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from yolo_detector import Detection
import config as cfg


def draw_roi(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Draw the track ROI as a semi-transparent filled polygon with border.

    Uses cv2.fillPoly on an overlay, then cv2.addWeighted for transparency.
    """
    if polygon is None:
        return frame

    overlay = frame.copy()
    pts = polygon.reshape((-1, 1, 2))

    # Fill with semi-transparent color
    cv2.fillPoly(overlay, [polygon], cfg.ROI_OVERLAY_COLOR)
    frame = cv2.addWeighted(overlay, cfg.ROI_OVERLAY_ALPHA, frame, 1 - cfg.ROI_OVERLAY_ALPHA, 0)

    # Draw border
    cv2.polylines(frame, [polygon], isClosed=True,
                  color=cfg.ROI_BORDER_COLOR, thickness=cfg.ROI_BORDER_THICKNESS)

    return frame


def draw_rail_lines(
    frame: np.ndarray,
    left_line: Optional[Tuple],
    right_line: Optional[Tuple]
) -> np.ndarray:
    """Draw detected rail lines."""
    if left_line is not None:
        cv2.line(frame, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]),
                 cfg.RAIL_LINE_COLOR, cfg.RAIL_LINE_THICKNESS)

    if right_line is not None:
        cv2.line(frame, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]),
                 cfg.RAIL_LINE_COLOR, cfg.RAIL_LINE_THICKNESS)

    return frame


def draw_vanishing_point(
    frame: np.ndarray,
    vp: Optional[Tuple[int, int]]
) -> np.ndarray:
    """Draw the estimated vanishing point as a crosshair."""
    if vp is None:
        return frame

    x, y = vp
    cv2.circle(frame, (x, y), cfg.VP_RADIUS, cfg.VP_COLOR, 2)
    cv2.line(frame, (x - 12, y), (x + 12, y), cfg.VP_COLOR, 1)
    cv2.line(frame, (x, y - 12), (x, y + 12), cfg.VP_COLOR, 1)

    return frame


def draw_detections(
    frame: np.ndarray,
    danger_dets: List[Detection],
    warning_dets: List[Detection],
    safe_dets: List[Detection]
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and confidence scores.
      - DANGER detections: RED box + "DANGER" label
      - WARNING detections: YELLOW/ORANGE box + "WARNING" label
      - SAFE detections:   GREEN box + "Safe" label
    """
    WARNING_BOX_COLOR = (0, 165, 255) # Orange

    # Draw safe detections first (less prominent)
    for det in safe_dets:
        _draw_single_detection(frame, det, cfg.SAFE_BOX_COLOR, "Safe")

    # Draw warning detections
    for det in warning_dets:
        _draw_single_detection(frame, det, WARNING_BOX_COLOR, "WARNING")

    # Draw danger detections on top (more prominent)
    for det in danger_dets:
        _draw_single_detection(frame, det, cfg.DANGER_BOX_COLOR, "DANGER")

    return frame


def _draw_single_detection(
    frame: np.ndarray,
    det: Detection,
    color: Tuple[int, int, int],
    status_tag: str
) -> None:
    """Draw a single detection with box, label, confidence, and status."""
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cfg.BOX_THICKNESS)

    # Label background
    label = f"{det.class_name} {det.confidence:.2f}"
    (lw, lh), baseline = cv2.getTextSize(label, cfg.FONT, cfg.FONT_SCALE_SMALL, 1)

    # Label above box
    label_y = max(y1 - 8, lh + 5)
    cv2.rectangle(frame, (x1, label_y - lh - 5), (x1 + lw + 6, label_y + 3), color, -1)
    cv2.putText(frame, label, (x1 + 3, label_y - 2),
                cfg.FONT, cfg.FONT_SCALE_SMALL, (255, 255, 255), 1, cv2.LINE_AA)

    # Status tag below box
    tag = f"[{status_tag}]"
    (tw, th), _ = cv2.getTextSize(tag, cfg.FONT, cfg.FONT_SCALE_SMALL, 1)
    tag_y = min(y2 + th + 8, frame.shape[0] - 5)
    cv2.putText(frame, tag, (x1, tag_y),
                cfg.FONT, cfg.FONT_SCALE_SMALL, color, 1, cv2.LINE_AA)

    # Center point marker
    cx, cy = det.center
    cv2.circle(frame, (cx, cy), 4, color, -1)


def draw_status_bar(
    frame: np.ndarray,
    status_text: str,
    status_color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Draw the track status as a centered bar at the top of the frame.
    """
    h, w = frame.shape[:2]

    (tw, th), _ = cv2.getTextSize(status_text, cfg.FONT, cfg.FONT_SCALE_LARGE, cfg.FONT_THICKNESS)

    # Background bar
    bar_h = th + 24
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Status text (centered)
    tx = (w - tw) // 2
    ty = th + 12
    cv2.putText(frame, status_text, (tx, ty),
                cfg.FONT, cfg.FONT_SCALE_LARGE, status_color, cfg.FONT_THICKNESS, cv2.LINE_AA)

    return frame


def draw_info_overlay(
    frame: np.ndarray,
    fps: float,
    frame_number: int,
    total_frames: int,
    danger_count: int,
    safe_count: int
) -> np.ndarray:
    """
    Draw FPS, frame counter, and detection summary in corners.
    """
    h, w = frame.shape[:2]

    # --- Top-left: FPS and Frame ---
    y_offset = 60  # Below status bar

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (12, y_offset),
                cfg.FONT, cfg.FONT_SCALE_MEDIUM, (255, 255, 255), 1, cv2.LINE_AA)

    frame_text = f"Frame: {frame_number}/{total_frames}"
    cv2.putText(frame, frame_text, (12, y_offset + 25),
                cfg.FONT, cfg.FONT_SCALE_MEDIUM, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Bottom-left: Detection summary ---
    summary_y = h - 20
    det_text = f"On Track: {danger_count}  |  Off Track: {safe_count}"
    cv2.putText(frame, det_text, (12, summary_y),
                cfg.FONT, cfg.FONT_SCALE_SMALL, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def draw_alerts(
    frame: np.ndarray,
    danger_dets: List[Detection],
    warning_dets: List[Detection]
) -> np.ndarray:
    """
    Draw prominent alert indicators when obstacles are on the track.
    Flashing red border effect for critical alerts.
    """
    if not danger_dets and not warning_dets:
        return frame

    h, w = frame.shape[:2]

    # Border flash
    border_thickness = 6
    if danger_dets:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                      cfg.DANGER_BOX_COLOR, border_thickness)
        alert_text = "OBSTACLE VERY NEAR"
        bg_color = (0, 0, 180)
        fg_color = (255, 255, 255)
    else:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                      (0, 165, 255), border_thickness) # Orange
        alert_text = "OBSTACLE AHEAD"
        bg_color = (0, 165, 255)
        fg_color = (0, 0, 0)

    # Alert text at bottom center
    (tw, th), _ = cv2.getTextSize(alert_text, cfg.FONT, cfg.FONT_SCALE_LARGE, cfg.FONT_THICKNESS)
    tx = (w - tw) // 2
    ty = h - 40

    # Background for readability
    cv2.rectangle(frame, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), bg_color, -1)
    cv2.putText(frame, alert_text, (tx, ty),
                cfg.FONT, cfg.FONT_SCALE_LARGE, fg_color, cfg.FONT_THICKNESS, cv2.LINE_AA)

    return frame
