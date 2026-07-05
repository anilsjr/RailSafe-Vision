"""
=============================================================================
track_detector.py — OpenCV Track Detection & ROI Generation
=============================================================================

Uses Canny edge detection, Hough Line Transform, representative line fitting,
and vanishing point calculation to dynamically construct the railway track ROI.
"""

import cv2
import numpy as np
import config as cfg

class TrackDetector:
    """Detects railway tracks using OpenCV edge/line detection and vanishing point estimation."""

    def __init__(self, frame_width, frame_height):
        self.W = frame_width
        self.H = frame_height

        self._smoothed_polygon = None
        self._frame_count = 0
        self._default_polygon = self._build_default_polygon()
        self._last_valid_polygon = self._default_polygon.copy()

        self._last_valid_left = None
        self._last_valid_right = None
        self._last_valid_vp = None

        print("============================================================")
        print("Initialized OpenCV-based Track Detector & ROI Generator")
        print("============================================================")

    def detect(self, frame):
        """
        Process one frame using OpenCV. Returns:
          - roi_polygon: smoothed track ROI
          - left_line: tuple of (x1, y1, x2, y2)
          - right_line: tuple of (x1, y1, x2, y2)
          - vp: tuple of (x, y) vanishing point
          - edges: masked edges image
        """
        self._frame_count += 1
        
        roi_polygon = None
        left_line = None
        right_line = None
        vp = None
        masked_edges = np.zeros_like(frame[:,:,0])

        # 1. Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, cfg.GAUSSIAN_BLUR_KERNEL, cfg.GAUSSIAN_BLUR_SIGMA)
        
        # 2. Canny Edge Detection
        edges = cv2.Canny(blurred, cfg.CANNY_LOW_THRESHOLD, cfg.CANNY_HIGH_THRESHOLD)
        
        # 3. Apply ROI Mask for Edge Detection
        mask = np.zeros_like(edges)
        top_y = int(self.H * cfg.EDGE_ROI_TOP_FRACTION)
        bottom_y = int(self.H * cfg.EDGE_ROI_BOTTOM_FRACTION)
        
        pts = np.array([
            [int(self.W * 0.05), bottom_y],
            [int(self.W * 0.35), top_y],
            [int(self.W * 0.65), top_y],
            [int(self.W * 0.95), bottom_y]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 4. Hough Line Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            cfg.HOUGH_RHO,
            cfg.HOUGH_THETA,
            cfg.HOUGH_THRESHOLD,
            minLineLength=cfg.HOUGH_MIN_LINE_LENGTH,
            maxLineGap=cfg.HOUGH_MAX_LINE_GAP
        )
        
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                angle = np.abs(np.arctan(slope) * 180 / np.pi)
                
                # Filter by angle and length
                if cfg.RAIL_MIN_ANGLE_DEG <= angle <= cfg.RAIL_MAX_ANGLE_DEG:
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if length >= cfg.RAIL_MIN_LENGTH_PX:
                        # Extrapolate to bottom y = H - 1
                        x_bottom = x1 + (self.H - 1 - y1) * (x2 - x1) / (y2 - y1)
                        
                        # Refined spatial filtering for the train's own track
                        if slope < 0 and (0.15 * self.W <= x_bottom <= 0.48 * self.W):
                            left_lines.append((x1, y1, x2, y2, slope, length))
                        elif slope > 0 and (0.52 * self.W <= x_bottom <= 0.85 * self.W):
                            right_lines.append((x1, y1, x2, y2, slope, length))

                            
        # 5. Fit Representative Lines
        def fit_representative_line(lines_list):
            if not lines_list:
                return None
            points = []
            for x1, y1, x2, y2, _, _ in lines_list:
                points.append([x1, y1])
                points.append([x2, y2])
            points = np.array(points, dtype=np.float32)
            vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            m = vy[0] / (vx[0] + 1e-6)
            c = y0[0] - m * x0[0]
            return m, c

        left_line_fit = fit_representative_line(left_lines)
        right_line_fit = fit_representative_line(right_lines)
        
        if left_line_fit is not None and right_line_fit is not None:
            m_l, c_l = left_line_fit
            m_r, c_r = right_line_fit
            
            # Intersection (Vanishing Point)
            if np.abs(m_l - m_r) > 1e-4:
                vp_x = (c_r - c_l) / (m_l - m_r)
                vp_y = m_l * vp_x + c_l
                
                # Validate VP location
                if (cfg.VP_X_MIN_FRACTION * self.W <= vp_x <= cfg.VP_X_MAX_FRACTION * self.W and
                    cfg.VP_Y_MIN_FRACTION * self.H <= vp_y <= cfg.VP_Y_MAX_FRACTION * self.H):
                    
                    vp = (int(vp_x), int(vp_y))
                    
                    # Extrapolate to bottom
                    x_left_bottom = (self.H - 1 - c_l) / (m_l + 1e-6)
                    x_right_bottom = (self.H - 1 - c_r) / (m_r + 1e-6)
                    
                    raw_polygon = np.array([
                        [int(vp_x - cfg.ROI_TOP_HALF_WIDTH_PX), int(vp_y)],
                        [int(vp_x + cfg.ROI_TOP_HALF_WIDTH_PX), int(vp_y)],
                        [int(x_right_bottom + cfg.ROI_BOTTOM_EXPANSION_PX), self.H - 1],
                        [int(x_left_bottom - cfg.ROI_BOTTOM_EXPANSION_PX), self.H - 1]
                    ], dtype=np.int32)
                    
                    # Smooth polygon
                    roi_polygon = self._smooth_polygon(raw_polygon)
                    self._last_valid_polygon = roi_polygon
                    
                    # Prepare lines for display
                    y_top = int(vp_y)
                    x_left_top = (y_top - c_l) / (m_l + 1e-6)
                    x_right_top = (y_top - c_r) / (m_r + 1e-6)
                    
                    left_line = (int(x_left_top), y_top, int(x_left_bottom), self.H - 1)
                    right_line = (int(x_right_top), y_top, int(x_right_bottom), self.H - 1)
                    
                    self._last_valid_left = left_line
                    self._last_valid_right = right_line
                    self._last_valid_vp = vp

        # Fallback to last valid or default
        if roi_polygon is None:
            roi_polygon = self._last_valid_polygon
            left_line = self._last_valid_left
            right_line = self._last_valid_right
            vp = self._last_valid_vp

        return roi_polygon, left_line, right_line, vp, masked_edges

    def _build_default_polygon(self):
        """Default fallback polygon."""
        return np.array([
            [int(self.W * 0.42), int(self.H * 0.35)], 
            [int(self.W * 0.58), int(self.H * 0.35)], 
            [int(self.W * 0.90), self.H - 1],         
            [int(self.W * 0.10), self.H - 1],         
        ], dtype=np.int32)

    def _smooth_polygon(self, raw_polygon):
        """EMA smoothing for polygon vertices."""
        if self._smoothed_polygon is None or self._frame_count <= cfg.SMOOTHING_WARMUP_FRAMES:
            self._smoothed_polygon = raw_polygon.astype(np.float64)
            return raw_polygon

        if len(self._smoothed_polygon) != len(raw_polygon):
            self._smoothed_polygon = raw_polygon.astype(np.float64)
            return raw_polygon

        alpha = cfg.TEMPORAL_SMOOTHING_ALPHA
        raw = raw_polygon.astype(np.float64)
        prev = self._smoothed_polygon

        self._smoothed_polygon = alpha * raw + (1 - alpha) * prev
        return self._smoothed_polygon.astype(np.int32)
