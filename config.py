"""
=============================================================================
config.py — Central Configuration for Railway Track Obstacle Detection System
=============================================================================

All tunable parameters are defined here.
Modify paths and thresholds before running the pipeline.
"""

import numpy as np
import os
from dotenv import load_dotenv

# Load workspace .env settings
load_dotenv()

# ===========================================================================
# MODEL PATHS — Loaded from environment or fallback
# ===========================================================================

# Custom YOLOv11s model trained on Fallen Trees
FALLEN_TREE_MODEL_PATH = os.getenv("FALLEN_TREE_MODEL_PATH", "models/best.pt")

# Official YOLOv11s COCO pretrained model
COCO_MODEL_PATH = os.getenv("COCO_MODEL_PATH", "yolo11s.pt")

# ===========================================================================
# VIDEO PATHS
# ===========================================================================

INPUT_VIDEO_PATH = os.getenv("INPUT_VIDEO_PATH", "input/Reference_Use_the_uploaded_im.mp4")
OUTPUT_VIDEO_PATH = os.getenv("OUTPUT_VIDEO_PATH", "output_track_obstacle_detection.mp4")

# ===========================================================================
# YOLO DETECTION SETTINGS
# ===========================================================================

# Confidence threshold for YOLO detections
YOLO_CONFIDENCE_THRESHOLD = 0.35

# IoU threshold for Non-Maximum Suppression
YOLO_NMS_IOU_THRESHOLD = 0.45

# Maximum detections per frame per model
YOLO_MAX_DETECTIONS = 30

# Device: 0 for GPU (CUDA), "cpu" for CPU, "mps" for macOS Metal
# Try loading from .env, default to cpu
raw_device = os.getenv("YOLO_DEVICE", "cpu")
try:
    # If it is a digit string, cast to int for CUDA index
    YOLO_DEVICE = int(raw_device) if raw_device.isdigit() else raw_device
except Exception:
    YOLO_DEVICE = raw_device


# Input image size for YOLO inference
YOLO_IMG_SIZE = 640

# ===========================================================================
# COCO CLASS IDS OF INTEREST
# ===========================================================================
# Only these COCO classes will be kept from the COCO model.
# Format: {class_id: class_name}

COCO_CLASSES_OF_INTEREST = {

    1:  "Bicycle",
    2:  "Car",
    3:  "Motorcycle",
    5:  "Bus",
    7:  "Truck",
    14: "Bird",
    15: "Cat",
    16: "Dog",
    17: "Horse",
    18: "Sheep",
    19: "Cow",
    20: "Elephant",
    21: "Bear",
    22: "Zebra",
    23: "Giraffe",
    24: "Backpack",
    25: "Umbrella",
    28: "Suitcase",
}

# ===========================================================================
# TRACK DETECTION — OpenCV Parameters
# ===========================================================================

# --- Preprocessing ---
GAUSSIAN_BLUR_KERNEL = (7, 7)       # Must be odd numbers
GAUSSIAN_BLUR_SIGMA = 2             # Gaussian sigma

# --- Canny Edge Detection ---
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# --- Morphological Operations ---
# Dilation kernel to connect broken edge fragments
MORPH_KERNEL_SIZE = (3, 3)
MORPH_DILATE_ITERATIONS = 1
MORPH_ERODE_ITERATIONS = 0

# --- Region of Interest Mask for Edge Detection ---
# Only process the lower portion of the frame where rails are visible.
# Expressed as a fraction of frame height (0.0 = top, 1.0 = bottom).
EDGE_ROI_TOP_FRACTION = 0.35        # Ignore the top 35% of the frame
EDGE_ROI_BOTTOM_FRACTION = 1.0      # Process up to the bottom

# --- Hough Line Transform (Probabilistic) ---
HOUGH_RHO = 1                       # Distance resolution in pixels
HOUGH_THETA = np.pi / 180           # Angular resolution in radians
HOUGH_THRESHOLD = 50                 # Accumulator threshold
HOUGH_MIN_LINE_LENGTH = 80          # Minimum line length in pixels
HOUGH_MAX_LINE_GAP = 30             # Maximum gap between line segments

# ===========================================================================
# RAIL LINE FILTERING
# ===========================================================================

# Angle constraints for rail lines (in degrees from horizontal).
# Railway rails appear as diagonal lines converging toward the vanishing point.
# We filter lines by their angle to remove horizontal/near-horizontal noise.

# Minimum angle from horizontal for a line to be considered a rail
RAIL_MIN_ANGLE_DEG = 20

# Maximum angle from horizontal (near-vertical lines are usually not rails)
RAIL_MAX_ANGLE_DEG = 85

# Minimum line length (in pixels) to keep — rejects short noisy segments
RAIL_MIN_LENGTH_PX = 60

# ===========================================================================
# VANISHING POINT ESTIMATION
# ===========================================================================

# The vanishing point is the intersection of left-rail and right-rail lines.
# We use RANSAC-like median estimation to be robust against outliers.

# Expected vanishing point Y-coordinate range (as fraction of frame height).
# The VP should be in the upper portion of the frame.
VP_Y_MIN_FRACTION = 0.05
VP_Y_MAX_FRACTION = 0.55

# Expected vanishing point X-coordinate range (as fraction of frame width).
# The VP should be roughly centered.
VP_X_MIN_FRACTION = 0.15
VP_X_MAX_FRACTION = 0.85

# ===========================================================================
# TRACK ROI POLYGON
# ===========================================================================

# The ROI polygon is a trapezoid formed by:
#   - Top: the vanishing point (with slight lateral expansion)
#   - Bottom: the extrapolated rail positions at the bottom of the frame
#
# Bottom expansion: how many pixels to extend outward beyond the rail endpoints
ROI_BOTTOM_EXPANSION_PX = 0

# Top width: lateral pixels from vanishing point to form the top edge
ROI_TOP_HALF_WIDTH_PX = 10

# ===========================================================================
# PROXIMITY ALERT SETTINGS
# ===========================================================================

# Y-coordinate threshold for transitioning from WARNING (far) to DANGER (near)
# Expressed as a fraction of frame height (0.0 = top, 1.0 = bottom)
DANGER_Y_THRESHOLD_FRACTION = 0.60


# ===========================================================================
# TEMPORAL SMOOTHING
# ===========================================================================

# Exponential Moving Average (EMA) alpha for polygon vertex smoothing.
# Lower alpha = more smoothing (less flicker) but slower adaptation.
# Higher alpha = faster adaptation but more jitter.
# Range: 0.0 (frozen) to 1.0 (no smoothing)
TEMPORAL_SMOOTHING_ALPHA = 0.15

# Number of initial frames to skip smoothing (allow the system to "warm up")
SMOOTHING_WARMUP_FRAMES = 5

# Maximum allowed movement (in pixels) per frame for any vertex.
# If a vertex jumps more than this, we clamp it (outlier rejection).
MAX_VERTEX_JUMP_PX = 60

# ===========================================================================
# DETECTION MERGING (NMS across models)
# ===========================================================================

# IoU threshold for cross-model NMS (to remove duplicate detections)
CROSS_MODEL_NMS_IOU_THRESHOLD = 0.50

# ===========================================================================
# VISUALIZATION
# ===========================================================================

# Track ROI overlay
ROI_OVERLAY_COLOR = (0, 200, 80)     # BGR — semi-transparent green
ROI_OVERLAY_ALPHA = 0.25             # Transparency (0=invisible, 1=opaque)
ROI_BORDER_COLOR = (0, 255, 100)     # BGR — bright green border
ROI_BORDER_THICKNESS = 2

# Rail lines
RAIL_LINE_COLOR = (255, 180, 0)      # BGR — bright blue
RAIL_LINE_THICKNESS = 2

# Vanishing point
VP_COLOR = (0, 255, 255)             # BGR — yellow/cyan
VP_RADIUS = 8

# Bounding boxes
DANGER_BOX_COLOR = (0, 0, 255)       # BGR — RED
SAFE_BOX_COLOR = (0, 220, 0)         # BGR — GREEN
BOX_THICKNESS = 2

# Text
FONT = 0                             # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE = 0.9
FONT_SCALE_MEDIUM = 0.65
FONT_SCALE_SMALL = 0.5
FONT_THICKNESS = 2

# Status bar colors
STATUS_CLEAR_COLOR = (0, 200, 0)      # Green
STATUS_WARNING_COLOR = (0, 200, 255)  # Orange/Yellow
STATUS_DANGER_COLOR = (0, 0, 255)     # Red

# ===========================================================================
# PERFORMANCE
# ===========================================================================

# Process every Nth frame for track detection (saves CPU).
# YOLO runs on every frame; track detection can be less frequent since
# rails don't move rapidly.
TRACK_DETECTION_INTERVAL = 3

# Resize factor for track detection processing (not for YOLO).
# 1.0 = full resolution, 0.5 = half resolution (faster).
TRACK_PROCESSING_SCALE = 0.6

# ===========================================================================
# LOGGING
# ===========================================================================

# Print progress every N frames
LOG_INTERVAL = 10

# Enable/disable debug visualization (shows intermediate CV steps)
DEBUG_MODE = False
