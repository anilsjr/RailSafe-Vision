"""
=============================================================================
run_inference.py — Entry Point for Railway Track Obstacle Detection
=============================================================================

Usage:
    python run_inference.py

Before running, update the paths in config.py:
    - FALLEN_TREE_MODEL_PATH  (your custom .pt file)
    - COCO_MODEL_PATH          (yolo11s.pt — auto-downloads if missing)
    - INPUT_VIDEO_PATH          (input railway video)
    - OUTPUT_VIDEO_PATH         (where to save the result)

Requirements:
    pip install opencv-python numpy ultralytics
"""

import cv2
import time
import sys
import config as cfg
from pipeline import InferencePipeline


def main():
    print("\n" + "=" * 70)
    print("   RAILWAY TRACK OBSTACLE DETECTION SYSTEM")
    print("   OpenCV Track ROI + Dual YOLO11 Inference Pipeline")
    print("=" * 70)

    # ==================================================================
    # Open Input Video
    # ==================================================================
    print(f"\n📹 Input Video : {cfg.INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(cfg.INPUT_VIDEO_PATH)

    if not cap.isOpened():
        print("❌ ERROR: Could not open input video.")
        print(f"   Path: {cfg.INPUT_VIDEO_PATH}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    print(f"   Resolution  : {width} x {height}")
    print(f"   FPS         : {fps:.1f}")
    print(f"   Total Frames: {total_frames}")

    # ==================================================================
    # Setup Output Video Writer
    # ==================================================================
    print(f"\n💾 Output Video: {cfg.OUTPUT_VIDEO_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cfg.OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ ERROR: Failed to create output video writer.")
        sys.exit(1)

    # ==================================================================
    # Initialize Pipeline
    # ==================================================================
    pipeline = InferencePipeline(width, height, total_frames)

    # ==================================================================
    # Inference Loop
    # ==================================================================
    print("\n" + "─" * 70)
    print("  Running inference...")
    print("─" * 70)

    t_total_start = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process frame through the complete pipeline
            annotated_frame = pipeline.process_frame(frame)

            # Write to output video
            out.write(annotated_frame)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user.")

    # ==================================================================
    # Cleanup
    # ==================================================================
    cap.release()
    out.release()

    t_total = time.time() - t_total_start

    print("\n" + "=" * 70)
    print("  ✅ INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Frames Processed : {frame_count}")
    print(f"  Total Time       : {t_total:.1f}s")
    print(f"  Average FPS      : {frame_count / max(t_total, 0.001):.1f}")
    print(f"  Output Saved     : {cfg.OUTPUT_VIDEO_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
