import cv2
from ultralytics import YOLO

# ======================================================
# Paths
# ======================================================

MODEL_PATH = "D:\\Railway_Track\\best.pt"
VIDEO_PATH = "D:\Railway_Track\kling_20260702_VIDEO_Ultra_real_5180_0.mp4"
OUTPUT_VIDEO = "D:\\Railway_Track\\output_detection.avi"

# ======================================================
# Load Model
# ======================================================

print("Loading model...")
model = YOLO(MODEL_PATH)

# ======================================================
# Open Video
# ======================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open input video.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 30

print("=" * 60)
print(f"Resolution   : {width} x {height}")
print(f"FPS          : {fps}")
print(f"Total Frames : {total_frames}")
print("=" * 60)

# ======================================================
# Video Writer
# ======================================================

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    raise RuntimeError("❌ Failed to create output video.")

print("Running inference...")

frame_number = 0

# ======================================================
# Inference Loop
# ======================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # -----------------------------
    # YOLO Prediction
    # -----------------------------
    results = model.predict(
        source=frame,
        conf=0.3,      # Confidence Threshold
        iou=0.4,       # NMS IoU Threshold
        device=0,       # GPU
        verbose=False,
        max_det=20      # Maximum detections per frame
    )

    result = results[0]

    # Draw Bounding Boxes + Labels + Confidence
    annotated_frame = result.plot(
        conf=True,
        labels=True,
        line_width=3,
        font_size=1
    )

    # Save output frame
    out.write(annotated_frame)

    # Print progress
    if frame_number % 10 == 0:
        print(f"Processed {frame_number}/{total_frames} frames")

# ======================================================
# Cleanup
# ======================================================

cap.release()
out.release()

print("\n" + "=" * 60)
print("✅ Inference Completed Successfully!")
print(f"Frames Processed : {frame_number}")
print(f"Saved Video      : {OUTPUT_VIDEO}")
print("=" * 60)