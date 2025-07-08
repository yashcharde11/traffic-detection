import cv2
import os
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Define input video path and snapshot directory
video_path = "data/traffic_video.mp4"
snapshot_dir = os.path.join("output", "snapshots")
os.makedirs(snapshot_dir, exist_ok=True)

# Vehicle classes of interest from COCO
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    7: 'truck'
}

# Snapshot trigger threshold
VEHICLE_THRESHOLD = 2  # Lowered temporarily for testing

# Speed estimation helper
previous_boxes = {}

def estimate_speed(cls_id, current_box, previous_box):
    if not previous_box:
        return 0.0
    x1, y1, x2, y2 = current_box
    px1, py1, px2, py2 = previous_box
    curr_area = (x2 - x1) * (y2 - y1)
    prev_area = (px2 - px1) * (py2 - py1)
    speed = abs(curr_area - prev_area) / (prev_area + 1e-6)
    return round(speed * 100, 2)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f" Failed to open video: {video_path}")
    exit()

frame_id = 0
max_frames = 300  # Limit frames for faster debugging

#  Test snapshot write before loop
ret, test_frame = cap.read()
if ret:
    test_test_path = os.path.join(snapshot_dir, "test_frame.jpg")
    if cv2.imwrite(test_test_path, test_frame):
        print(f" Snapshot test successful: {test_test_path}")
    else:
        print(f" Snapshot test failed to write: {test_test_path}")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to frame 0

# Start frame-by-frame processing
while cap.isOpened() and frame_id < max_frames:
    ret, frame = cap.read()
    if not ret:
        print(" End of video reached.")
        break

    frame_id += 1
    results = model(frame)[0]
    annotated = frame.copy()
    vehicle_counts = defaultdict(int)
    current_boxes = {}

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls_id = result
        cls_id = int(cls_id)
        if cls_id in VEHICLE_CLASSES and conf > 0.5:
            label = VEHICLE_CLASSES[cls_id]
            vehicle_counts[label] += 1
            box = [int(x1), int(y1), int(x2), int(y2)]
            color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Speed estimation
            speed = estimate_speed(cls_id, box, previous_boxes.get(cls_id))
            current_boxes[cls_id] = box

            # Label text
            text = f"{label} {conf:.2f} | Speed: {speed}"
            cv2.putText(annotated, text, (box[0], box[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Snapshot trigger
    total = sum(vehicle_counts.values())
    if total >= VEHICLE_THRESHOLD:
        count_str = "_".join([f"{k}{v}" for k, v in vehicle_counts.items()])
        fname = f"snapshot_{frame_id}_{count_str}.jpg"
        full_path = os.path.join(snapshot_dir, fname)
        saved = cv2.imwrite(full_path, annotated)
        if saved:
            print(f" Snapshot saved at frame {frame_id}: {full_path}")
        else:
            print(f" Failed to save snapshot at: {full_path}")

    previous_boxes = current_boxes

    # Optional: Display live output
    # cv2.imshow("Live", annotated)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
print(" Done! Snapshots saved to:", os.path.abspath(snapshot_dir))
