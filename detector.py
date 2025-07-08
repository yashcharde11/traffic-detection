from ultralytics import YOLO
import cv2

# COCO Class IDs
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    7: "truck"
}

class VehicleDetector:
    def __init__(self, model_type="yolov8n"):
        self.model = YOLO(f"{model_type}.pt")

    def detect_vehicles(self, image):
        results = self.model.predict(image)[0]
        detections = []
        counts = {"car": 0, "truck": 0, "motorcycle": 0}

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if cls_id in VEHICLE_CLASSES and conf > 0.5:
                label = VEHICLE_CLASSES[cls_id]
                counts[label] += 1
                detections.append({
                    "box": box.xyxy[0].tolist(),
                    "label": label,
                    "confidence": conf
                })
        return detections, counts

    def draw_results(self, image, detections, counts):
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            text = f"{det['label']} {det['confidence']:.2f}"
            cv2.putText(annotated, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # Display count summary
        y = 30
        for vehicle, count in counts.items():
            cv2.putText(annotated, f"{vehicle}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            y += 30
        return annotated
