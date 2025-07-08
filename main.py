from detector import VehicleDetector
from utils import load_images, save_results

if __name__ == "__main__":
    image_folder = "data/"
    output_folder = "output/processed_images/"
    
    model_type = "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, etc.

    detector = VehicleDetector(model_type)
    images = load_images(image_folder)

    for img_path, img in images:
        results, counts = detector.detect_vehicles(img)
        output_img = detector.draw_results(img, results, counts)
        save_results(output_img, img_path, counts, output_folder)

    print("Processing completed.")
