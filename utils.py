import os
import cv2

def load_images(folder):
    images = []
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, file_name)
            img = cv2.imread(path)
            images.append((file_name, img))
    return images

def save_results(image, filename, counts, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"processed_{filename}")
    cv2.imwrite(out_path, image)
    print(f"Saved: {out_path}")
