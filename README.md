🚦 Traffic Vehicle Detection System
This project is part of an internship assessment to build an automated computer vision system that detects, classifies, and counts vehicles in traffic images and video. The system is powered by YOLOv8 (pre-trained), and includes innovations like real-time snapshot generation and mock speed estimation.

📌 Objective
To detect vehicles (cars, trucks, motorcycles) in traffic images with bounding boxes, classification, confidence scores, and total counts. The system should save annotated images and support real-time video processing.

🔧 Technology Stack
Language: Python 3.8+

Libraries: OpenCV, NumPy, Matplotlib

ML Framework: Ultralytics YOLOv8 (pre-trained weights)

📁 Project Structure

traffic-detection-assignment/

├── README.md

├── requirements.txt

├── main.py                          # Image detection script

├── innovative_video_processor.py   # Video + Innovation logic

├── detector.py                     # Detection helper functions

├── utils.py                        # Utilities (if needed)

├── data/

│   ├── test_images/                # 10 test images

│   └── traffic_video.mp4           # Video file (for innovation)

├── output/

│   ├── processed_images/           # Annotated test image results

│   └── snapshots/                  # Snapshots from video processing

└── docs/

    ├── technical_report.pdf        # 2-page report
    
    └── presentation_slides.pdf     # Final presentation (10 minutes)
    
🛠 Setup Instructions

Step 1: Clone the Repo

git clone https://github.com/yashcharde11/traffic-detection

cd traffic-detection-assignment

Step 2: Install Dependencies

pip install -r requirements.txt

Or install manually:
pip install opencv-python numpy matplotlib torch torchvision ultralytics

▶️ How to Run
🔍 For Test Images
python main.py
Loads test images from data/test_images/

Detects vehicles

Saves output to output/processed_images/

🎥 For Video (with Innovations)
python innovative_video_processor.py
Processes video: data/traffic_video.mp4

Detects vehicles per frame

Saves automatic snapshots to output/snapshots/

Estimates mock vehicle speed using bounding box size changes

✨ Features
✅ Core
Vehicle detection with YOLOv8

Classification into cars, trucks, motorcycles

Confidence scoring for each prediction

Total vehicle count by category

Annotated image outputs

💡 Bonus Innovations

1. Automatic Traffic Snapshot Generator: Saves video frames when vehicle count exceeds threshold.

2. Speed Estimation Using Object Size: Estimates relative speed of vehicles by comparing bounding box size change across frames (mock logic).

📊 Sample Output
Images with bounding boxes and labels

Confidence scores shown next to each vehicle

Vehicle counts stored in filename/summaries

Snapshots saved automatically for heavy traffic scenes in video

📄 Deliverables
Code (Python scripts)

requirements.txt

10 test image outputs (annotated)

Video snapshots (with speed estimates)

Technical report (docs/technical_report.pdf)

Presentation slides (docs/presentation_slides.pdf)

🧠 Model Selection
YOLOv8 (n model) from Ultralytics chosen for its:

Speed and real-time capability

Accuracy on COCO vehicle classes

Simplicity of integration without training

📈 Summary Report Example

Image: test1.jpg

Cars: 4

Trucks: 1

Motorcycles: 2
