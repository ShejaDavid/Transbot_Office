🧠 Transbot Office Object Detection
A real-time YOLOv8 object detection system for the Yahboom Transbot SE Robot, designed to recognize and localize common office items. The system can detect objects through live camera feed or uploaded media, display results via a Streamlit interface, and is ready for future integration with ROS2 modules for full robotic autonomy.

Table of Contents
	1. Project Overview
	2. Installation
  3. How to Run
	4. Examples
  5. Expected Outputs
	6. Optional: Training / Fine-tuning
  7. Troubleshooting
      
1️⃣ Project Overview

This project uses YOLOv8 for detecting common office objects in images. It contains:
- Dataset preprocessing scripts
- Training scripts (train.py)
- Pretrained models in the export/ folder
- Run outputs (training metrics, validation results) in the runs/ folder
- Streamlit UI for real-time detection
  
2️⃣ Installation

Requirements:

- Python 3.11+
- pip
- Git
- Git LFS (for large files)
  
Steps:
•	1. Clone the repository:
git clone https://github.com/ShejaDavid/Transbot_Office.git
cd Transbot_Office

•	2. Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate  # Mac/Linux

# OR

venv\Scripts\activate  # Windows

•	3. Install required packages:

pip install -r requirements.txt

•	4. Initialize Git LFS:
git lfs install

Dependencies include:

YOLOv8 and related libraries:
- ultralytics==8.3.222
- torch>=2.2.2
- opencv-python>=4.7.0

UI:
- streamlit>=1.25.0

Data and metrics:
- numpy>=1.26.4
- matplotlib>=3.7.2
- seaborn>=0.12.2
- scikit-learn>=1.3.2
- PyYAML>=6.0
  
3️⃣ How to Run

Launch Streamlit UI:

streamlit run app.py

Command-line options (optional):
Run inference:
python detect.py --weights runs/detect/train/weights/best.pt --source data/images/test/
Evaluate model performance:

python val.py --weights runs/detect/train/weights/best.pt --data data.yaml

4️⃣ Examples

Training Metrics:

- Precision, recall, F1 score: runs/detect/train/BoxP_curve.png
- Confusion matrix: runs/detect/train/confusion_matrix.png

Sample Predictions:

- Original: data/images/test/sample1.jpg
- Predicted: runs/detect/val/val_batch0_pred.jpg
  
5️⃣ Expected Outputs

1. runs/detect/train/ folder with:
- BoxP_curve.png
- BoxR_curve.png
- BoxF1_curve.png
- confusion_matrix.png
  
2. runs/detect/val/ folder with validation predictions
3. results.csv
4. Streamlit interface showing live results
   
6️⃣ Optional: Training / Fine-tuning

The model comes pre-trained for office object detection. To fine-tune with your own data:

1. Place images and labels in the dataset folder
2. Run:
python train.py --data data.yaml --cfg yolov8n.yaml --weights yolov8n.pt

3. Adjust parameters in train.py or data.yaml as needed.
   
7️⃣ Troubleshooting

- Streamlit not found: Activate virtual environment and install streamlit
- Module not found: Verify dependencies are installed
- CUDA errors: Check PyTorch GPU setup
- Large file push fails: Exclude venv/ and use Git LFS
- Missing images: Confirm dataset in data/ folder

Author: Alpha Robotics
GitHub: https://github.com/ShejaDavid/Transbot_Office

