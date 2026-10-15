<div align="center">

# 🧠 NeuroScan

### Brain Tumor Detection Using YOLOv8

<p>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Status-Research%20Project-orange?style=for-the-badge" alt="Status">
</p>

<p>
  <b>NeuroScan</b> is a computer vision project for detecting tumor regions in brain MRI images using the YOLOv8 object detection framework.
</p>

</div>

---

## 📌 Overview

**NeuroScan** is an experimental computer vision project developed to explore the use of deep learning and object detection for identifying tumor regions in brain MRI images.

The project uses **YOLOv8s** to detect tumor regions and generate bounding boxes around detected areas.

The original dataset contains MRI images belonging to different tumor categories. For this project, the detection task was simplified into a single object class:

```text
0 → Tumor
```

The project focuses on the complete workflow of an object detection system:

* Dataset preparation
* YOLO annotation
* Model training
* Model prediction
* Bounding box visualization
* Confidence threshold analysis
* Model evaluation
* Result visualization

---

## 🎯 Project Goals

The main goals of NeuroScan are:

* Learn and implement YOLO-based object detection.
* Prepare MRI images for YOLO training.
* Detect tumor regions using bounding boxes.
* Analyze the effect of confidence thresholds.
* Evaluate the trained model using standard object detection metrics.
* Build a reproducible computer vision project.
* Document the complete machine learning workflow.

---

## 🧠 Model

The project uses:

<table>
<tr>
<th>Component</th>
<th>Configuration</th>
</tr>

<tr>
<td>Model</td>
<td>YOLOv8s</td>
</tr>

<tr>
<td>Task</td>
<td>Object Detection</td>
</tr>

<tr>
<td>Classes</td>
<td>1 — Tumor</td>
</tr>

<tr>
<td>Image Size</td>
<td>640 × 640</td>
</tr>

<tr>
<td>Epochs</td>
<td>10</td>
</tr>

<tr>
<td>Batch Size</td>
<td>16</td>
</tr>

<tr>
<td>Optimizer</td>
<td>Ultralytics Default</td>
</tr>

<tr>
<td>Device</td>
<td>CPU</td>
</tr>

</table>

---


<h2>📦 Libraries & Versions</h2>

<p>
NeuroScan was developed and tested using the following environment and libraries:
</p>

<table>
<thead>
<tr>
<th>Library</th>
<th>Version</th>
<th>Purpose</th>
</tr>
</thead>

<tbody>

<tr>
<td><b>Python</b></td>
<td>3.11.9</td>
<td>Programming Language</td>
</tr>

<tr>
<td><b>Ultralytics</b></td>
<td>8.4.150</td>
<td>YOLOv8 Object Detection</td>
</tr>

<tr>
<td><b>PyTorch</b></td>
<td>2.11.0</td>
<td>Deep Learning Framework</td>
</tr>

<tr>
<td><b>OpenCV</b></td>
<td>4.13.0.92</td>
<td>Image Processing</td>
</tr>

<tr>
<td><b>NumPy</b></td>
<td>1.26.4</td>
<td>Numerical Computing</td>
</tr>

<tr>
<td><b>Matplotlib</b></td>
<td>3.11.0</td>
<td>Data Visualization</td>
</tr>

<tr>
<td><b>Pillow</b></td>
<td>12.2.0</td>
<td>Image Processing</td>
</tr>

</tbody>
</table>

<br>

<h3>⚙️ Environment</h3>

<table>
<thead>
<tr>
<th>Component</th>
<th>Configuration</th>
</tr>
</thead>

<tbody>

<tr>
<td><b>Python</b></td>
<td>3.11.9</td>
</tr>

<tr>
<td><b>Operating System</b></td>
<td>Windows</td>
</tr>

<tr>
<td><b>Deep Learning Framework</b></td>
<td>PyTorch</td>
</tr>

<tr>
<td><b>Object Detection Framework</b></td>
<td>Ultralytics YOLOv8</td>
</tr>

<tr>
<td><b>Model</b></td>
<td>YOLOv8s</td>
</tr>

<tr>
<td><b>Input Image Size</b></td>
<td>640 × 640</td>
</tr>

<tr>
<td><b>Training Device</b></td>
<td>CPU</td>
</tr>

</tbody>
</table>

<br>

<h3>📥 Installation</h3>

<p>Install the required packages using:</p>

<pre><code>pip install ultralytics==8.4.150
pip install torch==2.11.0
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pillow</code></pre>

<p>Or install all dependencies from:</p>

<pre><code>pip install -r requirements.txt</code></pre>

<br>

<blockquote>
<b>Note:</b> Exact package versions are documented to improve reproducibility and reduce compatibility issues when running NeuroScan on another system.
</blockquote>



---


## 📂 Dataset

The dataset was converted into the YOLO object detection format.

The dataset contains two main types of images:

* Images containing tumor regions
* Images without tumor regions

Tumor images were manually annotated using bounding boxes.

The final detection task uses a single class:

```yaml
names:
  0: Tumor
```

### YOLO Dataset Structure

```text
YOLO_Dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

---

## 🏷️ Annotation

Each tumor region is represented using a YOLO-format label.

Example:

```text
0 0.512 0.483 0.231 0.194
```

The values represent:

```text
class_id
x_center
y_center
width
height
```

All coordinates are normalized between `0` and `1`.

---

## 🚀 Training

The model was trained using the Ultralytics YOLO framework.

Example training configuration:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="dataset/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    name="BrainTumor/YOLOv8s"
)
```

After training, the best model weights are stored as:

```text
best.pt
```

---

## 🔍 Prediction

The trained model can be used to detect tumors in new MRI images.

Example:

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")

results = model.predict(
    "test_image.jpg",
    conf=0.2,
    imgsz=640
)
```

The model returns:

* Detected object
* Bounding box
* Confidence score
* Class ID

---

## 📦 Bounding Box Visualization

For each detected tumor, NeuroScan can display a bounding box around the detected region.

The prediction result contains:

```text
Class
Confidence
Bounding Box
```

Example:

```text
Tumor
Confidence: 87.4%
```

This allows the prediction to be visually inspected rather than relying only on a numerical result.

---

## 🎚️ Confidence Threshold Analysis

One of the experiments performed in this project was analyzing the effect of the confidence threshold.

Different thresholds can be used during prediction:

```text
0.05
0.10
0.25
0.50
0.75
0.90
```

A lower threshold allows more predictions to pass through the detection filter, while a higher threshold makes the prediction filter more strict.

This experiment helps demonstrate how the number of displayed detections changes when the confidence threshold is modified.

> **Important:** Confidence threshold is a prediction filtering parameter and should not be interpreted as model accuracy.

---

## 📊 Model Evaluation

The trained YOLOv8s model was evaluated using standard object detection metrics.

The latest evaluation results were:

<table>
<tr>
<th>Metric</th>
<th>Result</th>
</tr>

<tr>
<td>Precision</td>
<td><b>79.08%</b></td>
</tr>

<tr>
<td>Recall</td>
<td><b>81.82%</b></td>
</tr>

<tr>
<td>mAP50</td>
<td><b>80.49%</b></td>
</tr>

<tr>
<td>mAP50-95</td>
<td><b>53.54%</b></td>
</tr>

</table>

### Metrics

**Precision**

Measures how many of the detected objects were actually correct according to the evaluation annotations.

**Recall**

Measures how many of the annotated target objects were successfully detected.

**mAP50**

Mean Average Precision calculated using an IoU threshold of 0.50.

**mAP50-95**

Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.

---

## 📈 Evaluation Visualization

The evaluation results can be visualized using a Radar Chart or other comparison charts.

The four main evaluation metrics are:

```text
Precision
Recall
mAP50
mAP50-95
```

The visualization provides a compact overview of the model's performance across multiple evaluation criteria.

---

## 🧪 Test Results

The model was also tested on individual MRI images.

For each image, the system can determine whether a tumor detection passes the selected confidence threshold.

Example output:

```text
Tumor detected
Confidence: 87.4%
```

or:

```text
No Tumor
```

The model can also display the original image together with the detected bounding box.

---

## 📁 Project Structure

```text
NeuroScan/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── dataset/
│   └── data.yaml
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── confidence_test.py
│
├── results/
│   ├── evaluation.png
│   ├── confidence.png
│   └── predictions/
│
├── docs/
│   └── NeuroScan_Report.pdf
│
└── weights/
    └── best.pt
```

> Large datasets and model weights may be excluded from the Git repository because of file-size limitations.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Soheilsamandar/NeuroScan.git
```

Enter the project directory:

```bash
cd NeuroScan
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Main libraries used in the project:

```text
ultralytics
torch
opencv-python
numpy
matplotlib
Pillow
```

The exact versions can be specified in `requirements.txt` to make the project reproducible.

---

## ▶️ Usage

### Train

```bash
python src/train.py
```

### Predict

```bash
python src/predict.py
```
---

## 📚 Project Documentation

A complete Persian project report is included separately.

The report contains:

* Project introduction
* Dataset preparation
* YOLO annotation
* Training process
* Model information
* Prediction experiments
* Confidence threshold experiments
* Bounding box visualization
* Test results
* Model evaluation
* Evaluation metrics
* Final results

## 🖼️ Model Output

The following images show examples of NeuroScan predictions on brain MRI images.

The model output includes the detected tumor region, bounding box, class label, and confidence score.

### 🔍 Detection Examples

<table>
<tr>
<td align="center">
<img src="img\test1.png" width="400">
<br>
<b>Detection Example 1</b>
</td>

<td align="center">
<img src="img\test2.png" width="400">
<br>
<b>Detection Example 2</b>
</td>
</tr>

<tr>
<td align="center">
<img src="img\test3.png" width="400">
<br>
<b>Detection Example 3</b>
</td>

<td align="center">
<img src="img\test4.png" width="400">
<br>
<b>Detection Example 4</b>
</td>
</tr>
</table>

---

### 📊 Evaluation Results

The following visualization summarizes the evaluation metrics obtained from the trained YOLOv8s model.

<p align="center">
  <img src="img\bar.png" width="700">
</p>

The evaluation includes:

* Precision
* Recall
* mAP50
* mAP50-95

---

### 🎚️ Confidence Threshold Experiment

Different confidence thresholds were tested to observe their effect on the model's predictions.

<p align="center">
  <img src="img\Confidence Threshold Experiment.png" width="700">
</p>

This experiment demonstrates how changing the confidence threshold affects the detections displayed by the model.

## ⚠️ Limitations

This project has several limitations.

* The dataset size and composition affect the model's performance.
* The model uses a single `Tumor` class.
* Results depend on the quality and accuracy of the annotations.
* Confidence threshold changes the displayed predictions.
* The model has not been clinically validated.
* The evaluation results are based on the project dataset and should not be interpreted as medical diagnostic performance.

---

## 🏥 Medical Disclaimer

**NeuroScan is an educational and research-oriented computer vision project.**

It is **not a medical device**, and its predictions should not be used for medical diagnosis, treatment decisions, or clinical decision-making.

The results presented in this repository describe the performance of an experimental machine learning model on the project's dataset.

---

## 🔮 Future Improvements

Possible future improvements include:

* Increasing the size and diversity of the dataset
* Improving annotation quality
* Training for more epochs
* Testing larger YOLO architectures
* Comparing different YOLO versions
* Improving data augmentation
* Evaluating on an independent dataset
* Adding a graphical user interface
* Improving visualization and reporting
* Comparing multiple confidence thresholds
* Separating tumor types into multiple classes

---

## 🛠️ Technologies

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python">
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square">
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=flat-square&logo=opencv">
<img src="https://img.shields.io/badge/NumPy-Numerical-orange?style=flat-square&logo=numpy">
<img src="https://img.shields.io/badge/Matplotlib-Visualization-blue?style=flat-square">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch">

</p>

---

## 👨‍💻 Author

<div align="center">

### Soheil Samandar

Computer Engineering Student
Robotics & Artificial Intelligence Enthusiast
Computer Vision & Machine Learning

<br>

<a href="https://github.com/Soheilsamandar">
  <img src="https://img.shields.io/badge/GitHub-SoheilSamandar-black?style=for-the-badge&logo=github">
</a>

</div>

---

<div align="center">

### 🧠 NeuroScan

**Computer Vision • Deep Learning • YOLOv8 • Medical Image Research**

</div>
