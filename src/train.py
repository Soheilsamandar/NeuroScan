import os
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(
    data=r"YOLO_Dataset\data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    workers=4,
    device="cpu",
    project="BrainTumor",
    name="YOLOv8s",
    patience=5,
)

### Info from model
model = YOLO(r"runs\detect\BrainTumor\YOLOv8s\weights\best.pt")
print("Model Loaded Successfully")
print("============================")
print("========== Classes ==========")
for id, name in model.names.items():
    print(f"Class ID {id}: {name}")
print("============================")
parameters = sum(p.numel() for p in model.model.parameters())
print(f"Total Parameters: {parameters:,}")
print("============================================")
print("Input Image Size: 640x640")
print("========== Training Configuration ==========")
print("Epochs: 10")
print("Image Size: 640")
print("Batch Size: 16")
print("Device: CPU")
print("Optimizer: YOLO Default")
print("============================================")
print("Model Path:")
print(r"runs\detect\BrainTumor\YOLOv8s\weights\best.pt")
print("Model Size:")
size = os.path.getsize(r"runs\detect\BrainTumor\YOLOv8s\weights\best.pt") / (
    1024 * 1024
)
print(f"{size:.2f} MB")
