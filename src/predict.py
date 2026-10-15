import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO(r"runs\detect\BrainTumor\YOLOv8s\weights\best.pt")
show = True
conf_me = 0.05
img_tst_count = 100
yes_list, no_list, noimg = [], [], 0
highest_confidence = 0
highest_image = ""
for i in range(2, img_tst_count):
    image_path = f"image test\image({i}).jpg"
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        noimg += 1
        continue
    results = model.predict(source=image_path, conf=conf_me, save=False, show=False)
    result = results[0]
    original_img = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
    if len(result.boxes) == 0:
        status = "No Tumor Detected"
        no_list.append(image_path)
        annotated_img = original_img.copy()
    else:
        yes_list.append(image_path)
        confidences = result.boxes.conf.cpu().numpy()
        best_index = confidences.argmax()
        best_confidence = confidences[best_index]
        best_box = result.boxes.xyxy[best_index].cpu().numpy()
        x1, y1, x2, y2 = map(int, best_box)
        annotated_img = original_img.copy()
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"Tumor {best_confidence * 100:.2f}%"
        cv2.putText(
            annotated_img,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
        status = f"Tumor: {best_confidence * 100:.2f}%"
        if best_confidence > highest_confidence:
            highest_confidence = best_confidence
            highest_image = image_path
    print(f"image({i}) -> {status}")
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(7, 5))
        axes[0].imshow(original_img)
        axes[0].set_title("Before Detection")
        axes[0].axis("off")
        axes[1].imshow(annotated_img)
        axes[1].set_title("Best Detection Only")
        axes[1].axis("off")
        plt.suptitle(f"image({i}) - {status}")
        plt.tight_layout()
        plt.show()


### Info from the all images
total_images = len(yes_list) + len(no_list)
print("=" * 40)
print("FINAL RESULTS")
print("=" * 40)
print(f"Total Images: {total_images}")
print(f"Tumor Detected: {len(yes_list)}")
print(f"No Tumor Detected: {len(no_list)}")
print(f"No Image Path: {noimg}")
if highest_confidence > 0:
    print(f"Highest Confidence: " f"{highest_confidence * 100:.2f}%")
    print(f"Highest Confidence Image: {highest_image}")
else:
    print("Highest Confidence: 0.00%")
print("=" * 40)
img = cv2.imread(highest_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = model.predict(source=highest_image, conf=0.1, save=False, show=False)
result = results[0]
if len(result.boxes) > 0:
    confidences = result.boxes.conf.cpu().numpy()
    best_index = confidences.argmax()
    box = result.boxes.xyxy[best_index].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    confidence = confidences[best_index]
    cv2.putText(
        img,
        f"Tumor {confidence * 100:.2f}%",
        (x1, max(y1 - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        1,
    )
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f"Highest Confidence Image - " f"{confidence * 100:.2f}%")
    plt.axis("off")
    plt.show()


from ultralytics import YOLO

model = YOLO(r"runs\detect\BrainTumor\YOLOv8s\weights\best.pt")
results = model.val(
    data=r"dataset\YOLO_Dataset\data.yaml", imgsz=640, conf=0.02, verbose=True
)
print("\n===== RESULTS =====")
pr = results.box.mp
recall = results.box.mr
map501 = results.box.map50
map5095 = results.box.map
print("Precision:", pr)
print("Recall:", recall)
print("mAP50:", map501)
print("mAP50-95:", map5095)
