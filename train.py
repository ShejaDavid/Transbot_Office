import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import torch

# Configuration
project_name = "transbot_office_train"
data_path = "data.yaml"            # Points to your dataset
model_name = "yolov8n.pt"          # Use pretrained weights for transfer learning
epochs = 30
img_size = 640

# 2️⃣ Model Training
model = YOLO(model_name)
print("🚀 Starting YOLOv8 training with transfer learning...")
results = model.train(
    data=data_path,
    epochs=epochs,
    imgsz=img_size,
    batch=8,
    name=project_name
)

# 3️⃣ Load Best Weights
weights_path = os.path.join("runs", "detect", project_name, "weights", "best.pt")
if os.path.exists(weights_path):
    print(f"\n✅ Training complete. Best model saved at: {weights_path}")
else:
    print("\n⚠️ Could not find best.pt file — check your runs/detect folder.")

model = YOLO(weights_path)

# 4️⃣ Model Evaluation
print("\n📊 Evaluating model on validation set...")
val_results = model.val()
print(val_results)

# 5️⃣ Extract Detection Metrics
map50 = val_results.box.map50
precision = val_results.box.mp
recall = val_results.box.mr

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"mAP@50: {map50:.4f}")

# 6️⃣ Generate Predictions for Confusion Matrix & F1
preds, truths = [], []
for batch in val_results:
    for pred, gt in zip(batch.boxes.cls.cpu().numpy(), batch.orig_shape):  # Access class IDs
        preds.append(pred)
    for gt_label in batch.names.values():
        truths.append(gt_label)

# Dummy handling in case evaluation set is empty
if len(preds) > 0 and len(truths) > 0:
    preds = np.array(preds, dtype=int)
    truths = np.array(truths, dtype=int)

    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average="macro")
    conf_mat = confusion_matrix(truths, preds)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Plot F1 and Accuracy
    metrics = {"Accuracy": acc, "Macro F1-score": f1, "Precision": precision, "Recall": recall, "mAP@50": map50}
    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values(), color=["skyblue", "salmon", "lime", "orange", "violet"])
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    plt.show()

    # Save metrics to file
    with open("evaluation_results.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
else:
    print("⚠️ No predictions or labels found for evaluation set.")

# 7️⃣ Plot YOLO loss and precision curves (from training logs)
results_path = os.path.join("runs", "detect", project_name)
if os.path.exists(results_path):
    print(f"\n📈 Opening YOLO training results in: {results_path}")
    try:
        from ultralytics.utils.plotting import plot_results
        plot_results(file=os.path.join(results_path, "results.csv"))
    except Exception as e:
        print(f"⚠️ Could not plot YOLO results: {e}")
else:
    print("⚠️ Training results folder not found.")
