from ultralytics import YOLO

# Load your previously trained model
model = YOLO("runs/detect/transbot_office_train/weights/best.pt")


# Continue training (fine-tuning)
model.train(
    data="data.yaml",
    epochs=10,        # additional fine-tuning epochs
    imgsz=640,
    batch=8,
    name="train_finetune",   # new run folder name
    lr0=0.0005,       # smaller learning rate for fine-tuning
    patience=30       # early stop if no improvement
)
