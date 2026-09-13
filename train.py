from pathlib import Path
from ultralytics import RTDETR

PROJECT_ROOT = Path("/kaggle/working/SafeSite")
DATA_YAML = Path("/kaggle/working/Hard-Hat-Workers-Clean/data.yaml")

MODEL = "rtdetr-l.pt"
EPOCHS = 20
IMG_SIZE = 640
BATCH = 8
PROJECT = PROJECT_ROOT / "rtdetr_training_repro"

print("Starting RT-DETR training...")
print(f"Dataset: {DATA_YAML}")
print(f"Model: {MODEL}")
print(f"Epochs: {EPOCHS}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH}")

model = RTDETR(MODEL)

results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    project=str(PROJECT),
    name="run",
    device=0,
)

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best weights: {PROJECT / 'run' / 'weights' / 'best.pt'}")
print(f"Last weights: {PROJECT / 'run' / 'weights' / 'last.pt'}")
