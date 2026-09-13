from pathlib import Path
from ultralytics import RTDETR

PROJECT_ROOT = Path("/kaggle/working/SafeSite")
DATA_YAML = Path("/kaggle/working/Hard-Hat-Workers-Clean/data.yaml")
WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
OUTPUT = PROJECT_ROOT / "test_evaluation_repro"

print("Loading trained RT-DETR model...")
print(f"Weights: {WEIGHTS}")
print(f"Dataset: {DATA_YAML}")

model = RTDETR(str(WEIGHTS))

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    project=str(OUTPUT),
    name="evaluation",
)

print("=" * 60)
print("TEST EVALUATION COMPLETE")
print("=" * 60)

print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
print(f"mAP@50:    {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")
