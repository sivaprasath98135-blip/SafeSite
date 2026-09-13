
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from ultralytics import RTDETR
from PIL import Image
import tempfile
import os
import io
import urllib.request

# ============================================================
# SafeSite API
# ============================================================

MODEL_PATH = "https://github.com/sivaprasath98135-blip/SafeSite/releases/download/v1.0/best.pt"

CLASS_NAMES = {
    0: "head",
    1: "helmet",
    2: "person"
}

CONFIDENCE_THRESHOLD = 0.25

app = FastAPI(
    title="SafeSite",
    description="Construction-site safety detection using RT-DETR.",
    version="1.0.0"
)

# Load model once when the server starts.
MODEL_FILE = "/tmp/best.pt"

if not os.path.exists(MODEL_FILE):
    urllib.request.urlretrieve(MODEL_PATH, MODEL_FILE)

model = RTDETR(MODEL_FILE)


def run_detection(image_bytes: bytes):
    """Run RT-DETR detection on uploaded image bytes."""

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # Temporary file because Ultralytics accepts image paths reliably.
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as tmp:

        image.save(tmp.name, format="JPEG")
        temp_path = tmp.name

    try:
        results = model.predict(
            source=temp_path,
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            device=0,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is not None:
            boxes = result.boxes

            for box, cls, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy()
            ):
                class_id = int(cls)
                confidence = float(conf)

                detections.append({
                    "class_id": class_id,
                    "class_name": CLASS_NAMES.get(
                        class_id,
                        str(class_id)
                    ),
                    "confidence": round(confidence, 4),
                    "bbox": [
                        round(float(box[0]), 2),
                        round(float(box[1]), 2),
                        round(float(box[2]), 2),
                        round(float(box[3]), 2)
                    ]
                })

        return detections, image.size

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def summarize_detections(detections):
    """Create a simple safety-oriented summary."""

    counts = {
        "head": 0,
        "helmet": 0,
        "person": 0
    }

    for detection in detections:
        name = detection["class_name"]

        if name in counts:
            counts[name] += 1

    return counts


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "RT-DETR-L",
        "classes": list(CLASS_NAMES.values()),
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):
    """Detect heads, helmets and persons in an uploaded image."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    detections, image_size = run_detection(image_bytes)
    counts = summarize_detections(detections)

    return {
        "filename": file.filename,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "detection_count": len(detections),
        "counts": counts,
        "detections": detections
    }


@app.post("/ask")
async def ask(
    question: str = Form(...),
    file: UploadFile = File(...)
):
    q = question.lower().strip()

    image_related_terms = [
        "helmet", "hard hat", "hardhat",
        "person", "worker", "people",
        "head", "safe", "safety",
        "violation", "compliance"
    ]

    if not any(term in q for term in image_related_terms):
        return {
            "question": question,
            "intent": "unrelated_question",
            "answer": (
                "This question is outside the image-detection capabilities "
                "of SafeSite, so the detection model was not called."
            ),
            "counts": {},
            "detections": [],
            "note": "No image detection was required for this question."
        }
    """
    Answer a simple safety question using detections
    from the uploaded image.
    """

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    detections, image_size = run_detection(image_bytes)
    counts = summarize_detections(detections)

    q = question.lower().strip()

    # --------------------------------------------------------
    # Intent routing
    # --------------------------------------------------------

    if "helmet" in q or "hard hat" in q or "hardhat" in q:
        helmet_count = counts["helmet"]

        if helmet_count > 0:
            answer = (
                f"I detected {helmet_count} helmet detection"
                f"{'s' if helmet_count != 1 else ''} in the image."
            )
        else:
            answer = (
                "I did not detect a helmet in the image. "
                "This does not prove that no helmet is present; "
                "the detector can miss small, occluded, or distant objects."
            )

        intent = "helmet_check"

    elif "person" in q or "worker" in q or "people" in q:
        person_count = counts["person"]

        answer = (
            f"I detected {person_count} person detection"
            f"{'s' if person_count != 1 else ''} in the image."
        )

        intent = "person_check"

    elif "head" in q:
        head_count = counts["head"]

        answer = (
            f"I detected {head_count} head detection"
            f"{'s' if head_count != 1 else ''} in the image."
        )

        intent = "head_check"

    elif (
        "safe" in q
        or "safety" in q
        or "violation" in q
        or "compliance" in q
    ):
        if counts["person"] == 0:
            answer = (
                "No person detections were found. "
                "The image cannot be reliably assessed for worker "
                "helmet compliance from this detector alone."
            )
        elif counts["helmet"] >= counts["person"]:
            answer = (
                "The detector found helmets and persons in the image. "
                "However, helmet detection alone should not be treated "
                "as definitive safety compliance because detections "
                "can be missed or incorrectly localized."
            )
        else:
            answer = (
                "The detector found fewer helmets than person detections. "
                "This may indicate possible helmet-compliance concerns, "
                "but it should be reviewed by a human because the model "
                "can produce false negatives."
            )

        intent = "safety_assessment"

    else:
        answer = (
            "I can analyze the image for detected heads, helmets, "
            "and persons. Try asking about helmets, workers, heads, "
            "or general safety."
        )

        intent = "general_detection"

    return {
        "question": question,
        "intent": intent,
        "answer": answer,
        "counts": counts,
        "detections": detections,
        "note": (
            "Results are model predictions and should not be treated "
            "as a definitive safety determination."
        )
    }
