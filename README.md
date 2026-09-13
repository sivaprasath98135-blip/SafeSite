# SafeSite

## RT-DETR Construction-Site Safety Detection System

SafeSite is a computer-vision safety monitoring system built around an RT-DETR-L object detector.

The system detects three classes:

- `head`
- `helmet`
- `person`

It also provides a FastAPI service for image detection and safety-oriented question answering.

---

## 1. Project Overview

The project demonstrates an end-to-end computer-vision workflow:

1. Dataset preparation
2. Dataset validation
3. RT-DETR model training
4. Test-set evaluation
5. Failure-case analysis
6. REST API deployment
7. Safety-oriented response logic

The trained RT-DETR-L `best.pt` weights are available in the GitHub Release:

https://github.com/sivaprasath98135-blip/SafeSite/releases/tag/v1.0

## 2. Dataset

The final cleaned dataset contains three classes:

```text
0: head
1: helmet
2: person
```

Dataset split:

| Split | Images |
|---|---:|
| Train | 4,915 |
| Validation | 1,412 |
| Test | 706 |

Ground-truth test instances:

| Class | Instances |
|---|---:|
| head | 726 |
| helmet | 1,917 |
| person | 64 |
| Total | 2,707 |

One test sample contained mixed segmentation/detection label rows and was ignored by the Ultralytics evaluation pipeline. Therefore the reported test metrics use 705 usable test images.

---

## 3. Model

Model: **RT-DETR-L**

Training configuration:

| Parameter | Value |
|---|---:|
| Epochs | 20 |
| Image size | 640 |
| Batch size | 8 |
| GPUs | 2 x Tesla T4 |
| Workers | 2 |

---

## 4. Test Results

| Metric | Result |
|---|---:|
| Precision | 0.6102 |
| Recall | 0.6622 |
| mAP@50 | 0.6662 |
| mAP@50-95 | 0.4692 |

Class-level results:

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| head | 0.875 | 0.978 | 0.979 | 0.694 |
| helmet | 0.899 | 0.978 | 0.986 | 0.693 |
| person | 0.057 | 0.031 | 0.0335 | 0.0208 |

The model performs strongly on head and helmet detection, while the person class is substantially weaker.

---

## 5. API

### Health check

```http
GET /health
```

### Object detection

```http
POST /detect
```

Returns image dimensions, detection count, class counts, bounding boxes, and confidence scores.

### Safety question answering

```http
POST /ask
```

Accepts an image and a question. The API supports simple intents for helmet, person/worker, head, and general safety questions.

---
## 5.1 Example API Usage

### `/detect`

Example request:

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example.jpg"
Example response:

```json
{
  "filename": "example.jpg",
  "image_width": 640,
  "image_height": 480,
  "detection_count": 2,
  "counts": {
    "head": 1,
    "helmet": 1,
    "person": 0
  },
  "detections": [
    {
      "class_id": 1,
      "class_name": "helmet",
      "confidence": 0.91,
      "bbox": [120.0, 80.0, 220.0, 180.0]
    }
  ]
}



```
### `/ask`

Example request:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "question=Are there any helmets?" \
  -F "file=@example.jpg"
Example response:

```json
{
  "question": "Are there any helmets?",
  "intent": "helmet_check",
  "answer": "I detected 1 helmet detection in the image.",
  "counts": {
    "head": 1,
    "helmet": 1,
    "person": 0
  },
  "detections": []
}

### Important

These are **example responses only**. The numbers are illustrative; we're not claiming that this exact image produced those results.

After the `/ask` example, your existing:


## 6. Running the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## 7. Safety Limitation

SafeSite is a computer-vision prototype. Its predictions should not be treated as definitive proof of workplace safety compliance.

The model can produce false negatives, false positives, localization errors, and reduced performance on small or occluded objects.

Human review remains important for safety-critical decisions.

---

## 8. Failure Analysis

Five representative failure cases were automatically identified by comparing predictions against ground-truth annotations.

### Case 1 — Person false negative

Image: `005304_jpg.rf.b9669bca89a84d84a04080a275fe4311.jpg`

The ground-truth person object was missed.

### Case 2 — Helmet false negative

Image: `005332_jpg.rf.2daa87e6fe3f78d1f722179eb72fdc6e.jpg`

The ground-truth helmet object was missed.

### Case 3 — Head false negative

Image: `005435_jpg.rf.2543fe45bf8e123be3fae36607b23c49.jpg`

The ground-truth head object was missed.

### Case 4 — Unmatched helmet detection

Image: `005526_jpg.rf.4ad9acabcfacadd60ab51de611d12e86.jpg`

Helmet prediction confidence: **0.8785**

IoU with nearby ground-truth box: **0.8916**

This is best interpreted as an unmatched/duplicate detection around an existing helmet rather than a completely nonexistent object.

### Case 5 — Helmet localization error

Image: `005813_jpg.rf.9e4f8cd6ecebf6464e555350ca09da50.jpg`

Helmet prediction confidence: **0.9201**

IoU: **0.7427**

The prediction has the correct class but is not localized as tightly as desired.

---

## 9. Project Structure

```text
SafeSite/
├── app.py
├── requirements.txt
├── weights/
│   ├── best.pt
│   └── last.pt
├── failure_cases/
│   ├── failure_case_1.jpg
│   ├── failure_case_2.jpg
│   ├── failure_case_3.jpg
│   ├── failure_case_4.jpg
│   ├── failure_case_5.jpg
│   └── failure_cases.csv
├── prediction_contact_sheet.jpg
├── rtdetr_training/
└── test_evaluation/
```

---

## 10. Reproducibility

The final model weights are included in `weights/`.

The cleaned dataset configuration is stored in:

```text
Hard-Hat-Workers-Clean/data.yaml
```

The API can be run using the supplied requirements file and trained model.

---

## 11. Conclusion

SafeSite demonstrates an end-to-end RT-DETR-based construction-site safety detection pipeline. The final model performs strongly on helmet and head detection, while the person class remains a significant weakness.

The evaluation results and failure cases make these limitations explicit. Additional person-focused training data, broader validation, threshold calibration, and human review would be appropriate before safety-critical deployment.
