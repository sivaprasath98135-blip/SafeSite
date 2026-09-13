# SafeSite — Technical Memo

## 1. Executive Summary
SafeSite is a construction-site safety detection system based on the RT-DETR object detection architecture. The system detects three classes: head, helmet, and person. A cleaned Hard Hat Workers dataset was used for training, validation, and testing. The final model was trained for 20 epochs at 640-pixel image size using two Tesla T4 GPUs.

The final test evaluation achieved 0.6102 precision, 0.6622 recall, 0.6662 mAP@50, and 0.4692 mAP@50-95. Helmet and head detection performed strongly, while person detection remained substantially weaker because the person class was sparsely represented in the evaluated test set.

## 2. Objective
The objective of SafeSite is to provide an automated computer-vision system capable of identifying construction-site safety-relevant objects from images. The system focuses on detecting workers and head/helmet-related classes and exposes the trained detector through a FastAPI service.

## 3. Dataset Preparation
The source dataset was obtained from the Roboflow Hard Hat Workers project. The project was cleaned and reorganized into three detection classes:
- head
- helmet
- person

Dataset split sizes:
- Training: 4,915 images and 4,915 labels
- Validation: 1,412 images and 1,412 labels
- Test: 706 images and 706 labels

Label validation found zero invalid label rows across the training, validation, and test splits.

Ground-truth test instances were distributed as follows:
- head: 726
- helmet: 1,917
- person: 64
- Total: 2,707

During Ultralytics test evaluation, one test sample contained incompatible mixed segmentation/detection annotation content and was ignored. Therefore, the reported test metrics are based on 705 usable test images and 2,704 evaluated instances. This limitation is documented rather than hidden.

## 4. Model and Training
Model architecture: RT-DETR-L.

Training configuration:
- Epochs: 20
- Image size: 640
- Batch size: 8
- Devices: two Tesla T4 GPUs
- Workers: 2

The trained model contains approximately 31.99 million parameters and requires approximately 105.3 GFLOPs according to the Ultralytics model summary.

Training completed in approximately 1.382 hours.

The best RT-DETR-L checkpoint is provided as the best.pt asset in the GitHub Release v1.0.

## 5. Evaluation
Validation metrics:
- Precision: 0.958
- Recall: 0.634
- mAP@50: 0.663
- mAP@50-95: 0.462

Validation class performance:
- head: precision 0.912, recall 0.949, mAP@50 0.972, mAP@50-95 0.676
- helmet: precision 0.962, recall 0.953, mAP@50 0.986, mAP@50-95 0.691
- person: precision 1.000, recall 0.000, mAP@50 0.0324, mAP@50-95 0.0198

Test metrics:
- Precision: 0.6102
- Recall: 0.6622
- mAP@50: 0.6662
- mAP@50-95: 0.4692

Test class performance:
- head: precision 0.875, recall 0.978, mAP@50 0.979, mAP@50-95 0.694
- helmet: precision 0.899, recall 0.978, mAP@50 0.986, mAP@50-95 0.693
- person: precision 0.057, recall 0.0312, mAP@50 0.0335, mAP@50-95 0.0208

The results show that the detector is highly effective for the head and helmet classes. Person detection is considerably weaker, which is consistent with the much smaller number of person examples in the test annotations.

## 6. Failure Analysis
Five representative failure cases were selected programmatically from the test predictions.

### Case 1 — Missed Person
Image: 005304_jpg.rf.b9669bca89a84d84a04080a275fe4311.jpg

A ground-truth person instance was not matched by a prediction. This represents a false-negative detection for the person class.

### Case 2 — Missed Helmet
Image: 005332_jpg.rf.2daa87e6fe3f78d1f722179eb72fdc6e.jpg

A ground-truth helmet instance was missed by the detector. This demonstrates that helmet detection can still fail in individual image conditions despite strong overall helmet metrics.

### Case 3 — Missed Head
Image: 005435_jpg.rf.2543fe45bf8e123be3fae36607b23c49.jpg

A ground-truth head instance was not matched by a prediction. This represents a false-negative head detection.

### Case 4 — Unmatched/Duplicate Helmet Detection
Image: 005526_jpg.rf.4ad9acabcfacadd60ab51de611d12e86.jpg

A helmet prediction with confidence approximately 0.8785 was not matched under the selected matching procedure. The prediction had a high overlap of approximately 0.8916 with a nearby ground-truth helmet, so this case is more accurately described as an unmatched or duplicate-style detection around an existing helmet rather than a completely nonexistent helmet.

### Case 5 — Helmet Localization Error
Image: 005813_jpg.rf.9e4f8cd6ecebf6464e555350ca09da50.jpg

A helmet prediction had confidence approximately 0.9201 and IoU approximately 0.7427 with the corresponding ground-truth object. Because this is below the 0.75 IoU threshold used in the failure analysis, it was classified as a localization error.

## 7. API Architecture
SafeSite exposes the trained detector through FastAPI.

Available endpoints:
- GET /health — checks whether the API is available.
- POST /detect — accepts an uploaded image and returns detected objects, confidence scores, bounding boxes, image dimensions, and class counts.
- POST /ask — accepts an image and a natural-language question. A simple intent-routing layer interprets questions related to helmets, workers/persons, heads, safety/compliance, or general detections and returns a structured answer together with the detections.

The API downloads the trained best.pt checkpoint from the GitHub Release v1.0 on first startup and loads it locally.

## 8. Safety and Reliability Considerations
SafeSite is a computer-vision assistance system and should not be treated as a replacement for trained safety personnel, site procedures, or certified protective-equipment inspection.

The model can miss objects, particularly small, distant, occluded, or visually ambiguous workers. The weak person-class test performance demonstrates that the current model should not be relied upon as the sole mechanism for worker detection.

Predictions should therefore be interpreted as decision-support information rather than a guarantee of site safety or regulatory compliance.

## 9. Reproducibility
The project directory contains the trained model checkpoints, API implementation, dependency list, README documentation, technical memo, prediction contact sheet, failure-case images, and training artifacts.

The primary trained checkpoint is provided as the best.pt asset in the GitHub Release v1.0.

Required Python dependencies are listed in SafeSite/requirements.txt.

The project uses the cleaned dataset configuration and the RT-DETR training configuration described in this memo.

## 10. Conclusion
SafeSite demonstrates a complete construction-site object-detection workflow covering dataset preparation, model training, quantitative evaluation, visual prediction analysis, failure-case analysis, and API deployment.

The strongest results are obtained for helmet and head detection, with test mAP@50 values of 0.986 and 0.979 respectively. Person detection remains the main limitation, with a test mAP@50 of 0.0335. Future improvement should therefore prioritize better representation and annotation coverage for the person class, additional challenging examples, and further evaluation under varied construction-site conditions.

Overall, the project provides a reproducible RT-DETR-based foundation for construction-site safety monitoring while clearly documenting its current limitations.
