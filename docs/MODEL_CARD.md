# Model card

**Purpose:** academic railway-image inspection and an evidence-backed CNN/transfer-learning comparison. Not a railway operations or automated safety system.

**Classification:** two classes in this exact order: `Defective`, `Non-defective`. Newly trained small CNN, MobileNetV3 Small feature extraction, and MobileNetV3 Small fine-tuning. Input: 128×128 RGB stretched to square. ImageNet normalization is embedded in the ONNX graph. Confidence is softmax output, not a calibrated probability of track safety.

**Localization:** inherited YOLOv8n, one class `railway`. It localizes regions represented by the reference labels; no crack/defect subtype is inferred. Inherited YOLO11s exposes Animal, Debris, Human, Object, Stone, Tree. Input: 416×416 RGB, letterboxed with 114 padding; confidence threshold default 0.40; class-aware NMS IoU 0.45; up to 100 output boxes.

**Dataset:** the requested Kaggle version 2. The manifest records source paths, decoded-pixel hashes and retained splits. Original partition labels are preserved. Test duplicates take priority, then validation, then training. Exact duplicates are removed, not visually similar captures. This is not an independent capture-group benchmark.

**Validation:** actual classification metrics and histories are in `dist/data/experiments.json`. Precision/recall/F1 use defective as positive. The selected checkpoint is the epoch with lowest validation loss. Each model is evaluated once on the held-out test split after selection. No verified independent detector mAP is available.

**Known limitations:** small dataset, capture-session correlation, unknown robustness to lighting/weather/camera changes, no OOD rejection, potential label noise, and uncertain lineage for some inherited reference annotations. Image resizing can obscure small faults. Misclassifications and false negatives are expected. Inherited obstacle examples are demonstration inputs, not evidence of held-out performance. The UI never marks a track operationally safe.

**Export:** classification exports embed normalization and softmax. YOLO exports return raw class scores and boxes; browser postprocessing applies clipping and NMS. Browser resampling differs slightly from Pillow/OpenCV and may move scores near a threshold. See verification results for measured numerical checks.

**Privacy:** inference runs in WASM on the visitor's device. The app does not upload images. Browser history stores filenames, timestamps, settings, and outputs; no image bytes. The host receives ordinary requests for static model and application assets.
