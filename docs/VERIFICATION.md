# Verification — 2026-10-27

## Completed checks

- `npm test`: four geometry tests passed (IoU, class-aware NMS, reverse letterboxing, and rejection of an incompatible output shape).
- `npm run check`: both browser JavaScript modules passed syntax checks.
- `npm run build`: all five models, external weight chunks, sample images, and three measured experiments are present; no model asset exceeds 25 MiB.
- `python ml/verify_exports.py`: all five ONNX graphs passed checker validation and returned finite predictions. Classification exports were compared against their selected PyTorch checkpoints on a deterministic RGB tensor. Maximum absolute probability error was **3.58 × 10⁻⁷**. This verifies the export math, not model accuracy on unseen routes.
- Dataset audit: **380 unique decoded-image hashes**, with **296 train / 62 validation / 22 test** records. Three exact duplicates removed. Near-duplicate/capture-group leakage was not ruled out.
- `CHROMIUM_PATH=/usr/bin/chromium node tests/browser.mjs`: passed actual WASM inference for **all five models**, JSON/CSV downloads, image upload and invalid-format rejection, bounding-box visibility, expanded image view, navigation, history persistence after reload, history clearing, and a 390 px mobile viewport without horizontal page overflow. No browser JavaScript errors.
- Desktop (1440 px) and mobile (390 px) screenshots reviewed. A mobile sidebar overflow issue found in the first review was fixed and retested.

## Measured classification results

All results use the same 22 test images (11 per class). Defective is positive.

| Model | Accuracy | Precision | Recall | F1 | Confusion matrix (actual rows) |
| --- | ---: | ---: | ---: | ---: | --- |
| Scratch CNN | 72.73% | 85.71% | 54.55% | 66.67% | `[[6,5],[1,10]]` |
| MobileNetV3 frozen | 72.73% | 77.78% | 63.64% | 70.00% | `[[7,4],[2,9]]` |
| MobileNetV3 fine-tuned | 72.73% | 77.78% | 63.64% | 70.00% | `[[7,4],[2,9]]` |

Fine-tuning improved validation loss but did not improve this test confusion matrix over frozen features. These short CPU experiments establish functioning baselines, not state-of-the-art performance. No test-driven tuning or detector mAP claim was made.

## Limits

Browser testing used Linux Chromium; Safari, Firefox, WebGPU, and physical mobile devices were not tested. The application uses single-threaded WASM for portability. Independent detector mAP, fresh Roboflow export/retraining, long-duration load tests, and route-level safety validation were not completed. The Roboflow site returned 403 and the supplied URL did not specify a pinned export.

Raw export checks are in `dist/data/verification.json`; epoch-level metrics and hardware configuration are in `dist/data/experiments.json`.
