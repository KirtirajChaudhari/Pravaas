# Lab 4 — CNN and Transfer Learning

Mapping to the supplied M.Tech AI–ML 2026–27 practical assignment.

| Assignment objective | Implementation and evidence |
| --- | --- |
| Understand CNN convolution, activation, pooling, dense layers | `SmallCNN` in `ml/train.py`: 3 convolution/ReLU/max-pool stages, adaptive pooling, dropout and a dense classifier |
| Implement and train a CNN | Randomly initialized small CNN, with recorded 12-epoch history |
| Apply transfer learning | Torchvision MobileNetV3 Small with ImageNet pretrained weights |
| Compare feature extraction and fine-tuning | Frozen backbone with trained head, followed by unfreezing the last 3 feature blocks at lower learning rate |
| Accuracy, precision, recall, F1, confusion matrix | Calculated from held-out predictions with scikit-learn; displayed in Model lab |
| Compare compute, training time and performance | Parameter count, trainable count, stage duration, and inference latency recorded in experiment JSON |

## Experimental controls

Same source partitions, input dimensions, class order, preprocessing, and seed (42). Validation loss selects each checkpoint. Test images are not used to update weights or select checkpoints. Exact image hashes are checked before training; known remaining limitations include near-duplicates and capture-session similarity.

Report metrics from `dist/data/experiments.json`. Discuss false negatives (actual defective → predicted non-defective) and false positives separately. The classifier is not a detector: detector mAP and image-level classification accuracy cannot be substituted for each other.

Training from scratch describes the CNN's weights. Transfer learning starts with ImageNet weights; the two YOLO demonstration models use inherited project checkpoints. These origins are explicitly separated in the UI and model registry.
