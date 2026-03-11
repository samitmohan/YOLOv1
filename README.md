# YOLOv1 - From Scratch in PyTorch

A from-scratch implementation of [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (Redmon et al., 2016) in PyTorch, trained on Pascal VOC.

**Blog post**: https://samitmohan.github.io/2025/11/21/yolo.html

## Architecture

```
Input (448x448x3)
    |
ResNet34 Backbone (pretrained on ImageNet)
    |
3x Conv2d + BatchNorm + LeakyReLU (detection head)
    |
1x1 Conv (or FC layers)
    |
Output: 7x7x30  ->  S*S*(5B+C)
                     S=7 grid, B=2 boxes/cell, C=20 classes
```

Each grid cell predicts:
- **2 bounding boxes**: (x_offset, y_offset, sqrt_w, sqrt_h, confidence) per box
- **20 class probabilities**: conditional on object presence

Single forward pass produces all detections - no region proposals needed.

## Key Implementation Details

- **Backbone**: ResNet34 pretrained on ImageNet (replaces the paper's custom darknet for faster convergence)
- **Loss**: Multi-part MSE loss with lambda_coord=5 and lambda_noobj=0.5 (Section 2.2 of paper)
  - Localization loss on (x, y, sqrt_w, sqrt_h) for responsible predictor boxes only
  - Objectness loss weighted by IoU for responsible boxes
  - No-object confidence loss to push non-responsible boxes toward zero
  - Classification loss (MSE on class probabilities)
- **NMS**: Per-class non-maximum suppression at inference
- **Evaluation**: mAP with VOC-style precision envelope (area under PR curve)

## Project Structure

```
implementation/
  config/voc.yaml       # All hyperparameters
  models/yolo.py        # YOLOV1 model (ResNet34 backbone + detection head)
  loss/loss.py          # YOLOv1 multi-part loss function
  dataset/voc.py        # Pascal VOC dataset loader + augmentations
  tools/train.py        # Training loop with SGD + MultiStepLR
  tools/inference.py    # Inference, NMS, mAP evaluation
  utils/visualise.py    # Bounding box + grid visualization
  yolov1.py             # Entry point
```

## Setup

```bash
cd implementation
pip install -r requirements.txt
```

### Dataset

Download Pascal VOC 2007 and 2012:

```bash
mkdir -p data && cd data

# VOC 2007 trainval
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar && mv VOCdevkit/VOC2007 VOC2007

# VOC 2012 trainval
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_11-May-2012.tar && mv VOCdevkit/VOC2012 VOC2012

# VOC 2007 test
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar && mv VOCdevkit/VOC2007 VOC2007-test
```

## Usage

### Train

```bash
python yolov1.py --mode train --config config/voc.yaml
```

Training runs for 135 epochs with SGD (lr=0.001, momentum=0.9, weight_decay=5e-4).
Learning rate is halved at epochs [50, 75, 100, 125].

### Inference (sample detections)

```bash
python yolov1.py --mode infer --config config/voc.yaml
```

Saves predicted bounding boxes and class probability grid maps to `samples/`.

### Evaluate mAP

```bash
python yolov1.py --mode evaluate --config config/voc.yaml
```

Computes per-class AP and mean AP on VOC2007 test set.

## Configuration

All hyperparameters are in `config/voc.yaml`:

| Parameter | Value | Description |
|---|---|---|
| Grid size (S) | 7 | Image divided into 7x7 grid |
| Boxes per cell (B) | 2 | Each cell predicts 2 bounding boxes |
| Classes (C) | 20 | Pascal VOC classes |
| Input size | 448x448 | Standard YOLO input resolution |
| Batch size | 64 | Training batch size |
| Epochs | 135 | Total training epochs |
| Optimizer | SGD | momentum=0.9, weight_decay=5e-4 |

## References

- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) - Redmon et al., 2016
- [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
