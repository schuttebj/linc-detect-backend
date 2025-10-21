# Vehicle Classifier Training Guide

This directory contains training scripts for the EfficientNet-B0 vehicle classifier used in the two-stage classification pipeline.

## Overview

The two-stage classification system:
1. **YOLO11n** (Stage 1): Detects vehicle bounding boxes
2. **EfficientNet-B0** (Stage 2): Classifies refined vehicle types

This approach provides better accuracy than YOLO alone, especially for distinguishing similar vehicles (SUV vs pickup, delivery van vs box truck).

## Vehicle Classes

The classifier recognizes 9 refined vehicle types:

| Class | Description | Toll Class |
|-------|-------------|------------|
| `car` | Sedans, hatchbacks, coupes | Class 1 |
| `suv` | Consumer SUVs, crossovers | Class 1 |
| `pickup` | Consumer pickup trucks (F-150, Hilux, etc) | Class 1 |
| `van` | Passenger vans, minibuses | Class 1 |
| `delivery_van` | Commercial delivery vans (2 axles) | Class 2 |
| `box_truck` | Single-unit box trucks | Class 2 |
| `semi` | Semi-trucks, articulated vehicles | Class 2+ |
| `bus` | Buses | Class 2+ |
| `motorcycle` | Motorcycles | Class 1 |

## Training Data

### Option 1: MIO-TCD Dataset (Recommended for Initial Training)

**MIO-TCD (Moving Objects Classification)** is a large-scale traffic dataset with 11 vehicle classes.

**Download**: http://podoce.dinf.usherbrooke.ca/challenge/dataset/

**Size**: ~650,000 images

**Prepare the dataset:**

```bash
# 1. Download and extract MIO-TCD dataset
# Directory structure should be:
# miotcd/
#   articulated_truck/
#   bus/
#   car/
#   motorcycle/
#   pickup_truck/
#   single_unit_truck/
#   work_van/
#   ...

# 2. Prepare data for training
python training/prepare_miotcd_data.py \
  --miotcd_root /path/to/miotcd \
  --output_dir training/data/vehicles \
  --train_ratio 0.8

# This creates:
# training/data/vehicles/
#   train/
#     car/
#     suv/
#     pickup/
#     van/
#     delivery_van/
#     box_truck/
#     semi/
#     bus/
#     motorcycle/
#   val/
#     (same structure)
```

### Option 2: Custom Toll Footage (For Fine-Tuning)

Once you have toll gate footage, organize images manually:

```
training/data/toll_vehicles/
  train/
    car/ (50-100 images)
    suv/ (50-100 images)
    pickup/ (50-100 images)
    ...
  val/
    car/ (10-20 images)
    suv/ (10-20 images)
    ...
```

**Tips for custom data:**
- Include varied lighting conditions (day/night)
- Include different weather (clear/rain)
- Capture from your actual camera angle
- Label carefully - quality > quantity

## Training

### Initial Training (MIO-TCD)

```bash
# Train from scratch on MIO-TCD
python training/train_vehicle_classifier.py \
  --data_dir training/data/vehicles \
  --output_dir models \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001

# With GPU (recommended):
python training/train_vehicle_classifier.py \
  --data_dir training/data/vehicles \
  --output_dir models \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001 \
  --device cuda
```

**Expected results:**
- Training time: 2-3 hours (GPU), 12-24 hours (CPU)
- Validation accuracy: 90-95%
- Model size: ~20MB

### Fine-Tuning (Toll Footage)

After initial training, fine-tune on your toll gate images:

```bash
# Fine-tune existing model
python training/train_vehicle_classifier.py \
  --data_dir training/data/toll_vehicles \
  --output_dir models \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.0001

# Model will be saved as: models/efficientnet_b0_vehicles.pth
```

**Expected results:**
- Training time: 30-60 minutes (GPU)
- Validation accuracy: 95-98% (on your specific setup)

## Model Deployment

### 1. Automatic Loading

The trained model is automatically loaded by the backend if it exists:

```python
# In detector.py
self.vehicle_classifier = VehicleClassifier(
    model_path="models/efficientnet_b0_vehicles.pth"  # Auto-loads if exists
)
```

### 2. Hot-Swap Models

You can replace the model without code changes:

```bash
# Train new model
python training/train_vehicle_classifier.py --data_dir training/data/vehicles

# Copy to deployment location
cp models/efficientnet_b0_vehicles_best.pth models/efficientnet_b0_vehicles.pth

# Restart backend
docker-compose restart backend
# Or: docker-compose -f docker-compose.gpu.yml restart backend
```

### 3. Fallback Behavior

If no trained model is found, the system automatically falls back to YOLO classification:

```
⚠️  No trained model found. Using fallback to YOLO classification.
```

This allows the system to run without the EfficientNet model while you're training.

## Training Parameters

### Recommended Settings

| Parameter | Initial Training | Fine-Tuning | Description |
|-----------|------------------|-------------|-------------|
| `--epochs` | 50 | 20 | Number of training epochs |
| `--batch_size` | 32-64 (GPU), 16-32 (CPU) | 16-32 | Images per batch |
| `--lr` | 0.001 | 0.0001 | Learning rate (lower for fine-tuning) |
| `--device` | auto | auto | cuda/cpu (auto-detects) |

### Data Augmentation

The training script automatically applies:
- Random horizontal flip
- Random crop
- Color jitter (brightness, contrast, saturation)
- Random rotation (±5°)

This improves generalization and reduces need for large datasets.

## Monitoring Training

### During Training

Watch the console output:

```
Epoch 1/50:
  Train Loss: 1.2345, Train Acc: 0.7123
  Val Loss: 0.9876, Val Acc: 0.7892
✅ Saved new best model: models/efficientnet_b0_vehicles_best.pth (val_acc=0.7892)
```

### Success Criteria

- **Validation accuracy > 90%**: Good general performance
- **Validation accuracy > 95%**: Excellent performance (after fine-tuning)
- **Train/Val gap < 5%**: Not overfitting
- **Train/Val gap > 10%**: Overfitting - need more data or regularization

### Troubleshooting

**Problem: Validation accuracy plateaus at ~70%**
- Solution: Train longer (more epochs) or check data quality

**Problem: Training accuracy 99%, validation accuracy 70%**
- Solution: Overfitting - reduce model size or add more training data

**Problem: Training is very slow**
- Solution: Reduce batch size or use GPU

**Problem: Out of memory**
- Solution: Reduce batch size (try 16 or 8)

## Continuous Improvement Workflow

### Week 1-4: Collect Feedback
```bash
# System runs with initial MIO-TCD model
# Collect misclassifications via feedback system
```

### Week 5: Export Hard Cases
```bash
# From your database, export images where:
# - Confidence < 0.7
# - Feedback corrections were needed
# - Organize into training/data/hard_cases/
```

### Week 6: Retrain
```bash
# Combine original + hard cases
python training/train_vehicle_classifier.py \
  --data_dir training/data/combined \
  --epochs 30
```

### Week 7: A/B Test
```bash
# Deploy new model to test instance
# Compare accuracy metrics

# If better: deploy to production
cp models/efficientnet_b0_vehicles_new.pth models/efficientnet_b0_vehicles.pth
```

### Week 8: Repeat

This cycle continuously improves accuracy over time.

## Model Versions

Track model versions for reproducibility:

```bash
# Save with version number
cp models/efficientnet_b0_vehicles_best.pth models/efficientnet_b0_vehicles_v2.0_20250120.pth

# Keep training logs
python training/train_vehicle_classifier.py ... | tee logs/training_v2.0.log
```

## Upgrading Model Architecture

The system supports easy upgrades:

```python
# Current: EfficientNet-B0 (20MB, 77% ImageNet)
# Upgrade options:
#   - EfficientNet-B1 (30MB, 79% ImageNet)
#   - EfficientNet-B2 (36MB, 80% ImageNet)
#   - EfficientNet-B3 (48MB, 82% ImageNet)

# To upgrade: modify train_vehicle_classifier.py
model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
# Update classifier dimensions accordingly
```

## Performance Expectations

### MIO-TCD Training (Initial)
- **Dataset**: 650K images
- **Training time**: 2-3 hours (GPU)
- **Validation accuracy**: 90-95%
- **Real-world accuracy**: 88-92% (before fine-tuning)

### Toll Footage Fine-Tuning
- **Dataset**: 500-1000 images (per class)
- **Training time**: 30-60 minutes (GPU)
- **Validation accuracy**: 95-98%
- **Real-world accuracy**: 93-97% (on your specific setup)

### Combined with Tripline (Production)
- **Vehicle type accuracy**: 95-97%
- **Axle count accuracy**: 98-99% (tripline)
- **Combined toll class accuracy**: 95-99.6% ✅

## Resources

- **MIO-TCD Dataset**: http://podoce.dinf.usherbrooke.ca/challenge/dataset/
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946
- **PyTorch Docs**: https://pytorch.org/docs/stable/torchvision/models.html
- **Our Training Script**: `train_vehicle_classifier.py`

## Support

For training issues:
1. Check logs in console output
2. Verify data directory structure
3. Try smaller batch size if OOM
4. Use CPU if GPU unavailable (`--device cpu`)
5. Check CUDA installation: `nvidia-smi`

## Next Steps

1. **Download MIO-TCD dataset**
2. **Prepare data**: `python training/prepare_miotcd_data.py ...`
3. **Train model**: `python training/train_vehicle_classifier.py ...`
4. **Deploy model**: Copy to `models/` directory
5. **Test accuracy**: Use feedback system
6. **Fine-tune**: Add toll footage and retrain
7. **Achieve 99.6%**: Combine with tripline counting! 🎯

