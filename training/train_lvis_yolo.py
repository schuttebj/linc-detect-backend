#!/usr/bin/env python3
"""
LVIS YOLO12 Training Script for RunPod/Cloud GPU

Train YOLO12 on LVIS dataset for enhanced vehicle classification.
This model will natively distinguish between car, pickup, semi, bus, etc.

Usage:
    python train_lvis_yolo.py --epochs 100 --batch 16
    
Or quick test:
    python train_lvis_yolo.py --epochs 10 --batch 8 --name quicktest
"""

import argparse
import torch
from ultralytics import YOLO
from pathlib import Path


def main(args):
    """Train YOLO12 on LVIS dataset."""
    
    print("="*60)
    print("LVIS YOLO12 Vehicle Classifier Training")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Version: {torch.version.cuda}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  No GPU detected - training will be VERY slow!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print("="*60)
    
    # Load base model
    print(f"\n📥 Loading base model: {args.model}")
    model = YOLO(f"{args.model}.pt")
    
    # Parse cache argument
    cache_value = args.cache
    if cache_value.lower() in ['false', 'no', '0']:
        cache_value = False
    elif cache_value.lower() in ['true', 'yes', '1', 'ram']:
        cache_value = 'ram'
    elif cache_value.lower() == 'disk':
        cache_value = 'disk'
    
    # Training configuration
    print(f"\n📋 Training Configuration:")
    print(f"   Dataset: LVIS (auto-download if needed)")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch}")
    print(f"   Image Size: {args.imgsz}")
    print(f"   Device: {args.device}")
    print(f"   Workers: {args.workers}")
    print(f"   Cache: {cache_value}")
    print(f"   Output: {args.project}/{args.name}")
    
    if args.epochs < 50:
        print(f"\n⚠️  WARNING: Training with only {args.epochs} epochs.")
        print(f"   For production use, recommend 100+ epochs.")
    
    print("\n" + "="*60)
    print("Starting training...")
    print("This will download LVIS dataset (~20GB) if not present")
    print("="*60 + "\n")
    
    # Train
    results = model.train(
        # Dataset
        data="lvis.yaml",           # LVIS dataset (auto-downloads)
        
        # Training params
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        
        # Device
        device=args.device,
        workers=args.workers,
        
        # Output
        project=args.project,
        name=args.name,
        
        # Optimization
        patience=args.patience,     # Early stopping
        save=True,                  # Save checkpoints
        cache=cache_value,          # Cache setting (False, 'disk', or 'ram')
        
        # Advanced
        pretrained=True,
        optimizer='AdamW',
        lr0=0.01,                   # Initial learning rate
        lrf=0.01,                   # Final learning rate
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,                # No rotation (vehicles are upright)
        translate=0.1,
        scale=0.5,
        flipud=0.0,                 # No vertical flip (vehicles don't flip)
        fliplr=0.5,                 # Horizontal flip OK
        mosaic=1.0,
        
        # Validation
        val=True,
        plots=True,
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    
    # Print results
    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    last_model_path = Path(args.project) / args.name / "weights" / "last.pt"
    
    print(f"\n📊 Results:")
    print(f"   Best model: {best_model_path}")
    print(f"   Last model: {last_model_path}")
    print(f"   Results directory: {Path(args.project) / args.name}")
    
    if best_model_path.exists():
        print(f"\n✅ Model ready for deployment!")
        print(f"\nNext steps:")
        print(f"1. Copy {best_model_path} to your backend:")
        print(f"   scp {best_model_path} your_server:backend/models/yolo12n_lvis.pt")
        print(f"\n2. Update backend to use new model:")
        print(f"   In detector.py: model_path='yolo12n_lvis'")
        print(f"\n3. Deploy and test!")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO12 on LVIS for vehicle classification")
    
    # Model
    parser.add_argument("--model", default="yolo12n", help="Base model (yolo12n, yolo12s, yolo12m)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (10 for quick test, 100+ for production)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if OOM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    # Device
    parser.add_argument("--device", default="0", help="CUDA device (0, 1, 2...) or 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers")
    
    # Output
    parser.add_argument("--project", default="lvis_training", help="Project directory")
    parser.add_argument("--name", default="yolo12n_vehicles", help="Experiment name")
    
    # Options
    parser.add_argument("--cache", type=str, default="False", help="Cache images: 'ram', 'disk', or 'False'")
    
    args = parser.parse_args()
    
    main(args)

