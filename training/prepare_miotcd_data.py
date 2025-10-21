"""
Prepare MIO-TCD dataset for vehicle classifier training.
Maps MIO-TCD categories to our refined vehicle types.

MIO-TCD Dataset: http://podoce.dinf.usherbrooke.ca/challenge/dataset/
"""

import shutil
from pathlib import Path
import argparse
import random

# Map MIO-TCD classes to our vehicle types
MIOTCD_MAPPING = {
    # MIO-TCD class -> Our class
    'articulated_truck': 'semi',
    'bicycle': None,  # Skip
    'bus': 'bus',
    'car': 'car',
    'motorcycle': 'motorcycle',
    'motorized_vehicle': 'van',  # Generic motorized -> van
    'non-motorized_vehicle': None,  # Skip
    'pedestrian': None,  # Skip
    'pickup_truck': 'pickup',
    'single_unit_truck': 'box_truck',
    'work_van': 'delivery_van',
}

def prepare_miotcd_dataset(
    miotcd_root: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Organize MIO-TCD images into train/val split for our vehicle classifier.
    
    Args:
        miotcd_root: Path to MIO-TCD dataset root directory
        output_dir: Output directory for organized data
        train_ratio: Ratio of data for training (rest for validation)
        seed: Random seed for reproducible splits
    """
    
    random.seed(seed)
    
    miotcd_path = Path(miotcd_root)
    output_path = Path(output_dir)
    
    print(f"Preparing MIO-TCD dataset from: {miotcd_path}")
    print(f"Output directory: {output_path}")
    print(f"Train/Val split: {train_ratio:.0%}/{(1-train_ratio):.0%}")
    print("="*60)
    
    # Create output structure
    for split in ['train', 'val']:
        for vehicle_class in set(MIOTCD_MAPPING.values()):
            if vehicle_class:
                class_dir = output_path / split / vehicle_class
                class_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_images = 0
    total_copied = 0
    class_stats = {}
    
    # Process each MIO-TCD class
    for miotcd_class, our_class in MIOTCD_MAPPING.items():
        if our_class is None:
            print(f"⏭️  Skipping: {miotcd_class}")
            continue
        
        source_dir = miotcd_path / miotcd_class
        if not source_dir.exists():
            print(f"⚠️  Directory not found: {miotcd_class}, skipping")
            continue
        
        # Find all images
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(list(source_dir.glob(ext)))
        
        if not images:
            print(f"⚠️  No images found in: {miotcd_class}, skipping")
            continue
        
        total_images += len(images)
        
        # Shuffle for random split
        random.shuffle(images)
        
        # Split train/val
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train
        for img in train_images:
            dest = output_path / 'train' / our_class / img.name
            shutil.copy(img, dest)
            total_copied += 1
        
        # Copy to val
        for img in val_images:
            dest = output_path / 'val' / our_class / img.name
            shutil.copy(img, dest)
            total_copied += 1
        
        # Track stats
        if our_class not in class_stats:
            class_stats[our_class] = {'train': 0, 'val': 0}
        class_stats[our_class]['train'] += len(train_images)
        class_stats[our_class]['val'] += len(val_images)
        
        print(f"✅ {miotcd_class:25} -> {our_class:15} ({len(train_images):4} train, {len(val_images):4} val)")
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Total images copied: {total_copied}")
    print()
    
    # Print class distribution
    print("Class Distribution:")
    print("-"*60)
    print(f"{'Class':<15} {'Train':>10} {'Val':>10} {'Total':>10} {'Train %':>10}")
    print("-"*60)
    
    total_train = 0
    total_val = 0
    
    for vehicle_class in sorted(class_stats.keys()):
        stats = class_stats[vehicle_class]
        train_count = stats['train']
        val_count = stats['val']
        total_count = train_count + val_count
        train_pct = (train_count / total_count * 100) if total_count > 0 else 0
        
        total_train += train_count
        total_val += val_count
        
        print(f"{vehicle_class:<15} {train_count:>10} {val_count:>10} {total_count:>10} {train_pct:>9.1f}%")
    
    print("-"*60)
    print(f"{'TOTAL':<15} {total_train:>10} {total_val:>10} {total_train+total_val:>10}")
    print()
    
    # Check for class imbalance
    counts = [sum(stats.values()) for stats in class_stats.values()]
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            print("⚠️  WARNING: Dataset is imbalanced (ratio > 3:1)")
            print("   Consider using weighted sampling during training")
        else:
            print("✅ Dataset is reasonably balanced")
    
    print()
    print("Next steps:")
    print(f"  1. Train model: python training/train_vehicle_classifier.py --data_dir {output_dir}")
    print(f"  2. Model will be saved to: models/efficientnet_b0_vehicles.pth")
    print(f"  3. Restart backend to load new model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MIO-TCD dataset for vehicle classification")
    parser.add_argument("--miotcd_root", required=True, help="Path to MIO-TCD dataset root directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for organized data")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    prepare_miotcd_dataset(args.miotcd_root, args.output_dir, args.train_ratio, args.seed)

