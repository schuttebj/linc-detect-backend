"""
Training script for EfficientNet-B0 vehicle classifier.
Uses MIO-TCD dataset initially, can be fine-tuned with toll footage later.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import argparse
from tqdm import tqdm

def train_classifier(
    data_dir: str,
    output_dir: str = "models",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'auto'
):
    """
    Train EfficientNet-B0 vehicle classifier.
    
    Expected directory structure:
        data_dir/
            train/
                car/
                suv/
                pickup/
                van/
                delivery_van/
                box_truck/
                semi/
                bus/
                motorcycle/
            val/
                car/
                suv/
                ...
    
    Args:
        data_dir: Root directory containing train/ and val/ subdirectories
        output_dir: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        device: 'auto', 'cuda', or 'cpu'
    """
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Training on device: {device}")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(f'{data_dir}/train', train_transform)
    val_dataset = datasets.ImageFolder(f'{data_dir}/val', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model with pre-trained ImageNet weights
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
        
        train_acc = train_correct.double() / len(train_dataset)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  "):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
        
        val_acc = val_correct.double() / len(val_dataset)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model_path = f'{output_dir}/efficientnet_b0_vehicles_best.pth'
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved new best model: {model_path} (val_acc={val_acc:.4f})")
        
        scheduler.step(avg_val_loss)
        
        # Early stopping if learning rate gets too small
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning rate: {current_lr:.6f}")
        if current_lr < 1e-6:
            print("Learning rate too small, stopping training")
            break
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}/efficientnet_b0_vehicles_best.pth")
    
    # Rename to final model name
    best_model = Path(output_dir) / "efficientnet_b0_vehicles_best.pth"
    final_model = Path(output_dir) / "efficientnet_b0_vehicles.pth"
    if best_model.exists():
        import shutil
        shutil.copy(best_model, final_model)
        print(f"✅ Copied best model to: {final_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 vehicle classifier")
    parser.add_argument("--data_dir", required=True, help="Path to training data (with train/ and val/ subdirectories)")
    parser.add_argument("--output_dir", default="models", help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cuda/cpu)")
    
    args = parser.parse_args()
    train_classifier(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr, args.device)

