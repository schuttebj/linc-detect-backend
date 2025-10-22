"""
EfficientNet-B0 vehicle type classifier.
Classifies cropped vehicles into refined types for accurate toll classification.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class VehicleClassifier:
    """EfficientNet-B0 for refined vehicle classification."""
    
    # Refined vehicle types for toll classification
    # Order must match PyTorch ImageFolder alphabetical loading
    VEHICLE_CLASSES = [
        'box_truck',     # Single-unit box trucks
        'bus',           # Buses
        'car',           # Sedans, hatchbacks, coupes
        'delivery_van',  # Commercial delivery vans (2 axles)
        'motorcycle',    # Motorcycles
        'pickup',        # Consumer pickup trucks (F-150, Hilux, etc)
        'semi',          # Semi-trucks, articulated vehicles
    ]
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """Initialize classifier with optional pre-trained weights."""
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load EfficientNet-B0
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(1280, len(self.VEHICLE_CLASSES))
        
        # Load trained weights if available
        if model_path and Path(model_path).exists():
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=True)
                )
                print(f"✅ Loaded vehicle classifier from {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load model from {model_path}: {e}")
                print(f"⚠️  Using fallback to YOLO classification.")
                self.model = None
        else:
            print(f"⚠️  No trained model found at {model_path}. Using fallback to YOLO classification.")
            self.model = None  # Will use YOLO fallback
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def classify(self, vehicle_crop: np.ndarray) -> Tuple[str, float]:
        """
        Classify vehicle type from cropped image.
        
        Args:
            vehicle_crop: BGR image crop from YOLO detection
        
        Returns:
            (vehicle_type, confidence)
        """
        if self.model is None:
            print("⚠️  EfficientNet model not loaded, returning unknown")
            return "unknown", 0.0
        
        try:
            # Convert BGR to RGB
            if len(vehicle_crop.shape) == 3:
                vehicle_crop = vehicle_crop[:, :, ::-1]
            
            # Convert to PIL and transform
            pil_image = Image.fromarray(vehicle_crop)
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top 3 predictions for debugging
                top3_conf, top3_pred = torch.topk(probabilities, min(3, len(self.VEHICLE_CLASSES)))
                
                confidence, predicted = torch.max(probabilities, 0)
                vehicle_type = self.VEHICLE_CLASSES[predicted.item()]
                
                # Debug logging
                print(f"🔍 EfficientNet Classification:")
                print(f"   Top prediction: {vehicle_type} ({confidence.item():.3f})")
                for i in range(len(top3_pred)):
                    print(f"   #{i+1}: {self.VEHICLE_CLASSES[top3_pred[i].item()]} ({top3_conf[i].item():.3f})")
            
            return vehicle_type, confidence.item()
        except Exception as e:
            print(f"⚠️  Classification error: {e}")
            import traceback
            traceback.print_exc()
            return "unknown", 0.0
    
    def classify_with_top3(self, vehicle_crop: np.ndarray) -> Tuple[str, float, list]:
        """
        Classify vehicle type and return top 3 predictions for debugging.
        
        Args:
            vehicle_crop: BGR image crop from YOLO detection
        
        Returns:
            (vehicle_type, confidence, top3_predictions)
            where top3_predictions is a list of dicts: [{"class": "pickup", "confidence": 0.85}, ...]
        """
        if self.model is None:
            print("⚠️  EfficientNet model not loaded, returning unknown")
            return "unknown", 0.0, []
        
        try:
            # Convert BGR to RGB
            if len(vehicle_crop.shape) == 3:
                vehicle_crop = vehicle_crop[:, :, ::-1]
            
            # Convert to PIL and transform
            pil_image = Image.fromarray(vehicle_crop)
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top 3 predictions
                top3_conf, top3_pred = torch.topk(probabilities, min(3, len(self.VEHICLE_CLASSES)))
                
                confidence, predicted = torch.max(probabilities, 0)
                vehicle_type = self.VEHICLE_CLASSES[predicted.item()]
                
                # Format top 3 for response
                top3_list = []
                for i in range(len(top3_pred)):
                    top3_list.append({
                        "class": self.VEHICLE_CLASSES[top3_pred[i].item()],
                        "confidence": float(top3_conf[i].item())
                    })
                
                # Debug logging
                print(f"🔍 EfficientNet Classification:")
                print(f"   Top prediction: {vehicle_type} ({confidence.item():.3f})")
                for i, pred in enumerate(top3_list):
                    print(f"   #{i+1}: {pred['class']} ({pred['confidence']:.3f})")
            
            return vehicle_type, confidence.item(), top3_list
        except Exception as e:
            print(f"⚠️  Classification error: {e}")
            import traceback
            traceback.print_exc()
            return "unknown", 0.0, []
    
    def classify_batch(self, crops: list) -> list:
        """Batch classification for efficiency."""
        if self.model is None:
            return [("unknown", 0.0) for _ in crops]
        
        try:
            # Process batch
            batch_tensors = []
            for crop in crops:
                if len(crop.shape) == 3:
                    crop = crop[:, :, ::-1]
                pil_image = Image.fromarray(crop)
                batch_tensors.append(self.transform(pil_image))
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, 1)
            
            results = []
            for pred, conf in zip(predictions, confidences):
                results.append((self.VEHICLE_CLASSES[pred.item()], conf.item()))
            
            return results
        except Exception as e:
            print(f"⚠️  Batch classification error: {e}")
            return [("unknown", 0.0) for _ in crops]

