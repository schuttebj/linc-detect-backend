"""
YOLO11n vehicle detection wrapper.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from PIL import Image

from .rules import toll_class, estimate_axles_from_detection

# Allow Ultralytics classes for PyTorch 2.6+ weights_only loading
torch.serialization.add_safe_globals([DetectionModel])


class VehicleDetector:
    """Wrapper for YOLO11n vehicle detection model."""
    
    # Vehicle classes we care about from COCO dataset
    VEHICLE_CLASSES = {
        "car", "truck", "bus", "motorcycle", "train"
    }
    
    def __init__(self, model_path: str = "yolo11n", confidence: float = 0.25):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: YOLO model name (e.g., "yolo11n")
            confidence: Confidence threshold for detections
        """
        self.confidence = confidence
        self.model_path = model_path
        
        print(f"Initializing YOLO11 model: {model_path}")
        
        # Try multiple locations for the model
        # 1. Project models directory (where download_model.py copies it)
        project_model = Path("models") / f"{model_path}.pt"
        # 2. Cache directory
        cache_model = Path.home() / '.cache' / 'ultralytics' / f"{model_path}.pt"
        
        if project_model.exists():
            print(f"✅ Found model in project directory: {project_model}")
            self.model = YOLO(str(project_model))
        elif cache_model.exists():
            print(f"✅ Found model in cache: {cache_model}")
            self.model = YOLO(str(cache_model))
        else:
            print(f"Model not found locally, using model name (YOLO will auto-download)")
            self.model = YOLO(f"{model_path}.pt")
        
        print(f"✅ YOLO11 model '{model_path}' loaded and ready!")
    
    def detect_vehicles(
        self, 
        image: np.ndarray,
        imgsz: int = 640
    ) -> List[Dict]:
        """
        Detect vehicles in an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            imgsz: Input image size for model
        
        Returns:
            List of detected vehicles with metadata
        """
        # Run inference
        results = self.model.predict(
            image, 
            imgsz=imgsz, 
            conf=self.confidence, 
            verbose=False
        )[0]
        
        detections = []
        image_height, image_width = image.shape[:2]
        
        for box in results.boxes:
            class_id = int(box.cls.item())
            class_name = results.names[class_id].lower()
            
            # Only process vehicle classes
            if class_name not in self.VEHICLE_CLASSES:
                continue
            
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf.item())
            
            # Calculate box dimensions
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Estimate axles from bounding box heuristics
            estimated_axles = estimate_axles_from_detection(
                class_name,
                bbox_height,
                bbox_width,
                image_height
            )
            
            # Determine toll class
            predicted_class = toll_class(class_name, estimated_axles, False)
            
            detection = {
                "vehicle_type": class_name,
                "confidence": confidence,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": bbox_width,
                    "height": bbox_height
                },
                "axle_count": estimated_axles,
                "predicted_class": predicted_class
            }
            
            detections.append(detection)
        
        return detections
    
    def detect_and_classify(
        self, 
        image_path: str
    ) -> Tuple[Optional[Dict], np.ndarray]:
        """
        Detect and classify a vehicle from an image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (classification_result, annotated_image)
        """
        # Load image
        image = np.array(Image.open(image_path))
        
        # Convert RGB to BGR for OpenCV compatibility
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = image[:, :, ::-1].copy()
        else:
            image_bgr = image
        
        # Detect vehicles
        detections = self.detect_vehicles(image_bgr)
        
        # Get the most confident detection
        if not detections:
            return None, image
        
        # Sort by confidence and take the best
        best_detection = max(detections, key=lambda d: d["confidence"])
        
        # Annotate image
        annotated_image = self._annotate_image(image, best_detection)
        
        return best_detection, annotated_image
    
    def _annotate_image(
        self, 
        image: np.ndarray, 
        detection: Dict
    ) -> np.ndarray:
        """
        Draw bounding box and labels on image.
        
        Args:
            image: Input image
            detection: Detection dictionary
        
        Returns:
            Annotated image
        """
        import cv2
        
        img = image.copy()
        bbox = detection["bbox"]
        
        # Draw bounding box
        cv2.rectangle(
            img,
            (bbox["x1"], bbox["y1"]),
            (bbox["x2"], bbox["y2"]),
            (0, 255, 0),
            2
        )
        
        # Prepare label
        label = f"{detection['vehicle_type']} ({detection['predicted_class']}) - {detection['axle_count']} axles"
        conf_label = f"Conf: {detection['confidence']:.2f}"
        
        # Draw label background
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            img,
            (bbox["x1"], bbox["y1"] - label_height - 25),
            (bbox["x1"] + label_width, bbox["y1"]),
            (0, 255, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            label,
            (bbox["x1"], bbox["y1"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        cv2.putText(
            img,
            conf_label,
            (bbox["x1"], bbox["y1"] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        return img

