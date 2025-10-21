"""
YOLO11n vehicle detection wrapper.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image

from .rules import toll_class, estimate_axles_from_detection

# Allow required classes for PyTorch 2.6+ weights_only loading of YOLO models
# We trust Ultralytics models from official GitHub releases
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([
        DetectionModel,
        nn.modules.container.Sequential,
        nn.modules.conv.Conv2d,
        nn.modules.batchnorm.BatchNorm2d,
        nn.modules.activation.SiLU,
        nn.modules.pooling.MaxPool2d,
        nn.modules.upsampling.Upsample,
    ])
except Exception as e:
    print(f"Warning: Could not add safe globals: {e}")
    print("Model loading will proceed with default PyTorch settings")


def detect_trailer_connection(
    vehicle1: Dict,
    vehicle2: Dict,
    image_width: int
) -> bool:
    """
    Detect if two vehicles are connected (e.g., bus + trailer, truck + trailer).
    
    Criteria:
    1. Horizontally aligned (similar y-coordinates)
    2. Close proximity (small horizontal gap)
    3. Not overlapping (would indicate same vehicle detected twice)
    
    Args:
        vehicle1: First vehicle detection (left/front vehicle)
        vehicle2: Second vehicle detection (right/rear vehicle)
        image_width: Total image width for normalization
    
    Returns:
        True if vehicles appear to be connected
    """
    bbox1 = vehicle1["bbox"]
    bbox2 = vehicle2["bbox"]
    
    # Check if vehicle2 is to the right of vehicle1
    if bbox2["x1"] <= bbox1["x1"]:
        return False
    
    # Calculate horizontal gap between vehicles
    horizontal_gap = bbox2["x1"] - bbox1["x2"]
    
    # Check for overlap (same vehicle detected twice)
    if horizontal_gap < 0:
        overlap_ratio = abs(horizontal_gap) / min(bbox1["width"], bbox2["width"])
        if overlap_ratio > 0.5:  # Significant overlap = same vehicle
            return False
    
    # Normalize gap relative to image width (angle-invariant)
    relative_gap = horizontal_gap / image_width
    
    # Connected vehicles have small gap (< 5% of image width)
    # Separate vehicles in traffic have larger gaps
    if relative_gap > 0.05:
        return False  # Too far apart
    
    # Check vertical alignment (y-centers should be similar)
    y_center1 = (bbox1["y1"] + bbox1["y2"]) / 2
    y_center2 = (bbox2["y1"] + bbox2["y2"]) / 2
    y_diff = abs(y_center1 - y_center2)
    
    # Allow some vertical misalignment (road slope, camera angle)
    max_y_diff = max(bbox1["height"], bbox2["height"]) * 0.3
    if y_diff > max_y_diff:
        return False  # Not aligned
    
    # Check bottom alignment (wheels should be on same ground plane)
    bottom_diff = abs(bbox1["y2"] - bbox2["y2"])
    max_bottom_diff = max(bbox1["height"], bbox2["height"]) * 0.2
    if bottom_diff > max_bottom_diff:
        return False  # Not on same ground level
    
    # Passed all checks - likely connected!
    return True


def merge_connected_vehicles(
    vehicles: List[Dict],
    image_width: int,
    image_height: int,
    full_image: np.ndarray
) -> List[Dict]:
    """
    Merge vehicles that are connected (e.g., bus + trailer).
    
    Args:
        vehicles: List of vehicle detections
        image_width: Image width
        image_height: Image height
        full_image: Full image for cropping merged vehicles
    
    Returns:
        List of merged vehicle detections
    """
    if len(vehicles) < 2:
        return vehicles
    
    # Sort vehicles left to right
    sorted_vehicles = sorted(vehicles, key=lambda v: v["bbox"]["x1"])
    
    merged = []
    skip_indices = set()
    
    for i, vehicle1 in enumerate(sorted_vehicles):
        if i in skip_indices:
            continue
        
        # Check if this vehicle connects to the next one
        connected_group = [vehicle1]
        
        for j in range(i + 1, len(sorted_vehicles)):
            if j in skip_indices:
                continue
            
            vehicle2 = sorted_vehicles[j]
            
            # Check if vehicle2 connects to the last vehicle in the group
            if detect_trailer_connection(connected_group[-1], vehicle2, image_width):
                connected_group.append(vehicle2)
                skip_indices.add(j)
            else:
                break  # No more connections in this direction
        
        # Merge the connected group
        if len(connected_group) > 1:
            # This is a vehicle + trailer(s)
            merged_vehicle = merge_vehicle_group(connected_group, image_width, image_height, full_image)
            merged.append(merged_vehicle)
        else:
            # Single vehicle, no trailer
            merged.append(vehicle1)
    
    return merged


def merge_vehicle_group(
    vehicles: List[Dict],
    image_width: int,
    image_height: int,
    full_image: np.ndarray
) -> Dict:
    """
    Merge a group of connected vehicles into one detection.
    
    Args:
        vehicles: List of connected vehicle detections
        image_width: Image width
        image_height: Image height
        full_image: Full image for cropping
    
    Returns:
        Merged vehicle detection with combined axles
    """
    # Combine bounding boxes
    x1 = min(v["bbox"]["x1"] for v in vehicles)
    y1 = min(v["bbox"]["y1"] for v in vehicles)
    x2 = max(v["bbox"]["x2"] for v in vehicles)
    y2 = max(v["bbox"]["y2"] for v in vehicles)
    
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Sum axle counts
    total_axles = sum(v["axle_count"] for v in vehicles)
    
    # Use primary vehicle's type (usually the first/largest)
    primary_vehicle = max(vehicles, key=lambda v: v["bbox"]["width"] * v["bbox"]["height"])
    vehicle_type = primary_vehicle["vehicle_type"]
    
    # Average confidence
    avg_confidence = sum(v["confidence"] for v in vehicles) / len(vehicles)
    
    # Create merged crop
    vehicle_crop = full_image[y1:y2, x1:x2, :].copy() if (y2 > y1 and x2 > x1) else None
    
    # Reclassify with combined axles
    predicted_class = toll_class(
        vehicle_type,
        total_axles,
        has_trailer=True,  # Mark as having trailer
        bbox_height=bbox_height,
        bbox_width=bbox_width,
        image_height=image_height,
        vehicle_crop=vehicle_crop
    )
    
    return {
        "vehicle_type": f"{vehicle_type}+trailer",  # Indicate it's a combination
        "confidence": avg_confidence,
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": bbox_width,
            "height": bbox_height
        },
        "axle_count": total_axles,
        "predicted_class": predicted_class,
        "has_trailer": True,
        "component_count": len(vehicles)  # How many vehicles were merged
    }


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
        
        # Temporarily disable weights_only for trusted YOLO model loading
        # PyTorch 2.6+ requires this for Ultralytics models
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        
        try:
            if project_model.exists():
                print(f"✅ Found model in project directory: {project_model}")
                self.model = YOLO(str(project_model))
            elif cache_model.exists():
                print(f"✅ Found model in cache: {cache_model}")
                self.model = YOLO(str(cache_model))
            else:
                print(f"Model not found locally, using model name (YOLO will auto-download)")
                self.model = YOLO(f"{model_path}.pt")
        finally:
            # Restore original torch.load
            torch.load = original_load
        
        print(f"✅ YOLO11 model '{model_path}' loaded and ready!")
        
        # NEW: Initialize EfficientNet classifier for refined vehicle types
        from .vehicle_classifier import VehicleClassifier
        classifier_path = Path("models/efficientnet_b0_vehicles.pth")
        self.vehicle_classifier = VehicleClassifier(
            model_path=str(classifier_path) if classifier_path.exists() else None
        )
    
    def detect_vehicles(
        self, 
        image: np.ndarray,
        imgsz: int = 640
    ) -> List[Dict]:
        """
        Detect vehicles in an image using two-stage classification.
        Stage 1: YOLO11n detects vehicle bounding boxes
        Stage 2: EfficientNet-B0 classifies vehicle type (if trained model available)
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            imgsz: Input image size for model
        
        Returns:
            List of detected vehicles with metadata
        """
        # Run YOLO inference (Stage 1: Detection)
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
            yolo_class = results.names[class_id].lower()
            
            # Only process vehicle classes
            if yolo_class not in self.VEHICLE_CLASSES:
                continue
            
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf.item())
            
            # Calculate box dimensions
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Crop vehicle from image for classification
            vehicle_crop = image[y1:y2, x1:x2, :].copy() if (y2 > y1 and x2 > x1) else None
            
            # Stage 2: EfficientNet classification for refined vehicle type
            if vehicle_crop is not None and vehicle_crop.size > 0:
                refined_type, classifier_conf = self.vehicle_classifier.classify(vehicle_crop)
                
                # Use refined type if confident, otherwise fallback to YOLO
                if refined_type != "unknown" and classifier_conf > 0.5:
                    vehicle_type = refined_type
                    # Average confidences from both models
                    confidence = (confidence + classifier_conf) / 2
                else:
                    # Fallback: map YOLO class to refined types
                    vehicle_type = self._map_yolo_to_refined(yolo_class)
            else:
                vehicle_type = self._map_yolo_to_refined(yolo_class)
            
            # Estimate axles from bounding box heuristics
            estimated_axles = estimate_axles_from_detection(
                vehicle_type,
                bbox_height,
                bbox_width,
                image_height
            )
            
            # Determine toll class with refined vehicle type
            predicted_class = toll_class(
                vehicle_type, 
                estimated_axles, 
                False,
                bbox_height=bbox_height,
                bbox_width=bbox_width,
                image_height=image_height,
                vehicle_crop=vehicle_crop
            )
            
            detection = {
                "vehicle_type": vehicle_type,
                "yolo_class": yolo_class,  # Keep original for debugging
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
    
    def _map_yolo_to_refined(self, yolo_class: str) -> str:
        """
        Map YOLO's generic classes to refined types (fallback).
        Used when EfficientNet classifier is not available or has low confidence.
        """
        mapping = {
            'car': 'car',  # Could be car/suv/pickup, but default to car
            'truck': 'box_truck',  # Could be delivery_van/box_truck/semi
            'bus': 'bus',
            'motorcycle': 'motorcycle',
            'train': 'semi'  # Trains are rare, likely misclassified semis
        }
        return mapping.get(yolo_class, 'car')
    
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
        
        # Check for trailer connections and merge connected vehicles
        image_height, image_width = image_bgr.shape[:2]
        merged_detections = merge_connected_vehicles(
            detections, 
            image_width, 
            image_height,
            image_bgr
        )
        
        # Sort by confidence and take the best (might be a merged vehicle now)
        best_detection = max(merged_detections, key=lambda d: d["confidence"])
        
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

