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
    """Wrapper for YOLO vehicle detection model (supports COCO and LVIS datasets)."""
    
    # COCO vehicle classes (80 total classes, only these are vehicles)
    COCO_VEHICLE_CLASSES = {
        "car", "truck", "bus", "motorcycle", "train"
    }
    
    # LVIS vehicle class IDs and names (17 classes trained in our custom model)
    LVIS_VEHICLE_CLASSES = {
        7: 'car',                # ambulance (treat as car for toll purposes)
        93: 'motorcycle',        # bicycle (treat as motorcycle - Class 1)
        172: 'bus',              # bus
        177: 'car',              # taxi (treat as car)
        206: 'car',              # car
        336: 'car',              # police car (treat as car)
        440: 'box_truck',        # fire truck (treat as box truck)
        482: 'box_truck',        # garbage truck (treat as box truck)
        691: 'delivery_van',     # minivan / delivery van
        700: 'motorcycle',       # motor scooter
        702: 'motorcycle',       # motorcycle
        799: 'pickup',           # ⭐ pickup truck
        921: 'bus',              # school bus
        1106: 'box_truck',       # tow truck
        1112: 'motorcycle',      # dirt bike
        1113: 'semi',            # ⭐ semi truck (articulated)
        1122: 'box_truck',       # truck (generic)
    }
    
    def __init__(self, model_path: str = "yolo12n", confidence: float = 0.25):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: YOLO model name (e.g., "yolo12n" for COCO, "yolo12n_lvis" for LVIS-trained)
            confidence: Confidence threshold for detections
        """
        self.confidence = confidence
        self.model_path = model_path
        
        print(f"Initializing YOLO model: {model_path}")
        
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
        
        print(f"✅ YOLO model '{model_path}' loaded and ready!")
        
        # Detect if this is LVIS-trained model (has 1203 classes) or COCO (80 classes)
        self.is_lvis_model = self._detect_model_type()
    
    def _detect_model_type(self) -> bool:
        """Detect if model is LVIS-trained or COCO-trained."""
        try:
            # Check class names in model
            class_names = self.model.names
            
            # LVIS has 1203 classes, COCO has 80
            if len(class_names) > 1000:
                print(f"✅ Detected LVIS model ({len(class_names)} classes)")
                return True
            else:
                print(f"✅ Detected COCO model ({len(class_names)} classes)")
                return False
        except:
            print("⚠️  Could not detect model type, assuming COCO")
            return False
    
    def detect_vehicles(
        self, 
        image: np.ndarray,
        imgsz: int = 640,
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect vehicles in an image using YOLO.
        
        Supports both COCO-trained (generic car/truck/bus) and LVIS-trained (pickup/semi/etc) models.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            imgsz: Input image size for model
            confidence_threshold: Unused (kept for API compatibility)
        
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
        
        # Sort boxes by area (largest first) to prioritize main vehicle over background
        boxes_with_area = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            boxes_with_area.append((box, area))
        
        # Sort by area descending
        boxes_with_area.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 YOLO detected {len(boxes_with_area)} objects, processing largest first...")
        
        for box, area in boxes_with_area:
            class_id = int(box.cls.item())
            
            # Process based on model type (COCO or LVIS)
            if self.is_lvis_model:
                # LVIS model: Check if class_id is in our vehicle mapping
                if class_id not in self.LVIS_VEHICLE_CLASSES:
                    continue
                
                # Get refined vehicle type directly from LVIS
                vehicle_type = self.LVIS_VEHICLE_CLASSES[class_id]
                yolo_class = vehicle_type  # For debug info
                
                print(f"\n🚗 LVIS Detection: class_id={class_id} → {vehicle_type}")
                
            else:
                # COCO model: Use generic classes and map them
                yolo_class = results.names[class_id].lower()
                
                # Only process vehicle classes
                if yolo_class not in self.COCO_VEHICLE_CLASSES:
                    continue
                
                print(f"\n🚗 COCO Detection: {yolo_class}")
                vehicle_type = None  # Will be mapped later
            
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf.item())
            
            # Calculate box dimensions
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Crop vehicle from image for reference
            vehicle_crop = image[y1:y2, x1:x2, :].copy() if (y2 > y1 and x2 > x1) else None
            
            # Debug info
            debug_info = {
                "model_type": "LVIS" if self.is_lvis_model else "COCO",
                "class_id": class_id,
                "yolo_class": yolo_class,
                "yolo_confidence": confidence,
                "crop_size": vehicle_crop.shape if vehicle_crop is not None else None,
            }
            
            # Map COCO classes to refined types if needed
            if not self.is_lvis_model:
                vehicle_type = self._map_yolo_to_refined(yolo_class)
                debug_info["mapped_type"] = vehicle_type
                print(f"   Mapped to: {vehicle_type}")
            else:
                debug_info["lvis_direct"] = True
            
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
                "predicted_class": predicted_class,
                "debug": debug_info  # Add debug information
            }
            
            detections.append(detection)
        
        return detections
    
    def _map_yolo_to_refined(self, yolo_class: str) -> str:
        """
        Map COCO's generic classes to refined types.
        
        Note: This is a fallback for COCO models. LVIS models have native refined types.
        For best results, use a LVIS-trained model which has native pickup/semi detection.
        """
        mapping = {
            'car': 'car',           # COCO lumps car/suv/pickup together
            'truck': 'box_truck',   # COCO lumps all trucks together
            'bus': 'bus',
            'motorcycle': 'motorcycle',
            'train': 'semi'         # Rare misclassification
        }
        return mapping.get(yolo_class, 'car')
    
    def detect_and_classify(
        self, 
        image_path: str,
        confidence_threshold: float = 0.5  # Kept for API compatibility
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

