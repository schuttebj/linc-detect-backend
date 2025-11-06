"""
YOLO11n vehicle detection with CLIP classification.
Uses YOLO for bounding box detection and CLIP for vehicle type and color classification.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from PIL import Image

from .rules import toll_class, estimate_axles_from_detection

# Lazy import CLIP to avoid loading it if not needed (saves memory)
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIPProcessor = None
    CLIPModel = None


# ============================================================================
# CLIP Classification Labels
# ============================================================================

# Vehicle type labels for CLIP classification
# Simple, clear categories based on visual differences
CLIP_VEHICLE_LABELS = [
    "a photo of a car",
    "a photo of a pickup",
    "a photo of an SUV",
    "a photo of a van",
    "a photo of a bus",
    "a photo of a semitruck",
    "a photo of a motorcycle",
    "a photo of a trailer"
]

# For two-round refinement (optional future enhancement)
CLIP_VAN_SUBCATEGORIES = [
    "a photo of a minivan",
    "a photo of a small commercial van",
    "a photo of a large commercial van"
]

# Color labels for CLIP classification
CLIP_COLOR_LABELS = [
    "a red vehicle",
    "a blue vehicle",
    "a white vehicle",
    "a black vehicle",
    "a gray vehicle",
    "a silver vehicle",
    "a green vehicle",
    "a yellow vehicle",
    "an orange vehicle",
    "a brown vehicle",
    "a gold vehicle",
    "a purple vehicle"
]

# Mapping from CLIP labels to standardized vehicle types for toll classification
CLIP_TO_VEHICLE_TYPE = {
    "car": "car",
    "pickup": "pickup",
    "SUV": "suv",
    "van": "van",
    "bus": "bus",
    "semitruck": "semi",
    "motorcycle": "motorcycle",
    "trailer": "trailer"
}

# Van subcategory mapping (for two-round refinement)
CLIP_VAN_TO_TYPE = {
    "minivan": "light_van",
    "small commercial van": "light_van",
    "large commercial van": "heavy_van"
}

# Vehicle type groups for South African toll classification
LIGHT_VEHICLE_GROUP = {"car", "pickup", "suv", "light_van", "motorcycle"}  # Class 1
HEAVY_2AXLE_GROUP = {"van", "heavy_van", "bus"}  # Class 2 (2-axle heavy) - vans default to heavy unless refined
HEAVY_MULTAXLE_GROUP = {"semi", "trailer"}  # Class 3-4 (3+ axles)


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
    
    def __init__(
        self, 
        model_path: str = "yolo12n", 
        confidence: float = 0.25,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        use_clip: bool = True,
        use_wheel_detection: bool = True
    ):
        """
        Initialize the vehicle detector with YOLO and CLIP.
        
        Args:
            model_path: YOLO model name (e.g., "yolo12n" for COCO, "yolo12n_lvis" for LVIS-trained)
            confidence: Confidence threshold for YOLO detections
            clip_model_name: CLIP model name ("openai/clip-vit-base-patch32" or "openai/clip-vit-large-patch14")
            use_clip: Whether to use CLIP for classification (if False, falls back to YOLO-based classification)
            use_wheel_detection: Whether to use segmentation model for wheel detection
        """
        self.confidence = confidence
        self.model_path = model_path
        self.use_clip = use_clip
        self.clip_model_name = clip_model_name
        self.use_wheel_detection = use_wheel_detection
        
        print(f"Initializing YOLO model: {model_path}")
        print(f"📋 CLIP Labels: {len(CLIP_VEHICLE_LABELS)} vehicle categories")
        
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
        
        # Initialize CLIP models if enabled
        self.clip_model = None
        self.clip_processor = None
        if use_clip:
            self.load_clip_model(clip_model_name)
        
        # Initialize wheel detection segmentation model if enabled
        self.seg_model = None
        if use_wheel_detection:
            self.load_wheel_detection_model()
    
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
    
    def load_wheel_detection_model(self):
        """
        Load YOLO detection model for wheel detection.
        Uses YOLOv8 trained on Open Images V7 dataset (600 classes including wheels).
        """
        print("Loading wheel detection model (YOLOv8n-OIv7)...")
        try:
            # Use YOLOv8n trained on Open Images V7 (has wheel/tire class)
            self.seg_model = YOLO("yolov8n-oiv7.pt")
            
            # Check if model has wheel/tire class
            if hasattr(self.seg_model, 'names'):
                class_names = self.seg_model.names
                wheel_classes = [name for name in class_names.values() if 'wheel' in name.lower() or 'tire' in name.lower()]
                if wheel_classes:
                    print(f"✅ Wheel detection model loaded! Found classes: {wheel_classes}")
                else:
                    print(f"⚠️  Model loaded but no wheel class found. Available: {list(class_names.values())[:20]}")
            else:
                print("✅ Wheel detection model loaded!")
        except Exception as e:
            print(f"⚠️  Failed to load wheel detection model: {e}")
            print("⚠️  Will use heuristic-based axle estimation")
            self.seg_model = None
            self.use_wheel_detection = False
    
    def load_clip_model(self, model_name: str):
        """
        Load CLIP model for vehicle and color classification.
        
        Args:
            model_name: CLIP model name (e.g., "openai/clip-vit-base-patch32" or "openai/clip-vit-large-patch14")
        """
        if not CLIP_AVAILABLE:
            print("❌ CLIP not available (transformers not installed)")
            print("⚠️  Falling back to YOLO-based classification")
            self.use_clip = False
            return
        
        print(f"Loading CLIP model: {model_name}")
        try:
            self.clip_model = CLIPModel.from_pretrained(model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model_name = model_name
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
                print(f"✅ CLIP model '{model_name}' loaded on GPU!")
            else:
                print(f"✅ CLIP model '{model_name}' loaded on CPU!")
        except Exception as e:
            print(f"❌ Failed to load CLIP model: {e}")
            print("⚠️  Falling back to YOLO-based classification")
            self.use_clip = False
    
    def classify_vehicle_with_clip(
        self, 
        vehicle_crop: np.ndarray, 
        top_k: int = 5,
        labels: List[str] = None
    ) -> List[Dict[str, float]]:
        """
        Classify vehicle type using CLIP model.
        
        Args:
            vehicle_crop: Cropped vehicle image (numpy array, BGR format)
            top_k: Number of top predictions to return
            labels: Optional custom label list (defaults to CLIP_VEHICLE_LABELS)
        
        Returns:
            List of dicts with 'label' and 'confidence' keys, sorted by confidence
        """
        if self.clip_model is None or self.clip_processor is None:
            return []
        
        if labels is None:
            labels = CLIP_VEHICLE_LABELS
        
        try:
            # Convert BGR to RGB
            if len(vehicle_crop.shape) == 3 and vehicle_crop.shape[2] == 3:
                vehicle_crop_rgb = vehicle_crop[:, :, ::-1].copy()
            else:
                vehicle_crop_rgb = vehicle_crop
            
            # Convert to PIL Image
            pil_image = Image.fromarray(vehicle_crop_rgb)
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=labels, 
                images=pil_image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            
            # Calculate probabilities
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
            
            # Get top-k results
            sorted_indices = torch.argsort(probs, descending=True)[:top_k]
            
            results = []
            for idx in sorted_indices:
                label = labels[idx].replace("a photo of a ", "").replace("a photo of an ", "")
                confidence = probs[idx].item()
                results.append({
                    "label": label,
                    "confidence": confidence
                })
            
            return results
        
        except Exception as e:
            print(f"❌ CLIP vehicle classification failed: {e}")
            return []
    
    def refine_van_classification(
        self, 
        vehicle_crop: np.ndarray, 
        top_k: int = 3
    ) -> List[Dict[str, float]]:
        """
        Second-round refinement for van classification to distinguish light vs heavy vans.
        
        Args:
            vehicle_crop: Cropped vehicle image
            top_k: Number of top predictions
        
        Returns:
            List of van subcategory predictions
        """
        return self.classify_vehicle_with_clip(vehicle_crop, top_k, CLIP_VAN_SUBCATEGORIES)
    
    def detect_wheels_and_count_axles(
        self,
        vehicle_crop: np.ndarray,
        vehicle_type: str = "unknown"
    ) -> tuple[int, dict, list]:
        """
        Detect wheels using segmentation model and count axles.
        Only counts wheels that are touching the ground.
        
        Args:
            vehicle_crop: Cropped vehicle image
            vehicle_type: Type of vehicle (for fallback estimation)
        
        Returns:
            Tuple of (axle_count, debug_info, wheel_boxes)
        """
        if not self.use_wheel_detection or self.seg_model is None:
            # Fallback to heuristic estimation
            print(f"   ⚠️  Wheel detection disabled: use_wheel={self.use_wheel_detection}, model={self.seg_model is not None}")
            from classification.rules import estimate_axles_from_detection
            img_height = vehicle_crop.shape[0]
            bbox_height = vehicle_crop.shape[0]
            bbox_width = vehicle_crop.shape[1]
            axles = estimate_axles_from_detection(vehicle_type, bbox_height, bbox_width, img_height)
            return axles, {"method": "heuristic", "wheels_detected": 0, "reason": "model_not_loaded"}, []
        
        try:
            # OPTIMIZATION: Only search bottom 50% of vehicle crop (wheels are always in lower half)
            img_height = vehicle_crop.shape[0]
            img_width = vehicle_crop.shape[1]
            crop_start_y = img_height // 2
            bottom_half = vehicle_crop[crop_start_y:, :, :]
            
            print(f"   🔍 Searching for wheels in bottom 50% of crop ({bottom_half.shape[0]}x{bottom_half.shape[1]})")
            
            # Run detection with very low confidence to catch all wheels
            # Open Images V7 model uses detection, not segmentation
            results = self.seg_model(bottom_half, conf=0.08, verbose=False)
            
            if not results or len(results) == 0:
                # No detections, use heuristic fallback
                print(f"   ⚠️  Wheel detection returned no results")
                from classification.rules import estimate_axles_from_detection
                bbox_height = vehicle_crop.shape[0]
                bbox_width = vehicle_crop.shape[1]
                axles = estimate_axles_from_detection(vehicle_type, bbox_height, bbox_width, img_height)
                return axles, {"method": "heuristic_fallback", "reason": "no_wheels_detected"}, []
            
            result = results[0]
            
            # Get class names from model
            class_names = result.names if hasattr(result, 'names') else {}
            
            # Find wheel class ID (COCO has 'wheel' or might use other terms)
            wheel_class_id = None
            for class_id, class_name in class_names.items():
                if 'wheel' in class_name.lower() or 'tire' in class_name.lower():
                    wheel_class_id = class_id
                    break
            
            if wheel_class_id is None or not hasattr(result, 'boxes') or result.boxes is None:
                # No wheel class found or no boxes, use heuristic
                print(f"   ⚠️  No wheel class found. Classes: {list(class_names.values())}")
                from classification.rules import estimate_axles_from_detection
                img_height = vehicle_crop.shape[0]
                bbox_height = vehicle_crop.shape[0]
                bbox_width = vehicle_crop.shape[1]
                axles = estimate_axles_from_detection(vehicle_type, bbox_height, bbox_width, img_height)
                return axles, {"method": "heuristic_fallback", "reason": "no_wheel_class", "available_classes": list(class_names.values())[:10]}, []
            
            # Extract wheel detections
            boxes = result.boxes
            wheel_boxes = []
            
            # Debug: show what was detected
            detected_classes = []
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = class_names.get(class_id, f"class_{class_id}")
                confidence = float(box.conf[0])
                detected_classes.append(f"{class_name}({confidence:.2f})")
            
            print(f"   🔍 Detected: {', '.join(detected_classes[:5])}")  # Show first 5
            
            # Look for ALL wheel-related classes (Wheel, Tire, Bicycle wheel)
            wheel_class_ids = []
            for class_id, class_name in class_names.items():
                if 'wheel' in class_name.lower() or 'tire' in class_name.lower():
                    wheel_class_ids.append(class_id)
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                if class_id in wheel_class_ids:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    # Convert coordinates back to full crop (add crop_start_y offset)
                    y1_full = y1 + crop_start_y
                    y2_full = y2 + crop_start_y
                    center_x = (x1 + x2) / 2
                    center_y = (y1_full + y2_full) / 2
                    
                    wheel_boxes.append({
                        'x1': x1, 'y1': y1_full, 'x2': x2, 'y2': y2_full,
                        'center_x': center_x, 'center_y': center_y,
                        'confidence': confidence
                    })
            
            if len(wheel_boxes) == 0:
                # No wheels detected, use heuristic
                print(f"   ⚠️  No wheels found in {len(boxes)} detections. Looking for class IDs: {wheel_class_ids}")
                from classification.rules import estimate_axles_from_detection
                img_height = vehicle_crop.shape[0]
                bbox_height = vehicle_crop.shape[0]
                bbox_width = vehicle_crop.shape[1]
                axles = estimate_axles_from_detection(vehicle_type, bbox_height, bbox_width, img_height)
                return axles, {"method": "heuristic_fallback", "reason": "no_wheels_found", "total_detections": len(boxes)}, []
            
            # Strategy: Count ALL detected wheels (no ground filtering)
            # Assumption: We see vehicle from side, so visible wheels ≈ number of axles
            # Group wheels by Y-position to identify axles
            
            # Sort wheels by Y position
            wheel_boxes_sorted = sorted(wheel_boxes, key=lambda w: w['center_y'])
            
            # Group wheels into axles by Y-position clustering
            axle_groups = []
            y_threshold = img_height * 0.08  # Wheels within 8% height are on same axle
            
            for wheel in wheel_boxes_sorted:
                # Try to add to existing axle group
                added = False
                for group in axle_groups:
                    # Check if wheel's Y is close to group's average Y
                    avg_y = sum(w['center_y'] for w in group) / len(group)
                    if abs(wheel['center_y'] - avg_y) < y_threshold:
                        group.append(wheel)
                        added = True
                        break
                
                if not added:
                    # Create new axle group
                    axle_groups.append([wheel])
            
            # Count axles based on groups
            axle_count = len(axle_groups)
            
            # Minimum 2 axles for any vehicle
            axle_count = max(2, axle_count)
            
            # Cap at reasonable maximum (some trucks can have 5-6 axles)
            axle_count = min(axle_count, 10)
            
            debug_info = {
                "method": "wheel_detection_clustering",
                "total_wheels_detected": len(wheel_boxes),
                "axle_groups": len(axle_groups),
                "axle_count": axle_count,
                "wheels_per_group": [len(g) for g in axle_groups],
                "wheel_positions": [[int(w['center_x']), int(w['center_y'])] for w in wheel_boxes],
                "confidences": [round(w['confidence'], 2) for w in wheel_boxes]
            }
            
            print(f"   ✅ Detected {len(wheel_boxes)} wheels in {len(axle_groups)} groups → {axle_count} axles")
            return axle_count, debug_info, wheel_boxes
            
        except Exception as e:
            print(f"❌ Wheel detection failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to heuristic
            from classification.rules import estimate_axles_from_detection
            img_height = vehicle_crop.shape[0]
            bbox_height = vehicle_crop.shape[0]
            bbox_width = vehicle_crop.shape[1]
            axles = estimate_axles_from_detection(vehicle_type, bbox_height, bbox_width, img_height)
            return axles, {"method": "heuristic_fallback", "reason": f"exception: {str(e)}"}, []
    
    def classify_color_with_clip(
        self, 
        vehicle_crop: np.ndarray, 
        top_k: int = 3
    ) -> List[Dict[str, float]]:
        """
        Classify vehicle color using CLIP model.
        
        Args:
            vehicle_crop: Cropped vehicle image (numpy array, BGR format)
            top_k: Number of top predictions to return
        
        Returns:
            List of dicts with 'color' and 'confidence' keys, sorted by confidence
        """
        if self.clip_model is None or self.clip_processor is None:
            return []
        
        try:
            # Convert BGR to RGB
            if len(vehicle_crop.shape) == 3 and vehicle_crop.shape[2] == 3:
                vehicle_crop_rgb = vehicle_crop[:, :, ::-1].copy()
            else:
                vehicle_crop_rgb = vehicle_crop
            
            # Convert to PIL Image
            pil_image = Image.fromarray(vehicle_crop_rgb)
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=CLIP_COLOR_LABELS, 
                images=pil_image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            
            # Calculate probabilities
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
            
            # Get top-k results
            sorted_indices = torch.argsort(probs, descending=True)[:top_k]
            
            results = []
            for idx in sorted_indices:
                color = CLIP_COLOR_LABELS[idx].replace("a ", "").replace(" vehicle", "")
                confidence = probs[idx].item()
                results.append({
                    "color": color,
                    "confidence": confidence
                })
            
            return results
        
        except Exception as e:
            print(f"❌ CLIP color classification failed: {e}")
            return []
    
    def resolve_ambiguous_classification(
        self, 
        vehicle_results: List[Dict[str, float]],
        bbox_height: float,
        bbox_width: float,
        image_height: float
    ) -> Tuple[str, str]:
        """
        Resolve ambiguous vehicle classifications using smart logic.
        
        Handles cases where top results are close (e.g., pickup: 0.52, SUV: 0.48).
        
        Args:
            vehicle_results: Top K vehicle classification results
            bbox_height: Bounding box height
            bbox_width: Bounding box width
            image_height: Image height
        
        Returns:
            Tuple of (selected_vehicle_type, decision_reason)
        """
        if not vehicle_results or len(vehicle_results) == 0:
            return "car", "No CLIP results, defaulting to car"
        
        # Single result - use it
        if len(vehicle_results) == 1:
            label = vehicle_results[0]["label"]
            vehicle_type = CLIP_TO_VEHICLE_TYPE.get(label, "car")
            return vehicle_type, f"Single result: {label}"
        
        # Get top 2 results
        top1 = vehicle_results[0]
        top2 = vehicle_results[1]
        
        label1 = top1["label"]
        label2 = top2["label"]
        conf1 = top1["confidence"]
        conf2 = top2["confidence"]
        
        # Check confidence gap
        conf_gap = conf1 - conf2
        
        # Clear winner (>15% gap)
        if conf_gap > 0.15:
            vehicle_type = CLIP_TO_VEHICLE_TYPE.get(label1, "car")
            return vehicle_type, f"Clear winner: {label1} ({conf1:.3f})"
        
        # Check if both are in same vehicle group
        type1 = CLIP_TO_VEHICLE_TYPE.get(label1, label1)
        type2 = CLIP_TO_VEHICLE_TYPE.get(label2, label2)
        
        # Both in light vehicle group (Class 1)
        if type1 in LIGHT_VEHICLE_GROUP and type2 in LIGHT_VEHICLE_GROUP:
            # Use aspect ratio as tiebreaker for pickups vs cars/SUVs
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
            
            # Long/wide vehicles (bakkies/pickups tend to be longer)
            if aspect_ratio > 2.2 and "pickup" in [type1, type2]:
                return "pickup", f"Aspect ratio tiebreaker: {aspect_ratio:.2f} suggests bakkie/pickup"
            
            # Use higher confidence (they're all Class 1 anyway)
            vehicle_type = CLIP_TO_VEHICLE_TYPE.get(label1, "car")
            return vehicle_type, f"Light vehicle group (Class 1), using top: {label1} ({conf1:.3f})"
        
        # Both in heavy 2-axle group (Class 2)
        if type1 in HEAVY_2AXLE_GROUP and type2 in HEAVY_2AXLE_GROUP:
            vehicle_type = CLIP_TO_VEHICLE_TYPE.get(label1, "bus")
            return vehicle_type, f"Heavy 2-axle (Class 2), using top: {label1} ({conf1:.3f})"
        
        # Contradicting classes (light vs heavy) - critical distinction!
        if (type1 in LIGHT_VEHICLE_GROUP and (type2 in HEAVY_2AXLE_GROUP or type2 in HEAVY_MULTAXLE_GROUP)) or \
           ((type1 in HEAVY_2AXLE_GROUP or type1 in HEAVY_MULTAXLE_GROUP) and type2 in LIGHT_VEHICLE_GROUP):
            
            # Use relative size to help decide
            relative_size = bbox_height / image_height if image_height > 0 else 0
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
            
            # Large object in frame suggests heavy vehicle
            if relative_size > 0.5 and conf_gap < 0.15:
                # If heavy vehicle is in top 2 and object is large, prefer heavy
                if type1 in HEAVY_2AXLE_GROUP or type1 in HEAVY_MULTAXLE_GROUP:
                    heavy_type = type1
                else:
                    heavy_type = type2
                return heavy_type, f"Large object ({relative_size:.2f}) + aspect {aspect_ratio:.2f} suggests heavy vehicle"
            
            # Otherwise, trust CLIP's top pick
            vehicle_type = CLIP_TO_VEHICLE_TYPE.get(label1, "car")
            return vehicle_type, f"Light vs Heavy conflict, trusting CLIP top: {label1} ({conf1:.3f})"
        
        # Default: use top result
        vehicle_type = CLIP_TO_VEHICLE_TYPE.get(label1, "car")
        return vehicle_type, f"Using top result: {label1} ({conf1:.3f})"
    
    def detect_vehicles(
        self, 
        image: np.ndarray,
        imgsz: int = 640,
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect vehicles in an image using YOLO + CLIP.
        
        YOLO provides bounding boxes, CLIP classifies vehicle type and color.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            imgsz: Input image size for YOLO model
            confidence_threshold: Unused (kept for API compatibility)
        
        Returns:
            List of detected vehicles with metadata including CLIP classifications
        """
        # Run YOLO inference (Stage 1: Bounding Box Detection)
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
            
            # Process based on model type (COCO or LVIS) - just for filtering vehicles
            is_vehicle = False
            if self.is_lvis_model:
                # LVIS model: Check if class_id is in our vehicle mapping
                if class_id in self.LVIS_VEHICLE_CLASSES:
                    is_vehicle = True
                    yolo_class = self.LVIS_VEHICLE_CLASSES[class_id]
            else:
                # COCO model: Use generic classes
                yolo_class = results.names[class_id].lower()
                if yolo_class in self.COCO_VEHICLE_CLASSES:
                    is_vehicle = True
            
            if not is_vehicle:
                continue
            
            print(f"\n🚗 Vehicle Detection: YOLO class={yolo_class}")
            
            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            yolo_confidence = float(box.conf.item())
            
            # Calculate box dimensions
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Crop vehicle from image for CLIP classification
            vehicle_crop = image[y1:y2, x1:x2, :].copy() if (y2 > y1 and x2 > x1) else None
            
            if vehicle_crop is None or vehicle_crop.size == 0:
                print("⚠️  Invalid crop, skipping...")
                continue
            
            # Stage 2: CLIP Classification (if enabled)
            vehicle_type = None
            clip_vehicle_results = []
            clip_van_refinement = []
            clip_color_results = []
            decision_reason = "YOLO only used for detection"
            clip_inference_time = 0.0
            
            if self.use_clip and vehicle_crop is not None:
                clip_start = time.time()
                
                # Primary CLIP classification - simple, clear categories
                clip_vehicle_results = self.classify_vehicle_with_clip(vehicle_crop, top_k=5)
                
                # Classify color with CLIP
                clip_color_results = self.classify_color_with_clip(vehicle_crop, top_k=3)
                
                if clip_vehicle_results:
                    # Get initial vehicle type
                    vehicle_type, decision_reason = self.resolve_ambiguous_classification(
                        clip_vehicle_results,
                        bbox_height,
                        bbox_width,
                        image_height
                    )
                    
                    # Optional: Refine van classification (minivan vs small commercial vs large commercial)
                    if vehicle_type == "van" and clip_vehicle_results[0]["confidence"] > 0.3:
                        clip_van_refinement = self.refine_van_classification(vehicle_crop, top_k=3)
                        if clip_van_refinement and clip_van_refinement[0]["confidence"] > 0.5:
                            refined_type = CLIP_VAN_TO_TYPE.get(clip_van_refinement[0]["label"], "van")
                            if refined_type != "van":
                                vehicle_type = refined_type
                                decision_reason += f" | Refined: {clip_van_refinement[0]['label']} ({clip_van_refinement[0]['confidence']:.3f})"
                    
                    clip_inference_time = (time.time() - clip_start) * 1000  # milliseconds
                    print(f"   CLIP: {vehicle_type} | {decision_reason}")
                    print(f"   CLIP Inference: {clip_inference_time:.1f}ms")
                else:
                    clip_inference_time = (time.time() - clip_start) * 1000
                    vehicle_type = "unknown"
                    decision_reason = "CLIP failed to classify"
                    print(f"   ⚠️  {decision_reason}")
            else:
                vehicle_type = "unknown"
                print(f"   ⚠️ CLIP disabled")
            
            # Count axles using wheel detection (with heuristic fallback)
            estimated_axles, axle_debug, wheel_boxes = self.detect_wheels_and_count_axles(
                vehicle_crop,
                vehicle_type
            )
            
            # Store wheel boxes for later drawing (converted to full image coordinates)
            wheel_boxes_full_img = []
            if wheel_boxes and len(wheel_boxes) > 0:
                for wheel in wheel_boxes:
                    wheel_boxes_full_img.append({
                        'x1': int(x1 + wheel['x1']),
                        'y1': int(y1 + wheel['y1']),
                        'x2': int(x1 + wheel['x2']),
                        'y2': int(y1 + wheel['y2']),
                        'center_x': int(x1 + wheel['center_x']),
                        'center_y': int(y1 + wheel['center_y'])
                    })
            
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
            
            # Extract primary color
            primary_color = clip_color_results[0]["color"] if clip_color_results else None
            
            # Build detection result
            detection = {
                "vehicle_type": vehicle_type,
                "yolo_class": yolo_class,  # YOLO detection (for debugging only, not used for classification)
                "confidence": yolo_confidence,  # YOLO bbox confidence
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
                "primary_color": primary_color,
                "clip_vehicle_results": clip_vehicle_results,
                "clip_van_refinement": clip_van_refinement if clip_van_refinement else None,
                "clip_color_results": clip_color_results,
                "clip_inference_time_ms": clip_inference_time,
                "clip_model": self.clip_model_name if self.use_clip else None,
                "axle_detection": axle_debug,
                "wheel_boxes": wheel_boxes_full_img,  # For drawing wheels on annotated image
                "debug": {
                    "model_type": "Detection only" if self.use_clip else ("LVIS" if self.is_lvis_model else "COCO"),
                    "class_id": class_id,
                    "yolo_class": yolo_class,
                    "yolo_confidence": yolo_confidence,
                    "crop_size": vehicle_crop.shape if vehicle_crop is not None else None,
                    "decision": decision_reason,
                    "clip_enabled": self.use_clip,
                    "wheel_detection_enabled": self.use_wheel_detection
                }
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
        pil_image = Image.open(image_path)
        
        # Convert RGBA to RGB if needed (handle transparency)
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        image = np.array(pil_image)
        
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
        
        # Draw detected wheels (if any)
        wheel_boxes = detection.get("wheel_boxes", [])
        if wheel_boxes:
            for wheel in wheel_boxes:
                # Draw yellow bounding box around wheel
                cv2.rectangle(
                    img,
                    (wheel['x1'], wheel['y1']),
                    (wheel['x2'], wheel['y2']),
                    (0, 255, 255),  # Yellow in BGR
                    2
                )
                # Draw center point
                cv2.circle(
                    img,
                    (wheel['center_x'], wheel['center_y']),
                    3,
                    (0, 255, 255),  # Yellow in BGR
                    -1
                )
        
        return img

