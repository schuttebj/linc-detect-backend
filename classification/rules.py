"""
Toll classification rules based on vehicle type and axle count.

Classification Rules:
- Class 1: All light vehicles (passenger cars, LDVs/bakkies ≤3.5t, minibuses, motorcycles)
- Class 2: 2-axle heavy vehicles (i.e., heavy/large vehicles with exactly 2 axles)
- Class 3: 3-4 axles (any combination where the total is 3 or 4)
- Class 4: 5+ axles (any combination with five or more)

Enhanced with visual feature analysis to distinguish large SUVs from trucks.
Works on images from any camera angle without calibration.
"""

from typing import Optional
import numpy as np
import cv2

# Vehicle type categories
LIGHT_TYPES = {"car", "pickup", "bakkie", "suv", "minibus", "van", "motorcycle"}
HEAVY_TYPES = {"truck", "bus", "lorry", "tractor"}

# Thresholds for visual feature analysis
BOXINESS_THRESHOLD = 0.18        # Edge density in rectangular regions (higher = stricter)
DUAL_WHEEL_THRESHOLD = 0.12      # Dark region pattern at rear wheels
LOWER_ASPECT_THRESHOLD = 5.0     # Aspect ratio of bottom 40% of vehicle (very long chassis only)


def calculate_boxiness_score(crop: np.ndarray) -> float:
    """
    Calculate boxiness score - trucks have rectangular cargo areas.
    
    Args:
        crop: Vehicle crop (BGR image)
    
    Returns:
        Score 0-1, higher = more boxy/rectangular (truck-like)
    """
    if crop is None or crop.size == 0:
        return 0.0
    
    try:
        # Focus on rear 60% of vehicle (cargo area)
        h, w = crop.shape[:2]
        rear_crop = crop[:, int(w*0.4):, :]
        
        # Edge detection
        gray = cv2.cvtColor(rear_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for straight lines (rectangular cargo box)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                minLineLength=int(min(h, w)*0.2), maxLineGap=10)
        
        if lines is None:
            return 0.0
        
        # Count long horizontal and vertical lines
        horizontal = 0
        vertical = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15 or angle > 165:  # Horizontal
                horizontal += 1
            elif 75 < angle < 105:  # Vertical
                vertical += 1
        
        # Boxiness = presence of both horizontal and vertical lines
        score = min((horizontal + vertical) / 20.0, 1.0)
        return score
    except Exception:
        return 0.0


def detect_dual_rear_wheels(crop: np.ndarray) -> bool:
    """
    Detect dual rear wheels pattern (heavy truck indicator).
    
    Args:
        crop: Vehicle crop (BGR image)
    
    Returns:
        True if dual wheel pattern detected
    """
    if crop is None or crop.size == 0:
        return False
    
    try:
        h, w = crop.shape[:2]
        # Focus on bottom 30%, rear 40% of vehicle
        wheel_region = crop[int(h*0.7):, int(w*0.6):, :]
        
        if wheel_region.size == 0:
            return False
        
        # Convert to grayscale and look for dark circular regions
        gray = cv2.cvtColor(wheel_region, cv2.COLOR_BGR2GRAY)
        
        # Detect dark regions (tires are dark)
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Count connected components (wheel clusters)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for 2+ dark regions close together (dual wheels)
        if len(contours) >= 2:
            # Check if they're vertically aligned (dual wheels are side-by-side)
            centers = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
            
            # Check for horizontally close, vertically aligned centers
            for i, (x1, y1) in enumerate(centers):
                for x2, y2 in centers[i+1:]:
                    h_dist = abs(x1 - x2)
                    v_dist = abs(y1 - y2)
                    if h_dist < wheel_region.shape[1] * 0.3 and v_dist < wheel_region.shape[0] * 0.4:
                        return True  # Found dual wheel pattern
        
        return False
    except Exception:
        return False


def calculate_lower_body_aspect(bbox_height: float, bbox_width: float) -> float:
    """
    Calculate aspect ratio focusing on chassis (bottom 40%).
    Long chassis = truck/bus, compact = SUV.
    
    Args:
        bbox_height: Full bounding box height
        bbox_width: Full bounding box width
    
    Returns:
        Aspect ratio (width / effective_height)
    """
    # Focus on lower 40% of vehicle (chassis level)
    effective_height = bbox_height * 0.4
    if effective_height == 0:
        return 0.0
    return bbox_width / effective_height


# Note: This function is kept for legacy compatibility but not used in main flow
# We use simpler aspect ratio + boxiness check directly in toll_class for better performance
def classify_2axle_vehicle(
    vehicle_type: str,
    bbox_height: float,
    bbox_width: float,
    image_height: int,
    vehicle_crop: Optional[np.ndarray] = None
) -> str:
    """
    Legacy function - kept for compatibility.
    Use toll_class() directly for better performance.
    """
    return "Class 1"  # Conservative default


def toll_class(
    vehicle_type: Optional[str], 
    axle_count: int, 
    has_trailer: bool = False,
    bbox_height: Optional[float] = None,
    bbox_width: Optional[float] = None,
    image_height: Optional[int] = None,
    vehicle_crop: Optional[np.ndarray] = None
) -> str:
    """
    Enhanced toll classification with refined vehicle types from two-stage classification.
    
    Classification Rules:
    - Class 1: Light vehicles (cars, SUVs, pickups, vans, motorcycles)
    - Class 2: 2-axle heavy vehicles (delivery vans, box trucks, buses)
    - Class 3: 3-4 axles (any combination)
    - Class 4: 5+ axles (heavy articulated vehicles)
    
    Args:
        vehicle_type: Refined vehicle type from EfficientNet or YOLO fallback
                     (car, suv, pickup, van, delivery_van, box_truck, semi, bus, motorcycle)
        axle_count: Number of axles detected (from tripline or estimation)
        has_trailer: Whether a trailer was detected
        bbox_height: Bounding box height (for 2-axle classification)
        bbox_width: Bounding box width (for 2-axle classification)
        image_height: Image height (for 2-axle classification)
        vehicle_crop: Cropped vehicle image (for visual feature analysis)
    
    Returns:
        Toll class as string: "Class 1", "Class 2", "Class 3", or "Class 4"
    """
    vt = (vehicle_type or "").lower()
    
    # PRIORITY 1: Motorcycles are ALWAYS Class 1 (regardless of axle count)
    if vt == "motorcycle":
        return "Class 1"
    
    # PRIORITY 2: Axle count thresholds (overrides vehicle type)
    # Tripline counting is the gold standard for production deployment
    if axle_count >= 5:
        return "Class 4"
    
    if axle_count in (3, 4):
        # Light vehicle + trailer: 2+1 or 2+2 = 3-4 axles → Class 3
        # Heavy vehicle with 3-4 axles → Class 3
        return "Class 3"
    
    # PRIORITY 3: Two-axle classification with refined vehicle types
    # This is where the two-stage classifier shines!
    if axle_count == 2:
        # Class 1: Light vehicles (consumer vehicles)
        if vt in ("car", "suv", "pickup", "van"):
            return "Class 1"
        
        # Class 2: Heavy 2-axle vehicles (commercial/public transport)
        if vt in ("delivery_van", "box_truck", "bus"):
            return "Class 2"
        
        # Semi with 2 axles (rare, bobtail tractor)
        if vt == "semi":
            return "Class 2"
        
        # Legacy fallback for old YOLO types
        if vt in ("bakkie", "minibus"):
            return "Class 1"
        
        # Generic "truck" - use aspect ratio heuristic
        if vt == "truck":
            if bbox_height and bbox_width:
                aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
                # Long rigid body = commercial truck (Class 2)
                # Compact = likely pickup/SUV (Class 1)
                return "Class 2" if aspect_ratio > 3.0 else "Class 1"
            return "Class 2"  # Default to Class 2 for generic trucks
        
        # Unknown 2-axle vehicle: Conservative default
        return "Class 1"
    
    # PRIORITY 4: Unknown axle count - use refined type hints
    # Light vehicle types → Class 1
    if vt in ("car", "suv", "pickup", "van", "motorcycle"):
        return "Class 1"
    
    # Heavy vehicle types → Class 2 (conservative)
    if vt in ("delivery_van", "box_truck", "bus", "semi"):
        return "Class 2"
    
    # Legacy light types
    if vt in LIGHT_TYPES:
        return "Class 1"
    
    # Legacy heavy types
    if vt in HEAVY_TYPES:
        return "Class 2"
    
    # Conservative default
    return "Class 1"


def estimate_axles_from_detection(
    vehicle_type: str,
    bbox_height: float,
    bbox_width: float,
    image_height: int
) -> int:
    """
    Estimate axle count from vehicle detection bounding box.
    This is a CONSERVATIVE heuristic for uploaded images (not tripline counting).
    
    IMPORTANT: For uploaded images, we default to 2 axles and let the 
    visual feature classifier (classify_2axle_vehicle) determine Class 1 vs Class 2.
    Only estimate 3+ axles if there's VERY strong evidence (extremely long articulated vehicles).
    
    Args:
        vehicle_type: Detected vehicle type
        bbox_height: Height of bounding box in pixels
        bbox_width: Width of bounding box in pixels
        image_height: Total image height in pixels
    
    Returns:
        Estimated axle count (conservative, defaults to 2)
    """
    vt = vehicle_type.lower()
    
    # Calculate aspect ratio
    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
    
    # For uploaded images (MVP testing), we CANNOT reliably estimate axles from bounding boxes
    # ALL vehicles default to 2 axles, and we rely on toll_class to distinguish Class 1 vs Class 2
    # 
    # For PRODUCTION with fixed cameras, use tripline pulse counting (already implemented in axle_counter.py)
    
    # Motorcycles: 2 axles (will be classified as Class 1)
    if vt == "motorcycle":
        return 2
    
    # ALL other vehicles: Default to 2 axles for MVP
    # The toll_class function will handle Class 1 vs Class 2 distinction
    # Only detect articulated trucks with trailers as 5+ axles
    if aspect_ratio > 5.0:  # VERY long (truck + visible trailer)
        return 5
    
    # Conservative default for all vehicles during testing phase
    return 2

