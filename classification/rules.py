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
BOXINESS_THRESHOLD = 0.15        # Edge density in rectangular regions
DUAL_WHEEL_THRESHOLD = 0.12      # Dark region pattern at rear wheels
LOWER_ASPECT_THRESHOLD = 3.5     # Aspect ratio of bottom 40% of vehicle (chassis length)


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


def classify_2axle_vehicle(
    vehicle_type: str,
    bbox_height: float,
    bbox_width: float,
    image_height: int,
    vehicle_crop: Optional[np.ndarray] = None
) -> str:
    """
    Classify 2-axle vehicles as light (Class 1) or heavy (Class 2) using multi-feature voting.
    
    Uses angle-invariant visual features to distinguish:
    - Light: Cars, SUVs, pickups, bakkies, minibuses (Class 1)
    - Heavy: Rigid trucks, buses (Class 2)
    
    Args:
        vehicle_type: YOLO-detected type (may be inaccurate for SUVs)
        bbox_height: Bounding box height in pixels
        bbox_width: Bounding box width in pixels
        image_height: Total image height in pixels
        vehicle_crop: Cropped image of vehicle (optional, for visual features)
    
    Returns:
        "Class 1" or "Class 2"
    """
    vt = vehicle_type.lower()
    votes_heavy = 0
    votes_light = 0
    
    # Feature 1: Dual rear wheels (STRONGEST signal if present)
    # Heavy trucks have visible dual rear wheels
    has_dual_wheels = False
    if vehicle_crop is not None:
        has_dual_wheels = detect_dual_rear_wheels(vehicle_crop)
        if has_dual_wheels:
            votes_heavy += 3  # Very strong signal - heavy trucks have dual wheels
        else:
            votes_light += 1  # Absence of dual wheels favors light vehicle
    
    # Feature 2: Boxiness score (STRONG signal)
    # Trucks have rectangular cargo boxes, SUVs are rounded/curved
    if vehicle_crop is not None:
        boxiness = calculate_boxiness_score(vehicle_crop)
        if boxiness > BOXINESS_THRESHOLD:
            votes_heavy += 2  # Strong rectangular body = truck
        else:
            votes_light += 2  # Rounded body = SUV/car
    
    # Feature 3: Lower-body aspect ratio (MODERATE signal)
    # Can vary with camera angle, so weighted less
    lower_aspect = calculate_lower_body_aspect(bbox_height, bbox_width)
    if lower_aspect > LOWER_ASPECT_THRESHOLD:
        votes_heavy += 1
    else:
        votes_light += 1
    
    # Feature 4: YOLO label (WEAK signal, tiebreaker only)
    # Often incorrect for SUVs, so minimal weight
    if vt in ("truck", "bus", "lorry"):
        votes_heavy += 1
    elif vt in ("car", "suv", "van", "pickup"):
        votes_light += 1
    
    # Decision: Majority vote wins
    # Weights: Dual wheels(3), Boxiness(2), Aspect(1), Label(1)
    # If tied, default to Class 1 (conservative for toll charging)
    if votes_heavy > votes_light:
        return "Class 2"  # Heavy 2-axle vehicle
    else:
        return "Class 1"  # Light vehicle (includes large SUVs, pickups)


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
    Determine toll class based on vehicle type, axle count, and visual features.
    
    Args:
        vehicle_type: Detected vehicle type (e.g., "car", "truck", "bus")
        axle_count: Number of axles detected
        has_trailer: Whether a trailer was detected
        bbox_height: Bounding box height (for 2-axle classification)
        bbox_width: Bounding box width (for 2-axle classification)
        image_height: Image height (for 2-axle classification)
        vehicle_crop: Cropped vehicle image (for visual feature analysis)
    
    Returns:
        Toll class as string: "Class 1", "Class 2", "Class 3", or "Class 4"
    """
    vt = (vehicle_type or "").lower()
    
    # Always trust clear axle thresholds first
    if axle_count >= 5:
        return "Class 4"
    
    if axle_count in (3, 4):
        return "Class 3"
    
    # Two-axle cases - use visual feature classification
    if axle_count == 2:
        # If we have size info, use intelligent classification with visual features
        if bbox_height and bbox_width and image_height:
            return classify_2axle_vehicle(
                vt, bbox_height, bbox_width, image_height, vehicle_crop
            )
        
        # Fallback without size info (shouldn't happen with image classification)
        if vt in HEAVY_TYPES:
            return "Class 2"
        return "Class 1"
    
    # One axle (motorcycle) or unknown axle_count with type hints
    if vt in LIGHT_TYPES or vt == "motorcycle":
        return "Class 1"
    
    # Fallback: if we only know "truck/bus" but axle_count failed, be conservative
    if vt in HEAVY_TYPES:
        return "Class 2"  # Safe default
    
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
    This is a heuristic fallback when tripline counting isn't available.
    
    Args:
        vehicle_type: Detected vehicle type
        bbox_height: Height of bounding box in pixels
        bbox_width: Width of bounding box in pixels
        image_height: Total image height in pixels
    
    Returns:
        Estimated axle count
    """
    vt = vehicle_type.lower()
    
    # Calculate relative size
    relative_height = bbox_height / image_height
    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
    
    # Motorcycle
    if vt == "motorcycle":
        return 2
    
    # Light vehicles (cars, SUVs, vans)
    if vt in LIGHT_TYPES:
        return 2
    
    # Heavy vehicles - use size heuristics
    if vt in ("bus", "truck"):
        # Very long vehicles (articulated trucks) - assume 5+ axles
        if aspect_ratio > 3.0:
            return 5  # Likely articulated with trailer
        # Long vehicles - but be conservative
        elif aspect_ratio > 2.2:
            return 3  # Could be 3-4 axles
        # Standard heavy vehicles
        else:
            return 2  # Standard 2-axle truck/bus
    
    # Default for unknown types
    return 2

