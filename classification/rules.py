"""
Toll classification rules based on vehicle type and axle count.

Classification Rules:
- Class 1: All light vehicles (passenger cars, LDVs/bakkies ≤3.5t, minibuses, motorcycles)
- Class 2: 2-axle heavy vehicles (i.e., heavy/large vehicles with exactly 2 axles)
- Class 3: 3-4 axles (any combination where the total is 3 or 4)
- Class 4: 5+ axles (any combination with five or more)

Enhanced with size-based classification to distinguish large SUVs from trucks.
"""

from typing import Optional

# Vehicle type categories
LIGHT_TYPES = {"car", "pickup", "bakkie", "suv", "minibus", "van", "motorcycle"}
HEAVY_TYPES = {"truck", "bus", "lorry", "tractor"}

# Size thresholds for distinguishing light vs heavy 2-axle vehicles
HEAVY_VEHICLE_HEIGHT_THRESHOLD = 0.45  # Relative to image height
HEAVY_VEHICLE_ASPECT_THRESHOLD = 2.5   # Width/height ratio for long vehicles


def classify_2axle_vehicle(
    vehicle_type: str,
    bbox_height: float,
    bbox_width: float,
    image_height: int
) -> str:
    """
    Classify 2-axle vehicles as light (Class 1) or heavy (Class 2) using voting logic.
    
    This solves the problem of large SUVs being mislabeled as trucks by YOLO.
    Uses size-based heuristics to distinguish:
    - Light: Cars, SUVs, pickups, bakkies, minibuses (Class 1)
    - Heavy: Rigid trucks, buses (Class 2)
    
    Args:
        vehicle_type: YOLO-detected type (may be inaccurate for SUVs)
        bbox_height: Bounding box height in pixels
        bbox_width: Bounding box width in pixels
        image_height: Total image height in pixels
    
    Returns:
        "Class 1" or "Class 2"
    """
    vt = vehicle_type.lower()
    votes_heavy = 0
    
    # Calculate proportions
    relative_height = bbox_height / image_height if image_height > 0 else 0
    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
    
    # Vote 1: Size-based (very tall or very long = heavy)
    if relative_height > HEAVY_VEHICLE_HEIGHT_THRESHOLD:
        votes_heavy += 1
    
    # Vote 2: Aspect ratio (very long = heavy rigid truck or bus)
    if aspect_ratio > HEAVY_VEHICLE_ASPECT_THRESHOLD:
        votes_heavy += 1
    
    # Vote 3: YOLO label as weak signal
    # Only count if it says "truck" or "bus", not if it says "car"
    if vt in ("truck", "bus", "lorry"):
        votes_heavy += 1
    
    # Decision: Need 2+ votes to classify as heavy
    # This means: Large SUV labeled as "truck" but not tall/long → Class 1
    #             Real truck that's tall or long → Class 2
    if votes_heavy >= 2:
        return "Class 2"  # Heavy 2-axle vehicle
    else:
        return "Class 1"  # Light vehicle (includes large SUVs, pickups)


def toll_class(
    vehicle_type: Optional[str], 
    axle_count: int, 
    has_trailer: bool = False,
    bbox_height: Optional[float] = None,
    bbox_width: Optional[float] = None,
    image_height: Optional[int] = None
) -> str:
    """
    Determine toll class based on vehicle type, axle count, and size.
    
    Args:
        vehicle_type: Detected vehicle type (e.g., "car", "truck", "bus")
        axle_count: Number of axles detected
        has_trailer: Whether a trailer was detected
        bbox_height: Bounding box height (for 2-axle classification)
        bbox_width: Bounding box width (for 2-axle classification)
        image_height: Image height (for 2-axle classification)
    
    Returns:
        Toll class as string: "Class 1", "Class 2", "Class 3", or "Class 4"
    """
    vt = (vehicle_type or "").lower()
    
    # Always trust clear axle thresholds first
    if axle_count >= 5:
        return "Class 4"
    
    if axle_count in (3, 4):
        return "Class 3"
    
    # Two-axle cases - use size-based classification
    if axle_count == 2:
        # If we have size info, use intelligent classification
        if bbox_height and bbox_width and image_height:
            return classify_2axle_vehicle(
                vt, bbox_height, bbox_width, image_height
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

