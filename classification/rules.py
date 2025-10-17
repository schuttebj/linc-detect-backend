"""
Toll classification rules based on vehicle type and axle count.

Classification Rules:
- Class 1: All light vehicles (passenger cars, LDVs/bakkies ≤3.5t, minibuses, motorcycles)
- Class 2: 2-axle heavy vehicles (i.e., heavy/large vehicles with exactly 2 axles)
- Class 3: 3-4 axles (any combination where the total is 3 or 4)
- Class 4: 5+ axles (any combination with five or more)
"""

from typing import Optional

# Vehicle type categories
LIGHT_TYPES = {"car", "pickup", "bakkie", "suv", "minibus", "van", "motorcycle"}
HEAVY_TYPES = {"truck", "bus", "lorry", "tractor"}


def toll_class(
    vehicle_type: Optional[str], 
    axle_count: int, 
    has_trailer: bool = False
) -> str:
    """
    Determine toll class based on vehicle type and axle count.
    
    Args:
        vehicle_type: Detected vehicle type (e.g., "car", "truck", "bus")
        axle_count: Number of axles detected
        has_trailer: Whether a trailer was detected
    
    Returns:
        Toll class as string: "Class 1", "Class 2", "Class 3", or "Class 4"
    """
    vt = (vehicle_type or "").lower()
    
    # Always trust clear axle thresholds first
    if axle_count >= 5:
        return "Class 4"
    
    if axle_count in (3, 4):
        return "Class 3"
    
    # Two-axle cases
    if axle_count == 2:
        if vt in HEAVY_TYPES:
            return "Class 2"
        # Light vehicles with small trailers can still be 2 axles (rare) -> still Class 1
        return "Class 1"
    
    # One axle (motorcycle) or unknown axle_count with type hints
    if vt in LIGHT_TYPES or vt == "motorcycle":
        return "Class 1"
    
    # Fallback: if we only know "truck/bus" but axle_count failed, be conservative
    if vt in HEAVY_TYPES:
        return "Class 2"  # Safe default; tracking will promote to 3/4/5+ when pulses arrive
    
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
        # Very long vehicles (articulated trucks)
        if aspect_ratio > 3.5:
            return 5  # Likely articulated
        # Long vehicles
        elif aspect_ratio > 2.5:
            return 4
        # Medium vehicles
        elif aspect_ratio > 1.8:
            return 3
        # Standard heavy vehicles
        else:
            return 2
    
    # Default for unknown types
    return 2

