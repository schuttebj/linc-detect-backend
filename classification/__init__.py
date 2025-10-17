"""Classification module for vehicle toll classification."""

from .detector import VehicleDetector
from .rules import toll_class, estimate_axles_from_detection

__all__ = ["VehicleDetector", "toll_class", "estimate_axles_from_detection"]

