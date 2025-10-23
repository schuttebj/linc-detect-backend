"""Classification module for vehicle toll classification."""

from .detector import VehicleDetector
from .axle_counter import AxleCounter
from .stream_processor import StreamProcessor
from .rules import toll_class, estimate_axles_from_detection

__all__ = ["VehicleDetector", "AxleCounter", "StreamProcessor", "toll_class", "estimate_axles_from_detection"]

