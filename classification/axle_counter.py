"""
Tripline pulse counting for axle detection from video streams.
"""

import collections
from typing import Optional, Tuple, Dict
import numpy as np
import cv2


class AxleCounter:
    """Counts vehicle axles using tripline intensity pulse detection."""
    
    def __init__(
        self,
        tripline_y: int = 420,
        roi_x0: int = 200,
        roi_x1: int = 1100,
        hist_len: int = 40,
        pulse_threshold: int = 18,
        min_gap_frames: int = 3
    ):
        """
        Initialize axle counter.
        
        Args:
            tripline_y: Y-coordinate of the tripline
            roi_x0: Left X-coordinate of region of interest
            roi_x1: Right X-coordinate of region of interest
            hist_len: Number of frames to keep in history
            pulse_threshold: Intensity drop threshold to detect pulse
            min_gap_frames: Minimum frames between pulses to avoid double counting
        """
        self.tripline_y = tripline_y
        self.roi_x0 = roi_x0
        self.roi_x1 = roi_x1
        self.hist_len = hist_len
        self.pulse_threshold = pulse_threshold
        self.min_gap_frames = min_gap_frames
        
        # State tracking
        self.intensity_hist = collections.deque(maxlen=hist_len)
        self.current_axle_pulses = 0
        self.last_pulse_frame = -9999
        self.frame_idx = 0
        self.crossing_vehicle = None
        self.last_detection_frame = -9999
    
    def reset(self):
        """Reset counter state for new vehicle."""
        self.current_axle_pulses = 0
        self.last_pulse_frame = -9999
        self.crossing_vehicle = None
    
    def process_frame(
        self,
        frame: np.ndarray,
        detections: list
    ) -> Optional[Dict]:
        """
        Process a video frame to detect axle pulses.
        
        Args:
            frame: Video frame (BGR format)
            detections: List of vehicle detections in frame
        
        Returns:
            Classification result if vehicle finished crossing, None otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute average intensity on the tripline ROI
        line = gray[self.tripline_y:self.tripline_y+1, self.roi_x0:self.roi_x1]
        avg_intensity = float(np.mean(line))
        self.intensity_hist.append(avg_intensity)
        
        # Check if vehicle is crossing the tripline
        crossing = False
        vehicle_type = None
        
        for detection in detections:
            bbox = detection["bbox"]
            vehicle_type = detection["vehicle_type"]
            
            # Check if vehicle bbox overlaps tripline
            if bbox["y1"] <= self.tripline_y <= bbox["y2"] and \
               (bbox["x2"] > self.roi_x0 and bbox["x1"] < self.roi_x1):
                crossing = True
                self.crossing_vehicle = detection
                self.last_detection_frame = self.frame_idx
                break
        
        # Detect axle pulses while crossing
        if crossing and len(self.intensity_hist) >= 3:
            # Look for downward spike in intensity
            intensity_drop = self.intensity_hist[-3] - self.intensity_hist[-1]
            frames_since_pulse = self.frame_idx - self.last_pulse_frame
            
            if intensity_drop > self.pulse_threshold and frames_since_pulse > self.min_gap_frames:
                self.current_axle_pulses += 1
                self.last_pulse_frame = self.frame_idx
        
        # Check if vehicle finished crossing
        result = None
        frames_since_detection = self.frame_idx - self.last_detection_frame
        
        # Vehicle left if: no longer crossing AND we had pulses AND haven't seen vehicle for a bit
        if not crossing and self.current_axle_pulses > 0 and frames_since_detection > 5:
            # Finalize classification
            from .rules import toll_class
            
            axle_count = self.current_axle_pulses
            vehicle_type = self.crossing_vehicle.get("vehicle_type", "unknown") if self.crossing_vehicle else "unknown"
            
            result = {
                "vehicle_type": vehicle_type,
                "axle_count": axle_count,
                "predicted_class": toll_class(vehicle_type, axle_count, False),
                "confidence": self.crossing_vehicle.get("confidence", 0.0) if self.crossing_vehicle else 0.0,
                "frame_end": self.frame_idx
            }
            
            # Reset for next vehicle
            self.reset()
        
        self.frame_idx += 1
        return result
    
    def get_visualization_data(self) -> Dict:
        """
        Get current state for visualization.
        
        Returns:
            Dictionary with visualization data
        """
        return {
            "tripline_y": self.tripline_y,
            "roi_x0": self.roi_x0,
            "roi_x1": self.roi_x1,
            "current_pulses": self.current_axle_pulses,
            "avg_intensity": self.intensity_hist[-1] if self.intensity_hist else 0.0
        }
    
    def draw_tripline(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tripline visualization on frame.
        
        Args:
            frame: Input frame
        
        Returns:
            Frame with tripline drawn
        """
        annotated = frame.copy()
        
        # Draw tripline
        cv2.line(
            annotated,
            (self.roi_x0, self.tripline_y),
            (self.roi_x1, self.tripline_y),
            (0, 255, 0),
            2
        )
        
        # Draw ROI boundaries
        cv2.line(
            annotated,
            (self.roi_x0, self.tripline_y - 20),
            (self.roi_x0, self.tripline_y + 20),
            (0, 255, 0),
            2
        )
        cv2.line(
            annotated,
            (self.roi_x1, self.tripline_y - 20),
            (self.roi_x1, self.tripline_y + 20),
            (0, 255, 0),
            2
        )
        
        # Draw status text
        avg_intensity = self.intensity_hist[-1] if self.intensity_hist else 0.0
        status_text = f"Pulses: {self.current_axle_pulses} | Intensity: {avg_intensity:.1f}"
        
        cv2.putText(
            annotated,
            status_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        return annotated

