"""
Tripline pulse counting for axle detection from video streams.
Enhanced with adaptive threshold and bogie detection.
"""

import collections
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2


class AxleCounter:
    """
    Counts vehicle axles using adaptive tripline intensity pulse detection.
    
    Key improvements:
    - Adaptive threshold based on signal statistics
    - Pulse clustering to handle bogie/tandem axles
    - Refractory period to prevent double-counting
    - Time-gap analysis for dual wheels vs separate axles
    """
    
    def __init__(
        self,
        tripline_y: int = 420,
        roi_x0: int = 200,
        roi_x1: int = 1100,
        hist_len: int = 40,
        fps: int = 25,
        adaptive_k: float = 2.0,
        hard_min_drop: int = 12,
        refractory_frames: int = 3,
        dual_wheel_gap_ms: int = 90,
        bogie_gap_ms: int = 350
    ):
        """
        Initialize adaptive axle counter.
        
        Args:
            tripline_y: Y-coordinate of the tripline
            roi_x0: Left X-coordinate of region of interest
            roi_x1: Right X-coordinate of region of interest
            hist_len: Number of frames to keep in history
            fps: Frame rate for time-based calculations
            adaptive_k: Multiplier for std in adaptive threshold (1.5-2.5)
            hard_min_drop: Absolute minimum intensity drop
            refractory_frames: Frames to wait after pulse (prevents double-count)
            dual_wheel_gap_ms: Max gap (ms) to consider dual wheels (merge into 1)
            bogie_gap_ms: Max gap (ms) to consider bogie/tandem (count as 2)
        """
        self.tripline_y = tripline_y
        self.roi_x0 = roi_x0
        self.roi_x1 = roi_x1
        self.hist_len = hist_len
        self.fps = fps
        self.adaptive_k = adaptive_k
        self.hard_min_drop = hard_min_drop
        self.refractory_frames = refractory_frames
        self.dual_wheel_gap_ms = dual_wheel_gap_ms
        self.bogie_gap_ms = bogie_gap_ms
        
        # State tracking
        self.intensity_hist = collections.deque(maxlen=hist_len)
        self.pulse_frames: List[int] = []  # Frame indices of detected pulses
        self.last_pulse_frame = -9999
        self.frame_idx = 0
        self.crossing_vehicle = None
        self.last_detection_frame = -9999
    
    def reset(self):
        """Reset counter state for new vehicle."""
        self.pulse_frames = []
        self.last_pulse_frame = -9999
        self.crossing_vehicle = None
        self.intensity_hist.clear()
    
    def _calculate_adaptive_threshold(self) -> float:
        """
        Calculate adaptive threshold based on signal statistics.
        
        Returns:
            Adaptive threshold value
        """
        if len(self.intensity_hist) < 10:
            return self.hard_min_drop
        
        values = np.array(list(self.intensity_hist))
        median_val = np.median(values)
        std_val = np.std(values)
        
        # Adaptive threshold: median - k * std
        adaptive = self.adaptive_k * std_val
        
        # Use maximum of adaptive and hard minimum
        return max(adaptive, self.hard_min_drop)
    
    def process_frame(
        self,
        frame: np.ndarray,
        detections: list
    ) -> Optional[Dict]:
        """
        Process a video frame to detect axle pulses with adaptive threshold.
        
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
        
        # Detect axle pulses while crossing using adaptive threshold
        if crossing and len(self.intensity_hist) >= 3:
            # Calculate adaptive threshold
            threshold = self._calculate_adaptive_threshold()
            
            # Look for downward spike in intensity
            intensity_drop = self.intensity_hist[-3] - self.intensity_hist[-1]
            frames_since_pulse = self.frame_idx - self.last_pulse_frame
            
            # Check if drop exceeds adaptive threshold and refractory period
            if intensity_drop > threshold and frames_since_pulse > self.refractory_frames:
                self.pulse_frames.append(self.frame_idx)
                self.last_pulse_frame = self.frame_idx
        
        # Check if vehicle finished crossing
        result = None
        frames_since_detection = self.frame_idx - self.last_detection_frame
        
        # Vehicle left if: no longer crossing AND we had pulses AND haven't seen vehicle for a bit
        if not crossing and len(self.pulse_frames) > 0 and frames_since_detection > 5:
            # Finalize classification with pulse clustering
            from .rules import toll_class
            
            # Count axles from pulse frames using clustering
            axle_count = self._count_axles_from_pulses(self.pulse_frames)
            vehicle_type = self.crossing_vehicle.get("vehicle_type", "unknown") if self.crossing_vehicle else "unknown"
            
            # Get bbox info for visual feature classification
            bbox = self.crossing_vehicle.get("bbox") if self.crossing_vehicle else None
            bbox_height = bbox.get("height") if bbox else None
            bbox_width = bbox.get("width") if bbox else None
            image_height = frame.shape[0] if frame is not None else None
            
            # Extract vehicle crop for visual analysis
            vehicle_crop = None
            if frame is not None and bbox:
                x1, y1 = bbox.get("x1"), bbox.get("y1")
                x2, y2 = bbox.get("x2"), bbox.get("y2")
                if all(v is not None for v in [x1, y1, x2, y2]) and y2 > y1 and x2 > x1:
                    vehicle_crop = frame[y1:y2, x1:x2, :].copy()
            
            result = {
                "vehicle_type": vehicle_type,
                "axle_count": axle_count,
                "predicted_class": toll_class(
                    vehicle_type, 
                    axle_count, 
                    False,
                    bbox_height=bbox_height,
                    bbox_width=bbox_width,
                    image_height=image_height,
                    vehicle_crop=vehicle_crop
                ),
                "confidence": self.crossing_vehicle.get("confidence", 0.0) if self.crossing_vehicle else 0.0,
                "frame_end": self.frame_idx,
                "pulse_count": len(self.pulse_frames)  # Debug info
            }
            
            # Reset for next vehicle
            self.reset()
        
        self.frame_idx += 1
        return result
    
    def _count_axles_from_pulses(self, pulse_frames: List[int]) -> int:
        """
        Count axles from pulse frames using time-gap clustering.
        
        This implements bogie/tandem detection:
        - Very close pulses (< dual_wheel_gap_ms) = same axle (merge)
        - Close pulses (< bogie_gap_ms) = tandem/bogie (count as 2)
        - Far pulses = separate axles
        
        Args:
            pulse_frames: List of frame indices where pulses were detected
        
        Returns:
            Estimated axle count
        """
        if not pulse_frames:
            return 0
        
        if len(pulse_frames) == 1:
            return 1
        
        # Convert frame gaps to milliseconds
        ms_per_frame = 1000.0 / self.fps
        gaps_ms = [
            (pulse_frames[i] - pulse_frames[i-1]) * ms_per_frame
            for i in range(1, len(pulse_frames))
        ]
        
        axle_count = 0
        i = 0
        
        while i < len(gaps_ms):
            # Merge ultra-close pulses: dual tires/noise < dual_wheel_gap_ms
            if gaps_ms[i] < self.dual_wheel_gap_ms:
                # Eat all tiny gaps as one axle (dual wheels)
                while i < len(gaps_ms) and gaps_ms[i] < self.dual_wheel_gap_ms:
                    i += 1
                axle_count += 1
            # Bogie/tandem: second axle close but not tiny
            elif gaps_ms[i] < self.bogie_gap_ms:
                # This is a bogie/tandem - count as 2 axles
                axle_count += 2
                i += 1
            else:
                # Normal axle spacing
                axle_count += 1
                i += 1
        
        # Don't forget the last axle (after last gap)
        axle_count += 1
        
        return axle_count
    
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
            "pulse_count": len(self.pulse_frames),
            "avg_intensity": self.intensity_hist[-1] if self.intensity_hist else 0.0,
            "adaptive_threshold": self._calculate_adaptive_threshold() if len(self.intensity_hist) >= 10 else self.hard_min_drop
        }
    
    def draw_tripline(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tripline visualization on frame with adaptive threshold info.
        
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
        
        # Draw status text with adaptive threshold
        avg_intensity = self.intensity_hist[-1] if self.intensity_hist else 0.0
        threshold = self._calculate_adaptive_threshold() if len(self.intensity_hist) >= 10 else self.hard_min_drop
        
        status_text = f"Pulses: {len(self.pulse_frames)} | Intensity: {avg_intensity:.1f} | Threshold: {threshold:.1f}"
        
        cv2.putText(
            annotated,
            status_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated

