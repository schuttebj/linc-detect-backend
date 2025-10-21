"""
Multi-format video stream processor for real-time vehicle classification.
Supports RTSP, RTMP, HTTP/HLS, and USB cameras.

Integrates with tripline axle counting for production deployment.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Callable, List
from pathlib import Path
import time


class StreamProcessor:
    """Process live video streams with vehicle classification."""
    
    SUPPORTED_PROTOCOLS = ['rtsp://', 'rtmp://', 'http://', 'https://']
    
    def __init__(self, detector, axle_counter=None):
        """
        Initialize stream processor.
        
        Args:
            detector: VehicleDetector instance (with YOLO + EfficientNet)
            axle_counter: AxleCounter instance (optional, for tripline counting)
        """
        self.detector = detector
        self.axle_counter = axle_counter
        self.cap = None
        self.is_processing = False
        self.fps = 25
        self.width = 0
        self.height = 0
    
    def open_stream(self, source: str) -> bool:
        """
        Open video stream from various sources.
        
        Args:
            source: Stream URL or device index
                - RTSP: "rtsp://192.168.1.100/stream"
                - RTMP: "rtmp://server/live/stream"
                - HTTP: "http://server/stream.m3u8"
                - USB: "0" or 0 for device index
        
        Returns:
            True if stream opened successfully
        """
        # Convert string device index to int
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        print(f"Opening stream: {source}")
        
        # Open stream
        self.cap = cv2.VideoCapture(source)
        
        # Configure for network streams
        if isinstance(source, str):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # RTSP-specific optimizations
            if source.startswith('rtsp://'):
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open stream: {source}")
            return False
        
        # Get stream properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 25
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Stream opened: {self.width}x{self.height} @ {self.fps}fps")
        return True
    
    def process_stream(
        self,
        duration: Optional[int] = None,
        callback: Optional[Callable] = None,
        save_output: Optional[str] = None
    ) -> List[Dict]:
        """
        Process stream and classify vehicles.
        
        Args:
            duration: Max duration in seconds (None = infinite)
            callback: Function to call with each classification result
            save_output: Path to save annotated video (optional)
        
        Returns:
            List of classification results
        """
        if not self.cap or not self.cap.isOpened():
            raise ValueError("Stream not opened. Call open_stream() first.")
        
        self.is_processing = True
        results = []
        
        # Video writer for saving output
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, self.fps, (self.width, self.height))
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while self.is_processing:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️  Stream ended or connection lost")
                    break
                
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    print(f"✅ Duration limit reached: {duration}s")
                    break
                
                # Detect vehicles (two-stage: YOLO + EfficientNet)
                detections = self.detector.detect_vehicles(frame)
                
                # Process with axle counter if available (for production)
                classification = None
                if self.axle_counter:
                    classification = self.axle_counter.process_frame(frame, detections)
                    
                    if classification:
                        # Vehicle finished crossing tripline
                        results.append(classification)
                        
                        # Call callback (e.g., for WebSocket broadcast)
                        if callback:
                            callback(classification)
                else:
                    # Without axle counter, emit each detection
                    for detection in detections:
                        results.append(detection)
                        if callback:
                            callback(detection)
                
                # Annotate frame with detections
                if self.axle_counter:
                    annotated = self.axle_counter.draw_tripline(frame)
                else:
                    annotated = frame.copy()
                
                annotated = self._annotate_frame(annotated, detections)
                
                # Save to video
                if writer:
                    writer.write(annotated)
                
                frame_count += 1
                
                # Print progress every 5 seconds
                if frame_count % (self.fps * 5) == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_count} frames ({current_fps:.1f} fps, {len(results)} vehicles)")
        
        finally:
            if writer:
                writer.release()
            
            elapsed = time.time() - start_time
            if elapsed > 0:
                avg_fps = frame_count / elapsed
                print(f"✅ Processed {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} fps)")
                print(f"✅ Detected {len(results)} vehicles")
        
        return results
    
    def stop(self):
        """Stop processing stream."""
        self.is_processing = False
    
    def close(self):
        """Close stream and release resources."""
        self.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detections on frame.
        
        Args:
            frame: Input frame
            detections: List of vehicle detections
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            vehicle_type = det.get('vehicle_type', 'unknown')
            predicted_class = det.get('predicted_class', 'Unknown')
            confidence = det.get('confidence', 0.0)
            
            # Draw bounding box
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                (0, 255, 0),
                2
            )
            
            # Prepare labels
            label = f"{vehicle_type} ({predicted_class})"
            conf_label = f"{confidence:.2f}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_height = label_size[1] + 10
            
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1'] - label_height - 5),
                (bbox['x1'] + label_size[0] + 80, bbox['y1']),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated,
                label,
                (bbox['x1'] + 2, bbox['y1'] - label_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            cv2.putText(
                annotated,
                conf_label,
                (bbox['x1'] + label_size[0] + 10, bbox['y1'] - label_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        # Add FPS counter
        fps_text = f"FPS: {self.fps}"
        cv2.putText(
            annotated,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return annotated
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()

