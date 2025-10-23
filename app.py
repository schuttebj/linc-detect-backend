"""
FastAPI backend for vehicle classification system.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor

from classification import VehicleDetector
from database import get_database, Database


# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings from environment variables."""
    database_url: str = "postgresql://admin:localpass123@localhost:5432/vehicle_classification"
    cors_origins: str = "http://localhost:5173,http://localhost:5174"
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = 0.25
    upload_dir: str = "./uploads"
    max_upload_size: int = 10485760  # 10MB
    
    class Config:
        env_file = ".env"


settings = Settings()


# Create uploads directory
UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


# Global detector instance
detector: Optional[VehicleDetector] = None

# Thread pool for async stream processing
stream_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    global detector
    print("🚀 Starting up...")
    print(f"Loading YOLO model: {settings.yolo_model}")
    
    # Use just the model name without .pt - Ultralytics will handle download
    model_name = settings.yolo_model.replace('.pt', '')
    detector = VehicleDetector(
        model_path=model_name,
        confidence=settings.yolo_confidence
    )
    print("✅ Model loaded successfully!")
    
    # Initialize database
    db = await get_database()
    print("✅ Database connected!")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")
    if db:
        await db.disconnect()


# Create FastAPI app
app = FastAPI(
    title="Vehicle Classification API",
    description="YOLO11n-based vehicle classification for toll systems",
    version="1.0.0",
    lifespan=lifespan
)


# CORS middleware
origins = settings.cors_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# ============================================================================
# Pydantic Models
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""
    classification_id: str
    corrected_class: str
    corrected_axles: Optional[int] = None
    notes: Optional[str] = None


class ClassificationResponse(BaseModel):
    """Response model for classification results."""
    id: str
    vehicle_type: str
    axle_count: int
    predicted_class: str
    confidence: float
    processing_time: float
    image_url: str
    bbox: dict
    timestamp: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Vehicle Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db = await get_database()
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "database_connected": db.pool is not None
    }


@app.post("/api/classify/video")
async def classify_video(file: UploadFile = File(...)):
    """
    Classify vehicles from an uploaded video using tripline pulse counting.
    
    Args:
        file: Uploaded video file (mp4, avi, mov)
    
    Returns:
        List of classification results with accurate axle counts
    """
    if not detector:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    import cv2
    from classification.axle_counter import AxleCounter
    
    # Save uploaded video
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".mp4"
    video_path = settings.upload_dir / f"{file_id}{file_ext}"
    
    contents = await file.read()
    video_path.write_bytes(contents)
    
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize axle counter with tripline
        axle_counter = AxleCounter(fps=fps)
        
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect vehicles in frame
            detections = detector.detect_vehicles(frame)
            
            # Process frame with axle counter
            classification_result = axle_counter.process_frame(frame, detections)
            
            if classification_result:
                # Vehicle finished crossing - save result
                timestamp = frame_idx / fps
                
                # Save to database
                db = await get_database()
                classification_id = await db.save_classification(
                    vehicle_type=classification_result["vehicle_type"],
                    axle_count=classification_result["axle_count"],
                    predicted_class=classification_result["predicted_class"],
                    confidence=classification_result["confidence"],
                    processing_time=0.0,
                    image_url=None,
                    annotated_image_url=None
                )
                
                results.append({
                    "id": classification_id,
                    "timestamp": timestamp,
                    "vehicle_type": classification_result["vehicle_type"],
                    "axle_count": classification_result["axle_count"],
                    "predicted_class": classification_result["predicted_class"],
                    "confidence": classification_result["confidence"],
                    "pulse_count": classification_result["pulse_count"]
                })
            
            frame_idx += 1
        
        cap.release()
        
        return {
            "total_vehicles": len(results),
            "total_frames": total_frames,
            "fps": fps,
            "duration_seconds": total_frames / fps,
            "classifications": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")
    finally:
        # Cleanup
        if video_path.exists():
            video_path.unlink()


@app.post("/api/classify/image")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify a vehicle from an uploaded image.
    
    Args:
        file: Uploaded image file
    
    Returns:
        Classification result with metadata
    """
    if not detector:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file size
    contents = await file.read()
    if len(contents) > settings.max_upload_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".jpg"
    file_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Process image
    start_time = time.time()
    
    try:
        result, annotated_image = detector.detect_and_classify(str(file_path))
        processing_time = time.time() - start_time
        
        if result is None:
            raise HTTPException(status_code=400, detail="No vehicle detected in image")
        
        # Save annotated image
        annotated_path = UPLOAD_DIR / f"{file_id}_annotated{file_ext}"
        Image.fromarray(annotated_image).save(annotated_path)
        
        # Store in database
        db = await get_database()
        classification_id = await db.insert_classification(
            image_path=str(file_path),
            vehicle_type=result["vehicle_type"],
            axle_count=result["axle_count"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            processing_time=processing_time,
            bbox=result["bbox"]
        )
        
        return {
            "id": classification_id,
            "vehicle_type": result["vehicle_type"],
            "yolo_class": result.get("yolo_class", "unknown"),  # Add YOLO detection for debugging
            "axle_count": result["axle_count"],
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "processing_time": processing_time,
            "image_url": f"/uploads/{file_path.name}",
            "annotated_image_url": f"/uploads/{annotated_path.name}",
            "bbox": result["bbox"],
            "debug": result.get("debug", {})  # Include debug information
        }
        
    except Exception as e:
        # Clean up files on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/classifications")
async def get_classifications(limit: int = 50, offset: int = 0):
    """
    Get paginated classification results.
    
    Args:
        limit: Number of results to return
        offset: Number of results to skip
    
    Returns:
        List of classification results
    """
    db = await get_database()
    classifications = await db.get_classifications(limit=limit, offset=offset)
    
    # Convert to response format
    results = []
    for c in classifications:
        results.append({
            "id": str(c["id"]),
            "timestamp": c["timestamp"].isoformat() if c["timestamp"] else None,
            "vehicle_type": c["vehicle_type"],
            "axle_count": c["axle_count"],
            "predicted_class": c["predicted_class"],
            "corrected_class": c["corrected_class"],
            "confidence": c["confidence"],
            "processing_time": c["processing_time"],
            "image_url": f"/uploads/{Path(c['image_path']).name}" if c["image_path"] else None,
            "bbox": {
                "x1": c["bbox_x1"],
                "y1": c["bbox_y1"],
                "x2": c["bbox_x2"],
                "y2": c["bbox_y2"]
            } if c["bbox_x1"] is not None else None
        })
    
    return results


@app.get("/api/classifications/{classification_id}")
async def get_classification(classification_id: str):
    """Get a specific classification by ID."""
    db = await get_database()
    classification = await db.get_classification_by_id(classification_id)
    
    if not classification:
        raise HTTPException(status_code=404, detail="Classification not found")
    
    return {
        "id": str(classification["id"]),
        "timestamp": classification["timestamp"].isoformat() if classification["timestamp"] else None,
        "vehicle_type": classification["vehicle_type"],
        "axle_count": classification["axle_count"],
        "predicted_class": classification["predicted_class"],
        "corrected_class": classification["corrected_class"],
        "confidence": classification["confidence"],
        "processing_time": classification["processing_time"],
        "image_url": f"/uploads/{Path(classification['image_path']).name}" if classification["image_path"] else None,
        "bbox": {
            "x1": classification["bbox_x1"],
            "y1": classification["bbox_y1"],
            "x2": classification["bbox_x2"],
            "y2": classification["bbox_y2"]
        } if classification["bbox_x1"] is not None else None
    }


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback/correction for a classification.
    
    Args:
        feedback: Feedback data including corrected class
    
    Returns:
        Success message with feedback ID
    """
    db = await get_database()
    
    # Validate classification exists
    classification = await db.get_classification_by_id(feedback.classification_id)
    if not classification:
        raise HTTPException(status_code=404, detail="Classification not found")
    
    # Insert feedback
    feedback_id = await db.insert_feedback(
        classification_id=feedback.classification_id,
        corrected_class=feedback.corrected_class,
        corrected_axles=feedback.corrected_axles,
        notes=feedback.notes
    )
    
    return {
        "message": "Feedback submitted successfully",
        "feedback_id": feedback_id
    }


@app.get("/api/metrics")
async def get_metrics():
    """
    Get accuracy metrics and statistics.
    
    Returns:
        Dictionary with accuracy metrics and confusion matrix
    """
    db = await get_database()
    metrics = await db.get_accuracy_metrics()
    stats = await db.get_statistics()
    
    return {
        "accuracy": metrics,
        "statistics": stats
    }


@app.delete("/api/data/clear")
async def clear_data():
    """
    Clear all classification data from database.
    
    Returns:
        Success message
    """
    db = await get_database()
    await db.clear_all_data()
    
    # Optionally clear uploaded files
    for file in UPLOAD_DIR.glob("*"):
        if file.is_file():
            file.unlink()
    
    return {"message": "All data cleared successfully"}


@app.get("/api/statistics")
async def get_statistics():
    """Get general statistics about classifications."""
    db = await get_database()
    stats = await db.get_statistics()
    return stats


# ============================================================================
# Live Stream Processing Endpoints
# ============================================================================

@app.post("/api/classify/stream")
async def classify_stream(
    stream_url: str,
    duration: int = 30,
    use_tripline: bool = True
):
    """
    Classify vehicles from live stream (RTSP/RTMP/HTTP/USB camera).
    
    Args:
        stream_url: Stream URL (e.g., rtsp://192.168.1.100/stream) or device index (e.g., "0")
        duration: Processing duration in seconds
        use_tripline: Use tripline counting for accurate axle count (recommended for production)
    
    Returns:
        Classification results from stream
    """
    if not detector:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    def process_stream_sync():
        """Process stream in separate thread to avoid blocking."""
        from classification.stream_processor import StreamProcessor
        from classification.axle_counter import AxleCounter
        
        # Initialize stream processor
        axle_counter = AxleCounter(fps=25) if use_tripline else None
        processor = StreamProcessor(detector, axle_counter)
        
        try:
            if not processor.open_stream(stream_url):
                raise ValueError(f"Could not open stream: {stream_url}")
            
            # Process stream
            results = processor.process_stream(duration=duration)
            
            return results
        finally:
            processor.close()
    
    # Run in thread pool to avoid blocking event loop
    try:
        results = await asyncio.get_event_loop().run_in_executor(
            stream_executor,
            process_stream_sync
        )
        
        return {
            "stream_url": stream_url,
            "duration": duration,
            "total_vehicles": len(results),
            "use_tripline": use_tripline,
            "classifications": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream processing error: {str(e)}")


@app.get("/api/stream/formats")
async def get_supported_formats():
    """Get list of supported stream formats and examples."""
    return {
        "formats": [
            {
                "protocol": "RTSP",
                "example": "rtsp://192.168.1.100:554/stream",
                "description": "IP cameras, NVRs (most common for CCTV)",
                "notes": "Recommended for toll gate cameras"
            },
            {
                "protocol": "RTMP",
                "example": "rtmp://server.com/live/stream",
                "description": "Streaming servers (OBS, etc.)",
                "notes": "Good for remote testing"
            },
            {
                "protocol": "HTTP/HLS",
                "example": "http://server.com/stream.m3u8",
                "description": "HTTP live streaming",
                "notes": "Works with web cameras"
            },
            {
                "protocol": "USB Camera",
                "example": "0",
                "description": "Local USB camera (device index)",
                "notes": "For local testing only"
            }
        ],
        "tripline_info": {
            "recommended": True,
            "description": "Tripline counting provides 98%+ axle accuracy for fixed cameras",
            "note": "Essential for achieving 99.6% toll classification accuracy"
        }
    }


# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# ============================================================================
# Configuration & Debug Endpoints
# ============================================================================

# Global configuration
model_config = {
    "yolo_confidence": 0.25,      # YOLO detection confidence
}

@app.get("/api/config")
async def get_config():
    """Get current model configuration."""
    if not detector:
        return {"error": "Model not loaded"}
    
    return {
        "model_config": model_config,
        "model_type": "LVIS" if detector.is_lvis_model else "COCO",
        "model_loaded": detector.model is not None,
        "vehicle_classes": list(detector.LVIS_VEHICLE_CLASSES.values()) if detector.is_lvis_model else list(detector.COCO_VEHICLE_CLASSES)
    }

@app.post("/api/config")
async def update_config(config: dict):
    """Update model configuration dynamically."""
    if "yolo_confidence" in config:
        yolo_conf = float(config["yolo_confidence"])
        if 0.0 <= yolo_conf <= 1.0:
            model_config["yolo_confidence"] = yolo_conf
            # Update detector confidence
            if detector:
                detector.confidence = yolo_conf
        else:
            raise HTTPException(status_code=400, detail="YOLO confidence must be between 0.0 and 1.0")
    
    return {"message": "Configuration updated", "config": model_config}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time classification updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

