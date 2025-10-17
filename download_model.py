"""
Download YOLO11n model during build phase.
"""
from ultralytics import YOLO

print("Downloading YOLO11n model...")
model = YOLO("yolo11n.pt")
print("✅ Model downloaded successfully!")
print(f"Model location: {model.model_path if hasattr(model, 'model_path') else 'cached'}")

