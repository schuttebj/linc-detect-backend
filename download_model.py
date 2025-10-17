"""
Download YOLO11n model during build phase.
"""
import urllib.request
import os
from pathlib import Path

# Create cache directory that Ultralytics uses
cache_dir = Path.home() / '.cache' / 'ultralytics'
cache_dir.mkdir(parents=True, exist_ok=True)

model_file = cache_dir / 'yolo11n.pt'

if not model_file.exists():
    print("📥 Downloading YOLO11n model from GitHub...")
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    
    try:
        urllib.request.urlretrieve(url, model_file)
        print(f"✅ Model downloaded successfully to {model_file}")
        print(f"   File size: {model_file.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise
else:
    print(f"✅ Model already exists at {model_file}")

