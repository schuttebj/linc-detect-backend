"""
Download YOLO11n model during build phase and copy to project directory.
"""
import urllib.request
import shutil
from pathlib import Path

# Create cache directory
cache_dir = Path.home() / '.cache' / 'ultralytics'
cache_dir.mkdir(parents=True, exist_ok=True)

# Also create a models directory in the project
project_models_dir = Path('models')
project_models_dir.mkdir(exist_ok=True)

model_file = cache_dir / 'yolo11n.pt'
project_model = project_models_dir / 'yolo11n.pt'

# Download if not in cache
if not model_file.exists():
    print("📥 Downloading YOLO11n model from GitHub...")
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    
    try:
        urllib.request.urlretrieve(url, model_file)
        print(f"✅ Model downloaded to {model_file}")
        print(f"   File size: {model_file.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise
else:
    print(f"✅ Model already in cache: {model_file}")

# Copy to project models directory for runtime access
print(f"📋 Copying model to project directory...")
shutil.copy2(model_file, project_model)
print(f"✅ Model copied to {project_model}")
print(f"   This will be available at runtime!")

