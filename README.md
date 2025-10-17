# Vehicle Classification Backend

FastAPI-based backend for vehicle classification using YOLO11n.

## Features

- ✅ YOLO11n vehicle detection
- ✅ 4-class toll classification system
- ✅ PostgreSQL database for persistence
- ✅ REST API + WebSocket support
- ✅ Image upload and classification
- ✅ Video stream processing (coming soon)
- ✅ Accuracy tracking and feedback system

## Quick Start

### Local Development

1. **Install Python 3.11+**

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start PostgreSQL:**
   ```bash
   # Using Docker (from project root)
   docker-compose up -d postgres
   ```

5. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

6. **Run the server:**
   ```bash
   uvicorn app:app --reload --port 8000
   ```

7. **Access API docs:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### Classification

- `POST /api/classify/image` - Upload and classify an image
- `GET /api/classifications` - Get paginated classification results
- `GET /api/classifications/{id}` - Get specific classification

### Feedback & Metrics

- `POST /api/feedback` - Submit correction for a classification
- `GET /api/metrics` - Get accuracy metrics and confusion matrix
- `GET /api/statistics` - Get general statistics

### Data Management

- `DELETE /api/data/clear` - Clear all data

### Real-time

- `WebSocket /ws` - Real-time classification updates

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/vehicle_classification

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:5174

# Model
YOLO_MODEL=yolo11n.pt
YOLO_CONFIDENCE=0.25

# Storage
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=10485760
```

## Classification Rules

| Class | Rule | Examples |
|-------|------|----------|
| **Class 1** | Light vehicles (≤2 axles) | Cars, SUVs, motorcycles, minibuses |
| **Class 2** | 2-axle heavy vehicles | 2-axle bus, 2-axle rigid truck |
| **Class 3** | 3-4 axles | Bus + trailer, articulated trucks |
| **Class 4** | 5+ axles | Heavy articulated trucks |

## Project Structure

```
backend/
├── app.py                  # FastAPI main application
├── database.py             # PostgreSQL schema and queries
├── classification/
│   ├── detector.py         # YOLO11 wrapper
│   ├── axle_counter.py     # Tripline pulse counting
│   └── rules.py            # Classification rules
├── requirements.txt
├── Dockerfile
└── README.md
```

## Docker Deployment

```bash
# Build image
docker build -t vehicle-classification-backend .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e CORS_ORIGINS=https://your-frontend.com \
  vehicle-classification-backend
```

## Render.com Deployment

1. Create PostgreSQL database on Render
2. Create Web Service from this repository
3. Set environment variables:
   - `DATABASE_URL` (from PostgreSQL service)
   - `CORS_ORIGINS` (your frontend URL)
4. Deploy!

The model will download automatically on first startup (~5 minutes).

## Testing

```bash
# Test image classification
curl -X POST "http://localhost:8000/api/classify/image" \
  -F "file=@test_image.jpg"

# Get metrics
curl "http://localhost:8000/api/metrics"
```

## License

MIT

