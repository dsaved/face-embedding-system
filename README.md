# Face Recognition System

Real-time face detection and recognition system with video streaming support.

## Features

- Face detection and recognition from images/video
- Real-time webcam processing with WebSocket streaming
- SQLite database with face embeddings storage
- REST API for face operations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
python app/main.py
```

Open http://localhost:5001/video/demo for the video demo interface.

## API Usage

```bash
# Register a face
curl -X POST http://localhost:5001/api/v1/faces/register \
  -F "file=@person.jpg" \
  -F "person_name=John Doe"

# Detect faces in image
curl -X POST http://localhost:5001/api/v1/faces/detect \
  -F "file=@image.jpg"

# Search similar faces
curl -X POST http://localhost:5001/api/v1/faces/search \
  -F "file=@query.jpg"

