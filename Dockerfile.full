# Enhanced Dockerfile with Full ML Capabilities
# Face Vector Embedding System - Production Ready with Complete ML Stack
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for full ML stack
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    libatlas3-base \
    libavcodec-dev \
    libavformat-dev \
    libgfortran5 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libgtk-3-0 \
    libhdf5-dev \
    libjpeg-dev \
    liblapack-dev \
    libopenblas-dev \
    libopencv-dev \
    libpng-dev \
    libsm6 \
    libswscale-dev \
    libtiff-dev \
    libv4l-dev \
    libx264-dev \
    libxext6 \
    libxrender-dev \
    libxvidcore-dev \
    pkg-config \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-full.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy==1.24.3 && \
    pip install opencv-python==4.8.1.78 && \
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install tensorflow==2.13.0 && \
    pip install -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p uploads data models logs

# Download pre-trained models
RUN mkdir -p models && \
    # Download OpenCV DNN face detection models
    wget -q https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector.pbtxt \
         -O models/opencv_face_detector.pbtxt && \
    wget -q https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb \
         -O models/opencv_face_detector_uint8.pb && \
    # Download MTCNN weights (will be downloaded automatically by mtcnn library)
    python -c "import mtcnn; mtcnn.MTCNN()" || true

# Set environment variables for production
ENV FLASK_APP=app.main
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV FACE_DETECTION_MODEL=mtcnn
ENV FACE_ENCODING_MODEL=facenet
ENV DATABASE_URL=postgresql://face_user:face_password@postgres:5432/face_db
ENV REDIS_URL=redis://redis:6379/0
ENV UPLOAD_FOLDER=/app/uploads
ENV FAISS_INDEX_PATH=/app/data/face_index.faiss
ENV MODEL_PATH=/app/models

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/health', timeout=5)" || exit 1

# Use production-ready WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "--timeout", "120", "--max-requests", "1000", "app.main:app"]
