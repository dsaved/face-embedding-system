# Face Embedding System Dockerfile
# Face Vector Embedding System Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas3-base \
    libopenblas-dev \
    gfortran \
    wget \
    curl \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY setup.py .
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p uploads data models

# Download face detection models (optional, for better performance)
RUN mkdir -p models && \
    wget -q https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector.pbtxt \
         -O models/opencv_face_detector.pbtxt && \
    wget -q https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb \
         -O models/opencv_face_detector_uint8.pb || true

# Set environment variables
ENV FLASK_APP=app.main
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV FACE_DETECTION_MODEL=mock
ENV FACE_ENCODING_MODEL=mock
ENV DATABASE_URL=sqlite:///app/data/face_db.sqlite

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Copy test application
COPY test_app.py .

# Run the application using our tested app
CMD ["python", "test_app.py"]
