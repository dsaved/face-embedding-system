"""
Configuration management for the Face Embedding System.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///face_db.sqlite")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    DEBUG = os.getenv("DEBUG", "False") == "True"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    PORT = int(os.getenv("PORT", 5001))
    
    # Face processing configuration
    FACE_DETECTION_MODEL = os.getenv("FACE_DETECTION_MODEL", "opencv_dnn")  # opencv_dnn, mtcnn
    FACE_ENCODING_MODEL = os.getenv("FACE_ENCODING_MODEL", "facenet")  # facenet, arcface
    FACE_EMBEDDING_SIZE = int(os.getenv("FACE_EMBEDDING_SIZE", "128"))  # 128 or 512
    MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "50"))  # minimum face size in pixels
    FACE_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))
    
    # File upload settings
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "16777216"))  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Vector database settings
    USE_FAISS_INDEX = os.getenv("USE_FAISS_INDEX", "True") == "True"
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/face_index.faiss")
    
    # Redis cache settings
    CACHE_EMBEDDINGS = os.getenv("CACHE_EMBEDDINGS", "True") == "True"
    EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "3600"))  # 1 hour
    
    # Video Processing Configuration
    VIDEO_FRAME_SKIP = int(os.getenv("VIDEO_FRAME_SKIP", "2"))  # Process every 2nd frame
    VIDEO_RECOGNITION_INTERVAL = int(os.getenv("VIDEO_RECOGNITION_INTERVAL", "30"))  # Frames
    VIDEO_DETECTION_INTERVAL = int(os.getenv("VIDEO_DETECTION_INTERVAL", "5"))  # Frames
    VIDEO_MAX_TRACKED_FACES = int(os.getenv("VIDEO_MAX_TRACKED_FACES", "10"))
    VIDEO_MAX_FRAME_WIDTH = int(os.getenv("VIDEO_MAX_FRAME_WIDTH", "640"))  # Pixels
    VIDEO_CACHE_TTL = int(os.getenv("VIDEO_CACHE_TTL", "300"))  # 5 minutes
    VIDEO_CACHE_MAX_SIZE = int(os.getenv("VIDEO_CACHE_MAX_SIZE", "1000"))
    
    # WebSocket Configuration
    WEBSOCKET_TIMEOUT = int(os.getenv("WEBSOCKET_TIMEOUT", "60"))  # Seconds
    WEBSOCKET_PING_TIMEOUT = int(os.getenv("WEBSOCKET_PING_TIMEOUT", "60"))
    WEBSOCKET_PING_INTERVAL = int(os.getenv("WEBSOCKET_PING_INTERVAL", "25"))
    MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS", "10"))
    
    # Security settings
    API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "True") == "True"
    API_KEY = os.getenv("API_KEY", "sk-face123abc")  # Added single API key
    API_KEYS = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "True") == "True"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))  # requests per minute
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # File security
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    SCAN_UPLOADS = os.getenv("SCAN_UPLOADS", "True") == "True"
    
    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Audit logging
    AUDIT_LOG_ENABLED = os.getenv("AUDIT_LOG_ENABLED", "True") == "True"
    AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", "logs/audit.log")
    
    # Security headers
    SECURITY_HEADERS_ENABLED = os.getenv("SECURITY_HEADERS_ENABLED", "True") == "True"
