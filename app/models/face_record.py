"""
Database models for face embedding storage.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
import json
import numpy as np

Base = declarative_base()


class FaceRecord(Base):
    """Model for storing face records and embeddings."""
    
    __tablename__ = 'face_records'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(String(100), nullable=False, index=True)
    person_name = Column(String(255), nullable=False)
    embedding = Column(Vector(512), nullable=False)  # 512-dimensional vector
    confidence_score = Column(Float, default=0.0)
    image_path = Column(String(500))
    face_bbox = Column(Text)  # JSON string storing bounding box coordinates
    landmarks = Column(Text)  # JSON string storing facial landmarks
    encoding_model = Column(String(50), default='facenet')
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<FaceRecord(id={self.id}, person_id='{self.person_id}', name='{self.person_name}')>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'person_id': self.person_id,
            'person_name': self.person_name,
            'confidence_score': self.confidence_score,
            'image_path': self.image_path,
            'face_bbox': json.loads(self.face_bbox) if self.face_bbox else None,
            'landmarks': json.loads(self.landmarks) if self.landmarks else None,
            'encoding_model': self.encoding_model,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def set_bbox(self, bbox):
        """Set bounding box as JSON string."""
        if bbox:
            # Convert NumPy types to Python native types
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            elif isinstance(bbox, (list, tuple)):
                bbox = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in bbox]
            self.face_bbox = json.dumps(bbox)
        else:
            self.face_bbox = None
    
    def get_bbox(self):
        """Get bounding box from JSON string."""
        return json.loads(self.face_bbox) if self.face_bbox else None
    
    def set_landmarks(self, landmarks):
        """Set landmarks as JSON string."""
        if landmarks:
            # Convert NumPy types to Python native types
            if isinstance(landmarks, np.ndarray):
                landmarks = landmarks.tolist()
            elif isinstance(landmarks, dict):
                landmarks = {k: (v.tolist() if isinstance(v, np.ndarray) else 
                               [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in v] 
                               if isinstance(v, (list, tuple)) else v) 
                           for k, v in landmarks.items()}
            self.landmarks = json.dumps(landmarks)
        else:
            self.landmarks = None
    
    def get_landmarks(self):
        """Get landmarks from JSON string."""
        return json.loads(self.landmarks) if self.landmarks else None


class FaceProcessingLog(Base):
    """Model for logging face processing operations."""
    
    __tablename__ = 'face_processing_logs'
    
    id = Column(Integer, primary_key=True)
    operation_type = Column(String(50), nullable=False)  # detect, encode, identify, verify
    input_source = Column(String(500))  # file path or source identifier
    processing_time = Column(Float)  # time in seconds
    faces_detected = Column(Integer, default=0)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    operation_metadata = Column(Text)  # JSON string for additional data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<FaceProcessingLog(id={self.id}, operation='{self.operation_type}', success={self.success})>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'id': self.id,
            'operation_type': self.operation_type,
            'input_source': self.input_source,
            'processing_time': self.processing_time,
            'faces_detected': self.faces_detected,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': json.loads(self.operation_metadata) if self.operation_metadata else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def set_metadata(self, metadata):
        """Set metadata as JSON string."""
        self.operation_metadata = json.dumps(metadata) if metadata else None
    
    def get_metadata(self):
        """Get metadata from JSON string."""
        return json.loads(self.operation_metadata) if self.operation_metadata else None
