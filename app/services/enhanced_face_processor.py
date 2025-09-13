"""
Enhanced Face Processor with Full ML Capabilities
Supports multiple detection and encoding models for production use.
"""
import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import face_recognition
import dlib
from mtcnn import MTCNN
import tensorflow as tf
from keras_facenet import FaceNet
import torch
from facenet_pytorch import MTCNN as PytorchMTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import pickle

logger = logging.getLogger(__name__)

class EnhancedFaceDetector:
    """Enhanced face detector supporting multiple models."""
    
    def __init__(self, model_type: str = "mtcnn"):
        self.model_type = model_type.lower()
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the specified detection model."""
        try:
            if self.model_type == "mtcnn":
                self.detector = MTCNN(min_face_size=20, scale_factor=0.709, 
                                    steps_threshold=[0.6, 0.7, 0.7])
                logger.info("MTCNN face detector initialized")
                
            elif self.model_type == "opencv":
                model_path = os.getenv('MODEL_PATH', './models')
                prototxt_path = os.path.join(model_path, 'opencv_face_detector.pbtxt')
                model_path = os.path.join(model_path, 'opencv_face_detector_uint8.pb')
                
                if os.path.exists(prototxt_path) and os.path.exists(model_path):
                    self.detector = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
                    logger.info("OpenCV DNN face detector initialized")
                else:
                    raise FileNotFoundError("OpenCV model files not found")
                    
            elif self.model_type == "dlib":
                self.detector = dlib.get_frontal_face_detector()
                logger.info("Dlib face detector initialized")
                
            elif self.model_type == "pytorch_mtcnn":
                self.detector = PytorchMTCNN(keep_all=True, post_process=False, device='cpu')
                logger.info("PyTorch MTCNN face detector initialized")
                
            else:
                raise ValueError(f"Unsupported detector model: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_type} detector: {e}")
            # Fallback to face_recognition
            self.model_type = "face_recognition"
            logger.info("Falling back to face_recognition detector")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the image using the specified model."""
        try:
            if self.model_type == "mtcnn":
                return self._detect_mtcnn(image)
            elif self.model_type == "opencv":
                return self._detect_opencv(image)
            elif self.model_type == "dlib":
                return self._detect_dlib(image)
            elif self.model_type == "pytorch_mtcnn":
                return self._detect_pytorch_mtcnn(image)
            else:
                return self._detect_face_recognition(image)
                
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MTCNN."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)
        
        faces = []
        for detection in detections:
            bbox = detection['box']
            confidence = detection['confidence']
            landmarks = detection['keypoints']
            
            # Convert landmarks to list format
            landmark_points = [
                [landmarks['left_eye'][0], landmarks['left_eye'][1]],
                [landmarks['right_eye'][0], landmarks['right_eye'][1]],
                [landmarks['nose'][0], landmarks['nose'][1]],
                [landmarks['mouth_left'][0], landmarks['mouth_left'][1]],
                [landmarks['mouth_right'][0], landmarks['mouth_right'][1]]
            ]
            
            faces.append({
                'bbox': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                'confidence': confidence,
                'landmarks': landmark_points
            })
        
        return faces
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV DNN."""
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'landmarks': []  # OpenCV doesn't provide landmarks
                })
        
        return faces
    
    def _detect_dlib(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using Dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector(gray)
        
        faces = []
        for detection in detections:
            x1, y1, x2, y2 = detection.left(), detection.top(), detection.right(), detection.bottom()
            
            faces.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': 1.0,  # Dlib doesn't provide confidence scores
                'landmarks': []
            })
        
        return faces
    
    def _detect_pytorch_mtcnn(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using PyTorch MTCNN."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = self.detector.detect(rgb_image, landmarks=True)
        
        faces = []
        if boxes is not None:
            for box, prob, landmark in zip(boxes, probs, landmarks):
                if prob > 0.9:
                    faces.append({
                        'bbox': box.tolist(),
                        'confidence': float(prob),
                        'landmarks': landmark.tolist() if landmark is not None else []
                    })
        
        return faces
    
    def _detect_face_recognition(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using face_recognition library."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_landmarks = face_recognition.face_landmarks(rgb_image)
        
        faces = []
        for i, (top, right, bottom, left) in enumerate(face_locations):
            landmarks = []
            if i < len(face_landmarks):
                # Convert landmarks to our format
                for feature in face_landmarks[i].values():
                    landmarks.extend(feature)
            
            faces.append({
                'bbox': [left, top, right, bottom],
                'confidence': 1.0,
                'landmarks': landmarks
            })
        
        return faces

class EnhancedFaceEncoder:
    """Enhanced face encoder supporting multiple models."""
    
    def __init__(self, model_type: str = "facenet"):
        self.model_type = model_type.lower()
        self.encoder = None
        self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize the specified encoding model."""
        try:
            if self.model_type == "facenet":
                self.encoder = FaceNet()
                logger.info("FaceNet encoder initialized")
                
            elif self.model_type == "pytorch_facenet":
                self.encoder = InceptionResnetV1(pretrained='vggface2').eval()
                logger.info("PyTorch FaceNet encoder initialized")
                
            elif self.model_type == "face_recognition":
                # Uses dlib's face recognition model
                logger.info("face_recognition encoder initialized")
                
            else:
                raise ValueError(f"Unsupported encoder model: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_type} encoder: {e}")
            # Fallback to face_recognition
            self.model_type = "face_recognition"
            logger.info("Falling back to face_recognition encoder")
    
    def encode_face(self, face_image: np.ndarray) -> np.ndarray:
        """Encode a face image into a feature vector."""
        try:
            if self.model_type == "facenet":
                return self._encode_facenet(face_image)
            elif self.model_type == "pytorch_facenet":
                return self._encode_pytorch_facenet(face_image)
            else:
                return self._encode_face_recognition(face_image)
                
        except Exception as e:
            logger.error(f"Face encoding failed: {e}")
            return np.array([])
    
    def _encode_facenet(self, face_image: np.ndarray) -> np.ndarray:
        """Encode face using Keras FaceNet."""
        # Resize and preprocess
        face_image = cv2.resize(face_image, (160, 160))
        face_image = face_image.astype('float32')
        face_image = (face_image - 127.5) / 128.0
        face_image = np.expand_dims(face_image, axis=0)
        
        # Get embedding
        embedding = self.encoder.embeddings(face_image)
        return embedding[0]
    
    def _encode_pytorch_facenet(self, face_image: np.ndarray) -> np.ndarray:
        """Encode face using PyTorch FaceNet."""
        # Preprocess
        face_image = cv2.resize(face_image, (160, 160))
        face_image = np.transpose(face_image, (2, 0, 1))
        face_image = torch.tensor(face_image).float()
        face_image = (face_image - 127.5) / 128.0
        face_image = face_image.unsqueeze(0)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.encoder(face_image)
        
        return embedding.numpy()[0]
    
    def _encode_face_recognition(self, face_image: np.ndarray) -> np.ndarray:
        """Encode face using face_recognition library."""
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        
        if encodings:
            return encodings[0]
        else:
            return np.array([])

class EnhancedFaceProcessor:
    """Enhanced face processor with full ML capabilities."""
    
    def __init__(self, detection_model: str = None, encoding_model: str = None):
        # Get models from environment or use defaults
        self.detection_model = detection_model or os.getenv('FACE_DETECTION_MODEL', 'mtcnn')
        self.encoding_model = encoding_model or os.getenv('FACE_ENCODING_MODEL', 'facenet')
        
        # Initialize components
        self.detector = EnhancedFaceDetector(self.detection_model)
        self.encoder = EnhancedFaceEncoder(self.encoding_model)
        
        logger.info(f"Enhanced face processor initialized with {self.detection_model} detector and {self.encoding_model} encoder")
    
    def process_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Process an image to detect and encode faces."""
        faces = self.detector.detect_faces(image)
        
        processed_faces = []
        for face in faces:
            try:
                # Extract face region
                bbox = face['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Ensure bbox is within image bounds
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    face_image = image[y1:y2, x1:x2]
                    
                    # Encode face
                    embedding = self.encoder.encode_face(face_image)
                    
                    if len(embedding) > 0:
                        processed_faces.append({
                            'bbox': face['bbox'],
                            'confidence': face['confidence'],
                            'landmarks': face['landmarks'],
                            'embedding': embedding,
                            'embedding_model': self.encoding_model
                        })
                        
            except Exception as e:
                logger.error(f"Failed to process face: {e}")
                continue
        
        return processed_faces
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two face embeddings."""
        try:
            # Reshape for cosine similarity calculation
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def is_same_person(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.6) -> bool:
        """Determine if two embeddings represent the same person."""
        similarity = self.calculate_similarity(embedding1, embedding2)
        return similarity >= threshold

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Standalone function for calculating similarity between embeddings."""
    processor = EnhancedFaceProcessor()
    return processor.calculate_similarity(embedding1, embedding2)
