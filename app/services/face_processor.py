"""
Core face processing engine for detection, alignment, and encoding.
"""
import cv2
import numpy as np
import face_recognition
import logging
from typing import List, Tuple, Optional, Dict, Any
from mtcnn import MTCNN
import time
from keras_facenet import FaceNet

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection using different algorithms."""
    
    def __init__(self, model_type: str = "opencv_dnn"):
        self.model_type = model_type
        self._init_detector()
    
    def _init_detector(self):
        """Initialize the face detection model."""
        if self.model_type == "opencv_dnn":
            self._init_opencv_dnn()
        elif self.model_type == "mtcnn":
            self._init_mtcnn()
        else:
            raise ValueError(f"Unsupported face detection model: {self.model_type}")
    
    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN face detector."""
        try:
            # Load the DNN model
            self.net = cv2.dnn.readNetFromTensorflow(
                'models/opencv_face_detector_uint8.pb',
                'models/opencv_face_detector.pbtxt'
            )
            logger.info("OpenCV DNN face detector initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load OpenCV DNN model: {e}")
            # Fallback to face_recognition library which is more accurate than Haar cascade
            self.model_type = "face_recognition_hog"
            logger.info("Fallback to face_recognition HOG detector")
    
    def _init_mtcnn(self):
        """Initialize MTCNN face detector."""
        self.detector = MTCNN()
        logger.info("MTCNN face detector initialized successfully")
    
    def detect_faces(self, image: np.ndarray, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect faces in an image with enhanced filtering to reduce false positives.
        
        Args:
            image: Input image as numpy array
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        if self.model_type == "opencv_dnn":
            faces = self._detect_opencv_dnn(image, min_confidence)
        elif self.model_type == "mtcnn":
            faces = self._detect_mtcnn(image, min_confidence)
        elif self.model_type == "face_recognition_hog":
            faces = self._detect_face_recognition(image)
        elif self.model_type == "haar_cascade":
            faces = self._detect_haar_cascade(image)
        else:
            faces = []
        
        # Apply enhanced post-processing to reduce false positives
        faces = self._post_process_detections(image, faces)
        
        return faces
    
    def _detect_opencv_dnn(self, image: np.ndarray, min_confidence: float) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV DNN."""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype("int")
                faces.append({
                    'bbox': (x, y, x1 - x, y1 - y),
                    'confidence': float(confidence),
                    'landmarks': None
                })
        return faces
    
    def _detect_mtcnn(self, image: np.ndarray, min_confidence: float) -> List[Dict[str, Any]]:
        """Detect faces using MTCNN."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_faces(rgb_image)
        
        faces = []
        for face in result:
            if face['confidence'] > min_confidence:
                x, y, width, height = face['box']
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': face['confidence'],
                    'landmarks': face['keypoints']
                })
        return faces
    
    def _detect_haar_cascade(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using Haar cascade (fallback)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rect = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        faces = []
        for (x, y, w, h) in faces_rect:
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0,  # Haar cascade doesn't provide confidence
                'landmarks': None
            })
        return faces
    
    def _detect_face_recognition(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using face_recognition library (more accurate fallback)."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use HOG model (faster, good accuracy)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        faces = []
        for (top, right, bottom, left) in face_locations:
            faces.append({
                'bbox': (left, top, right - left, bottom - top),
                'confidence': 0.9,  # face_recognition doesn't provide confidence, use high value
                'landmarks': None
            })
        return faces
    
    def _post_process_detections(self, image: np.ndarray, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply post-processing to improve detection accuracy and reduce false positives."""
        if not faces:
            return faces
        
        h, w = image.shape[:2]
        valid_faces = []
        
        for face in faces:
            x, y, width, height = face['bbox']
            
            # 1. Size validation - remove very small faces (likely false positives)
            min_face_size = max(80, min(w, h) * 0.05)  # Minimum 80px or 5% of image dimension
            if width < min_face_size or height < min_face_size:
                logger.debug(f"Filtered small face: {width}x{height} < {min_face_size}")
                continue
            
            # 2. Aspect ratio validation - faces should be roughly square-ish
            aspect_ratio = width / height
            if aspect_ratio < 0.6 or aspect_ratio > 1.8:
                logger.debug(f"Filtered face with bad aspect ratio: {aspect_ratio}")
                continue
            
            # 3. Position validation - ensure face is within image bounds
            if x < 0 or y < 0 or x + width > w or y + height > h:
                logger.debug(f"Filtered face outside image bounds: {face['bbox']}")
                continue
            
            # 4. Verify it's actually a face using face_recognition for better accuracy
            if self._verify_face_with_encodings(image, face):
                valid_faces.append(face)
            else:
                logger.debug(f"Face verification failed for bbox: {face['bbox']}")
        
        # 5. Apply Non-Maximum Suppression to remove overlapping detections
        valid_faces = self._apply_nms(valid_faces, overlap_threshold=0.4)
        
        # 6. Sort by confidence and size (prefer larger, more confident faces)
        valid_faces.sort(key=lambda f: f['confidence'] * (f['bbox'][2] * f['bbox'][3]), reverse=True)
        
        # 7. Limit number of faces to prevent too many false positives
        max_faces = 3
        if len(valid_faces) > max_faces:
            valid_faces = valid_faces[:max_faces]
            logger.info(f"Limited detections to {max_faces} most confident faces")
        
        return valid_faces
    
    def _verify_face_with_encodings(self, image: np.ndarray, face: Dict[str, Any]) -> bool:
        """Verify if a detection is actually a face using face encodings."""
        try:
            x, y, width, height = face['bbox']
            
            # Extract face region with some padding
            padding = 20
            face_region = image[max(0, y-padding):min(image.shape[0], y+height+padding),
                              max(0, x-padding):min(image.shape[1], x+width+padding)]
            
            if face_region.size == 0 or face_region.shape[0] < 50 or face_region.shape[1] < 50:
                return False
            
            # Convert to RGB
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Try to generate face encoding
            encodings = face_recognition.face_encodings(rgb_face)
            
            # If we can generate an encoding, it's likely a real face
            return len(encodings) > 0
            
        except Exception as e:
            logger.debug(f"Face verification error: {e}")
            return False  # If verification fails, assume it's not a valid face
    
    def _apply_nms(self, faces: List[Dict[str, Any]], overlap_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if len(faces) <= 1:
            return faces
        
        # Convert to format expected by OpenCV
        boxes = []
        confidences = []
        
        for face in faces:
            x, y, w, h = face['bbox']
            boxes.append([x, y, w, h])
            confidences.append(face['confidence'])
        
        # Apply NMS
        try:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, overlap_threshold)
            
            if len(indices) > 0:
                # indices is returned as a list of lists, flatten it
                if isinstance(indices[0], list):
                    indices = [i[0] for i in indices]
                else:
                    indices = indices.flatten()
                
                return [faces[i] for i in indices]
        except Exception as e:
            logger.debug(f"NMS error: {e}")
            return faces
        
        return faces


class FaceEncoder:
    """Face encoding using different models."""
    
    def __init__(self, model_type: str = "facenet"):
        self.model_type = model_type
        self._init_encoder()
    
    def _init_encoder(self):
        """Initialize the face encoding model."""
        if self.model_type == "facenet":
            self.embedder = FaceNet()
            logger.info("FaceNet encoder initialized successfully")
        elif self.model_type == "face_recognition":
            # face_recognition library uses dlib
            logger.info("Face recognition (dlib) encoder initialized")
        else:
            raise ValueError(f"Unsupported face encoding model: {self.model_type}")
    
    def encode_faces(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Generate face encodings for detected faces.
        
        Args:
            image: Input image as numpy array
            face_locations: List of face bounding boxes (x, y, width, height)
            
        Returns:
            List of face encodings (128 or 512 dimensional vectors)
        """
        if self.model_type == "facenet":
            return self._encode_facenet(image, face_locations)
        elif self.model_type == "face_recognition":
            return self._encode_face_recognition(image, face_locations)
        else:
            return []
    
    def _encode_facenet(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Encode faces using FaceNet."""
        encodings = []
        for (x, y, w, h) in face_locations:
            # Extract face region
            face = image[y:y+h, x:x+w]
            if face.size == 0:
                continue
            
            # Resize to 160x160 (FaceNet input size)
            face_resized = cv2.resize(face, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Generate embedding
            embedding = self.embedder.embeddings([face_rgb])[0]
            encodings.append(embedding)
        
        return encodings
    
    def _encode_face_recognition(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Encode faces using face_recognition library."""
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert bounding boxes to face_recognition format (top, right, bottom, left)
        face_locations_dlib = []
        for (x, y, w, h) in face_locations:
            face_locations_dlib.append((y, x + w, y + h, x))
        
        # Generate encodings
        encodings = face_recognition.face_encodings(rgb_image, face_locations_dlib)
        return encodings


class FaceProcessor:
    """Main face processing pipeline."""
    
    def __init__(self, detection_model: str = "opencv_dnn", encoding_model: str = "facenet"):
        self.detector = FaceDetector(detection_model)
        self.encoder = FaceEncoder(encoding_model)
        logger.info(f"FaceProcessor initialized with detection: {detection_model}, encoding: {encoding_model}")
    
    def process_image(self, image: np.ndarray, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Complete face processing pipeline.
        
        Args:
            image: Input image as numpy array
            min_confidence: Minimum face detection confidence
            
        Returns:
            List of processed faces with embeddings and metadata
        """
        start_time = time.time()
        
        # Detect faces
        detected_faces = self.detector.detect_faces(image, min_confidence)
        
        if not detected_faces:
            return []
        
        # Extract face locations for encoding
        face_locations = [face['bbox'] for face in detected_faces]
        
        # Generate encodings
        encodings = self.encoder.encode_faces(image, face_locations)
        
        # Combine results
        processed_faces = []
        for i, face in enumerate(detected_faces):
            if i < len(encodings):
                processed_faces.append({
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'landmarks': face['landmarks'],
                    'embedding': encodings[i],
                    'processing_time': time.time() - start_time
                })
        
        logger.info(f"Processed {len(processed_faces)} faces in {time.time() - start_time:.3f}s")
        return processed_faces
    
    def process_image_file(self, image_path: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Process faces from an image file.
        
        Args:
            image_path: Path to image file
            min_confidence: Minimum face detection confidence
            
        Returns:
            List of processed faces with embeddings and metadata
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            return self.process_image(image, min_confidence)
        except Exception as e:
            logger.error(f"Error processing image file {image_path}: {e}")
            return []


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray, method: str = "cosine") -> float:
    """
    Calculate similarity between two face embeddings.
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        method: Similarity method ("cosine" or "euclidean")
        
    Returns:
        Similarity score (higher is more similar for cosine, lower for euclidean)
    """
    if method == "cosine":
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    elif method == "euclidean":
        # Euclidean distance (lower is more similar)
        return np.linalg.norm(embedding1 - embedding2)
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


def is_same_person(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.6) -> bool:
    """
    Determine if two embeddings represent the same person.
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        threshold: Similarity threshold
        
    Returns:
        True if same person, False otherwise
    """
    similarity = calculate_similarity(embedding1, embedding2, "cosine")
    return similarity > threshold
