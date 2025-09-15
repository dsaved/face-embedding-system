"""
Real-time video processing service for face recognition.
Handles video streams with optimized face detection, tracking, and recognition.
"""
import cv2
import numpy as np
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import hashlib
from dataclasses import dataclass
from .face_processor import FaceProcessor
from .database_service import DatabaseService

logger = logging.getLogger(__name__)


@dataclass
class TrackedFace:
    """Represents a tracked face across video frames."""
    id: int
    tracker: Any  # OpenCV tracker object
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    embedding: Optional[np.ndarray] = None
    identification: Optional[Dict] = None
    frames_tracked: int = 0
    last_recognition: float = 0
    confidence: float = 0.0
    
    def needs_recognition(self, recognition_interval: int = 30) -> bool:
        """Check if face needs re-recognition."""
        return (self.identification is None or 
                self.frames_tracked - self.last_recognition > recognition_interval)


class RecognitionCache:
    """Cache for face recognition results to avoid repeated processing."""
    
    def __init__(self, ttl: int = 300, max_size: int = 1000):
        self.cache = {}
        self.ttl = ttl
        self.max_size = max_size
        self.access_times = deque()
        
    def _generate_key(self, embedding: np.ndarray) -> str:
        """Generate cache key from face embedding."""
        return hashlib.md5(embedding.tobytes()).hexdigest()
    
    def get(self, embedding: np.ndarray) -> Optional[Dict]:
        """Get cached identification result."""
        key = self._generate_key(embedding)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                # Update access time
                self.access_times.append((time.time(), key))
                return entry['result']
            else:
                # Expired entry
                del self.cache[key]
        return None
    
    def set(self, embedding: np.ndarray, result: Dict):
        """Cache identification result."""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
            
        key = self._generate_key(embedding)
        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
        self.access_times.append((time.time(), key))
    
    def _evict_oldest(self):
        """Remove least recently used entries."""
        current_time = time.time()
        while self.access_times and current_time - self.access_times[0][0] > self.ttl:
            _, old_key = self.access_times.popleft()
            self.cache.pop(old_key, None)


class FaceTracker:
    """Manages face tracking across video frames."""
    
    def __init__(self):
        self.tracked_faces: List[TrackedFace] = []
        self.next_id = 0
        self.max_tracked_faces = 10
        
    def update(self, frame: np.ndarray) -> List[TrackedFace]:
        """Update all tracked faces with new frame."""
        active_faces = []
        
        for tracked_face in self.tracked_faces:
            # Skip tracking update if no tracker (fallback case)
            if tracked_face.tracker is None:
                logger.debug(f"Face {tracked_face.id} has no tracker, keeping as-is")
                # Keep face but mark as needing re-detection
                tracked_face.frames_tracked += 1
                active_faces.append(tracked_face)
                continue
                
            success, bbox = tracked_face.tracker.update(frame)
            if success:
                # Update bounding box
                tracked_face.bbox = tuple(map(int, bbox))
                tracked_face.frames_tracked += 1
                active_faces.append(tracked_face)
            else:
                logger.debug(f"Lost tracking for face {tracked_face.id}")
        
        self.tracked_faces = active_faces
        return self.tracked_faces
    
    def add_new_faces(self, frame: np.ndarray, detected_faces: List[Dict]):
        """Add new detected faces to tracking."""
        for face_data in detected_faces:
            bbox = face_data['bbox']
            
            # Check if face overlaps with existing tracked faces
            if not self._overlaps_existing(bbox):
                if len(self.tracked_faces) < self.max_tracked_faces:
                    self._create_tracker(frame, face_data)
    
    def _overlaps_existing(self, bbox: Tuple[int, int, int, int], 
                          threshold: float = 0.5) -> bool:
        """Check if bounding box overlaps with existing tracked faces."""
        x1, y1, w1, h1 = bbox
        
        for tracked_face in self.tracked_faces:
            x2, y2, w2, h2 = tracked_face.bbox
            
            # Calculate intersection over union (IoU)
            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            
            if xi2 <= xi1 or yi2 <= yi1:
                continue  # No intersection
                
            intersection = (xi2 - xi1) * (yi2 - yi1)
            union = w1 * h1 + w2 * h2 - intersection
            
            if intersection / union > threshold:
                return True
        
        return False
    
    def _create_tracker(self, frame: np.ndarray, face_data: Dict):
        """Create new tracker for detected face."""
        bbox = face_data['bbox']
        logger.debug(f"_create_tracker called with bbox: {bbox}, frame shape: {frame.shape}")
        
        try:
            # Validate bbox format and bounds
            x, y, w, h = bbox
            frame_h, frame_w = frame.shape[:2]
            
            # Check if bbox is valid
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                logger.debug(f"Invalid bbox dimensions: {bbox}")
                return
                
            # Ensure bbox is within frame bounds
            if x + w > frame_w or y + h > frame_h:
                logger.debug(f"Bbox extends beyond frame bounds: {bbox}, frame: {frame_w}x{frame_h}")
                # Clip bbox to frame bounds
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
                if w <= 0 or h <= 0:
                    logger.debug(f"Bbox clipped to invalid size, skipping tracker creation")
                    return
                bbox = (x, y, w, h)
                logger.debug(f"Clipped bbox to: {bbox}")
            
            # Create tracker (using CSRT for better accuracy)
            tracker = cv2.TrackerCSRT_create()
            logger.debug(f"Created CSRT tracker")
            
            # Ensure frame is in correct format (uint8) and clamp values
            if frame.dtype != np.uint8:
                # Clamp values to valid uint8 range before conversion
                frame_clamped = np.clip(frame, 0, 255)
                frame = frame_clamped.astype(np.uint8)
                logger.debug(f"Converted frame to uint8 with clamping")
            
            success = tracker.init(frame, bbox)
            logger.debug(f"Tracker init result: {success}")
            
            # Handle different OpenCV version return values
            if success is None:
                logger.debug(f"Tracker init returned None (possible OpenCV version issue)")
                # Try to create tracked face anyway since detection was successful
                success = True
            
            if success:
                tracked_face = TrackedFace(
                    id=self.next_id,
                    tracker=tracker,
                    bbox=bbox,
                    embedding=face_data.get('embedding'),
                    confidence=face_data.get('confidence', 0.0)
                )
                
                self.tracked_faces.append(tracked_face)
                self.next_id += 1
                logger.debug(f"Started tracking face {tracked_face.id}, total tracked faces: {len(self.tracked_faces)}")
            else:
                logger.debug(f"Failed to initialize tracker for bbox: {bbox}")
                
                # Fallback: Create tracked face without OpenCV tracker for this frame
                tracked_face = TrackedFace(
                    id=self.next_id,
                    tracker=None,  # No tracker, will rely on detection for next frames
                    bbox=bbox,
                    embedding=face_data.get('embedding'),
                    confidence=face_data.get('confidence', 0.0)
                )
                
                self.tracked_faces.append(tracked_face)
                self.next_id += 1
                logger.debug(f"Created face {tracked_face.id} without tracker as fallback, total tracked faces: {len(self.tracked_faces)}")
                
        except Exception as e:
            logger.error(f"Exception during tracker creation: {e}")
            # Emergency fallback: Create tracked face without tracker
            tracked_face = TrackedFace(
                id=self.next_id,
                tracker=None,
                bbox=bbox,
                embedding=face_data.get('embedding'),
                confidence=face_data.get('confidence', 0.0)
            )
            
            self.tracked_faces.append(tracked_face)
            self.next_id += 1
            logger.debug(f"Created face {tracked_face.id} as emergency fallback, total tracked faces: {len(self.tracked_faces)}")
    
    def get_face_by_id(self, face_id: int) -> Optional[TrackedFace]:
        """Get tracked face by ID."""
        for face in self.tracked_faces:
            if face.id == face_id:
                return face
        return None
    
    def cleanup_lost_faces(self, max_frames_without_update: int = 30):
        """Remove faces that haven't been updated recently."""
        active_faces = [
            face for face in self.tracked_faces 
            if face.frames_tracked > 0  # This gets reset if tracking fails
        ]
        
        removed_count = len(self.tracked_faces) - len(active_faces)
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} lost faces")
            
        self.tracked_faces = active_faces


class VideoStreamProcessor:
    """Main video stream processing class."""
    
    def __init__(self, 
                 face_processor: FaceProcessor = None,
                 db_service: DatabaseService = None):
        self.face_processor = face_processor or FaceProcessor()
        self.db_service = db_service or DatabaseService()
        
        # Optimized performance settings for real-time processing
        self.frame_skip = 1  # Process every frame for smooth real-time
        self.recognition_interval = 15  # Recognize every 15 frames (1.5 seconds at 10 FPS)
        self.detection_interval = 3   # Detect new faces every 3 frames
        self.optimal_frame_rate = 10  # Target 10 FPS for optimal performance
        
        # Use hybrid detection for best results
        self.detection_model = "hybrid"
        self.confidence_threshold = 0.6
        
        # Components
        self.face_tracker = FaceTracker()
        self.recognition_cache = RecognitionCache()
        
        # State
        self.frame_count = 0
        self.processing_stats = defaultdict(float)
        self.is_processing = False
        self._lock = threading.Lock()
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single video frame.
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            Dictionary with processing results
        """
        with self._lock:
            if self.is_processing:
                return {'error': 'Frame processing in progress'}
            self.is_processing = True
        
        try:
            start_time = time.time()
            self.frame_count += 1
            
            # Skip frames for performance
            if self.frame_count % self.frame_skip != 0:
                return self._quick_tracking_update(frame)
            
            results = self._full_frame_process(frame)
            
            # Update stats
            processing_time = time.time() - start_time
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] * 0.9) + 
                (processing_time * 0.1)
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {'error': str(e)}
        finally:
            self.is_processing = False
    
    def _quick_tracking_update(self, frame: np.ndarray) -> Dict[str, Any]:
        """Quick update using only face tracking."""
        tracked_faces = self.face_tracker.update(frame)
        
        return {
            'timestamp': time.time(),
            'frame_number': self.frame_count,
            'faces_tracked': len(tracked_faces),
            'faces': [
                {
                    'id': face.id,
                    'bbox': face.bbox,
                    'identification': face.identification,
                    'confidence': face.confidence,
                    'frames_tracked': face.frames_tracked
                }
                for face in tracked_faces
            ],
            'processing_type': 'tracking_only'
        }
    
    def _full_frame_process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Full frame processing with detection and recognition."""
        # Update existing tracked faces
        tracked_faces = self.face_tracker.update(frame)
        
        # Detect new faces periodically (or always if no faces are tracked)
        new_faces_detected = 0
        should_detect = (self.frame_count % self.detection_interval == 0 or 
                        len(tracked_faces) == 0 or 
                        self.frame_count <= 5)  # Always detect in first few frames
        
        if should_detect:
            logger.debug(f"Detecting faces on frame {self.frame_count} (tracked: {len(tracked_faces)})")
            detected_faces = self._detect_new_faces(frame)
            self.face_tracker.add_new_faces(frame, detected_faces)
            new_faces_detected = len(detected_faces)
            logger.debug(f"Detected {new_faces_detected} new faces")
            
            # Update tracked_faces to include newly added faces
            tracked_faces = self.face_tracker.tracked_faces
            logger.debug(f"Updated tracked faces count: {len(tracked_faces)}")
        
        # Perform recognition on faces that need it
        recognition_updates = 0
        for face in tracked_faces:
            if face.needs_recognition(self.recognition_interval):
                self._recognize_face(frame, face)
                recognition_updates += 1
        
        # Cleanup lost faces
        self.face_tracker.cleanup_lost_faces()
        
        return {
            'timestamp': time.time(),
            'frame_number': self.frame_count,
            'faces_tracked': len(tracked_faces),
            'new_faces_detected': new_faces_detected,
            'recognition_updates': recognition_updates,
            'faces': [
                {
                    'id': face.id,
                    'bbox': face.bbox,
                    'identification': face.identification,
                    'confidence': face.confidence,
                    'frames_tracked': face.frames_tracked,
                    'last_recognition_frame': face.last_recognition
                }
                for face in tracked_faces
            ],
            'processing_type': 'full_processing',
            'performance': {
                'avg_processing_time': self.processing_stats['avg_processing_time'],
                'cache_hit_rate': self._calculate_cache_hit_rate()
            }
        }
    
    def _detect_new_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect new faces in frame."""
        try:
            logger.debug(f"Starting face detection on frame {frame.shape}")
            
            # Use existing face detection with more permissive settings for video
            detected_faces = self.face_processor.detector.detect_faces(
                frame, min_confidence=0.5  # Lower threshold for video streams
            )
            
            logger.debug(f"Face detector returned {len(detected_faces)} faces")
            
            # Add embeddings for detected faces
            for i, face_data in enumerate(detected_faces):
                bbox = face_data['bbox']
                x, y, w, h = bbox
                
                logger.debug(f"Processing face {i+1}: bbox={bbox}, confidence={face_data.get('confidence', 'N/A')}")
                
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                if face_region.size > 0:
                    # Generate embedding using the encoder
                    try:
                        logger.debug(f"About to generate embedding for face {i+1} using encoder model: {self.face_processor.encoder.model_type}")
                        # Use the encoder to generate embedding for this face region
                        encodings = self.face_processor.encoder.encode_faces(frame, [(x, y, w, h)])
                        logger.debug(f"Encoder returned {len(encodings) if encodings else 0} embeddings for face {i+1}")
                        if encodings:
                            face_data['embedding'] = encodings[0]
                            logger.debug(f"Generated embedding for face {i+1} - shape: {encodings[0].shape}")
                            logger.debug(f"Embedding sample: {encodings[0][:5]}")
                            logger.debug(f"Embedding stats: min={encodings[0].min():.6f}, max={encodings[0].max():.6f}, mean={encodings[0].mean():.6f}, norm={np.linalg.norm(encodings[0]):.6f}")
                        else:
                            face_data['embedding'] = None
                            logger.debug(f"No embedding generated for face {i+1}")
                    except Exception as e:
                        logger.debug(f"Could not generate embedding for face {i+1}: {e}")
                        face_data['embedding'] = None
            
            logger.debug(f"Returning {len(detected_faces)} processed faces")
            return detected_faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _recognize_face(self, frame: np.ndarray, tracked_face: TrackedFace):
        """Perform face recognition on tracked face."""
        try:
            if tracked_face.embedding is None:
                # Extract embedding from current frame
                x, y, w, h = tracked_face.bbox
                face_region = frame[y:y+h, x:x+w]
                if face_region.size > 0:
                    # Generate embedding using the encoder
                    try:
                        encodings = self.face_processor.encoder.encode_faces(frame, [(x, y, w, h)])
                        if encodings:
                            tracked_face.embedding = encodings[0]
                        else:
                            tracked_face.embedding = None
                    except Exception as e:
                        logger.debug(f"Could not generate embedding for tracked face: {e}")
                        tracked_face.embedding = None
            
            if tracked_face.embedding is not None:
                # Check cache first
                cached_result = self.recognition_cache.get(tracked_face.embedding)
                if cached_result:
                    tracked_face.identification = cached_result
                else:
                    # Perform recognition
                    logger.debug(f"Searching for similar faces with embedding shape: {tracked_face.embedding.shape}")
                    logger.debug(f"Query embedding sample: {tracked_face.embedding[:5]}")
                    logger.debug(f"Query embedding stats: min={tracked_face.embedding.min():.6f}, max={tracked_face.embedding.max():.6f}, mean={tracked_face.embedding.mean():.6f}, norm={np.linalg.norm(tracked_face.embedding):.6f}")
                    similar_faces = self.db_service.find_similar_faces(
                        tracked_face.embedding, top_k=1, similarity_threshold=0.6
                    )
                    logger.debug(f"Found {len(similar_faces) if similar_faces else 0} similar faces")
                    
                    if similar_faces:
                        face_record, similarity_score = similar_faces[0]
                        logger.debug(f"Best match: person_id={face_record.person_id}, name={face_record.person_name}, similarity={similarity_score}")
                        identification = {
                            'person_id': face_record.person_id,
                            'person_name': face_record.person_name,
                            'similarity_score': float(similarity_score),
                            'face_record_id': face_record.id
                        }
                    else:
                        logger.debug("No similar faces found above threshold 0.6")
                        identification = {
                            'person_id': 'unknown',
                            'person_name': 'Unknown Person',
                            'similarity_score': 0.0,
                            'face_record_id': None
                        }
                    
                    tracked_face.identification = identification
                    self.recognition_cache.set(tracked_face.embedding, identification)
                
                tracked_face.last_recognition = self.frame_count
                
        except Exception as e:
            logger.error(f"Error recognizing face {tracked_face.id}: {e}")
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate recognition cache hit rate."""
        # Simplified calculation - in practice you'd track hits/misses
        return 0.0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            'frame_count': self.frame_count,
            'tracked_faces': len(self.face_tracker.tracked_faces),
            'avg_processing_time': self.processing_stats['avg_processing_time'],
            'cache_size': len(self.recognition_cache.cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def reset_processing(self):
        """Reset processor state."""
        with self._lock:
            self.frame_count = 0
            self.face_tracker = FaceTracker()
            self.recognition_cache = RecognitionCache()
            self.processing_stats.clear()
            logger.info("Video processor reset")


# Utility functions for video processing
def validate_frame(frame: np.ndarray) -> bool:
    """Validate video frame."""
    if frame is None or frame.size == 0:
        return False
    
    # Check dimensions
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return False
    
    # Check reasonable size
    height, width = frame.shape[:2]
    if width < 100 or height < 100 or width > 4096 or height > 4096:
        return False
    
    return True


def resize_frame_for_processing(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Resize frame for optimal processing speed."""
    height, width = frame.shape[:2]
    
    if width > max_width:
        # Calculate new height maintaining aspect ratio
        new_height = int(height * (max_width / width))
        frame = cv2.resize(frame, (max_width, new_height))
    
    return frame


def extract_face_region(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                       padding: float = 0.1) -> np.ndarray:
    """Extract face region from frame with optional padding."""
    x, y, w, h = bbox
    
    # Add padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    
    # Calculate padded coordinates
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    
    return frame[y1:y2, x1:x2]
