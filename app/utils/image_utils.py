"""
Utility functions for image processing and face operations.
"""
import cv2
import numpy as np
import os
import logging
from typing import Tuple, List, Optional, Union
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def resize_image(image: np.ndarray, max_width: int = 1024, max_height: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    # Calculate scaling factor
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height)
    
    # Only resize if image is larger than max dimensions
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image


def crop_face(image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.2) -> np.ndarray:
    """
    Crop face region from image with optional padding.
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, width, height)
        padding: Padding factor (0.2 = 20% padding)
        
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    height, width = image.shape[:2]
    
    # Calculate padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    # Apply padding with bounds checking
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(width, x + w + pad_w)
    y2 = min(height, y + h + pad_h)
    
    return image[y1:y2, x1:x2]


def draw_face_boxes(image: np.ndarray, faces: List[dict], 
                   draw_landmarks: bool = True, draw_confidence: bool = True) -> np.ndarray:
    """
    Draw bounding boxes and landmarks on faces.
    
    Args:
        image: Input image
        faces: List of face dictionaries with bbox, confidence, landmarks
        draw_landmarks: Whether to draw facial landmarks
        draw_confidence: Whether to draw confidence scores
        
    Returns:
        Image with drawn annotations
    """
    result_image = image.copy()
    
    for face in faces:
        bbox = face.get('bbox')
        confidence = face.get('confidence', 0.0)
        landmarks = face.get('landmarks')
        
        if bbox:
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            if draw_confidence:
                text = f"{confidence:.2f}"
                cv2.putText(result_image, text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw landmarks
            if draw_landmarks and landmarks:
                if isinstance(landmarks, dict):
                    # MTCNN format
                    for landmark_name, (lx, ly) in landmarks.items():
                        cv2.circle(result_image, (int(lx), int(ly)), 2, (255, 0, 0), -1)
                elif isinstance(landmarks, list):
                    # Other formats
                    for (lx, ly) in landmarks:
                        cv2.circle(result_image, (int(lx), int(ly)), 2, (255, 0, 0), -1)
    
    return result_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for better face detection.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    # Convert to float and normalize to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Apply histogram equalization to improve contrast
    if len(image.shape) == 3:
        # Convert to YUV, equalize Y channel, convert back
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        image = cv2.equalizeHist(image)
    
    return image


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert image to base64 string.
    
    Args:
        image: Input image
        
    Returns:
        Base64 encoded string
    """
    try:
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return ""


def base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Convert base64 string to image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        return image_bgr
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        return None


def validate_image_format(image_path: str) -> bool:
    """
    Validate if image format is supported.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if format is supported, False otherwise
    """
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if not os.path.exists(image_path):
        return False
    
    _, ext = os.path.splitext(image_path.lower())
    return ext in supported_formats


def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """
    Enhance image quality for better face detection.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    try:
        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Sharpen the image
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    except Exception as e:
        logger.error(f"Error enhancing image quality: {e}")
        return image


def calculate_image_quality_score(image: np.ndarray) -> float:
    """
    Calculate image quality score based on various metrics.
    
    Args:
        image: Input image
        
    Returns:
        Quality score between 0 and 1 (higher is better)
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance (focus measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (empirically determined threshold)
        focus_score = min(laplacian_var / 1000.0, 1.0)
        
        # Calculate brightness score
        mean_brightness = np.mean(gray)
        # Optimal brightness is around 100-150
        brightness_score = 1.0 - abs(mean_brightness - 125) / 125
        brightness_score = max(0, brightness_score)
        
        # Calculate contrast score
        contrast = gray.std()
        contrast_score = min(contrast / 64.0, 1.0)  # Normalize by typical contrast value
        
        # Weighted combination
        quality_score = (0.5 * focus_score + 0.3 * brightness_score + 0.2 * contrast_score)
        
        return float(np.clip(quality_score, 0.0, 1.0))
    
    except Exception as e:
        logger.error(f"Error calculating image quality score: {e}")
        return 0.5  # Return average score on error


def extract_face_patches(image: np.ndarray, faces: List[dict], 
                        patch_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """
    Extract standardized face patches from detected faces.
    
    Args:
        image: Input image
        faces: List of face dictionaries with bbox
        patch_size: Desired patch size (width, height)
        
    Returns:
        List of face patches
    """
    patches = []
    
    for face in faces:
        bbox = face.get('bbox')
        if not bbox:
            continue
        
        try:
            # Crop face with padding
            face_crop = crop_face(image, bbox, padding=0.3)
            
            # Resize to standard size
            face_patch = cv2.resize(face_crop, patch_size, interpolation=cv2.INTER_AREA)
            
            patches.append(face_patch)
        except Exception as e:
            logger.error(f"Error extracting face patch: {e}")
            continue
    
    return patches


def create_face_montage(face_patches: List[np.ndarray], 
                       grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a montage of face patches.
    
    Args:
        face_patches: List of face patch images
        grid_size: (rows, cols) for the grid. If None, automatically calculated
        
    Returns:
        Montage image
    """
    if not face_patches:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    num_faces = len(face_patches)
    
    if grid_size is None:
        # Calculate grid size automatically
        cols = int(np.ceil(np.sqrt(num_faces)))
        rows = int(np.ceil(num_faces / cols))
    else:
        rows, cols = grid_size
    
    # Get patch dimensions
    patch_height, patch_width = face_patches[0].shape[:2]
    
    # Create montage canvas
    montage_height = rows * patch_height
    montage_width = cols * patch_width
    montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)
    
    # Place face patches
    for i, patch in enumerate(face_patches):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        y1 = row * patch_height
        y2 = y1 + patch_height
        x1 = col * patch_width
        x2 = x1 + patch_width
        
        # Ensure patch has 3 channels
        if len(patch.shape) == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        
        montage[y1:y2, x1:x2] = patch
    
    return montage
