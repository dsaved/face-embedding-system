"""
Face recognition API endpoints with comprehensive security.
"""
import os
import cv2
import numpy as np
import uuid
import logging
import base64
import tempfile
from flask import Blueprint, request, jsonify, current_app, g
from werkzeug.utils import secure_filename
from typing import Dict, Any, Optional
import time
from app.services.face_processor import FaceProcessor, calculate_similarity
from app.services.database_service import DatabaseService
from app.config_file import Config
from app.security import (
    require_api_key, rate_limit, validate_input, audit_request,
    add_security_headers, log_security_event
)

logger = logging.getLogger(__name__)

# Error message constants
ERROR_NO_IMAGE = 'No image provided or could not load image'
ERROR_NO_FACES = 'No faces detected in image'
ERROR_MULTIPLE_FACES = 'Multiple faces detected. Please provide image with single face'
ERROR_MISSING_PARAMS = 'app_id, person_id and person_name are required'
ERROR_MISSING_PERSON_ID = 'app_id and person_id are required'
ERROR_MISSING_APP_ID = 'app_id is required'
ERROR_INVALID_FACE_OWNERSHIP = 'Face does not belong to the specified person'

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    return obj

# Create blueprint
face_api = Blueprint('face_api', __name__, url_prefix='/api/v1/faces')

# Optimized singleton pattern for services with caching
class OptimizedFaceProcessor:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OptimizedFaceProcessor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not OptimizedFaceProcessor._initialized:
            # Initialize with fastest model configuration
            self.processor = FaceProcessor(
                detection_model=Config.FACE_DETECTION_MODEL,
                encoding_model=Config.FACE_ENCODING_MODEL
            )
            OptimizedFaceProcessor._initialized = True
            logger.info("OptimizedFaceProcessor singleton initialized")
    
    @property 
    def detector(self):
        return self.processor.detector
    
    @property
    def encoder(self):
        return self.processor.encoder

# Global cached instance
_face_processor = None
_db_service = None

def get_face_processor():
    global _face_processor
    if _face_processor is None:
        _face_processor = OptimizedFaceProcessor()
    return _face_processor

def get_db_service():
    global _db_service  
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def save_uploaded_file(file) -> str:
    """Save uploaded file and return the path."""
    if not file or not allowed_file(file.filename):
        raise ValueError("Invalid file type")
    
    # Create upload directory
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Generate unique filename
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
    
    file.save(file_path)
    return file_path


def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """Decode base64 string to OpenCV image array."""
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64 to bytes
        image_data = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None



def cleanup_temp_file(temp_file_path: str, should_cleanup: bool):
    """Clean up temporary file safely."""
    if should_cleanup and temp_file_path and os.path.exists(temp_file_path):
        try:
            os.remove(temp_file_path)
            logger.debug(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup temporary file {temp_file_path}: {e}")


def get_image_from_request(request) -> tuple[Optional[np.ndarray], Optional[str], bool]:
    """
    Get image from request (file upload, base64, or file path).
    
    Returns:
        tuple: (image_array, temp_file_path_if_any, should_cleanup)
    """
    image = None
    temp_file_path = None
    should_cleanup = False
    
    # Handle file upload
    if 'image' in request.files:
        temp_file_path = save_uploaded_file(request.files['image'])
        image = cv2.imread(temp_file_path)
        should_cleanup = True
    # Handle JSON requests
    elif request.is_json:
        # Base64 image
        if 'image_base64' in request.json:
            image = decode_base64_image(request.json['image_base64'])
        # File path (existing behavior)
        elif 'image_path' in request.json:
            image = cv2.imread(request.json['image_path'])
    
    return image, temp_file_path, should_cleanup


@face_api.route('/detect', methods=['POST'])
@require_api_key
@rate_limit(limit=50, window=60)  # 50 requests per minute
def detect_faces():
    """
    Optimized face detection endpoint - removed database logging and unnecessary middleware for maximum performance.
    
    Expected input:
    - image file (multipart/form-data) or
    - image_base64 in JSON or
    - image_path in JSON
    
    Returns:
    - List of detected faces with bounding boxes and confidence scores
    """
    try:
        start_time = time.time()
        
        # Optimized image retrieval - skip temp files for base64 images
        image = None
        temp_file_path = None
        should_cleanup = False
        
        # Handle file upload (fastest path - direct memory)
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                # Read directly into memory without temp file
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Handle JSON requests
        elif request.is_json:
            # Base64 image (optimized)
            if 'image_base64' in request.json:
                image = decode_base64_image(request.json['image_base64'])
            # File path (fallback - still requires disk I/O)
            elif 'image_path' in request.json:
                image = cv2.imread(request.json['image_path'])
        
        if image is None:
            return jsonify({'error': ERROR_NO_IMAGE}), 400
        
        # Get optional parameters with faster default lookup
        min_confidence = Config.FACE_CONFIDENCE_THRESHOLD
        if request.is_json and 'min_confidence' in request.json:
            min_confidence = request.json['min_confidence']
        
        # Detect faces using cached processor
        detected_faces = get_face_processor().detector.detect_faces(image, min_confidence)
        
        processing_time = time.time() - start_time
        
        # Return optimized response (removed database logging for performance)
        return jsonify(convert_numpy_types({
            'success': True,
            'faces_detected': len(detected_faces),
            'faces': [
                {
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'landmarks': face['landmarks']
                }
                for face in detected_faces
            ],
            'processing_time': processing_time
        }))
    
    except Exception as e:
        logger.error(f"Face detection error: {e}")        
        return jsonify({'error': str(e)}), 500


@face_api.route('/encode', methods=['POST'])
@require_api_key
@rate_limit(limit=30, window=60)  # 30 requests per minute
@validate_input('image')
@audit_request
def encode_faces():
    """
    Generate face embeddings for detected faces.
    
    Expected input:
    - image file (multipart/form-data) or
    - image_base64 in JSON or
    - image_path in JSON
    
    Returns:
    - Face embeddings and metadata
    """
    try:
        start_time = time.time()
        
        # Get image from request
        image, temp_file_path, should_cleanup = get_image_from_request(request)
        
        if image is None:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': ERROR_NO_IMAGE}), 400
        
        # Process faces
        processed_faces = get_face_processor().process_image(image, Config.FACE_CONFIDENCE_THRESHOLD)
        
        processing_time = time.time() - start_time
        
        # Log operation
        input_source = temp_file_path if temp_file_path else "base64_image"
        get_db_service().log_operation(
            operation_type='encode',
            input_source=input_source,
            processing_time=processing_time,
            faces_detected=len(processed_faces),
            success=True
        )
        
        # Clean up temporary file if needed
        cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify(convert_numpy_types({
            'success': True,
            'faces_processed': len(processed_faces),
            'faces': [
                {
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'landmarks': face['landmarks'],
                    'embedding': face['embedding'],
                    'embedding_size': len(face['embedding'])
                }
                for face in processed_faces
            ],
            'processing_time': processing_time
        }))
    
    except Exception as e:
        logger.error(f"Face encoding error: {e}")
        # Clean up temporary file if needed (even on error)
        if 'temp_file_path' in locals() and 'should_cleanup' in locals():
            cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify({'error': str(e)}), 500


@face_api.route('/register', methods=['POST'])
@require_api_key
@rate_limit(limit=20, window=60)  # 20 registrations per minute
@validate_input('image')
@audit_request
def register_face():
    """
    Register a new face identity in the database with enhanced validation.
    
    Expected input:
    - image file (multipart/form-data) or image_base64/image_path in JSON
    - app_id: application identifier (required)
    - person_id: unique identifier within the app (required)
    - person_name: name of the person (required)
    - face_alias: optional alias for this face (e.g., "profile_pic", "id_photo")
    - is_primary: whether this should be the primary face (default: false)
    - validate_ownership: whether to validate against existing faces (default: true)
    - similarity_threshold: minimum similarity for validation (default: 0.5, range: 0.0-1.0)
    
    Enhanced validation features:
    - Multiple validation strategies (best match, average, statistical)
    - Accepts same person with different angles, lighting, expressions
    - Customizable similarity threshold for flexible validation
    - Detailed error messages with suggestions
    
    Returns:
    - Registration result with face record ID
    """
    try:
        start_time = time.time()
        
        # Get required parameters
        data = request.form if request.files else request.json
        app_id = data.get('app_id')
        person_id = data.get('person_id')
        person_name = data.get('person_name')
        face_alias = data.get('face_alias')
        is_primary_value = data.get('is_primary', 'false')
        is_primary = str(is_primary_value).lower() == 'true' if isinstance(is_primary_value, str) else bool(is_primary_value)
        validate_ownership_value = data.get('validate_ownership', 'true')
        validate_ownership = str(validate_ownership_value).lower() == 'true' if isinstance(validate_ownership_value, str) else bool(validate_ownership_value)
        
        # Enhanced: Allow custom similarity threshold for face validation
        similarity_threshold = float(data.get('similarity_threshold', 0.5))  # Lowered default from 0.7 to 0.5
        
        if not app_id or not person_id or not person_name:
            return jsonify({'error': ERROR_MISSING_PARAMS}), 400
        
        # Get image from request
        image, temp_file_path, should_cleanup = get_image_from_request(request)
        
        if image is None:
            # Clean up temporary file before returning error  
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': ERROR_NO_IMAGE}), 400
        
        processed_faces = get_face_processor().process_image(image, Config.FACE_CONFIDENCE_THRESHOLD)
        
        if not processed_faces:
            # Clean up temporary file before returning error
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': ERROR_NO_FACES}), 400
        
        if len(processed_faces) > 1:
            # Clean up temporary file before returning error
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': ERROR_MULTIPLE_FACES}), 400
        
        # Get the face data
        face_data = processed_faces[0]
        
        # Validate face ownership if requested and person has existing faces
        if validate_ownership:
            is_valid = get_db_service().validate_face_ownership(
                app_id=app_id,
                person_id=person_id,
                new_embedding=face_data['embedding'],
                similarity_threshold=similarity_threshold  # Use custom threshold
            )
            if not is_valid:
                # Check if person has existing faces
                existing_faces = get_db_service().get_person_faces(app_id, person_id)
                if existing_faces:  # Only reject if person already has faces
                    # Clean up temporary file before returning error
                    cleanup_temp_file(temp_file_path, should_cleanup)
                    return jsonify({
                        'error': ERROR_INVALID_FACE_OWNERSHIP,
                        'suggestion': 'Try lowering similarity_threshold or set validate_ownership=false',
                        'similarity_threshold_used': similarity_threshold
                    }), 400
        
        # Save to database
        input_source = temp_file_path if temp_file_path else "base64_image"
        face_record = get_db_service().add_face_record(
            app_id=app_id,
            person_id=person_id,
            person_name=person_name,
            face_alias=face_alias,
            embedding=face_data['embedding'],
            confidence_score=float(face_data['confidence']),
            image_path=input_source,
            bbox=convert_numpy_types(face_data['bbox']),
            landmarks=convert_numpy_types(face_data['landmarks']),
            encoding_model=Config.FACE_ENCODING_MODEL,
            is_primary=is_primary
        )
        
        processing_time = time.time() - start_time
        
        # Log operation
        get_db_service().log_operation(
            operation_type='register',
            input_source=input_source,
            processing_time=processing_time,
            faces_detected=1,
            success=True,
            metadata={'app_id': app_id, 'person_id': person_id, 'person_name': person_name, 'face_alias': face_alias}
        )
        
        # Clean up temporary file if needed
        cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify(convert_numpy_types({
            'success': True,
            'message': f'Face registered successfully for {person_name}',
            'face_record_id': face_record.id,
            'app_id': app_id,
            'person_id': person_id,
            'person_name': person_name,
            'face_alias': face_alias,
            'is_primary': is_primary,
            'confidence_score': face_data['confidence'],
            'processing_time': processing_time
        }))
    
    except Exception as e:
        logger.error(f"Face registration error: {e}")
        # Clean up temporary file if needed (even on error)
        if 'temp_file_path' in locals() and 'should_cleanup' in locals():
            cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify({'error': str(e)}), 500


@face_api.route('/identify', methods=['POST'])
@require_api_key
@rate_limit(limit=40, window=60)  # 40 identifications per minute
@validate_input('image')
@audit_request
def identify_face():
    """
    Identify a face against the database.
    
    Expected input:
    - image file (multipart/form-data) or image_base64/image_path in JSON
    - app_id: application identifier (optional, if not provided searches all apps)
    - Optional: similarity_threshold
    
    Returns:
    - Identified person with similarity score
    """
    try:
        start_time = time.time()
        
        # Get parameters
        data = request.form if request.files else request.json or {}
        app_id = data.get('app_id')  # Optional for backward compatibility
        
        # Get image from request
        image, temp_file_path, should_cleanup = get_image_from_request(request)
        
        if image is None:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': 'No image provided or could not load image'}), 400
        
        # Get optional parameters
        similarity_threshold = data.get('similarity_threshold')
        
        processed_faces = get_face_processor().process_image(image, Config.FACE_CONFIDENCE_THRESHOLD)
        
        if not processed_faces:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': 'No faces detected in image'}), 400
        
        # Identify each face
        results = []
        for face_data in processed_faces:
            identification_result = get_db_service().identify_person(
                face_data['embedding'], 
                similarity_threshold
            )
            
            if identification_result:
                face_record, similarity_score = identification_result
                results.append({
                    'bbox': face_data['bbox'] if face_data['bbox'] is not None else None,
                    'identified': True,
                    'person_id': face_record.person_id,
                    'person_name': face_record.person_name,
                    'similarity_score': similarity_score,
                    'confidence_score': face_data['confidence']
                })
            else:
                results.append({
                    'bbox': face_data['bbox'] if face_data['bbox'] is not None else None,
                    'identified': False,
                    'similarity_score': 0.0,
                    'confidence_score': face_data['confidence']
                })
        
        processing_time = time.time() - start_time
        
        # Log operation
        input_source = temp_file_path if temp_file_path else "base64_image"
        get_db_service().log_operation(
            operation_type='identify',
            input_source=input_source,
            processing_time=processing_time,
            faces_detected=len(processed_faces),
            success=True
        )
        
        # Clean up temporary file if needed
        cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify(convert_numpy_types({
            'success': True,
            'faces_processed': len(processed_faces),
            'results': results,
            'processing_time': processing_time
        }))
    
    except Exception as e:
        logger.error(f"Face identification error: {e}")
        # Clean up temporary file if needed (even on error)
        if 'temp_file_path' in locals() and 'should_cleanup' in locals():
            cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify({'error': str(e)}), 500


@face_api.route('/verify', methods=['POST'])
@require_api_key
@rate_limit(limit=60, window=60)  # 60 verifications per minute
@validate_input('image')
@audit_request
def verify_face():
    """
    Verify if a face belongs to a specific person (1:1 verification).
    
    Expected input:
    - image file (multipart/form-data) or image_base64/image_path in JSON
    - person_id: person to verify against
    - Optional: similarity_threshold
    
    Returns:
    - Verification result with similarity score
    """
    try:
        start_time = time.time()
        
        # Get required parameters
        data = request.form if request.files else request.json
        person_id = data.get('person_id')
        
        if not person_id:
            return jsonify({'error': ERROR_MISSING_PERSON_ID}), 400
        
        # Get image from request
        image, temp_file_path, should_cleanup = get_image_from_request(request)
        
        if image is None:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': 'No image provided or could not load image'}), 400
        
        # Get optional parameters
        similarity_threshold = data.get('similarity_threshold') if data.get('similarity_threshold') else None
        
        processed_faces = get_face_processor().process_image(image, Config.FACE_CONFIDENCE_THRESHOLD)
        
        if not processed_faces:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': 'No faces detected in image'}), 400
        
        if len(processed_faces) > 1:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': 'Multiple faces detected. Please provide image with single face'}), 400
        
        # Verify face
        face_data = processed_faces[0]
        is_verified, similarity_score = get_db_service().verify_person(
            person_id, 
            face_data['embedding'], 
            similarity_threshold
        )
        
        processing_time = time.time() - start_time
        
        # Log operation
        input_source = temp_file_path if temp_file_path else "base64_image"
        get_db_service().log_operation(
            operation_type='verify',
            input_source=input_source,
            processing_time=processing_time,
            faces_detected=1,
            success=True,
            metadata={'person_id': person_id, 'verified': is_verified}
        )
        
        # Clean up temporary file if needed
        cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify(convert_numpy_types({
            'success': True,
            'verified': is_verified,
            'person_id': person_id,
            'similarity_score': similarity_score,
            'confidence_score': face_data['confidence'],
            'processing_time': processing_time
        }))
    
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        # Clean up temporary file if needed (even on error)
        if 'temp_file_path' in locals() and 'should_cleanup' in locals():
            cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify({'error': str(e)}), 500


@face_api.route('/search', methods=['POST'])
@require_api_key
@rate_limit(limit=30, window=60)  # 30 searches per minute
@validate_input('image')
@audit_request
def search_similar_faces():
    """
    Search for similar faces in the database.
    
    Expected input:
    - image file (multipart/form-data) or image_base64/image_path in JSON
    - Optional: top_k (number of results)
    - Optional: similarity_threshold
    
    Returns:
    - List of similar faces with similarity scores
    """
    try:
        start_time = time.time()
        
        # Get image from request
        image, temp_file_path, should_cleanup = get_image_from_request(request)
        
        if image is None:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': 'No image provided or could not load image'}), 400
        
        # Get optional parameters
        top_k = 5
        similarity_threshold = None
        if request.is_json:
            top_k = request.json.get('top_k', 5)
            similarity_threshold = request.json.get('similarity_threshold')
        
        processed_faces = get_face_processor().process_image(image, Config.FACE_CONFIDENCE_THRESHOLD)
        
        if not processed_faces:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': 'No faces detected in image'}), 400
        
        # Search for similar faces (use first detected face)
        face_data = processed_faces[0]
        similar_faces = get_db_service().find_similar_faces(
            face_data['embedding'], 
            top_k=top_k, 
            similarity_threshold=similarity_threshold
        )
        
        processing_time = time.time() - start_time
        
        # Format results
        results = []
        for face_record, similarity_score in similar_faces:
            results.append({
                'face_record_id': face_record.id,
                'person_id': face_record.person_id,
                'person_name': face_record.person_name,
                'similarity_score': similarity_score,
                'confidence_score': face_record.confidence_score,
                'created_at': face_record.created_at.isoformat() if face_record.created_at else None
            })
        
        # Log operation
        input_source = temp_file_path if temp_file_path else "base64_image"
        get_db_service().log_operation(
            operation_type='search',
            input_source=input_source,
            processing_time=processing_time,
            faces_detected=1,
            success=True,
            metadata={'results_found': len(results)}
        )
        
        # Clean up temporary file if needed
        cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify(convert_numpy_types({
            'success': True,
            'query_face': {
                'bbox': face_data['bbox'],
                'confidence_score': face_data['confidence']
            },
            'similar_faces_found': len(results),
            'results': results,
            'processing_time': processing_time
        }))
    
    except Exception as e:
        logger.error(f"Face search error: {e}")
        # Clean up temporary file if needed (even on error)
        if 'temp_file_path' in locals() and 'should_cleanup' in locals():
            cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify({'error': str(e)}), 500


@face_api.route('/database/stats', methods=['GET'])
def get_database_stats():
    """
    Get database statistics.
    
    Returns:
    - Database statistics including record counts and recent activity
    """
    try:
        stats = get_db_service().get_database_stats()
        return jsonify(convert_numpy_types({
            'success': True,
            'stats': stats
        }))
    
    except Exception as e:
        logger.error(f"Database stats error: {e}")
        return jsonify({'error': str(e)}), 500


@face_api.route('/debug/person/<person_id>', methods=['GET'])
def debug_person_records(person_id: str):
    """
    Debug endpoint to inspect person records.
    
    Args:
        person_id: Person ID to inspect
    
    Returns:
        Person's face records with embedding info
    """
    try:
        from app.models.face_record import FaceRecord
        
        db_service = get_db_service()
        session = db_service.Session()
        
        records = session.query(FaceRecord).filter(
            FaceRecord.person_id == person_id
        ).all()
        
        debug_info = []
        for record in records:
            embedding_info = {
                'type': str(type(record.embedding)),
                'length': len(record.embedding) if hasattr(record.embedding, '__len__') else 'unknown',
                'sample': convert_numpy_types(record.embedding[:5]) if hasattr(record.embedding, '__getitem__') else 'not_indexable',
                'is_active': record.is_active
            }
            
            debug_info.append({
                'record_id': record.id,
                'person_id': record.person_id,
                'person_name': record.person_name,
                'confidence_score': record.confidence_score,
                'encoding_model': record.encoding_model,
                'created_at': record.created_at.isoformat() if record.created_at else None,
                'embedding_info': embedding_info
            })
        
        session.close()
        
        return jsonify(convert_numpy_types({
            'success': True,
            'person_id': person_id,
            'records_found': len(debug_info),
            'records': debug_info
        }))
    
    except Exception as e:
        logger.error(f"Debug person records error: {e}")
        return jsonify({'error': str(e)}), 500


@face_api.route('/<int:face_id>', methods=['DELETE'])
def delete_face_record(face_id: int):
    """
    Delete a face record from the database.
    
    Args:
        face_id: ID of the face record to delete
    
    Returns:
        Deletion result
    """
    try:
        success = get_db_service().delete_face_record(face_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Face record {face_id} deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Face record {face_id} not found'
            }), 404
    
    except Exception as e:
        logger.error(f"Face deletion error: {e}")
        return jsonify({'error': str(e)}), 500


@face_api.route('/person/faces', methods=['GET'])
@require_api_key
def get_person_faces():
    """
    Get all faces for a specific person in an app.
    
    Expected parameters:
    - app_id: application identifier (required)
    - person_id: person identifier (required)
    
    Returns:
    - List of face records for the person
    """
    try:
        app_id = request.args.get('app_id')
        person_id = request.args.get('person_id')
        
        if not app_id or not person_id:
            return jsonify({'error': ERROR_MISSING_PERSON_ID}), 400
        
        faces = get_db_service().get_person_faces(app_id, person_id)
        
        return jsonify({
            'success': True,
            'app_id': app_id,
            'person_id': person_id,
            'face_count': len(faces),
            'faces': [face.to_dict() for face in faces]
        })
    
    except Exception as e:
        logger.error(f"Get person faces error: {e}")
        return jsonify({'error': str(e)}), 500


@face_api.route('/app/persons', methods=['GET'])
@require_api_key  
def get_app_persons():
    """
    Get all persons in an app with their face counts.
    
    Expected parameters:
    - app_id: application identifier (required)
    
    Returns:
    - List of persons with face counts
    """
    try:
        app_id = request.args.get('app_id')
        
        if not app_id:
            return jsonify({'error': ERROR_MISSING_APP_ID}), 400
        
        persons = get_db_service().get_app_persons(app_id)
        
        return jsonify({
            'success': True,
            'app_id': app_id,
            'person_count': len(persons),
            'persons': persons
        })
    
    except Exception as e:
        logger.error(f"Get app persons error: {e}")
        return jsonify({'error': str(e)}), 500


@face_api.route('/person/primary', methods=['PUT'])
@require_api_key
def set_primary_face():
    """
    Set a face as the primary face for a person.
    
    Expected input (JSON):
    - app_id: application identifier (required)
    - person_id: person identifier (required)
    - face_record_id: face record ID to set as primary (required)
    
    Returns:
    - Success status
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON data required'}), 400
        
        app_id = data.get('app_id')
        person_id = data.get('person_id')
        face_record_id = data.get('face_record_id')
        
        if not app_id or not person_id or not face_record_id:
            return jsonify({'error': 'app_id, person_id, and face_record_id are required'}), 400
        
        success = get_db_service().set_primary_face(app_id, person_id, face_record_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Face {face_record_id} set as primary for person {person_id}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to set primary face. Check if face exists and belongs to the person.'
            }), 400
    
    except Exception as e:
        logger.error(f"Set primary face error: {e}")
        return jsonify({'error': str(e)}), 500


@face_api.route('/identify/best-match', methods=['POST'])
@require_api_key
@rate_limit(limit=40, window=60)
@validate_input('image')
@audit_request
def identify_best_match():
    """
    Identify faces with best match per person in an app.
    
    Expected input:
    - image file (multipart/form-data) or image_base64/image_path in JSON
    - app_id: application identifier (required)
    - Optional: similarity_threshold (default: 0.8)
    
    Returns:
    - Best matching face for each person found
    """
    try:
        start_time = time.time()
        
        # Get parameters
        data = request.form if request.files else request.json
        app_id = data.get('app_id')
        similarity_threshold = float(data.get('similarity_threshold', 0.8))
        
        if not app_id:
            return jsonify({'error': ERROR_MISSING_APP_ID}), 400
        
        # Get image from request
        image, temp_file_path, should_cleanup = get_image_from_request(request)
        
        if image is None:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': ERROR_NO_IMAGE}), 400
        
        processed_faces = get_face_processor().process_image(image, Config.FACE_CONFIDENCE_THRESHOLD)
        
        if not processed_faces:
            cleanup_temp_file(temp_file_path, should_cleanup)
            return jsonify({'error': ERROR_NO_FACES}), 400
        
        # Use the first detected face for identification
        query_embedding = processed_faces[0]['embedding']
        
        # Find best matches per person
        best_matches = get_db_service().find_best_match_per_person(
            app_id=app_id,
            query_embedding=query_embedding,
            similarity_threshold=similarity_threshold
        )
        
        processing_time = time.time() - start_time
        
        # Log operation
        get_db_service().log_operation(
            operation_type='identify_best_match',
            input_source=temp_file_path if temp_file_path else "base64_image",
            processing_time=processing_time,
            faces_detected=len(processed_faces),
            success=True,
            metadata={'app_id': app_id, 'matches_found': len(best_matches)}
        )
        
        # Clean up temporary file if needed
        cleanup_temp_file(temp_file_path, should_cleanup)
        
        return jsonify(convert_numpy_types({
            'success': True,
            'app_id': app_id,
            'faces_detected': len(processed_faces),
            'matches_found': len(best_matches),
            'similarity_threshold': similarity_threshold,
            'best_matches': best_matches,
            'processing_time': processing_time
        }))
    
    except Exception as e:
        logger.error(f"Best match identification error: {e}")
        if 'temp_file_path' in locals() and 'should_cleanup' in locals():
            cleanup_temp_file(temp_file_path, should_cleanup)
        return jsonify({'error': str(e)}), 500
