"""
Liveness Detection API Routes
Provides endpoints for advanced liveness detection and anti-spoofing
"""

from flask import Blueprint, request, jsonify, current_app
from flask_socketio import emit
import cv2
import numpy as np
import base64
import json
import logging
from typing import Dict, Any
import face_recognition
from app.services.liveness_detector import AdvancedLivenessDetector
from app.services.face_processor import FaceProcessor
from app.utils.image_utils import base64_to_image, image_to_base64, base64_to_image_rgb
import time

# Set up logging
logger = logging.getLogger(__name__)

# Blueprint for liveness routes
liveness_bp = Blueprint('liveness', __name__)

# Global liveness detector instance (in production, use proper session management)
liveness_detector = AdvancedLivenessDetector()
face_processor = FaceProcessor()

# Cache for face detection optimization
_face_detection_cache = {
    "last_detection_time": 0,
    "cached_face_bbox": None,
    "cached_landmarks": None,
    "detection_interval": 0.03,  # Run face detection every 30ms (even faster)
    "frame_skip_count": 0,
    "max_frame_skips": 1  # Skip only 1 frame for maximum responsiveness
}

def get_face_data_optimized(frame):
    """Optimized face detection with aggressive caching and frame skipping for maximum speed"""
    current_time = time.time()
    
    # Implement frame skipping for extra speed
    _face_detection_cache["frame_skip_count"] += 1
    
    # Check if we should skip this frame
    if (_face_detection_cache["frame_skip_count"] < _face_detection_cache["max_frame_skips"] and 
        _face_detection_cache["cached_face_bbox"] is not None):
        # Return cached results without processing
        return _face_detection_cache["cached_face_bbox"], _face_detection_cache["cached_landmarks"]
    
    # Reset skip counter
    _face_detection_cache["frame_skip_count"] = 0
    
    # Check time-based cache
    time_since_last = current_time - _face_detection_cache["last_detection_time"]
    if (time_since_last < _face_detection_cache["detection_interval"] and 
        _face_detection_cache["cached_face_bbox"] is not None):
        return _face_detection_cache["cached_face_bbox"], _face_detection_cache["cached_landmarks"]
    
    # Perform actual detection with ultra-aggressive optimization
    try:
        # Scale down for ultra-fast processing (160px instead of 240px)
        height, width = frame.shape[:2]
        if width > 160:
            scale_factor = 160.0 / width
            new_width = 160
            new_height = int(height * scale_factor)
            small_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            small_frame = frame
            scale_factor = 1.0
        
        # Convert to RGB for face_recognition (if not already)
        if len(small_frame.shape) == 3 and small_frame.shape[2] == 3:
            # Assume BGR, convert to RGB
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = small_frame
        
        # Use faster face detection model and reduced number of upsamples
        face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=0, model="hog")
        
        if not face_locations:
            # Keep previous cache if available, clear if not
            if _face_detection_cache["cached_face_bbox"] is None:
                _face_detection_cache["last_detection_time"] = current_time
            return _face_detection_cache["cached_face_bbox"], _face_detection_cache["cached_landmarks"]
        
        # Get the first (largest) face
        top, right, bottom, left = face_locations[0]
        
        # Scale back to original size
        if scale_factor != 1.0:
            top = int(top / scale_factor)
            right = int(right / scale_factor)
            bottom = int(bottom / scale_factor)
            left = int(left / scale_factor)
        
        face_bbox = (left, top, right, bottom)
        
        # Get landmarks for the detected face (on original frame for accuracy)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if face_encodings:
            # Get landmarks using the original frame for better accuracy
            if scale_factor != 1.0:
                original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
                landmarks = face_recognition.face_landmarks(original_rgb, [face_locations[0]])
            else:
                landmarks = face_recognition.face_landmarks(rgb_frame, [face_locations[0]])
            
            # Convert landmarks dictionary to numpy array (68 landmark points)
            if landmarks and len(landmarks) > 0:
                landmark_dict = landmarks[0]
                # Extract all landmark points in dlib's 68-point format
                landmark_points = []
                # Order: chin (0-16), right_eyebrow (17-21), left_eyebrow (22-26), 
                # nose_bridge (27-30), nose_tip (31-35), right_eye (36-41), left_eye (42-47),
                # top_lip (48-54), bottom_lip (55-59), inner_lip (60-67)
                point_order = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 
                              'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
                
                for feature in point_order:
                    if feature in landmark_dict:
                        landmark_points.extend(landmark_dict[feature])
                
                face_landmarks = np.array(landmark_points) if landmark_points else None
            else:
                face_landmarks = None
        else:
            face_landmarks = None
        
        # Update cache
        _face_detection_cache["cached_face_bbox"] = face_bbox
        _face_detection_cache["cached_landmarks"] = face_landmarks
        _face_detection_cache["last_detection_time"] = current_time
        
        return face_bbox, face_landmarks
        
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        # Return cached results if available
        return _face_detection_cache["cached_face_bbox"], _face_detection_cache["cached_landmarks"]

@liveness_bp.route('/api/liveness/start-session', methods=['POST'])
def start_liveness_session():
    """Start a new liveness detection session"""
    try:
        data = request.get_json() or {}
        
        # Initialize liveness session
        session_info = liveness_detector.start_liveness_session()
        
        # Add security parameters
        session_info.update({
            "anti_spoofing_enabled": True,
            "tampering_detection_enabled": True,
            "min_challenges_required": liveness_detector.min_challenges,
            "challenge_timeout": liveness_detector.challenge_timeout,
            "instructions": {
                "blink": "Please blink naturally when prompted",
                "head_turn": "Turn your head in the indicated direction",
                "smile": "Please smile when prompted", 
                "mouth_open": "Please open your mouth when prompted"
            }
        })
        
        logger.info(f"Started liveness session: {session_info['session_id']}")
        
        return jsonify({
            "success": True,
            "session_info": session_info
        })
        
    except Exception as e:
        logger.error(f"Error starting liveness session: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@liveness_bp.route('/api/liveness/process-frame', methods=['POST'])
def process_liveness_frame():
    """Process a frame for liveness detection"""
    try:
        data = request.get_json()
        
        if not data or 'frame_data' not in data:
            return jsonify({
                "success": False,
                "error": "No frame data provided"
            }), 400
        
        # Decode frame
        frame = base64_to_image_rgb(data['frame_data'])
        if frame is None:
            return jsonify({
                "success": False,
                "error": "Invalid frame data"
            }), 400
        
        # Validate frame format
        if frame.dtype != np.uint8:
            logger.error(f"Invalid frame dtype: {frame.dtype}")
            return jsonify({
                "success": False,
                "error": "Invalid frame data format"
            }), 400
        
        # Extract integrity token if provided
        integrity_token = data.get('integrity_token')
        
        # Detect faces and landmarks using face_recognition (expects RGB)
        try:
            face_locations = face_recognition.face_locations(frame, model="hog")
        except Exception as fr_error:
            logger.error(f"Face recognition error: {fr_error}")
            return jsonify({
                "success": False,
                "error": f"Face detection failed: {str(fr_error)}"
            }), 500
        
        if not face_locations:
            return jsonify({
                "success": False,
                "error": "No face detected in frame"
            })
        
        # Use the first detected face
        face_location = face_locations[0]
        top, right, bottom, left = face_location
        face_bbox = (left, top, right, bottom)
        
        # Get facial landmarks
        face_landmarks_list = face_recognition.face_landmarks(frame, face_locations)
        if not face_landmarks_list:
            return jsonify({
                "success": False,
                "error": "Could not extract facial landmarks"
            })
        
        # Convert landmarks to numpy array (68 landmark points)
        landmarks_dict = face_landmarks_list[0]
        landmarks_points = []
        
        # Order landmarks according to dlib's 68-point model
        landmark_order = [
            'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
            'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip'
        ]
        
        for feature in landmark_order:
            if feature in landmarks_dict:
                landmarks_points.extend(landmarks_dict[feature])
        
        # Convert to numpy array
        face_landmarks = np.array(landmarks_points)
        
        # Process frame for liveness
        liveness_result = liveness_detector.process_liveness_frame(
            frame=frame,
            face_landmarks=face_landmarks,
            face_bbox=face_bbox,
            integrity_token=integrity_token
        )
        
        # Ensure liveness_result is JSON serializable
        liveness_result = liveness_detector._ensure_json_serializable(liveness_result)
        
        # Check if all challenges are completed
        session_complete = (len(liveness_detector.active_challenges) == 0 and 
                          len(liveness_detector.completed_challenges) > 0)
        
        response = {
            "success": True,
            "liveness_result": liveness_result,
            "session_complete": bool(session_complete),
            "face_detected": True,
            "face_bbox": list(face_bbox)  # Ensure tuple is converted to list for JSON
        }
        
        # If session is complete, generate final result
        if session_complete:
            final_result = liveness_detector.finalize_liveness_result(capture_snapshot=True)
            
            # Ensure all values are JSON serializable
            final_result_dict = {
                "is_live": bool(final_result.is_live),
                "confidence": float(final_result.confidence),
                "challenges_passed": int(final_result.challenges_passed),
                "total_challenges": int(final_result.total_challenges),
                "anti_spoofing_score": float(final_result.anti_spoofing_score),
                "snapshot": final_result.snapshot,
                "details": liveness_detector._ensure_json_serializable(final_result.details)
            }
            
            response["final_result"] = final_result_dict
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing liveness frame: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@liveness_bp.route('/api/liveness/get-status', methods=['GET'])
def get_liveness_status():
    """Get current liveness detection status"""
    try:
        status = {
            "active_challenges": [
                {
                    "challenge_id": c.challenge_id,
                    "challenge_type": c.challenge_type,
                    "direction": getattr(c, 'direction', None),
                    "time_remaining": float(max(0, c.timeout - (time.time() - c.start_time))),
                    "attempts": int(c.attempts),
                    "max_attempts": int(c.max_attempts)
                } for c in liveness_detector.active_challenges
            ],
            "completed_challenges": [
                {
                    "challenge_type": c.challenge_type,
                    "direction": getattr(c, 'direction', None),
                    "completed": bool(c.completed)
                } for c in liveness_detector.completed_challenges
            ],
            "session_progress": {
                "total_challenges": int(len(liveness_detector.active_challenges) + len(liveness_detector.completed_challenges)),
                "completed": int(len(liveness_detector.completed_challenges)),
                "remaining": int(len(liveness_detector.active_challenges))
            }
        }
        
        # Ensure status is JSON serializable
        status = liveness_detector._ensure_json_serializable(status)
        
        return jsonify({
            "success": True,
            "status": status
        })
        
    except Exception as e:
        logger.error(f"Error getting liveness status: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@liveness_bp.route('/api/liveness/reset-session', methods=['POST'])
def reset_liveness_session():
    """Reset the current liveness detection session"""
    try:
        # Clear all session data
        liveness_detector.frame_history.clear()
        liveness_detector.landmark_history.clear()
        liveness_detector.texture_history.clear()
        liveness_detector.motion_history.clear()
        liveness_detector.active_challenges.clear()
        liveness_detector.completed_challenges.clear()
        
        logger.info("Liveness session reset")
        
        return jsonify({
            "success": True,
            "message": "Session reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Error resetting liveness session: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@liveness_bp.route('/api/liveness/validate-environment', methods=['POST'])
def validate_environment():
    """Validate the video environment for tampering"""
    try:
        data = request.get_json()
        
        if not data or 'frame_data' not in data:
            return jsonify({
                "success": False,
                "error": "No frame data provided"
            }), 400
        
        # Decode frame
        frame = base64_to_image_rgb(data['frame_data'])
        if frame is None:
            return jsonify({
                "success": False,
                "error": "Invalid frame data"
            }), 400
        
        # Detect tampering
        tampering_indicators = liveness_detector.detect_environment_tampering(frame)
        tampering_detected = any(tampering_indicators.values())
        
        # Additional browser environment checks
        browser_info = data.get('browser_info', {})
        environment_analysis = {
            "tampering_detected": tampering_detected,
            "tampering_indicators": tampering_indicators,
            "browser_analysis": _analyze_browser_environment(browser_info),
            "frame_integrity": _analyze_frame_integrity(frame),
            "recommendations": []
        }
        
        # Generate recommendations based on analysis
        if tampering_detected:
            environment_analysis["recommendations"].append("Environment tampering detected - please ensure no video filters or modifications are active")
        
        if environment_analysis["browser_analysis"]["suspicious_extensions"]:
            environment_analysis["recommendations"].append("Disable browser extensions that may interfere with video capture")
        
        return jsonify({
            "success": True,
            "environment_analysis": environment_analysis
        })
        
    except Exception as e:
        logger.error(f"Error validating environment: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def _analyze_browser_environment(browser_info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze browser environment for potential security issues"""
    analysis = {
        "user_agent_valid": True,
        "suspicious_extensions": False,
        "webrtc_available": True,
        "security_score": 1.0
    }
    
    # Check user agent
    user_agent = browser_info.get('user_agent', '')
    if not user_agent or len(user_agent) < 50:
        analysis["user_agent_valid"] = False
        analysis["security_score"] -= 0.2
    
    # Check for known problematic patterns
    suspicious_patterns = ['headless', 'bot', 'automation', 'phantom']
    if any(pattern in user_agent.lower() for pattern in suspicious_patterns):
        analysis["suspicious_extensions"] = True
        analysis["security_score"] -= 0.3
    
    # Check WebRTC capabilities
    webrtc_info = browser_info.get('webrtc', {})
    if not webrtc_info.get('supported', True):
        analysis["webrtc_available"] = False
        analysis["security_score"] -= 0.1
    
    return analysis

def _analyze_frame_integrity(frame: np.ndarray) -> Dict[str, Any]:
    """Analyze frame for integrity and authenticity"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate various integrity metrics
    analysis = {
        "resolution_appropriate": True,
        "compression_normal": True,
        "noise_level_normal": True,
        "integrity_score": 1.0
    }
    
    # Check resolution
    height, width = gray.shape
    if width < 320 or height < 240:
        analysis["resolution_appropriate"] = False
        analysis["integrity_score"] -= 0.1
    
    # Check compression artifacts
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:  # Too smooth
        analysis["compression_normal"] = False
        analysis["integrity_score"] -= 0.2
    
    # Check noise level
    noise_level = np.std(gray)
    if noise_level < 10 or noise_level > 50:  # Unusual noise patterns
        analysis["noise_level_normal"] = False
        analysis["integrity_score"] -= 0.1
    
    return analysis

# Socket.IO event handlers for real-time liveness detection
def init_liveness_socketio(socketio):
    """Initialize Socket.IO event handlers for liveness detection"""
    
    @socketio.on('start_liveness_session')
    def handle_start_liveness_session(data):
        """Handle liveness session start via WebSocket"""
        try:
            session_info = liveness_detector.start_liveness_session()
            emit('liveness_session_started', {
                "success": True,
                "session_info": session_info
            })
        except Exception as e:
            logger.error(f"Socket error starting liveness session: {e}")
            emit('liveness_error', {
                "success": False,
                "error": str(e)
            })
    
    @socketio.on('process_liveness_frame')
    def handle_process_liveness_frame(data):
        """Handle liveness frame processing via WebSocket - optimized for speed"""
        start_time = time.time()
        try:
            if 'frame_data' not in data:
                emit('liveness_error', {
                    "success": False,
                    "error": "No frame data provided"
                })
                return
            
            # Fast decode and process frame (with proper RGB format for face_recognition)
            decode_start = time.time()
            frame = base64_to_image_rgb(data['frame_data'])
            decode_time = time.time() - decode_start
            
            if frame is None:
                emit('liveness_error', {
                    "success": False,
                    "error": "Invalid frame data"
                })
                return
            
            # Quick validation - reduced checks for speed
            if frame.dtype != np.uint8 or len(frame.shape) != 3 or frame.shape[2] != 3:
                emit('liveness_error', {
                    "success": False,
                    "error": "Invalid frame format"
                })
                return
            
            # Get face landmarks using optimized detection (with aggressive caching)
            face_detect_start = time.time()
            face_bbox, face_landmarks = get_face_data_optimized(frame)
            face_detect_time = time.time() - face_detect_start
            
            if face_bbox is None or face_landmarks is None:
                emit('liveness_frame_result', {
                    "success": False,
                    "error": "No face detected"
                })
                return
            
            # Process liveness with optimized settings
            liveness_start = time.time()
            liveness_result = liveness_detector.process_liveness_frame(
                frame=frame,
                face_landmarks=face_landmarks,
                face_bbox=face_bbox,
                integrity_token=data.get('integrity_token')
            )
            liveness_time = time.time() - liveness_start
            
            # Quick completion check
            session_complete = (len(liveness_detector.active_challenges) == 0 and 
                              len(liveness_detector.completed_challenges) > 0)
            
            # Ensure liveness_result is JSON serializable (optimized)
            liveness_result = liveness_detector._ensure_json_serializable(liveness_result)
            
            # Build response efficiently
            response = {
                "success": True,
                "liveness_result": liveness_result,
                "session_complete": bool(session_complete),
                "face_bbox": list(face_bbox)  # Convert tuple to list for JSON
            }
            
            if session_complete:
                final_result = liveness_detector.finalize_liveness_result(capture_snapshot=True)
                
                # Ensure all final result values are JSON serializable
                final_result_dict = {
                    "is_live": bool(final_result.is_live),
                    "confidence": float(final_result.confidence),
                    "challenges_passed": int(final_result.challenges_passed),
                    "total_challenges": int(final_result.total_challenges),
                    "anti_spoofing_score": float(final_result.anti_spoofing_score),
                    "snapshot": final_result.snapshot,
                    "details": liveness_detector._ensure_json_serializable(final_result.details)
                }
                
                response["final_result"] = final_result_dict
                
                emit('liveness_session_complete', response)
            else:
                emit('liveness_frame_result', response)
            
            # Log performance metrics (only in debug mode for speed)
            if logger.isEnabledFor(logging.DEBUG):
                total_time = time.time() - start_time
                logger.debug(f"Frame processing performance - Total: {total_time:.3f}s, "
                            f"Decode: {decode_time:.3f}s, Face: {face_detect_time:.3f}s, "
                            f"Liveness: {liveness_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Socket error processing liveness frame: {e}")
            emit('liveness_error', {
                "success": False,
                "error": str(e)
            })
    
    @socketio.on('get_liveness_status')
    def handle_get_liveness_status():
        """Handle liveness status request via WebSocket"""
        try:
            status = {
                "active_challenges": [
                    {
                        "challenge_id": c.challenge_id,
                        "challenge_type": c.challenge_type,
                        "direction": getattr(c, 'direction', None),
                        "time_remaining": float(max(0, c.timeout - (time.time() - c.start_time))),
                        "attempts": int(c.attempts)
                    } for c in liveness_detector.active_challenges
                ],
                "completed_challenges": int(len(liveness_detector.completed_challenges)),
                "total_challenges": int(len(liveness_detector.active_challenges) + len(liveness_detector.completed_challenges))
            }
            
            # Ensure status is JSON serializable
            status = liveness_detector._ensure_json_serializable(status)
            
            emit('liveness_status', {
                "success": True,
                "status": status
            })
            
        except Exception as e:
            logger.error(f"Socket error getting liveness status: {e}")
            emit('liveness_error', {
                "success": False,
                "error": str(e)
            })
