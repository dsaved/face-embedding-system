"""
WebSocket endpoints for real-time video face recognition.
Handles video streaming and real-time face recognition communication.
"""
import cv2
import numpy as np
import base64
import json
import time
import logging
from typing import Dict, Any
from flask import Blueprint, request, current_app
from flask_socketio import SocketIO, emit, disconnect, join_room, leave_room
from app.services.video_processor import VideoStreamProcessor, validate_frame, resize_frame_for_processing
from app.services.face_processor import FaceProcessor
from app.services.database_service import DatabaseService
from app.security import require_api_key_ws, log_security_event

logger = logging.getLogger(__name__)

# Blueprint for video-related routes
video_bp = Blueprint('video', __name__)

@video_bp.route('/demo')
def video_demo():
    """Serve the optimized video recognition demo page."""
    from flask import render_template
    return render_template('video_demo.html')

# Global SocketIO instance (will be initialized in main.py)
socketio = None

# Active video sessions
active_sessions = {}


def init_socketio(app):
    """Initialize SocketIO with Flask app."""
    global socketio
    socketio = SocketIO(
        app, 
        cors_allowed_origins="*",
        async_mode='threading',
        logger=True,
        engineio_logger=True
    )
    
    # Register event handlers
    register_socketio_events()
    
    return socketio


def register_socketio_events():
    """Register all SocketIO event handlers."""
    
    @socketio.on('connect')
    def handle_connect(auth):
        """Handle client connection."""
        try:
            from app.config_file import Config
            
            # Only require API key if configured
            if Config.API_KEY_REQUIRED:
                # Validate API key from auth data
                api_key = auth.get('api_key') if auth else None
                
                if not api_key or not _validate_api_key(api_key):
                    logger.warning(f"Unauthorized WebSocket connection attempt from {request.remote_addr}")
                    log_security_event('websocket_unauthorized_connection', {
                        'remote_addr': request.remote_addr,
                        'user_agent': request.headers.get('User-Agent', '')
                    })
                    return False  # Reject connection
            else:
                api_key = "development"  # Use placeholder for non-authenticated sessions
            
            # Create video processor for this session
            session_id = request.sid
            processor = VideoStreamProcessor()
            
            active_sessions[session_id] = {
                'processor': processor,
                'connected_at': time.time(),
                'frames_processed': 0,
                'api_key': api_key
            }
            
            logger.info(f"Video session {session_id} connected")
            
            # Send connection success
            emit('connection_status', {
                'status': 'connected',
                'session_id': session_id,
                'timestamp': time.time()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
            return False
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        session_id = request.sid
        
        if session_id in active_sessions:
            session_info = active_sessions[session_id]
            duration = time.time() - session_info['connected_at']
            frames_processed = session_info['frames_processed']
            
            logger.info(f"Video session {session_id} disconnected. "
                       f"Duration: {duration:.2f}s, Frames: {frames_processed}")
            
            # Cleanup session
            del active_sessions[session_id]
    
    @socketio.on('video_frame')
    def handle_video_frame(data):
        """Handle incoming video frame for processing."""
        session_id = request.sid
        
        logger.debug(f"DEBUG: handle_video_frame called for session {session_id}")
        logger.debug(f"DEBUG: data type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
        
        if session_id not in active_sessions:
            logger.error(f"ERROR: Session {session_id} not found in active_sessions")
            emit('error', {'message': 'Session not found'})
            return

        try:
            logger.debug(f"DEBUG: Starting frame processing for session {session_id}")
            session_info = active_sessions[session_id]
            processor = session_info['processor']
            
            # Decode frame data
            logger.debug(f"DEBUG: About to decode frame data")
            frame = _decode_frame_data(data)
            if frame is None:
                logger.warning("Failed to decode frame data")
                emit('error', {'message': 'Invalid frame data'})
                return
            
            logger.debug(f"Successfully decoded frame with shape: {frame.shape}")
            
            # Resize frame for optimal processing
            frame = resize_frame_for_processing(frame, max_width=640)
            logger.debug(f"Resized frame to: {frame.shape}")
            
            # Process frame with the optimized video processor
            # Note: VideoStreamProcessor is pre-configured for optimal settings
            logger.debug(f"DEBUG: About to call processor.process_frame")
            results = processor.process_frame(frame)
            logger.debug(f"Frame processing complete, results: {type(results)}")
            
            # Update session stats
            session_info['frames_processed'] += 1
            
            # Send results back to client with face recognition results
            emit('face_recognition_result', {
                'success': True,
                'session_id': session_id,
                'faces': results.get('faces', []),
                'processing_time': results.get('processing_time', 0),
                'frame_count': session_info['frames_processed'],
                'total_faces': len(results.get('faces', [])),
                'identified_faces': sum(1 for face in results.get('faces', []) 
                                       if face.get('identification', {}).get('person_id') != 'unknown')
            })
            
            logger.debug(f"Sent recognition result with {len(results.get('faces', []))} faces")
            
        except Exception as e:
            logger.error(f"Error processing video frame for session {session_id}: {e}")
            emit('error', {'message': f'Frame processing error: {str(e)}'})

    @socketio.on('process_frame')
    def handle_process_frame(data):
        """Handle process_frame event (alias for video_frame for frontend compatibility)."""
        logger.debug(f"DEBUG: handle_process_frame called with data keys: {list(data.keys()) if isinstance(data, dict) else 'not dict'}")
        # This provides compatibility with the demo frontend
        return handle_video_frame(data)
    @socketio.on('start_video_stream')
    def handle_start_video_stream(data):
        """Handle start video stream request."""
        session_id = request.sid
        
        if session_id not in active_sessions:
            emit('error', {'message': 'Session not found'})
            return
        
        try:
            # Reset processor for new stream
            processor = active_sessions[session_id]['processor']
            processor.reset_processing()
            
            # Join stream room for broadcasting
            room = f"stream_{session_id}"
            join_room(room)
            
            emit('stream_status', {
                'status': 'started',
                'session_id': session_id,
                'room': room
            })
            
            logger.info(f"Video stream started for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error starting video stream: {e}")
            emit('error', {'message': f'Stream start error: {str(e)}'})
    
    @socketio.on('stop_video_stream')
    def handle_stop_video_stream():
        """Handle stop video stream request."""
        session_id = request.sid
        
        if session_id not in active_sessions:
            emit('error', {'message': 'Session not found'})
            return
        
        try:
            # Leave stream room
            room = f"stream_{session_id}"
            leave_room(room)
            
            # Get final stats
            processor = active_sessions[session_id]['processor']
            final_stats = processor.get_processing_stats()
            
            emit('stream_status', {
                'status': 'stopped',
                'session_id': session_id,
                'final_stats': final_stats
            })
            
            logger.info(f"Video stream stopped for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error stopping video stream: {e}")
            emit('error', {'message': f'Stream stop error: {str(e)}'})
    
    @socketio.on('get_session_stats')
    def handle_get_session_stats():
        """Get current session statistics."""
        session_id = request.sid
        
        if session_id not in active_sessions:
            emit('error', {'message': 'Session not found'})
            return
        
        try:
            session_info = active_sessions[session_id]
            processor = session_info['processor']
            
            stats = {
                'session_id': session_id,
                'connected_at': session_info['connected_at'],
                'frames_processed': session_info['frames_processed'],
                'processing_stats': processor.get_processing_stats(),
                'uptime': time.time() - session_info['connected_at']
            }
            
            emit('session_stats', stats)
            
        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            emit('error', {'message': f'Stats error: {str(e)}'})


def _validate_api_key(api_key: str) -> bool:
    """Validate API key for WebSocket connection."""
    try:
        from app.config_file import Config
        return api_key == Config.API_KEY
    except Exception:
        return False


def _decode_frame_data(data: Dict[str, Any]) -> np.ndarray:
    """Decode frame data from client."""
    try:
        logger.debug(f"Decoding frame data: keys={list(data.keys())}")
        
        # Handle different frame data formats
        frame_data = None
        if 'frame_data' in data:
            frame_data = data['frame_data']
        elif 'frame' in data:
            frame_data = data['frame']
        elif 'image' in data:
            frame_data = data['image']
        else:
            logger.error("No frame or image data found in request")
            return None
        
        logger.debug(f"Frame data type: {type(frame_data)}, length: {len(frame_data) if frame_data else 0}")
        
        # Remove data URL prefix if present
        if frame_data.startswith('data:image'):
            frame_data = frame_data.split(',')[1]
            logger.debug("Removed data URL prefix")
        
        # Decode base64
        img_bytes = base64.b64decode(frame_data)
        logger.debug(f"Decoded {len(img_bytes)} bytes from base64")
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.debug(f"Decoded frame shape: {frame.shape if frame is not None else None}")
        
        # Validate frame
        if validate_frame(frame):
            logger.debug("Frame validation passed")
            return frame
        else:
            logger.warning("Invalid frame received")
            return None
                
    except Exception as e:
        logger.error(f"Error decoding frame data: {e}")
        return None


# REST API endpoints for video-related operations
@video_bp.route('/sessions', methods=['GET'])
@require_api_key_ws
def get_active_sessions():
    """Get information about active video sessions."""
    try:
        sessions_info = {}
        
        for session_id, session_data in active_sessions.items():
            sessions_info[session_id] = {
                'connected_at': session_data['connected_at'],
                'frames_processed': session_data['frames_processed'],
                'uptime': time.time() - session_data['connected_at'],
                'stats': session_data['processor'].get_processing_stats()
            }
        
        return {
            'success': True,
            'active_sessions': len(active_sessions),
            'sessions': sessions_info
        }
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        return {'error': str(e)}, 500


@video_bp.route('/sessions/<session_id>', methods=['DELETE'])
@require_api_key_ws
def disconnect_session(session_id: str):
    """Disconnect a specific video session."""
    try:
        if session_id in active_sessions:
            # Disconnect the session
            socketio.disconnect(session_id)
            
            return {
                'success': True,
                'message': f'Session {session_id} disconnected'
            }
        else:
            return {
                'error': 'Session not found'
            }, 404
            
    except Exception as e:
        logger.error(f"Error disconnecting session {session_id}: {e}")
        return {'error': str(e)}, 500


@video_bp.route('/config', methods=['GET'])
@require_api_key_ws
def get_video_config():
    """Get video processing configuration."""
    try:
        from app.config_file import Config
        
        config = {
            'frame_skip': getattr(Config, 'VIDEO_FRAME_SKIP', 2),
            'recognition_interval': getattr(Config, 'VIDEO_RECOGNITION_INTERVAL', 30),
            'detection_interval': getattr(Config, 'VIDEO_DETECTION_INTERVAL', 5),
            'max_tracked_faces': getattr(Config, 'VIDEO_MAX_TRACKED_FACES', 10),
            'max_frame_width': getattr(Config, 'VIDEO_MAX_FRAME_WIDTH', 640),
            'websocket_timeout': getattr(Config, 'WEBSOCKET_TIMEOUT', 60)
        }
        
        return {
            'success': True,
            'config': config
        }
        
    except Exception as e:
        logger.error(f"Error getting video config: {e}")
        return {'error': str(e)}, 500


@video_bp.route('/test', methods=['POST'])
@require_api_key_ws
def test_video_processing():
    """Test video processing with uploaded frame."""
    try:
        # Get image from request (similar to existing face detection)
        from app.api.face_routes import get_image_from_request
        from app.config_file import Config
        
        image, temp_file_path, should_cleanup = get_image_from_request(request)
        
        if image is None:
            return {'error': 'No image provided or could not load image'}, 400
        
        # Create video processor
        processor = VideoStreamProcessor()
        
        # Process frame
        results = processor.process_frame(image)
        
        # Cleanup temporary file if needed
        if should_cleanup and temp_file_path:
            import os
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        return {
            'success': True,
            'results': results,
            'processing_stats': processor.get_processing_stats()
        }
        
    except Exception as e:
        logger.error(f"Error in video processing test: {e}")
        return {'error': str(e)}, 500


# Health check for video services
@video_bp.route('/health', methods=['GET'])
def video_health_check():
    """Health check for video processing services."""
    try:
        # Check if video processor can be created
        processor = VideoStreamProcessor()
        
        # Check OpenCV
        opencv_version = cv2.__version__
        
        return {
            'status': 'healthy',
            'active_sessions': len(active_sessions),
            'opencv_version': opencv_version,
            'video_processor': 'available',
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Video health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }, 500
