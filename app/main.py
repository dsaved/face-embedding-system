"""
Entry point for the Flask REST API with comprehensive security and WebSocket support.
"""
import os
import logging
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

# Import the Config class from the original config.py file
# We need to use a different import path since we now have a config package
from .config_file import Config
from app.security import add_security_headers, log_security_event

# Configure logging
log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Configure audit logging
audit_logger = logging.getLogger('audit')
audit_handler = logging.FileHandler(Config.AUDIT_LOG_FILE) if Config.AUDIT_LOG_ENABLED else logging.StreamHandler()
audit_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)


def create_app():
    """Create and configure Flask application with SocketIO support."""
    app = Flask(__name__, 
                static_folder='../static', 
                static_url_path='/static',
                template_folder='../templates')
    app.config.from_object(Config)

    # Set up SQLAlchemy
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

    # Initialize SocketIO for video streaming
    try:
        from app.api.video_routes import init_socketio
        socketio = init_socketio(app)
        app.socketio = socketio
    except ImportError as e:
        logging.warning(f"SocketIO not available: {e}")
        app.socketio = None

    # Security middleware
    @app.after_request
    def after_request(response):
        """Add security headers to all responses."""
        return add_security_headers(response)

    @app.before_request
    def before_request():
        """Log requests and perform security checks."""
        # Log all incoming requests
        if Config.AUDIT_LOG_ENABLED:
            log_security_event('HTTP_REQUEST', {
                'method': request.method,
                'path': request.path,
                'content_type': request.content_type,
                'content_length': request.content_length,
                'remote_addr': request.remote_addr
            })

    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file size limit exceeded."""
        log_security_event('FILE_SIZE_EXCEEDED', {
            'content_length': request.content_length,
            'limit': Config.MAX_CONTENT_LENGTH
        }, 'WARNING')
        return jsonify({'error': 'File too large'}), 413

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle rate limit exceeded."""
        return jsonify({'error': 'Rate limit exceeded'}), 429

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return jsonify({'error': 'Internal server error'}), 500

    # Register blueprints
    from app.api.face_routes import face_api
    app.register_blueprint(face_api)
    
    try:
        from app.api.video_routes import video_bp
        app.register_blueprint(video_bp, url_prefix='/video')
    except ImportError:
        logging.warning("Video routes not available")
    
    try:
        from app.api.liveness_routes import liveness_bp, init_liveness_socketio
        app.register_blueprint(liveness_bp)
        
        # Initialize liveness detection SocketIO handlers
        if app.socketio:
            init_liveness_socketio(app.socketio)
    except ImportError:
        logging.warning("Liveness detection routes not available")

    # Create directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)

    @app.route('/')
    def index():
        """Main endpoint with system information."""
        endpoints = {
            'detect': '/api/v1/faces/detect',
            'encode': '/api/v1/faces/encode',
            'register': '/api/v1/faces/register',
            'identify': '/api/v1/faces/identify',
            'verify': '/api/v1/faces/verify',
            'search': '/api/v1/faces/search',
            'stats': '/api/v1/faces/database/stats',
            'delete': '/api/v1/faces/{id}'
        }
        
        # Add video endpoints if available
        if hasattr(app, 'socketio') and app.socketio:
            endpoints.update({
                'video_health': '/api/v1/video/health',
                'video_sessions': '/api/v1/video/sessions',
                'websocket': '/socket.io/',
                'video_demo': '/templates/video_demo.html',
                'liveness_check': '/liveness'
            })
        
        return jsonify({
            'message': 'Face Vector Embedding System with Real-Time Video',
            'version': '2.0.0',
            'status': 'operational',
            'features': ['face_detection', 'face_recognition'] + (['real_time_video'] if hasattr(app, 'socketio') and app.socketio else []),
            'endpoints': endpoints
        })

    @app.route('/health')
    def health_check():
        """Detailed health check."""
        try:
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            stats = db_service.get_database_stats()
            
            health_data = {
                'status': 'healthy',
                'database': 'connected',
                'redis': 'connected' if hasattr(db_service, 'redis_client') and db_service.redis_client else 'disconnected',
                'faiss_index': 'loaded' if hasattr(db_service, 'faiss_index') and db_service.faiss_index else 'not_loaded',
                'face_records': stats.get('total_face_records', 0),
                'websocket': 'available' if hasattr(app, 'socketio') and app.socketio else 'unavailable'
            }
            
            # Add video processing health check
            try:
                from app.services.video_processor import VideoStreamProcessor
                # Just check if class can be imported
                VideoStreamProcessor
                health_data['video_processor'] = 'available'
            except Exception:
                health_data['video_processor'] = 'unavailable'
            
            return jsonify(health_data)
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 500

    # Add route to serve video demo
    @app.route('/demo')
    def video_demo():
        """Serve the video demo page."""
        try:
            with open('templates/video_demo.html', 'r') as f:
                return f.read()
        except FileNotFoundError:
            return jsonify({'error': 'Demo page not found'}), 404
    
    # Add route to serve liveness detection page
    @app.route('/liveness')
    def liveness_check():
        """Serve the advanced liveness detection page."""
        try:
            with open('templates/liveness_check.html', 'r') as f:
                return f.read()
        except FileNotFoundError:
            return jsonify({'error': 'Liveness check page not found'}), 404

    return app


# Create app instance
app = create_app()

if __name__ == '__main__':
    try:
        logger.info("Starting Face Embedding System...")
        logger.info(f"Server will run on http://0.0.0.0:{Config.PORT}")
        logger.info(f"Debug mode: {Config.DEBUG}")
        
        # Allow unsafe Werkzeug for PM2 deployment
        app.socketio.run(app, host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
