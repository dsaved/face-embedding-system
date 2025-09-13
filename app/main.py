"""
Entry point for the Flask REST API with comprehensive security.
"""
import os
import logging
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from app.config import Config
from app.security import add_security_headers, log_security_event

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log') if os.path.exists('logs') else logging.StreamHandler()
    ]
)

# Configure audit logging
audit_logger = logging.getLogger('audit')
audit_handler = logging.FileHandler(Config.AUDIT_LOG_FILE) if Config.AUDIT_LOG_ENABLED else logging.StreamHandler()
audit_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)

app = Flask(__name__)
app.config.from_object(Config)

# Set up SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = Config.DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
db = SQLAlchemy(app)

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

# Register blueprints
from app.api.face_routes import face_api
app.register_blueprint(face_api)

# Create upload directory
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)

@app.route('/')
def index():
    """Health check endpoint."""
    return jsonify({
        'message': 'Face Vector Embedding System',
        'version': '1.0.0',
        'status': 'operational',
        'endpoints': {
            'detect': '/api/v1/faces/detect',
            'encode': '/api/v1/faces/encode',
            'register': '/api/v1/faces/register',
            'identify': '/api/v1/faces/identify',
            'verify': '/api/v1/faces/verify',
            'search': '/api/v1/faces/search',
            'stats': '/api/v1/faces/database/stats',
            'delete': '/api/v1/faces/{id}'
        }
    })

@app.route('/health')
def health_check():
    """Detailed health check."""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        stats = db_service.get_database_stats()
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'redis': 'connected' if db_service.redis_client else 'disconnected',
            'faiss_index': 'loaded' if db_service.faiss_index else 'not_loaded',
            'face_records': stats.get('total_face_records', 0)
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=Config.DEBUG)
