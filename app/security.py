"""
Security module for the Face Embedding System.
Provides authentication, rate limiting, input validation, and audit logging.
"""
import os
import time
import hashlib
import logging
import functools
import ipaddress
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from flask import request, jsonify, g
from werkzeug.exceptions import RequestEntityTooLarge
import redis
from app.config import Config

# Setup logging
security_logger = logging.getLogger('security')
audit_logger = logging.getLogger('audit')

# Initialize Redis for rate limiting
redis_client = None
if Config.RATE_LIMIT_ENABLED:
    try:
        redis_client = redis.from_url(Config.REDIS_URL)
    except Exception as e:
        security_logger.warning(f"Redis connection failed for rate limiting: {e}")

class SecurityError(Exception):
    """Custom security exception."""
    pass

class RateLimitExceeded(SecurityError):
    """Rate limit exceeded exception."""
    pass

class AuthenticationError(SecurityError):
    """Authentication failed exception."""
    pass

class ValidationError(SecurityError):
    """Input validation failed exception."""
    pass

def get_client_ip() -> str:
    """Get the real client IP address."""
    # Check for forwarded headers first
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(',')[0].strip()
    
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    return request.remote_addr or 'unknown'

def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage/comparison."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def validate_api_key(api_key: str) -> bool:
    """Validate API key against configured keys."""
    if not Config.API_KEYS:
        return True  # No API keys configured, allow access
    
    # Support both plain text and hashed keys
    hashed_key = hash_api_key(api_key)
    return api_key in Config.API_KEYS or hashed_key in Config.API_KEYS

def check_rate_limit(identifier: str, limit: int = None, window: int = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if request is within rate limit.
    Returns (is_allowed, rate_limit_info)
    """
    if not Config.RATE_LIMIT_ENABLED or not redis_client:
        return True, {}
    
    limit = limit or Config.RATE_LIMIT_REQUESTS
    window = window or Config.RATE_LIMIT_WINDOW
    
    try:
        current_time = int(time.time())
        window_start = current_time - window
        
        # Remove old entries
        redis_client.zremrangebyscore(identifier, 0, window_start)
        
        # Count current requests
        current_requests = redis_client.zcard(identifier)
        
        rate_limit_info = {
            'limit': limit,
            'remaining': max(0, limit - current_requests),
            'reset_time': current_time + window,
            'window': window
        }
        
        if current_requests >= limit:
            return False, rate_limit_info
        
        # Add current request
        redis_client.zadd(identifier, {str(current_time): current_time})
        redis_client.expire(identifier, window)
        
        rate_limit_info['remaining'] = max(0, limit - current_requests - 1)
        return True, rate_limit_info
        
    except Exception as e:
        security_logger.error(f"Rate limiting error: {e}")
        return True, {}  # Allow request if rate limiting fails

def validate_file_upload(file) -> Dict[str, Any]:
    """Validate uploaded file for security."""
    if not file:
        raise ValidationError("No file provided")
    
    if not file.filename:
        raise ValidationError("No filename provided")
    
    # Check file extension
    filename = file.filename.lower()
    allowed_extensions = {ext.lower() for ext in Config.ALLOWED_EXTENSIONS}
    file_ext = filename.rsplit('.', 1)[1] if '.' in filename else ''
    
    if file_ext not in allowed_extensions:
        raise ValidationError(f"File type not allowed. Allowed: {', '.join(allowed_extensions)}")
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > Config.MAX_FILE_SIZE:
        raise ValidationError(f"File too large. Maximum size: {Config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
    
    if file_size == 0:
        raise ValidationError("Empty file not allowed")
    
    return {
        'filename': file.filename,
        'size': file_size,
        'extension': file_ext
    }

def validate_image_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate image input data."""
    validation_info = {}
    
    # Validate base64 image input
    if 'image_base64' in data:
        validation_info.update(_validate_base64_input(data['image_base64']))
    
    # Validate file path input
    elif 'image_path' in data:
        validation_info.update(_validate_file_path_input(data['image_path']))
    
    # Validate optional parameters
    _validate_optional_parameters(data)
    
    return validation_info

def _validate_base64_input(base64_data: str) -> Dict[str, Any]:
    """Validate base64 image input."""
    if not isinstance(base64_data, str):
        raise ValidationError("image_base64 must be a string")
    
    # Remove data URL prefix if present
    if base64_data.startswith('data:'):
        try:
            base64_data = base64_data.split(',', 1)[1]
        except IndexError:
            raise ValidationError("Invalid data URL format")
    
    # Estimate decoded size (base64 is ~33% larger than binary)
    estimated_size = len(base64_data) * 3 / 4
    if estimated_size > Config.MAX_FILE_SIZE:
        raise ValidationError(f"Base64 image too large. Maximum size: {Config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
    
    return {
        'input_type': 'base64',
        'estimated_size': estimated_size
    }

def _validate_file_path_input(image_path: str) -> Dict[str, Any]:
    """Validate file path input."""
    if not isinstance(image_path, str):
        raise ValidationError("image_path must be a string")
    
    # Basic path validation (prevent directory traversal)
    if '..' in image_path or image_path.startswith('/'):
        raise ValidationError("Invalid image path")
    
    return {
        'input_type': 'file_path',
        'path': image_path
    }

def _validate_optional_parameters(data: Dict[str, Any]) -> None:
    """Validate optional parameters."""
    if 'similarity_threshold' in data:
        threshold = data['similarity_threshold']
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValidationError("similarity_threshold must be a number between 0 and 1")
    
    if 'top_k' in data:
        top_k = data['top_k']
        if not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValidationError("top_k must be an integer between 1 and 100")
    
    if 'min_confidence' in data:
        confidence = data['min_confidence']
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValidationError("min_confidence must be a number between 0 and 1")

def log_security_event(event_type: str, details: Dict[str, Any], level: str = 'INFO'):
    """Log security-related events."""
    if not Config.AUDIT_LOG_ENABLED:
        return
    
    client_ip = get_client_ip()
    timestamp = datetime.now(timezone.utc).isoformat()
    log_message = f"SECURITY_EVENT: {event_type} | IP: {client_ip} | Time: {timestamp} | Details: {details}"
    
    if level == 'WARNING':
        audit_logger.warning(log_message)
    elif level == 'ERROR':
        audit_logger.error(log_message)
    else:
        audit_logger.info(log_message)

def require_api_key(f):
    """Decorator to require API key authentication."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not Config.API_KEY_REQUIRED:
            return f(*args, **kwargs)
        
        # Get API key from header or query parameter
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            log_security_event('AUTH_MISSING_API_KEY', {
                'endpoint': request.endpoint,
                'method': request.method
            }, 'WARNING')
            return jsonify({'error': 'API key required'}), 401
        
        if not validate_api_key(api_key):
            log_security_event('AUTH_INVALID_API_KEY', {
                'endpoint': request.endpoint,
                'method': request.method,
                'api_key_hash': hash_api_key(api_key)[:8] + '...'
            }, 'WARNING')
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Store API key info in request context
        g.api_key_hash = hash_api_key(api_key)[:8]
        
        return f(*args, **kwargs)
    
    return decorated_function

def rate_limit(limit: int = None, window: int = None, per: str = 'ip'):
    """Decorator to apply rate limiting."""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not Config.RATE_LIMIT_ENABLED:
                return f(*args, **kwargs)
            
            # Determine rate limit identifier
            if per == 'ip':
                identifier = f"rate_limit:ip:{get_client_ip()}"
            elif per == 'api_key':
                api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
                if api_key:
                    identifier = f"rate_limit:api_key:{hash_api_key(api_key)}"
                else:
                    identifier = f"rate_limit:ip:{get_client_ip()}"
            else:
                identifier = "rate_limit:global"
            
            is_allowed, rate_info = check_rate_limit(identifier, limit, window)
            
            if not is_allowed:
                log_security_event('RATE_LIMIT_EXCEEDED', {
                    'endpoint': request.endpoint,
                    'method': request.method,
                    'identifier_type': per,
                    'limit': rate_info.get('limit'),
                    'window': rate_info.get('window')
                }, 'WARNING')
                
                response = jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': rate_info.get('window', 60)
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(rate_info.get('window', 60))
                return response
            
            # Add rate limit headers to response
            response = f(*args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit'] = str(rate_info.get('limit', 'unknown'))
                response.headers['X-RateLimit-Remaining'] = str(rate_info.get('remaining', 'unknown'))
                response.headers['X-RateLimit-Reset'] = str(rate_info.get('reset_time', 'unknown'))
            
            return response
        
        return decorated_function
    return decorator

def validate_input(validation_type: str = 'image'):
    """Decorator to validate input data."""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                validation_info = {}
                
                # Validate file uploads
                if 'image' in request.files:
                    file_info = validate_file_upload(request.files['image'])
                    validation_info.update(file_info)
                
                # Validate JSON input
                if request.is_json and validation_type == 'image':
                    image_info = validate_image_input(request.json)
                    validation_info.update(image_info)
                
                # Store validation info in request context
                g.validation_info = validation_info
                
                return f(*args, **kwargs)
                
            except ValidationError as e:
                log_security_event('INPUT_VALIDATION_FAILED', {
                    'endpoint': request.endpoint,
                    'method': request.method,
                    'error': str(e)
                }, 'WARNING')
                return jsonify({'error': str(e)}), 400
            
        return decorated_function
    return decorator

def add_security_headers(response):
    """Add security headers to response."""
    if not Config.SECURITY_HEADERS_ENABLED:
        return response
    
    # Security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    
    # API-specific headers
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    
    return response

def audit_request(f):
    """Decorator to audit API requests."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        # Log request start
        log_security_event('API_REQUEST_START', {
            'endpoint': request.endpoint,
            'method': request.method,
            'content_type': request.content_type,
            'content_length': request.content_length,
            'api_key_hash': getattr(g, 'api_key_hash', None)
        })
        
        try:
            response = f(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Log successful request
            log_security_event('API_REQUEST_SUCCESS', {
                'endpoint': request.endpoint,
                'method': request.method,
                'status_code': getattr(response, 'status_code', 200),
                'processing_time': round(processing_time, 3),
                'api_key_hash': getattr(g, 'api_key_hash', None)
            })
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Log failed request
            log_security_event('API_REQUEST_ERROR', {
                'endpoint': request.endpoint,
                'method': request.method,
                'error': str(e),
                'processing_time': round(processing_time, 3),
                'api_key_hash': getattr(g, 'api_key_hash', None)
            }, 'ERROR')
            
            raise
    
    return decorated_function
