"""
Liveness Detection Configuration Parameters
Centralized configuration for all liveness detection settings to enable easy management
"""

import os

# =============================================================================
# ULTRA-FAST PERFORMANCE CONFIGURATION
# =============================================================================

# History Management (for ultra-fast processing)
FRAME_HISTORY_SIZE = 3  # Keep only last 3 frames for speed
LANDMARK_HISTORY_SIZE = 3  # Minimal landmark history
TEXTURE_HISTORY_SIZE = 3  # Minimal texture history  
MOTION_HISTORY_SIZE = 3  # Minimal motion history

# =============================================================================
# DETECTION THRESHOLDS (ULTRA-AGGRESSIVE for instant detection)
# =============================================================================

# Blink Detection
BLINK_THRESHOLD = 0.15  # Extremely sensitive for instant blink detection

# Head Movement Detection  
HEAD_MOVEMENT_THRESHOLD = 0.5  # Ultra-low threshold for immediate detection

# Texture Analysis
TEXTURE_VARIANCE_THRESHOLD = 10.0  # Very lenient for fastest processing

# =============================================================================
# CHALLENGE CONFIGURATION (MINIMAL for maximum speed)
# =============================================================================

# Challenge Counts
MIN_CHALLENGES = 1  # Single challenge for fastest completion
MAX_CHALLENGES = 1  # Only one challenge for maximum speed

# Challenge Timing
CHALLENGE_TIMEOUT = 15.0  # Reasonable timeout for user actions
DEFAULT_CHALLENGE_TIMEOUT = 15.0  # Reasonable timeout for user actions
MAX_ATTEMPTS_PER_CHALLENGE = 2  # Allow a couple of attempts

# Session Configuration
EXPECTED_DURATION_PER_CHALLENGE = 5.0  # 5 seconds per challenge (for session planning)

# =============================================================================
# FEATURE TOGGLES (DISABLED for maximum speed)
# =============================================================================

# Performance vs Security Trade-offs
INTEGRITY_CHECKS_ENABLED = False  # Disable for speed
TAMPERING_DETECTION_ENABLED = False  # Disable for speed  
DEPTH_ANALYSIS_ENABLED = False  # Disable for speed
TEMPORAL_ANALYSIS_ENABLED = False  # Disable for speed

# =============================================================================
# SECURITY AND VALIDATION SETTINGS
# =============================================================================

# Token Validation
MAX_INTEGRITY_TOKEN_AGE = 300.0  # Maximum age for integrity tokens (seconds)

# =============================================================================
# TEXTURE AND IMAGE ANALYSIS PARAMETERS
# =============================================================================

# Local Binary Pattern (LBP) Configuration
LBP_RADIUS = 3  # Radius for LBP calculation
LBP_N_POINTS = 24  # Number of points for LBP calculation

# Gabor Filter Configuration
GABOR_KERNEL_SIZE = (21, 21)  # Gabor kernel size
GABOR_SIGMA = 5  # Gabor sigma parameter
GABOR_PSI = 0.5  # Gabor psi parameter
GABOR_ORIENTATIONS = [0, 45, 90, 135]  # Gabor filter orientations (degrees)

# Edge Detection
CANNY_LOW_THRESHOLD = 50  # Canny edge detection low threshold
CANNY_HIGH_THRESHOLD = 150  # Canny edge detection high threshold

# =============================================================================
# TAMPERING DETECTION THRESHOLDS
# =============================================================================

# Compression Artifacts
COMPRESSION_LAPLACIAN_THRESHOLD = 10  # Low variance indicates artificial processing

# Edge Density Analysis
EDGE_DENSITY_LOW_THRESHOLD = 0.005  # Minimum expected edge density
EDGE_DENSITY_HIGH_THRESHOLD = 0.5  # Maximum expected edge density

# Lighting Analysis
HISTOGRAM_ENTROPY_THRESHOLD = 3.0  # Minimum entropy for natural lighting

# =============================================================================
# DEPTH ANALYSIS CONFIGURATION
# =============================================================================

# Facial Landmark Indices (68-point model)
NOSE_TIP_LANDMARK_INDEX = 30  # Nose tip landmark
LEFT_EYE_LANDMARK_INDEX = 36  # Left eye landmark
RIGHT_EYE_LANDMARK_INDEX = 45  # Right eye landmark

# Depth Calculation
DEPTH_DISTANCE_SCALE = 0.01  # Scale factor for distance-to-depth conversion
DEPTH_CIRCLE_RADIUS = 5  # Radius for depth map circles
DEPTH_GAUSSIAN_BLUR_SIZE = (15, 15)  # Gaussian blur kernel size for depth smoothing

# =============================================================================
# TEMPORAL ANALYSIS CONFIGURATION
# =============================================================================

# Movement Analysis
MINIMUM_FRAMES_FOR_TEMPORAL_ANALYSIS = 3  # Minimum frames needed for temporal analysis
MOVEMENT_SMOOTHNESS_THRESHOLD = 0.3  # Threshold for movement smoothness
ACCELERATION_THRESHOLD = 0.2  # Threshold for acceleration analysis

# =============================================================================
# CHALLENGE TYPE CONFIGURATION
# =============================================================================

# Available Challenge Types (optimized for speed)
AVAILABLE_CHALLENGE_TYPES = ['blink']  # Only blink for ultra-fast detection
DEFAULT_CHALLENGE_TYPE = 'blink'  # Default challenge for maximum speed

# Challenge Generation
CHALLENGE_ID_RANGE = (1000, 9999)  # Range for random challenge IDs

# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

# Debug Logging
ENABLE_PERFORMANCE_LOGGING = True  # Enable performance metrics logging
ENABLE_DEBUG_TEXTURE_ANALYSIS = False  # Disable for speed
ENABLE_DEBUG_TEMPORAL_ANALYSIS = False  # Disable for speed

# =============================================================================
# MATHEMATICAL CONSTANTS AND SAFETY VALUES
# =============================================================================

# Numerical Stability
EPSILON = 1e-10  # Small value to prevent division by zero
MAX_UINT8_VALUE = 255  # Maximum value for uint8 data type

# Array Processing
HIGH_FREQ_ANALYSIS_MARGIN = 4  # Margin for high-frequency analysis (quarters)

# =============================================================================
# ULTRA-FAST MODE OVERRIDES
# =============================================================================

class UltraFastConfig:
    """
    Configuration class for ultra-fast liveness detection mode
    Automatically sets all parameters for maximum speed
    """
    
    # Override all performance-critical settings
    FRAME_HISTORY_SIZE = 2  # Even smaller history for ultra-fast
    LANDMARK_HISTORY_SIZE = 2
    TEXTURE_HISTORY_SIZE = 2
    MOTION_HISTORY_SIZE = 2
    
    # Extremely aggressive thresholds
    BLINK_THRESHOLD = 0.1  # Even more sensitive
    HEAD_MOVEMENT_THRESHOLD = 0.3  # Lower threshold
    TEXTURE_VARIANCE_THRESHOLD = 5.0  # More lenient
    
    # Minimal challenges
    MIN_CHALLENGES = 1
    MAX_CHALLENGES = 1
    CHALLENGE_TIMEOUT = 12.0  # Give users enough time
    
    # Disable all heavy processing
    INTEGRITY_CHECKS_ENABLED = False
    TAMPERING_DETECTION_ENABLED = False
    DEPTH_ANALYSIS_ENABLED = False
    TEMPORAL_ANALYSIS_ENABLED = False
    ENABLE_DEBUG_TEXTURE_ANALYSIS = False
    ENABLE_DEBUG_TEMPORAL_ANALYSIS = False

# =============================================================================
# PRODUCTION MODE CONFIGURATION
# =============================================================================

class ProductionConfig:
    """
    Configuration class for production mode with balanced performance and security
    """
    
    # Balanced performance settings
    FRAME_HISTORY_SIZE = 5
    LANDMARK_HISTORY_SIZE = 5
    TEXTURE_HISTORY_SIZE = 5
    MOTION_HISTORY_SIZE = 5
    
    # More conservative thresholds
    BLINK_THRESHOLD = 0.2
    HEAD_MOVEMENT_THRESHOLD = 0.8
    TEXTURE_VARIANCE_THRESHOLD = 15.0
    
    # Multiple challenges for security
    MIN_CHALLENGES = 2
    MAX_CHALLENGES = 3
    CHALLENGE_TIMEOUT = 10.0
    
    # Enable security features
    INTEGRITY_CHECKS_ENABLED = True
    TAMPERING_DETECTION_ENABLED = True
    DEPTH_ANALYSIS_ENABLED = True
    TEMPORAL_ANALYSIS_ENABLED = True

# =============================================================================
# CONFIGURATION SELECTION
# =============================================================================

# Select active configuration mode
ACTIVE_CONFIG_MODE = "custom"  # Options: "ultra_fast", "production", "custom"

class CustomConfig:
    """Custom configuration using global constants"""
    
    # Core Parameters
    FRAME_HISTORY_SIZE = FRAME_HISTORY_SIZE
    LANDMARK_HISTORY_SIZE = LANDMARK_HISTORY_SIZE
    TEXTURE_HISTORY_SIZE = TEXTURE_HISTORY_SIZE
    MOTION_HISTORY_SIZE = MOTION_HISTORY_SIZE
    
    # Detection Thresholds
    BLINK_THRESHOLD = BLINK_THRESHOLD
    HEAD_MOVEMENT_THRESHOLD = HEAD_MOVEMENT_THRESHOLD
    TEXTURE_VARIANCE_THRESHOLD = TEXTURE_VARIANCE_THRESHOLD
    
    # Challenge Configuration
    MIN_CHALLENGES = MIN_CHALLENGES
    MAX_CHALLENGES = MAX_CHALLENGES
    CHALLENGE_TIMEOUT = CHALLENGE_TIMEOUT
    
    # Feature Toggles
    INTEGRITY_CHECKS_ENABLED = INTEGRITY_CHECKS_ENABLED
    TAMPERING_DETECTION_ENABLED = TAMPERING_DETECTION_ENABLED
    DEPTH_ANALYSIS_ENABLED = DEPTH_ANALYSIS_ENABLED
    TEMPORAL_ANALYSIS_ENABLED = TEMPORAL_ANALYSIS_ENABLED

def get_active_config():
    """
    Get the active configuration based on the selected mode
    """
    if ACTIVE_CONFIG_MODE == "ultra_fast":
        return UltraFastConfig()
    elif ACTIVE_CONFIG_MODE == "production":
        return ProductionConfig()
    else:
        # Return custom configuration
        return CustomConfig()

def load_config_from_file(config_file: str = None):
    """
    Load configuration from external file if provided
    """
    if config_file and os.path.exists(config_file):
        # Implementation for loading from external file
        # Could support JSON, YAML, or other formats
        pass
    return get_active_config()
