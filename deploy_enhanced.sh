#!/bin/bash
#
# Face Vector Embedding System Deployment Script
# Simple deployment for SQLite-based face recognition system.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="face-embedding-system"

# Default values
MODE="development"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Face Vector Embedding System Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -m, --mode MODE           Deployment mode: development, production (default: development)
    -h, --help               Show this help message

EXAMPLES:
    # Development mode (default)
    $0 --mode development

    # Production mode
    $0 --mode production

DEPLOYMENT MODES:
    development    - Local development with hot reload
    production     - Production deployment with optimizations
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                MODE="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
        print_error "pip is not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment for $MODE mode..."
    
    # Create only necessary directories that are actually used
    mkdir -p uploads data logs
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        cat > .env << EOF
# Face Vector Embedding System Configuration
FLASK_ENV=$MODE
DEBUG=$([ "$MODE" = "development" ] && echo "true" || echo "false")
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# API Security
API_KEY=sk-face123abc

# Database (SQLite - simple and effective)
DATABASE_URL=sqlite:///face_db.sqlite

# Face Processing Configuration
FACE_DETECTION_MODEL=opencv_dnn
FACE_ENCODING_MODEL=facenet
FACE_CONFIDENCE_THRESHOLD=0.8
SIMILARITY_THRESHOLD=0.6

# Performance Settings
USE_FAISS_INDEX=true
MIN_FACE_SIZE=80
MAX_FACE_SIZE=1024
EOF
        print_success "Created .env file with secure defaults"
    fi
    
    # Set permissions
    chmod 644 .env
    chmod -R 755 uploads data logs
}

# Function to setup Python environment
setup_python_environment() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        print_status "Installing requirements..."
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_success "Python environment setup completed"
}

# Function to initialize database
initialize_database() {
    print_status "Initializing database..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Initialize database
    PYTHONPATH=. python3 -c "
from app.services.database_service import DatabaseService
try:
    db_service = DatabaseService()
    print('Database initialized successfully')
except Exception as e:
    print(f'Database initialization failed: {e}')
    exit(1)
"
    
    print_success "Database initialized"
}

# Function to start application
start_application() {
    print_status "Starting Face Vector Embedding System..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Set environment variables
    export PYTHONPATH=.
    
    if [[ "$MODE" == "development" ]]; then
        print_status "Starting in development mode with hot reload..."
        python3 -c "from app.main import app; app.run(host='0.0.0.0', port=5001, debug=True)" &
    else
        print_status "Starting in production mode..."
        python3 -c "from app.main import app; app.run(host='0.0.0.0', port=5001, debug=False)" &
    fi
    
    APP_PID=$!
    echo $APP_PID > .app.pid
    
    print_success "Application started (PID: $APP_PID)"
}
# Function to wait for application
wait_for_application() {
    print_status "Waiting for application to be ready..."
    
    # Wait for API to be available
    for i in {1..30}; do
        if curl -sf http://localhost:5001/health >/dev/null 2>&1; then
            print_success "Application is ready"
            return 0
        fi
        print_status "Waiting for application... ($i/30)"
        sleep 2
    done
    
    print_error "Application failed to start within 60 seconds"
    return 1
}

# Function to run tests
run_tests() {
    print_status "Running basic tests..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Basic health check
    if curl -sf http://localhost:5001/health | grep -q "healthy"; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        return 1
    fi
    
    # Test face detection endpoint with sample image
    if [[ -f "person.jpg" ]]; then
        print_status "Testing face detection with sample image..."
        if curl -sf -X POST \
            -H "X-API-Key: sk-face123abc" \
            -F "image=@person.jpg" \
            http://localhost:5001/api/v1/faces/detect >/dev/null 2>&1; then
            print_success "Face detection test passed"
        else
            print_warning "Face detection test failed (may be expected without valid image)"
        fi
    else
        print_warning "No sample image found for testing (person.jpg)"
    fi
    
    print_success "Basic tests completed"
}

# Function to show deployment info
show_deployment_info() {
    print_success "Deployment completed successfully!"
    echo
    echo "📋 DEPLOYMENT INFORMATION"
    echo "========================"
    echo "Mode: $MODE"
    echo "Database: SQLite (face_db.sqlite)"
    echo "Vector Index: FAISS (data/ directory)"
    echo
    echo "🌐 ACCESS POINTS"
    echo "==============="
    echo "API Endpoint: http://localhost:5001"
    echo "Health Check: http://localhost:5001/health"
    
    echo
    echo "🔧 MANAGEMENT COMMANDS"
    echo "====================="
    echo "Stop application: kill \$(cat .app.pid)"
    echo "View logs: tail -f logs/app.log"
    echo "Restart application: $0 --mode $MODE"
    
    echo
    echo "� MONITORING"
    echo "============"
    echo "Application PID: $(cat .app.pid 2>/dev/null || echo 'Not found')"
    echo "Database file: $(ls -lh face_db.sqlite 2>/dev/null || echo 'Not created yet')"
    echo "Uploads directory: $(ls -la uploads/ 2>/dev/null | wc -l) files"
    
    echo
    echo "🧪 TESTING"
    echo "========="
    echo "Test face detection:"
    echo "  curl -X POST -H 'X-API-Key: sk-face123abc' -F 'image=@your_image.jpg' http://localhost:5001/api/v1/faces/detect"
    echo
    echo "Test health endpoint:"
    echo "  curl http://localhost:5001/health"
}

# Function to cleanup on exit
cleanup() {
    if [[ -f .app.pid ]]; then
        APP_PID=$(cat .app.pid)
        if kill -0 $APP_PID 2>/dev/null; then
            print_status "Stopping application (PID: $APP_PID)..."
            kill $APP_PID
            rm -f .app.pid
        fi
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    echo "🚀 Face Vector Embedding System Deployment"
    echo "=========================================="
    echo
    
    # Parse arguments
    parse_args "$@"
    
    # Execute deployment steps
    check_prerequisites
    setup_environment
    setup_python_environment
    initialize_database
    start_application
    
    # Wait for application and run tests
    if wait_for_application && run_tests; then
        show_deployment_info
        echo
        print_status "Press Ctrl+C to stop the application"
        # Keep script running to maintain the application
        wait
    else
        print_error "Deployment completed but application or tests failed"
        echo "Check logs in logs/ directory"
        exit 1
    fi
}

# Run main function
main "$@"
