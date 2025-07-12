#!/bin/bash

# AI-Powered Interactive Learning Assistant - Enhanced Startup Script
# This script sets up and launches the complete enhanced learning assistant system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE} $1 ${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

# Main startup function
main() {
    print_header "🎓 AI-Powered Interactive Learning Assistant - Enhanced Edition"
    
    # Check prerequisites
    check_prerequisites
    
    # Set up environment
    setup_environment
    
    # Install dependencies
    install_dependencies
    
    # Download and optimize models
    setup_models
    
    # Start the system
    start_system
}

check_prerequisites() {
    print_header "🔍 Checking Prerequisites"
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python version: $python_version"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed. Please install pip."
        exit 1
    fi
    
    # Check available space
    available_space=$(df . | tail -1 | awk '{print $4}')
    required_space=10485760  # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        print_warning "Low disk space. At least 10GB recommended for models and cache."
    fi
    
    # Check RAM
    if command -v free &> /dev/null; then
        ram_gb=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$ram_gb" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Performance may be impacted."
        fi
        print_status "Available RAM: ${ram_gb}GB"
    fi
    
    print_success "Prerequisites check completed"
}

setup_environment() {
    print_header "🛠️  Setting Up Environment"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Environment setup completed"
}

install_dependencies() {
    print_header "📦 Installing Dependencies"
    
    print_status "Installing Python packages..."
    pip install -r requirements.txt
    
    # Install additional OpenVINO optimizations if available
    if pip show openvino-dev &> /dev/null; then
        print_status "OpenVINO development tools detected"
    else
        print_status "Installing OpenVINO development tools..."
        pip install openvino-dev
    fi
    
    print_success "Dependencies installed successfully"
}

setup_models() {
    print_header "🤖 Setting Up AI Models"
    
    if [ -f "scripts/setup_models.py" ]; then
        print_status "Downloading and optimizing AI models..."
        python scripts/setup_models.py
        print_success "Models setup completed"
    else
        print_warning "Model setup script not found. Models will be downloaded on first use."
    fi
    
    # Create necessary directories
    mkdir -p models logs data/samples
    print_status "Created necessary directories"
}

start_system() {
    print_header "🚀 Starting the Enhanced Learning Assistant"
    
    echo "Choose startup option:"
    echo "1) Full System (API + Streamlit UI)"
    echo "2) API Server Only"
    echo "3) Enhanced Demo"
    echo "4) Quick Demo"
    echo "5) Development Mode"
    echo ""
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            start_full_system
            ;;
        2)
            start_api_only
            ;;
        3)
            run_enhanced_demo
            ;;
        4)
            run_quick_demo
            ;;
        5)
            start_development_mode
            ;;
        *)
            print_error "Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
}

start_full_system() {
    print_status "Starting full system with API and UI..."
    
    # Start API server in background
    print_status "Starting API server on port 8000..."
    cd src/api
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    API_PID=$!
    cd ../..
    
    # Wait for API to start
    sleep 5
    
    # Check if API is running
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "API server started successfully"
    else
        print_error "Failed to start API server"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
    
    # Start Streamlit UI
    print_status "Starting Streamlit UI on port 8501..."
    cd src/ui
    streamlit run app.py --server.port 8501 --server.headless true &
    UI_PID=$!
    cd ../..
    
    print_success "🎉 System started successfully!"
    echo ""
    echo "Access the application at:"
    echo "  🌐 Streamlit UI: http://localhost:8501"
    echo "  🔗 API Docs: http://localhost:8000/docs"
    echo "  📊 API Health: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop the system"
    
    # Wait for user interrupt
    trap 'print_status "Stopping services..."; kill $API_PID $UI_PID 2>/dev/null || true; exit 0' INT
    wait
}

start_api_only() {
    print_status "Starting API server only..."
    cd src/api
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
}

run_enhanced_demo() {
    print_status "Running enhanced demo..."
    python enhanced_demo.py
}

run_quick_demo() {
    print_status "Running quick demo..."
    python enhanced_demo.py --quick
}

start_development_mode() {
    print_status "Starting development mode..."
    
    echo "Development tools available:"
    echo "1) Run tests: pytest"
    echo "2) Code formatting: black src/"
    echo "3) Linting: flake8 src/"
    echo "4) Type checking: mypy src/"
    echo "5) Coverage report: pytest --cov=src"
    echo ""
    
    # Install development dependencies
    pip install pytest pytest-cov black flake8 mypy
    
    print_success "Development environment ready!"
    echo "Virtual environment activated. You can now run development commands."
    
    # Keep the shell active
    exec bash
}

# Health check function
health_check() {
    print_status "Performing health check..."
    
    # Check API
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "API server is healthy"
    else
        print_warning "API server is not responding"
    fi
    
    # Check UI
    if curl -s http://localhost:8501 > /dev/null; then
        print_success "Streamlit UI is accessible"
    else
        print_warning "Streamlit UI is not accessible"
    fi
    
    # Check models
    if [ -d "models" ] && [ "$(ls -A models)" ]; then
        print_success "AI models are available"
    else
        print_warning "AI models not found"
    fi
    
    # Check disk space
    available_space=$(df . | tail -1 | awk '{print $4}')
    available_gb=$((available_space / 1024 / 1024))
    
    if [ "$available_gb" -gt 5 ]; then
        print_success "Sufficient disk space: ${available_gb}GB"
    else
        print_warning "Low disk space: ${available_gb}GB"
    fi
}

# Handle command line arguments
case "${1:-}" in
    "health")
        health_check
        ;;
    "demo")
        source venv/bin/activate 2>/dev/null || true
        run_enhanced_demo
        ;;
    "quick-demo")
        source venv/bin/activate 2>/dev/null || true
        run_quick_demo
        ;;
    "api")
        source venv/bin/activate 2>/dev/null || true
        start_api_only
        ;;
    "dev")
        source venv/bin/activate 2>/dev/null || true
        start_development_mode
        ;;
    "help"|"-h"|"--help")
        echo "AI-Powered Interactive Learning Assistant - Enhanced Edition"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)    - Interactive setup and startup"
        echo "  health       - Perform system health check"
        echo "  demo         - Run enhanced demo"
        echo "  quick-demo   - Run quick demo"
        echo "  api          - Start API server only"
        echo "  dev          - Start development mode"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0           # Interactive startup"
        echo "  $0 demo      # Run demo"
        echo "  $0 api       # API only"
        echo "  $0 health    # Health check"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        print_status "Use '$0 help' for usage information"
        exit 1
        ;;
esac
