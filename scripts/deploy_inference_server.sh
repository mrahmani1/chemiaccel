#!/bin/bash
# Deployment script for ChemiAccel Inference Server
# Supports Docker and Kubernetes deployments with monitoring

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-docker}"  # docker, k8s, local

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_warning "Docker Compose not found, using 'docker compose'"
    fi
    
    # Check NVIDIA Docker runtime (for GPU support)
    if docker info 2>/dev/null | grep -q nvidia; then
        log_success "NVIDIA Docker runtime detected"
        GPU_AVAILABLE=true
    else
        log_warning "NVIDIA Docker runtime not available, using CPU-only mode"
        GPU_AVAILABLE=false
    fi
    
    # Check Kubernetes (if k8s deployment)
    if [ "$DEPLOYMENT_TYPE" = "k8s" ]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is not installed"
            exit 1
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to Kubernetes cluster"
            exit 1
        fi
        
        log_success "Kubernetes cluster connection verified"
    fi
    
    log_success "Prerequisites check completed"
}

# Function to build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build GPU image
    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "Building GPU-enabled image..."
        docker build \
            -f docker/Dockerfile.inference \
            --target production \
            -t "${DOCKER_REGISTRY}/chemi-accel/inference:gpu-${IMAGE_TAG}" \
            .
        
        if [ "$DOCKER_REGISTRY" != "localhost:5000" ]; then
            docker push "${DOCKER_REGISTRY}/chemi-accel/inference:gpu-${IMAGE_TAG}"
        fi
    fi
    
    # Build CPU image
    log_info "Building CPU-only image..."
    docker build \
        -f docker/Dockerfile.inference \
        --target cpu-only \
        -t "${DOCKER_REGISTRY}/chemi-accel/inference:cpu-${IMAGE_TAG}" \
        .
    
    if [ "$DOCKER_REGISTRY" != "localhost:5000" ]; then
        docker push "${DOCKER_REGISTRY}/chemi-accel/inference:cpu-${IMAGE_TAG}"
    fi
    
    log_success "Docker images built successfully"
}

# Function to deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Determine which profiles to use
    PROFILES=""
    
    if [ "$GPU_AVAILABLE" = true ]; then
        log_info "GPU available, deploying GPU-enabled service"
    else
        log_info "No GPU available, deploying CPU-only service"
        PROFILES="--profile cpu-only"
    fi
    
    # Add monitoring if requested
    if [ "${ENABLE_MONITORING:-false}" = "true" ]; then
        PROFILES="$PROFILES --profile monitoring"
        log_info "Enabling monitoring stack"
    fi
    
    # Add load balancer if requested
    if [ "${ENABLE_LOAD_BALANCER:-false}" = "true" ]; then
        PROFILES="$PROFILES --profile load-balancer"
        log_info "Enabling load balancer"
    fi
    
    # Stop existing services
    docker-compose -f docker/docker-compose.inference.yml down || true
    
    # Start services
    docker-compose -f docker/docker-compose.inference.yml up -d $PROFILES
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check health
    if check_service_health "localhost:8000"; then
        log_success "Docker deployment completed successfully"
    else
        log_error "Service health check failed"
        return 1
    fi
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Apply namespace and configurations
    kubectl apply -f k8s/inference-deployment.yaml
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/chemi-accel-inference-gpu -n chemi-accel || true
    kubectl wait --for=condition=available --timeout=300s deployment/chemi-accel-inference-cpu -n chemi-accel || true
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service chemi-accel-inference-service -n chemi-accel -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [ "$SERVICE_IP" = "pending" ] || [ -z "$SERVICE_IP" ]; then
        log_info "Load balancer IP pending, using port-forward for testing"
        kubectl port-forward -n chemi-accel service/chemi-accel-inference-service 8000:80 &
        PORT_FORWARD_PID=$!
        sleep 5
        SERVICE_URL="localhost:8000"
    else
        SERVICE_URL="$SERVICE_IP"
        log_info "Service available at: $SERVICE_URL"
    fi
    
    # Check health
    if check_service_health "$SERVICE_URL"; then
        log_success "Kubernetes deployment completed successfully"
    else
        log_error "Service health check failed"
        return 1
    fi
    
    # Clean up port-forward if used
    if [ -n "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
}

# Function to deploy locally (development)
deploy_local() {
    log_info "Starting local development server..."
    
    cd "$PROJECT_ROOT"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    pip install -e .[inference]
    
    # Set environment variables
    export CHEMI_ACCEL_ENV=development
    export CHEMI_ACCEL_LOG_LEVEL=DEBUG
    
    # Start server
    log_info "Starting inference server on localhost:8000"
    python -m chemi_accel.inference.server \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level DEBUG \
        --max-batch-size 100 &
    
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 10
    
    # Check health
    if check_service_health "localhost:8000"; then
        log_success "Local development server started successfully (PID: $SERVER_PID)"
        echo "Server is running at http://localhost:8000"
        echo "API documentation: http://localhost:8000/docs"
        echo "To stop: kill $SERVER_PID"
    else
        log_error "Local server health check failed"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi
}

# Function to check service health
check_service_health() {
    local service_url="$1"
    local max_attempts=10
    local attempt=1
    
    log_info "Checking service health at $service_url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://$service_url/health" > /dev/null 2>&1; then
            log_success "Service is healthy"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts failed, retrying in 5 seconds..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    log_error "Service health check failed after $max_attempts attempts"
    return 1
}

# Function to run load test
run_load_test() {
    local service_url="${1:-localhost:8000}"
    
    log_info "Running load test against $service_url..."
    
    cd "$PROJECT_ROOT"
    
    python -m pytest tests/load/test_inference_performance.py \
        --server "$service_url" \
        --users 5 \
        --duration 60 \
        -v
    
    log_success "Load test completed"
}

# Function to show status
show_status() {
    log_info "Checking deployment status..."
    
    case "$DEPLOYMENT_TYPE" in
        "docker")
            docker-compose -f docker/docker-compose.inference.yml ps
            ;;
        "k8s")
            kubectl get pods -n chemi-accel
            kubectl get services -n chemi-accel
            ;;
        "local")
            ps aux | grep "chemi_accel.inference.server" | grep -v grep || echo "No local server running"
            ;;
    esac
}

# Function to clean up deployment
cleanup() {
    log_info "Cleaning up deployment..."
    
    case "$DEPLOYMENT_TYPE" in
        "docker")
            docker-compose -f docker/docker-compose.inference.yml down -v
            docker system prune -f
            ;;
        "k8s")
            kubectl delete -f k8s/inference-deployment.yaml --ignore-not-found
            ;;
        "local")
            pkill -f "chemi_accel.inference.server" || true
            ;;
    esac
    
    log_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy      Deploy the inference server
    build       Build Docker images only
    test        Run load test
    status      Show deployment status
    cleanup     Clean up deployment
    help        Show this help message

Options:
    --type TYPE         Deployment type: docker, k8s, local (default: docker)
    --tag TAG           Docker image tag (default: latest)
    --registry URL      Docker registry URL (default: localhost:5000)
    --env ENV           Environment: development, staging, production (default: production)
    --gpu               Force GPU deployment (auto-detected by default)
    --no-gpu            Force CPU-only deployment
    --monitoring        Enable monitoring stack (Prometheus, Grafana)
    --load-balancer     Enable load balancer (Traefik/NGINX)

Environment Variables:
    DEPLOYMENT_TYPE     Deployment type (docker, k8s, local)
    IMAGE_TAG           Docker image tag
    DOCKER_REGISTRY     Docker registry URL
    ENVIRONMENT         Environment name
    ENABLE_MONITORING   Enable monitoring (true/false)
    ENABLE_LOAD_BALANCER Enable load balancer (true/false)

Examples:
    # Deploy with Docker Compose (GPU + monitoring)
    $0 deploy --type docker --monitoring

    # Deploy to Kubernetes
    $0 deploy --type k8s --tag v1.0.0

    # Start local development server
    $0 deploy --type local

    # Build images only
    $0 build --tag latest

    # Run load test
    $0 test

    # Check status
    $0 status

    # Clean up
    $0 cleanup
EOF
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        deploy|build|test|status|cleanup|help)
            COMMAND="$1"
            shift
            ;;
        --type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            DOCKER_REGISTRY="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --gpu)
            GPU_AVAILABLE=true
            shift
            ;;
        --no-gpu)
            GPU_AVAILABLE=false
            shift
            ;;
        --monitoring)
            ENABLE_MONITORING=true
            shift
            ;;
        --load-balancer)
            ENABLE_LOAD_BALANCER=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Default command
if [ -z "$COMMAND" ]; then
    COMMAND="deploy"
fi

# Main execution
case "$COMMAND" in
    "deploy")
        check_prerequisites
        build_images
        
        case "$DEPLOYMENT_TYPE" in
            "docker")
                deploy_docker
                ;;
            "k8s")
                deploy_kubernetes
                ;;
            "local")
                deploy_local
                ;;
            *)
                log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
                exit 1
                ;;
        esac
        ;;
    
    "build")
        check_prerequisites
        build_images
        ;;
    
    "test")
        run_load_test
        ;;
    
    "status")
        show_status
        ;;
    
    "cleanup")
        cleanup
        ;;
    
    "help")
        show_usage
        ;;
    
    *)
        log_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

log_success "Script completed successfully"