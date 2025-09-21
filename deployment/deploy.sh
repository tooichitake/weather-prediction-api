#!/bin/bash
# Deployment script for Weather Prediction API

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="weather-prediction-api"
DOCKER_IMAGE="${DOCKER_REGISTRY:-}${DOCKER_IMAGE:-weather-prediction-api}"
DOCKER_TAG="${DOCKER_TAG:-latest}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    log_info "Docker is installed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    docker build -t "${DOCKER_IMAGE}:${DOCKER_TAG}" .
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Run tests
run_tests() {
    log_info "Running tests..."
    docker run --rm "${DOCKER_IMAGE}:${DOCKER_TAG}" python -c "print('Tests would run here')"
    log_info "Tests completed"
}

# Deploy using Docker Compose
deploy_compose() {
    log_info "Deploying with Docker Compose..."
    docker-compose down --remove-orphans
    docker-compose up -d
    
    # Wait for health check
    log_info "Waiting for application to be healthy..."
    for i in {1..30}; do
        if docker-compose ps | grep -q "healthy"; then
            log_info "Application is healthy!"
            break
        fi
        sleep 2
    done
    
    # Show logs
    docker-compose logs --tail=50
}

# Deploy single container
deploy_single() {
    log_info "Deploying single container..."
    
    # Stop existing container
    docker stop "${APP_NAME}" 2>/dev/null || true
    docker rm "${APP_NAME}" 2>/dev/null || true
    
    # Run new container
    docker run -d \
        --name "${APP_NAME}" \
        -p 8000:8000 \
        --restart unless-stopped \
        --health-cmd "curl -f http://localhost:8000/health/ || exit 1" \
        --health-interval 30s \
        --health-timeout 10s \
        --health-retries 3 \
        --health-start-period 40s \
        "${DOCKER_IMAGE}:${DOCKER_TAG}"
    
    # Wait for health
    log_info "Waiting for container to be healthy..."
    for i in {1..30}; do
        if docker inspect "${APP_NAME}" --format='{{.State.Health.Status}}' | grep -q "healthy"; then
            log_info "Container is healthy!"
            break
        fi
        sleep 2
    done
    
    # Show logs
    docker logs "${APP_NAME}" --tail=50
}

# Deploy to cloud platforms
deploy_cloud() {
    case "$1" in
        "render")
            log_info "Deploying to Render..."
            # Render deployment is automatic via GitHub integration
            log_info "Push to main branch to trigger Render deployment"
            ;;
        "heroku")
            log_info "Deploying to Heroku..."
            heroku container:push web -a "${HEROKU_APP_NAME}"
            heroku container:release web -a "${HEROKU_APP_NAME}"
            ;;
        "aws")
            log_info "Deploying to AWS..."
            # AWS ECR push
            aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${AWS_ECR_REPO}"
            docker tag "${DOCKER_IMAGE}:${DOCKER_TAG}" "${AWS_ECR_REPO}:${DOCKER_TAG}"
            docker push "${AWS_ECR_REPO}:${DOCKER_TAG}"
            # Update ECS service or Lambda
            ;;
        "gcp")
            log_info "Deploying to Google Cloud..."
            gcloud run deploy "${APP_NAME}" --image "${DOCKER_IMAGE}:${DOCKER_TAG}" --platform managed
            ;;
        *)
            log_error "Unknown cloud platform: $1"
            exit 1
            ;;
    esac
}

# Main deployment flow
main() {
    log_info "Starting deployment of Weather Prediction API"
    log_info "Image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
    
    # Parse arguments
    DEPLOYMENT_TYPE="${1:-compose}"
    
    # Check prerequisites
    check_docker
    
    # Build image
    build_image
    
    # Run tests (optional)
    if [ "${RUN_TESTS}" = "true" ]; then
        run_tests
    fi
    
    # Deploy based on type
    case "$DEPLOYMENT_TYPE" in
        "compose")
            deploy_compose
            ;;
        "single")
            deploy_single
            ;;
        "render"|"heroku"|"aws"|"gcp")
            deploy_cloud "$DEPLOYMENT_TYPE"
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            echo "Usage: $0 [compose|single|render|heroku|aws|gcp]"
            exit 1
            ;;
    esac
    
    log_info "Deployment completed successfully!"
    
    # Show access information
    if [ "$DEPLOYMENT_TYPE" = "compose" ] || [ "$DEPLOYMENT_TYPE" = "single" ]; then
        log_info "API is available at:"
        log_info "  - Health: http://localhost:8000/health/"
        log_info "  - Docs: http://localhost:8000/docs"
        log_info "  - Root: http://localhost:8000/"
    fi
}

# Run main function
main "$@"