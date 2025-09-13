# Weather Prediction API - Deployment Guide

This guide provides comprehensive instructions for deploying the Weather Prediction API to various platforms and environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Checklist](#production-checklist)
6. [Monitoring & Logging](#monitoring--logging)
7. [Troubleshooting](#troubleshooting)
8. [Security Considerations](#security-considerations)

## Prerequisites

### System Requirements
- Python 3.11+ (3.12 recommended)
- Docker 20.10+ and Docker Compose 2.0+
- 2GB RAM minimum (4GB recommended)
- 1GB free disk space

### Required Files
- All model files in `models/` directory
- `requirements.txt` with all dependencies
- Environment configuration files

## Local Development

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/weather-prediction-api.git
cd weather-prediction-api
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Locally
```bash
# Development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 4. Verify Installation
```bash
# Health check
curl http://localhost:8000/health/

# API documentation
open http://localhost:8000/docs
```

## Docker Deployment

### 1. Build Docker Image

```bash
# Build with Docker
docker build -t weather-prediction-api:latest .

# Build with Docker Compose
docker-compose build
```

### 2. Run with Docker

#### Single Container
```bash
docker run -d \
  --name weather-api \
  -p 8000:8000 \
  --restart unless-stopped \
  weather-prediction-api:latest
```

#### Docker Compose (Recommended)
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Docker Compose Configuration

The `docker-compose.yml` file includes:
- Health checks
- Resource limits
- Volume mounts for logs
- Network configuration
- Optional Nginx reverse proxy

## Cloud Deployment

### Render.com (Recommended for Simplicity)

1. **Connect GitHub Repository**
   - Log in to [Render](https://render.com)
   - Create New > Web Service
   - Connect your GitHub repository

2. **Configure Service**
   - Name: `weather-prediction-api`
   - Runtime: Docker
   - Region: Choose closest to Sydney
   - Plan: Free tier or higher

3. **Environment Variables**
   ```
   PORT=8000
   PYTHONUNBUFFERED=1
   LOG_LEVEL=info
   ```

4. **Deploy**
   - Render will automatically build and deploy
   - Monitor deployment in dashboard
   - Access at `https://your-app.onrender.com`

### AWS Deployment

#### Option 1: AWS App Runner
```bash
# Install AWS CLI and configure
aws configure

# Create App Runner service
aws apprunner create-service \
  --service-name "weather-prediction-api" \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "weather-prediction-api:latest",
      "ImageConfiguration": {
        "Port": "8000"
      }
    }
  }'
```

#### Option 2: AWS ECS with Fargate
```bash
# Push to ECR
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin [ECR_URL]
docker tag weather-prediction-api:latest [ECR_URL]/weather-prediction-api:latest
docker push [ECR_URL]/weather-prediction-api:latest

# Deploy with ECS CLI
ecs-cli compose up --cluster-config my-cluster
```

### Google Cloud Run
```bash
# Configure gcloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Submit build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/weather-prediction-api

# Deploy
gcloud run deploy weather-prediction-api \
  --image gcr.io/YOUR_PROJECT_ID/weather-prediction-api \
  --platform managed \
  --region australia-southeast1 \
  --allow-unauthenticated
```

### Heroku
```bash
# Login to Heroku
heroku login

# Create app
heroku create weather-prediction-api

# Deploy using container
heroku container:push web -a weather-prediction-api
heroku container:release web -a weather-prediction-api

# Open app
heroku open -a weather-prediction-api
```

## Production Checklist

### Pre-deployment
- [ ] Run all tests: `pytest tests/`
- [ ] Check code quality: `black app/` and `flake8 app/`
- [ ] Update dependencies: `pip list --outdated`
- [ ] Review security: `safety check`
- [ ] Verify model files exist and are accessible
- [ ] Test Docker build locally

### Configuration
- [ ] Set production environment variables
- [ ] Configure proper logging level
- [ ] Enable HTTPS/TLS
- [ ] Set up domain name (optional)
- [ ] Configure CORS if needed
- [ ] Set rate limiting parameters

### Post-deployment
- [ ] Verify health endpoint
- [ ] Test all API endpoints
- [ ] Monitor initial performance
- [ ] Set up alerts
- [ ] Document deployment details

## Monitoring & Logging

### Application Metrics
The API exposes metrics through:
- `/health/` - System health and status
- Application logs with structured JSON format

### Recommended Monitoring Tools
1. **Uptime Monitoring**
   - UptimeRobot
   - Pingdom
   - StatusCake

2. **Application Performance**
   - New Relic
   - DataDog
   - CloudWatch (AWS)

3. **Log Aggregation**
   - ELK Stack
   - Splunk
   - CloudWatch Logs

### Setting Up Alerts
```python
# Example alert configuration
alerts:
  - name: "API Down"
    condition: "health_check_failed"
    threshold: 3
    action: "email"
  
  - name: "High Response Time"
    condition: "response_time > 1000ms"
    threshold: 5
    action: "slack"
```

## Troubleshooting

### Common Issues

#### 1. Models Not Loading
**Error**: `ModelLoadError: Model file not found`
**Solution**:
```bash
# Verify model files exist
ls -la models/rain_or_not/
ls -la models/precipitation_fall/

# Check file permissions
chmod -R 755 models/
```

#### 2. Port Already in Use
**Error**: `[Errno 48] Address already in use`
**Solution**:
```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 [PID]
```

#### 3. Weather API Connection Issues
**Error**: `Weather API timeout`
**Solution**:
- Check internet connectivity
- Verify Open Meteo API is accessible
- Increase timeout in configuration
- Implement retry logic

#### 4. High Memory Usage
**Solution**:
```bash
# Monitor memory
docker stats

# Adjust Docker limits
docker run -m 2g weather-prediction-api
```

### Debug Mode
Enable debug logging:
```bash
# Set environment variable
export LOG_LEVEL=debug

# Or in Docker
docker run -e LOG_LEVEL=debug weather-prediction-api
```

## Security Considerations

### API Security
1. **Rate Limiting**: Implemented at 100 requests/minute
2. **Input Validation**: All inputs are validated
3. **Error Handling**: No sensitive information in errors
4. **CORS**: Configure for your domains only

### Deployment Security
1. **Use HTTPS**: Always use TLS in production
2. **Secrets Management**: Use environment variables
3. **Container Security**: Run as non-root user
4. **Network Security**: Use private networks when possible

### Security Headers
Add these headers in production (via reverse proxy):
```nginx
add_header X-Content-Type-Options "nosniff";
add_header X-Frame-Options "DENY";
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "no-referrer-when-downgrade";
```

## Continuous Deployment

### GitHub Actions
The repository includes CI/CD workflows:
- `ci-cd.yml`: Full pipeline with testing and deployment
- `test.yml`: API testing workflow

### Deployment Script
Use the included deployment script:
```bash
# Deploy with Docker Compose
./deployment/deploy.sh compose

# Deploy to cloud
./deployment/deploy.sh render
```

## Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review API logs
3. Open an issue on GitHub
4. Contact support team

---

**Last Updated**: 2025-09-09  
**Version**: 2.0.0