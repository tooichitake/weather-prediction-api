# Weather Prediction API for Sydney

A robust, production-ready FastAPI-based machine learning service that provides weather predictions for Sydney, Australia with enterprise-grade features including error handling, rate limiting, and comprehensive deployment support.

[![CI/CD Pipeline](https://github.com/tooichitake/weather-prediction-api/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/tooichitake/weather-prediction-api/actions/workflows/ci-cd.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![LightGBM](https://img.shields.io/badge/ML-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243.svg)](https://numpy.org/)

[![Poetry](https://img.shields.io/badge/dependency%20management-poetry-blue.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg)](https://render.com/)
[![Open Meteo](https://img.shields.io/badge/Weather%20Data-Open%20Meteo-blue.svg)](https://open-meteo.com/)

## 🌟 Features

### Core Functionality
- **Rain Prediction**: Binary classification to predict if it will rain exactly 7 days from a given date
- **Precipitation Volume Prediction**: Regression to predict cumulated precipitation (mm) for the next 3 days

### Production Features
- **🔒 Robust Error Handling**: Comprehensive error handling with graceful fallbacks
- **⚡ Rate Limiting**: Built-in rate limiting (100 requests/minute)
- **🚀 High Performance**: Async request handling with connection pooling
- **📊 Health Monitoring**: Detailed health checks and system status
- **🐳 Docker Support**: Multi-stage Docker builds for optimized images
- **🔄 CI/CD Ready**: GitHub Actions workflows for automated testing and deployment
- **📝 Comprehensive Documentation**: API usage guides, deployment docs, and troubleshooting

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/weather-prediction-api.git
cd weather-prediction-api
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the API**
```bash
uvicorn app.main:app --reload
```

4. **Access the API**
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker Deployment

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Using Docker directly
docker build -t weather-prediction-api .
docker run -p 8000:8000 weather-prediction-api
```

## 📚 API Endpoints

### 1. Root Endpoint
- **GET** `/`
- Returns comprehensive API information including endpoints, models, and configuration

### 2. Health Check
- **GET** `/health/`
- Returns detailed system health including model status, external services, and cache statistics
```json
{
  "status": "healthy",
  "timestamp": "2025-09-09T10:30:45.123456",
  "version": "2.0.0",
  "checks": {
    "models": {
      "rain_model": {"loaded": true, "threshold": 0.3297},
      "precipitation_model": {"loaded": true}
    },
    "external_services": {
      "weather_api": {"status": "healthy", "message": "Connected"}
    }
  }
}
```

### 3. Rain Prediction
- **GET** `/predict/rain/?date=YYYY-MM-DD`
- Predicts if it will rain exactly 7 days from the input date
- Example response:
```json
{
  "input_date": "2025-09-09",
  "prediction": {
    "date": "2025-09-16",
    "will_rain": "TRUE",
    "confidence": 0.7823
  },
  "metadata": {
    "model_version": "2.0.0",
    "threshold": 0.3297
  }
}
```

### 4. Precipitation Fall Prediction
- **GET** `/predict/precipitation/fall?date=YYYY-MM-DD`
- Predicts cumulated precipitation for the next 3 days
- Example response:
```json
{
  "input_date": "2025-09-09",
  "prediction": {
    "start_date": "2025-09-10",
    "end_date": "2025-09-12",
    "precipitation_fall": "12.5"
  },
  "metadata": {
    "model_version": "2.0.0",
    "unit": "mm",
    "period_days": 3
  }
}
```

## 🤖 Models

### Rain Prediction Model
- **Algorithm**: LightGBM (Gradient Boosting)
- **Type**: Binary Classification
- **Threshold**: 0.3297 (optimized for F1 score)
- **Features**: Weather conditions, temperature, wind, precipitation, sunshine duration, etc.
- **Output**: "TRUE" or "FALSE" as string

### Precipitation Prediction Model
- **Algorithm**: LightGBM (Gradient Boosting)
- **Type**: Regression
- **Target**: 3-day cumulative precipitation (mm)
- **Features**: Similar to rain prediction model
- **Output**: Precipitation amount as string (e.g., "12.5")

## 🏗️ Architecture

### Technical Stack
- **Framework**: FastAPI 0.104.1
- **ML Library**: LightGBM 4.0.0+
- **Data Processing**: Pandas, NumPy
- **Weather Data**: Open Meteo API
- **Server**: Uvicorn with async support
- **Container**: Docker with multi-stage builds

### Key Features
- **Async Processing**: Non-blocking request handling
- **Connection Pooling**: Efficient external API calls
- **Caching**: LRU cache for weather data
- **Error Recovery**: Automatic retries with exponential backoff
- **Security**: Non-root container user, input validation

## 📦 Deployment

### Supported Platforms
- **Render** (Recommended) - Automatic deployment via GitHub
- **AWS** - ECS, App Runner, or Lambda
- **Google Cloud** - Cloud Run
- **Heroku** - Container deployment
- **Self-hosted** - Docker/Docker Compose

### Quick Deploy to Render
1. Fork this repository
2. Connect to Render
3. Use the included `render.yaml` configuration
4. Deploy!

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 🧪 Testing

### API Tests
```bash
# Run automated tests
python -m pytest tests/

# Quick manual test
curl http://localhost:8000/health/
curl "http://localhost:8000/predict/rain/?date=2025-09-09"

# Run integration tests
python test_api_integration.py
```

### CI/CD Pipeline
- Automated testing on push/PR
- Code quality checks (Black, Flake8, MyPy)
- Security scanning (Trivy, TruffleHog)
- Multi-platform Docker builds
- Automated deployment to staging/production

## 📖 Documentation

- [API Usage Guide](API_USAGE.md) - Detailed examples and best practices
- [Deployment Guide](DEPLOYMENT.md) - Comprehensive deployment instructions
- [Interactive API Docs](http://localhost:8000/docs) - Swagger UI documentation
- [ReDoc](http://localhost:8000/redoc) - Alternative documentation interface

## 🔧 Configuration

### Environment Variables
- `PORT`: API port (default: 8000)
- `LOG_LEVEL`: Logging level (default: info)
- `PYTHONUNBUFFERED`: Python output buffering (default: 1)

### Rate Limiting
- Default: 100 requests/minute per IP
- Configurable in `app/main.py`

## 🚨 Error Handling

The API provides detailed error responses:
```json
{
  "error": {
    "status_code": 422,
    "message": "Invalid date format. Use YYYY-MM-DD",
    "timestamp": "2025-09-09T10:30:45.123456"
  }
}
```

Common status codes:
- `400`: Bad Request
- `422`: Invalid input format
- `429`: Rate limit exceeded
- `502`: External API error
- `503`: Service unavailable

## 📊 Monitoring

### Health Metrics
- Model loading status
- External API connectivity
- Cache performance
- Request rate and errors

### Recommended Tools
- **Uptime**: UptimeRobot, Pingdom
- **APM**: New Relic, DataDog
- **Logs**: CloudWatch, ELK Stack

## 🛠️ Development

### Project Structure
```
weather-prediction-api/
├── app/
│   └── main.py              # FastAPI application
├── models/                  # Pre-trained ML models
│   ├── rain_or_not/
│   └── precipitation_fall/
├── deployment/              # Deployment scripts
├── .github/workflows/       # CI/CD pipelines
├── requirements.txt         # Python dependencies
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose config
├── render.yaml             # Render deployment config
├── API_USAGE.md            # API usage examples
├── DEPLOYMENT.md           # Deployment guide
└── README.md               # This file
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is part of the Advanced Machine Learning course (36120) at UTS.

## 🙏 Acknowledgments

- UTS Advanced Machine Learning course team
- Open Meteo API for weather data
- FastAPI and LightGBM communities

---

**Version**: 2.0.0  
**Last Updated**: 2025-09-09  
**Repository**: https://github.com/tooichitake/weather-prediction-api

[![GitHub issues](https://img.shields.io/github/issues/tooichitake/weather-prediction-api)](https://github.com/tooichitake/weather-prediction-api/issues)
[![GitHub stars](https://img.shields.io/github/stars/tooichitake/weather-prediction-api)](https://github.com/tooichitake/weather-prediction-api/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/tooichitake/weather-prediction-api)](https://github.com/tooichitake/weather-prediction-api/network)
[![GitHub last commit](https://img.shields.io/github/last-commit/tooichitake/weather-prediction-api)](https://github.com/tooichitake/weather-prediction-api/commits/main)