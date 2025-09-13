# Weather Prediction API - Usage Guide & Examples

## Overview

The Weather Prediction API provides machine learning-powered weather predictions for Sydney, Australia. This guide covers all endpoints with practical examples.

## Base URL

```
Local: http://localhost:8000
Production: https://your-api-domain.com
```

## Authentication

Currently, the API is open and does not require authentication. Rate limiting is applied at 100 requests per minute per IP.

## Endpoints

### 1. Root Endpoint - API Information

**GET** `/`

Returns comprehensive information about the API, available endpoints, and model details.

#### Example Request
```bash
curl -X GET "http://localhost:8000/"
```

#### Example Response
```json
{
  "project": "Weather Prediction API for Sydney",
  "description": "Machine Learning service providing rain prediction (7 days ahead) and precipitation volume forecast (3 days cumulative)",
  "location": {
    "city": "Sydney",
    "latitude": -33.8678,
    "longitude": 151.2073
  },
  "endpoints": {
    "/": "Project information (this endpoint)",
    "/health/": "Health check endpoint",
    "/predict/rain/": "Predict if it will rain in exactly 7 days",
    "/predict/precipitation/fall": "Predict cumulated precipitation for next 3 days"
  },
  "input_format": {
    "date": "YYYY-MM-DD format"
  },
  "models": {
    "rain_prediction": {
      "type": "Binary Classification",
      "algorithm": "LightGBM",
      "prediction_horizon": "Exactly 7 days ahead",
      "output": "Boolean (will_rain: true/false)",
      "threshold": 0.3297
    },
    "precipitation_prediction": {
      "type": "Regression",
      "algorithm": "LightGBM",
      "prediction_horizon": "Next 3 days cumulative",
      "output": "Float (precipitation in mm)"
    }
  },
  "github_repository": "https://github.com/tooichitake/weather-prediction-api"
}
```

### 2. Health Check

**GET** `/health/`

Provides detailed health status of the API and its components.

#### Example Request
```bash
curl -X GET "http://localhost:8000/health/"
```

#### Example Response
```json
{
  "status": "healthy",
  "timestamp": "2025-09-09T10:30:45.123456",
  "version": "2.0.0",
  "checks": {
    "models": {
      "rain_model": {
        "loaded": true,
        "threshold": 0.3297
      },
      "precipitation_model": {
        "loaded": true
      }
    },
    "external_services": {
      "weather_api": {
        "status": "healthy",
        "message": "Connected"
      }
    },
    "cache": {
      "size": 15,
      "hits": 234,
      "misses": 18
    }
  }
}
```

### 3. Rain Prediction

**GET** `/predict/rain/`

Predicts whether it will rain exactly 7 days from the given date.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| date | string | Yes | Date in YYYY-MM-DD format |

#### Example Request
```bash
curl -X GET "http://localhost:8000/predict/rain/?date=2025-09-09"
```

#### Example Response
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

### 4. Precipitation Volume Prediction

**GET** `/predict/precipitation/fall`

Predicts the cumulative precipitation (in mm) for the next 3 days from the given date.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| date | string | Yes | Date in YYYY-MM-DD format |

#### Example Request
```bash
curl -X GET "http://localhost:8000/predict/precipitation/fall?date=2025-09-09"
```

#### Example Response
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

## Error Handling

The API returns standardized error responses for all error conditions.

### Error Response Format
```json
{
  "error": {
    "status_code": 422,
    "message": "Invalid date format. Use YYYY-MM-DD",
    "timestamp": "2025-09-09T10:30:45.123456"
  }
}
```

### Common Error Codes

| Status Code | Description | Example |
|-------------|-------------|---------|
| 400 | Bad Request | Invalid request parameters |
| 422 | Unprocessable Entity | Invalid date format or range |
| 429 | Too Many Requests | Rate limit exceeded |
| 502 | Bad Gateway | External weather API error |
| 503 | Service Unavailable | Models not loaded |
| 504 | Gateway Timeout | External API timeout |

## Code Examples

### Python (using requests)
```python
import requests
from datetime import datetime

# Base URL
BASE_URL = "http://localhost:8000"

# Get API info
response = requests.get(f"{BASE_URL}/")
print(response.json())

# Predict rain
date = datetime.now().strftime("%Y-%m-%d")
response = requests.get(f"{BASE_URL}/predict/rain/", params={"date": date})
result = response.json()
print(f"Will it rain on {result['prediction']['date']}? {result['prediction']['will_rain']}")

# Predict precipitation volume
response = requests.get(f"{BASE_URL}/predict/precipitation/fall", params={"date": date})
result = response.json()
print(f"Expected precipitation: {result['prediction']['precipitation_fall']} mm")
```

### JavaScript (using fetch)
```javascript
const BASE_URL = 'http://localhost:8000';

// Get API info
async function getApiInfo() {
  const response = await fetch(`${BASE_URL}/`);
  const data = await response.json();
  console.log(data);
}

// Predict rain
async function predictRain(date) {
  const response = await fetch(`${BASE_URL}/predict/rain/?date=${date}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const data = await response.json();
  console.log(`Rain prediction for ${data.prediction.date}: ${data.prediction.will_rain}`);
  return data;
}

// Predict precipitation
async function predictPrecipitation(date) {
  const response = await fetch(`${BASE_URL}/predict/precipitation/fall?date=${date}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const data = await response.json();
  console.log(`Precipitation: ${data.prediction.precipitation_fall} mm`);
  return data;
}

// Usage
const today = new Date().toISOString().split('T')[0];
predictRain(today).catch(console.error);
predictPrecipitation(today).catch(console.error);
```

### cURL Examples
```bash
# Pretty print with jq
curl -s "http://localhost:8000/predict/rain/?date=2025-09-09" | jq .

# Save response to file
curl -o rain_prediction.json "http://localhost:8000/predict/rain/?date=2025-09-09"

# With custom headers
curl -H "Accept: application/json" \
     -H "User-Agent: MyApp/1.0" \
     "http://localhost:8000/predict/rain/?date=2025-09-09"

# Verbose output for debugging
curl -v "http://localhost:8000/predict/rain/?date=2025-09-09"
```

## Best Practices

### 1. Error Handling
Always implement proper error handling:
```python
try:
    response = requests.get(f"{BASE_URL}/predict/rain/", params={"date": date})
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        print("Rate limit exceeded. Please wait before retrying.")
    elif e.response.status_code == 422:
        print("Invalid date format. Use YYYY-MM-DD")
    else:
        print(f"HTTP error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

### 2. Rate Limiting
Respect the rate limits:
```python
import time
from datetime import datetime

def predict_with_retry(date, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/predict/rain/", params={"date": date})
            if response.status_code == 429:
                # Get retry-after header
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 3. Date Validation
Validate dates before sending requests:
```python
from datetime import datetime, timedelta

def validate_date(date_string):
    try:
        date = datetime.strptime(date_string, "%Y-%m-%d")
        # Check if date is within reasonable range (5 years)
        today = datetime.now()
        if abs((date - today).days) > 365 * 5:
            raise ValueError("Date must be within 5 years of today")
        return date_string
    except ValueError as e:
        raise ValueError(f"Invalid date: {e}")
```

## Integration Examples

### Weather Dashboard
```python
import requests
import pandas as pd
from datetime import datetime, timedelta

class WeatherDashboard:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def get_weekly_forecast(self, start_date):
        """Get rain predictions for the next 7 days"""
        forecasts = []
        for i in range(7):
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                response = requests.get(
                    f"{self.api_url}/predict/rain/",
                    params={"date": date}
                )
                if response.status_code == 200:
                    data = response.json()
                    forecasts.append({
                        "input_date": date,
                        "forecast_date": data["prediction"]["date"],
                        "will_rain": data["prediction"]["will_rain"],
                        "confidence": data["prediction"].get("confidence", 0)
                    })
            except Exception as e:
                print(f"Error for date {date}: {e}")
        
        return pd.DataFrame(forecasts)
    
    def get_precipitation_trend(self, start_date, days=30):
        """Get precipitation predictions for multiple dates"""
        predictions = []
        for i in range(0, days, 3):  # Every 3 days
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                response = requests.get(
                    f"{self.api_url}/predict/precipitation/fall",
                    params={"date": date}
                )
                if response.status_code == 200:
                    data = response.json()
                    predictions.append({
                        "input_date": date,
                        "period_start": data["prediction"]["start_date"],
                        "period_end": data["prediction"]["end_date"],
                        "precipitation_mm": float(data["prediction"]["precipitation_fall"])
                    })
            except Exception as e:
                print(f"Error for date {date}: {e}")
        
        return pd.DataFrame(predictions)

# Usage
dashboard = WeatherDashboard()
today = datetime.now()

# Get weekly rain forecast
weekly_forecast = dashboard.get_weekly_forecast(today)
print("Weekly Rain Forecast:")
print(weekly_forecast)

# Get precipitation trend
precipitation_trend = dashboard.get_precipitation_trend(today, days=30)
print("\nPrecipitation Trend:")
print(precipitation_trend)
```

## Performance Tips

1. **Use Connection Pooling**
```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.3)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Use session for all requests
response = session.get(f"{BASE_URL}/predict/rain/", params={"date": date})
```

2. **Implement Caching**
```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=100)
def get_rain_prediction(date):
    response = requests.get(f"{BASE_URL}/predict/rain/", params={"date": date})
    return response.json()

# Cache is automatically used for repeated calls with same date
result1 = get_rain_prediction("2025-09-09")  # API call
result2 = get_rain_prediction("2025-09-09")  # From cache
```

## Monitoring Your Integration

Track these metrics in your application:
- Request count and rate
- Response times
- Error rates by status code
- Cache hit rates

Example monitoring:
```python
import time
from collections import defaultdict

class ApiMonitor:
    def __init__(self):
        self.metrics = defaultdict(int)
        self.response_times = []
    
    def track_request(self, endpoint, status_code, duration):
        self.metrics[f"requests_{endpoint}"] += 1
        self.metrics[f"status_{status_code}"] += 1
        self.response_times.append(duration)
        
        if status_code >= 400:
            self.metrics[f"errors_{endpoint}"] += 1
    
    def get_stats(self):
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        return {
            "total_requests": sum(v for k, v in self.metrics.items() if k.startswith("requests_")),
            "error_rate": sum(v for k, v in self.metrics.items() if k.startswith("errors_")) / sum(v for k, v in self.metrics.items() if k.startswith("requests_")),
            "avg_response_time": avg_response_time,
            "metrics": dict(self.metrics)
        }

# Usage
monitor = ApiMonitor()

start = time.time()
response = requests.get(f"{BASE_URL}/predict/rain/", params={"date": "2025-09-09"})
duration = time.time() - start

monitor.track_request("rain", response.status_code, duration)
print(monitor.get_stats())
```

## Support

For additional support:
- Check the [API Documentation](http://localhost:8000/docs)
- Review the [Deployment Guide](./DEPLOYMENT.md)
- Open an issue on [GitHub](https://github.com/yourusername/weather-prediction-api)

---

**Version**: 2.0.0  
**Last Updated**: 2025-09-09