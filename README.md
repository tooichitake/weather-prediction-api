# Weather Prediction API for Sydney

A FastAPI-based machine learning service that provides weather predictions for Sydney, Australia.

## Features

- **Rain Prediction**: Binary classification to predict if it will rain exactly 7 days from a given date
- **Precipitation Volume Prediction**: Regression to predict cumulated precipitation (mm) for the next 3 days

## Endpoints

### 1. Root Endpoint
- **GET** `/`
- Returns project description, endpoints list, and GitHub repository link

### 2. Health Check
- **GET** `/health/`
- Returns status 200 with system health information

### 3. Rain Prediction
- **GET** `/predict/rain/?date=YYYY-MM-DD`
- Predicts if it will rain exactly 7 days from the input date
- Example response:
```json
{
  "input_date": "2023-01-01",
  "prediction": {
    "date": "2023-01-08",
    "will_rain": "TRUE"
  }
}
```

### 4. Precipitation Fall Prediction
- **GET** `/predict/precipitation/fall?date=YYYY-MM-DD`
- Predicts cumulated precipitation for the next 3 days
- Example response:
```json
{
  "input_date": "2023-01-01",
  "prediction": {
    "start_date": "2023-01-02",
    "end_date_date": "2023-01-04",
    "precipitation_fall": "9.8"
  }
}
```

## Installation

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn app.main:app --reload
```

3. Access the API:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs

## Docker

1. Build the Docker image:
```bash
docker build -t weather-prediction-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 weather-prediction-api
```

## Models

The API uses pre-trained LightGBM models:

1. **Rain Prediction Model**
   - Algorithm: LightGBM (Gradient Boosting)
   - Threshold: 0.3297 (optimized for F1 score)
   - Features: Weather conditions, temperature, wind, precipitation, etc.
   - Returns: "TRUE" or "FALSE" as string

2. **Precipitation Prediction Model**
   - Algorithm: LightGBM (Gradient Boosting)
   - Target: 3-day cumulative precipitation (mm)
   - Features: Similar to rain prediction model
   - Returns: Precipitation amount as string (e.g., "9.8")

## Technical Implementation

- **Feature Engineering**: Uses current day's weather data to predict future conditions
- **Data Source**: Real-time weather data from Open Meteo API
- **Categorical Features**: Month and weather code handled with LightGBM's native categorical support
- **Sydney Coordinates**: Latitude -33.8678, Longitude 151.2073

## Data Constraints

- Training data: Historical weather data before 2025
- Production: Accepts any date (historical or future)
- Location: Sydney, Australia

## Testing

### Quick Test
```bash
# Test health endpoint
curl http://localhost:8000/health/

# Test rain prediction
curl "http://localhost:8000/predict/rain/?date=2023-01-01"

# Test precipitation prediction
curl "http://localhost:8000/predict/precipitation/fall?date=2023-01-01"
```

### Test Scripts
```bash
# Simple test
python simple_test.py

# Interactive test
python interactive_test.py

# Format verification
python final_test.py
```

## API Response Format

The API strictly follows the assignment specifications:

- `will_rain`: Returns "TRUE" or "FALSE" as string (not boolean)
- `end_date_date`: Uses this exact field name for precipitation end date
- No extra fields in the response

## Deployment

The API is designed to be deployed on Render using the provided Dockerfile.

## Project Structure
```
api/
├── app/
│   └── main.py              # FastAPI application
├── models/                  # Pre-trained LightGBM models
│   ├── rain_or_not/
│   │   ├── rain_or_not_lightgbm.txt
│   │   └── rain_or_not_lightgbm_info.json
│   └── precipitation_fall/
│       ├── precipitation_lightgbm.txt
│       └── precipitation_lightgbm_info.json
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── README.md               # This file
└── test scripts...         # Various testing scripts
```

## License

This project is part of the Advanced Machine Learning course (36120) at UTS.