from fastapi import FastAPI, Query
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import json
import pandas as pd
import lightgbm as lgb
import requests

app = FastAPI(
    title="Weather Prediction API",
    description="Machine Learning models for weather prediction in Sydney",
    version="1.0.0"
)

# Model paths
RAIN_MODEL_PATH = os.path.join("models", "rain_or_not", "rain_or_not_lightgbm.txt")
RAIN_INFO_PATH = os.path.join("models", "rain_or_not", "rain_or_not_lightgbm_info.json")

PRECIPITATION_MODEL_PATH = os.path.join("models", "precipitation_fall", "precipitation_lightgbm.txt")
PRECIPITATION_INFO_PATH = os.path.join("models", "precipitation_fall", "precipitation_lightgbm_info.json")

# Sydney coordinates
SYDNEY_LAT = -33.8678
SYDNEY_LON = 151.2073

# Load models and info on startup
rain_model = None
rain_info = None
rain_threshold = 0.5
rain_best_iteration = None
rain_categorical_features = []

precipitation_model = None
precipitation_info = None
precipitation_best_iteration = None
precipitation_categorical_features = []

# Open Meteo API base URL
OPEN_METEO_API = "https://archive-api.open-meteo.com/v1/archive"

def load_models():
    """Load pre-trained models from the models directory"""
    global rain_model, rain_info, rain_threshold, rain_best_iteration, rain_categorical_features
    global precipitation_model, precipitation_info, precipitation_best_iteration, precipitation_categorical_features
    
    # Load rain prediction model
    rain_model = lgb.Booster(model_file=RAIN_MODEL_PATH)
    
    with open(RAIN_INFO_PATH, 'r') as f:
        rain_info = json.load(f)
        rain_threshold = rain_info.get('best_threshold', 0.5)
        rain_best_iteration = rain_info.get('best_iteration', -1)
        rain_categorical_features = rain_info.get('categorical_features', [])
    
    # Load precipitation model
    precipitation_model = lgb.Booster(model_file=PRECIPITATION_MODEL_PATH)
    
    with open(PRECIPITATION_INFO_PATH, 'r') as f:
        precipitation_info = json.load(f)
        precipitation_best_iteration = precipitation_info.get('best_iteration', -1)
        precipitation_categorical_features = precipitation_info.get('categorical_features', [])

# Load models on startup
load_models()

def fetch_weather_data(start_date: str, end_date: str):
    """Fetch historical weather data from Open Meteo API"""
    params = {
        'latitude': SYDNEY_LAT,
        'longitude': SYDNEY_LON,
        'start_date': start_date,
        'end_date': end_date,
        'daily': [
            'weather_code',
            'temperature_2m_max',
            'temperature_2m_min',
            'temperature_2m_mean',
            'apparent_temperature_max',
            'apparent_temperature_min',
            'apparent_temperature_mean',
            'precipitation_sum',
            'rain_sum',
            'precipitation_hours',
            'sunshine_duration',
            'daylight_duration',
            'wind_speed_10m_max',
            'wind_gusts_10m_max',
            'wind_direction_10m_dominant',
            'shortwave_radiation_sum',
            'et0_fao_evapotranspiration'
        ],
        'timezone': 'Australia/Sydney'
    }
    
    response = requests.get(OPEN_METEO_API, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def prepare_features(weather_data: dict, date: str):
    """Prepare features from weather data for model prediction"""
    daily_data = weather_data['daily']
    
    # Find the index of the requested date
    dates = daily_data['time']
    idx = dates.index(date)
    
    # Create feature dictionary
    row_data = {}
    
    # Extract numerical features
    numerical_features = [
        'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
        'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean',
        'rain_sum', 'precipitation_hours', 'sunshine_duration', 'daylight_duration',
        'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant',
        'shortwave_radiation_sum', 'et0_fao_evapotranspiration'
    ]
    
    for feature in numerical_features:
        value = daily_data[feature][idx]
        row_data[feature] = value if value is not None else 0
    
    # Add weather_code as categorical feature
    weather_code = daily_data['weather_code'][idx]
    row_data['weather_code'] = weather_code
    
    # Add month as categorical feature (as month name, matching training data) 
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    import calendar
    row_data['month'] = calendar.month_name[date_obj.month]
    
    return row_data

@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    Display project information, endpoints, and documentation
    """
    return {
        "project": "Weather Prediction API for Sydney",
        "description": "Machine Learning service providing rain prediction (7 days ahead) and precipitation volume forecast (3 days cumulative)",
        "location": {
            "city": "Sydney",
            "latitude": SYDNEY_LAT,
            "longitude": SYDNEY_LON
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
                "threshold": rain_threshold
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

@app.get("/health/")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "message": "Welcome to the Weather Prediction API! All systems operational.",
        "models_loaded": {
            "rain_model": rain_model is not None,
            "precipitation_model": precipitation_model is not None
        }
    }

@app.get("/predict/rain/", response_model=Dict[str, Any])
async def predict_rain(
    date: str = Query(..., description="Date in YYYY-MM-DD format", pattern="^\\d{4}-\\d{2}-\\d{2}$")
):
    """
    Predict if it will rain exactly 7 days from the given date
    """
    # Parse input date
    input_date = datetime.strptime(date, "%Y-%m-%d")
    prediction_date = input_date + timedelta(days=7)
    
    # Fetch weather data for the input date
    weather_data = fetch_weather_data(date, date)
    
    # Prepare features
    features = prepare_features(weather_data, date)
    
    # Create DataFrame from features
    feature_df = pd.DataFrame([features])
    
    # Ensure features are in the correct order for LightGBM
    expected_features = rain_info.get('feature_names', [])
    if expected_features:
        feature_df = feature_df[expected_features]
    
    # Convert categorical features to category dtype for LightGBM
    for cat_col in rain_categorical_features:
        if cat_col in feature_df.columns:
            feature_df[cat_col] = feature_df[cat_col].astype('category')
    
    # Get prediction probability using LightGBM
    prediction_proba = rain_model.predict(feature_df, num_iteration=rain_best_iteration)[0]
    
    # Apply threshold
    will_rain = bool(prediction_proba >= rain_threshold)
    
    return {
        "input_date": date,
        "prediction": {
            "date": prediction_date.strftime("%Y-%m-%d"),
            "will_rain": "TRUE" if will_rain else "FALSE"
        }
    }

@app.get("/predict/precipitation/fall", response_model=Dict[str, Any])
async def predict_precipitation_fall(
    date: str = Query(..., description="Date in YYYY-MM-DD format", pattern="^\\d{4}-\\d{2}-\\d{2}$")
):
    """
    Predict cumulated precipitation (in mm) for the next 3 days
    """
    # Parse input date
    input_date = datetime.strptime(date, "%Y-%m-%d")
    start_date = input_date + timedelta(days=1)
    end_date = input_date + timedelta(days=3)
    
    # Fetch weather data for the input date
    weather_data = fetch_weather_data(date, date)
    
    # Prepare features
    features = prepare_features(weather_data, date)
    
    # Create DataFrame from features
    feature_df = pd.DataFrame([features])
    
    # Ensure features are in the correct order for LightGBM
    expected_features = precipitation_info.get('feature_names', [])
    if expected_features:
        feature_df = feature_df[expected_features]
    
    # Convert categorical features to category dtype for LightGBM
    for cat_col in precipitation_categorical_features:
        if cat_col in feature_df.columns:
            feature_df[cat_col] = feature_df[cat_col].astype('category')
    
    # Get prediction using LightGBM
    prediction = precipitation_model.predict(feature_df, num_iteration=precipitation_best_iteration)[0]
    
    # Ensure non-negative prediction
    prediction = max(0, prediction)
    
    return {
        "input_date": date,
        "prediction": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date_date": end_date.strftime("%Y-%m-%d"),
            "precipitation_fall": f"{prediction:.1f}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)