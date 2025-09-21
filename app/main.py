from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import json
import pandas as pd
import lightgbm as lgb
import requests
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weather Prediction API",
    description="Machine Learning models for weather prediction in Sydney with robust error handling and high availability",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Model paths
RAIN_MODEL_PATH = os.path.join("models", "rain_or_not", "rain_or_not_lightgbm.txt")
RAIN_INFO_PATH = os.path.join("models", "rain_or_not", "rain_or_not_lightgbm_info.json")

PRECIPITATION_MODEL_PATH = os.path.join(
    "models", "precipitation_fall", "precipitation_lightgbm.txt"
)
PRECIPITATION_INFO_PATH = os.path.join(
    "models", "precipitation_fall", "precipitation_lightgbm_info.json"
)

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

# Rate limiting and retry configuration
API_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1

# Cache configuration
CACHE_TTL = 3600  # 1 hour

# Request tracking for rate limiting
request_timestamps = []
RATE_LIMIT_WINDOW = 60  # 1 minute
RATE_LIMIT_MAX_REQUESTS = 100


class ModelLoadError(Exception):
    """Custom exception for model loading errors"""

    pass


def load_models():
    """Load pre-trained models from the models directory with error handling"""
    global rain_model, rain_info, rain_threshold, rain_best_iteration, rain_categorical_features
    global precipitation_model, precipitation_info, precipitation_best_iteration, precipitation_categorical_features

    try:
        # Validate model files exist
        if not os.path.exists(RAIN_MODEL_PATH):
            raise ModelLoadError(f"Rain model file not found: {RAIN_MODEL_PATH}")
        if not os.path.exists(RAIN_INFO_PATH):
            raise ModelLoadError(f"Rain model info file not found: {RAIN_INFO_PATH}")
        if not os.path.exists(PRECIPITATION_MODEL_PATH):
            raise ModelLoadError(
                f"Precipitation model file not found: {PRECIPITATION_MODEL_PATH}"
            )
        if not os.path.exists(PRECIPITATION_INFO_PATH):
            raise ModelLoadError(
                f"Precipitation model info file not found: {PRECIPITATION_INFO_PATH}"
            )

        # Load rain prediction model
        logger.info("Loading rain prediction model...")
        rain_model = lgb.Booster(model_file=RAIN_MODEL_PATH)

        with open(RAIN_INFO_PATH, "r") as f:
            rain_info = json.load(f)
            rain_threshold = rain_info.get("best_threshold", 0.5)
            rain_best_iteration = rain_info.get("best_iteration", -1)
            rain_categorical_features = rain_info.get("categorical_features", [])
        logger.info(f"Rain model loaded successfully. Threshold: {rain_threshold}")

        # Load precipitation model
        logger.info("Loading precipitation model...")
        precipitation_model = lgb.Booster(model_file=PRECIPITATION_MODEL_PATH)

        with open(PRECIPITATION_INFO_PATH, "r") as f:
            precipitation_info = json.load(f)
            precipitation_best_iteration = precipitation_info.get("best_iteration", -1)
            precipitation_categorical_features = precipitation_info.get(
                "categorical_features", []
            )
        logger.info("Precipitation model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise ModelLoadError(f"Model loading failed: {str(e)}")


# Load models on startup with error handling
try:
    load_models()
except ModelLoadError as e:
    logger.critical(f"Failed to initialize models: {str(e)}")
    # Continue startup but mark models as unavailable
    rain_model = None
    precipitation_model = None


@lru_cache(maxsize=128)
def fetch_weather_data(start_date: str, end_date: str):
    """Fetch historical weather data from Open Meteo API with caching and retry logic"""
    params = {
        "latitude": SYDNEY_LAT,
        "longitude": SYDNEY_LON,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "apparent_temperature_mean",
            "precipitation_sum",
            "rain_sum",
            "precipitation_hours",
            "sunshine_duration",
            "daylight_duration",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ],
        "timezone": "Australia/Sydney",
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(OPEN_METEO_API, params=params, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Validate response structure
            if "daily" not in data or "time" not in data["daily"]:
                raise ValueError("Invalid response structure from weather API")

            return data

        except requests.exceptions.Timeout:
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Weather API timeout after multiple attempts",
                )
            time.sleep(RETRY_DELAY * (attempt + 1))

        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Weather API error: {str(e)}",
                )
            time.sleep(RETRY_DELAY * (attempt + 1))

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Invalid weather data received: {str(e)}",
            )


def prepare_features(weather_data: dict, date: str) -> Dict[str, Any]:
    """Prepare features from weather data for model prediction with validation"""
    try:
        daily_data = weather_data["daily"]

        # Find the index of the requested date
        dates = daily_data["time"]
        if date not in dates:
            raise ValueError(f"Date {date} not found in weather data")
        idx = dates.index(date)

        # Create feature dictionary
        row_data = {}

        # Extract numerical features with validation
        numerical_features = [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "apparent_temperature_mean",
            "rain_sum",
            "precipitation_hours",
            "sunshine_duration",
            "daylight_duration",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ]

        for feature in numerical_features:
            if feature not in daily_data:
                logger.warning(f"Feature {feature} not found in weather data, using 0")
                row_data[feature] = 0
            else:
                value = daily_data[feature][idx]
                # Handle None values and validate ranges
                if value is None:
                    logger.warning(f"Null value for {feature}, using 0")
                    row_data[feature] = 0
                else:
                    # Basic range validation for temperatures
                    if "temperature" in feature and (value < -50 or value > 60):
                        logger.warning(
                            f"Suspicious temperature value: {value} for {feature}"
                        )
                    row_data[feature] = float(value)

        # Add weather_code as categorical feature with validation
        if "weather_code" not in daily_data:
            raise ValueError("weather_code not found in weather data")
        weather_code = daily_data["weather_code"][idx]
        if weather_code is None:
            weather_code = 0  # Default weather code
        row_data["weather_code"] = int(weather_code)

        # Add month as categorical feature (as month name, matching training data)
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        import calendar

        row_data["month"] = calendar.month_name[date_obj.month]

        return row_data

    except (KeyError, IndexError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error processing weather data: {str(e)}",
        )


@app.get("/", response_model=Dict[str, Any])
async def root():
    """
    Display project information, endpoints, and documentation
    """
    return {
        "project": "Weather Prediction API for Sydney",
        "description": "Machine Learning service providing rain prediction (7 days ahead) and precipitation volume forecast (3 days cumulative)",
        "location": {"city": "Sydney", "latitude": SYDNEY_LAT, "longitude": SYDNEY_LON},
        "endpoints": {
            "/": "Project information (this endpoint)",
            "/health/": "Health check endpoint",
            "/predict/rain/": "Predict if it will rain in exactly 7 days",
            "/predict/precipitation/fall": "Predict cumulated precipitation for next 3 days",
        },
        "input_format": {"date": "YYYY-MM-DD format"},
        "models": {
            "rain_prediction": {
                "type": "Binary Classification",
                "algorithm": "LightGBM",
                "prediction_horizon": "Exactly 7 days ahead",
                "output": "Boolean (will_rain: true/false)",
                "threshold": rain_threshold,
            },
            "precipitation_prediction": {
                "type": "Regression",
                "algorithm": "LightGBM",
                "prediction_horizon": "Next 3 days cumulative",
                "output": "Float (precipitation in mm)",
            },
        },
        "github_repository": "https://github.com/tooichitake/weather-prediction-api",
    }


@app.get("/health/", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint with comprehensive system status
    """
    # Check model status
    rain_model_status = rain_model is not None
    precipitation_model_status = precipitation_model is not None

    # Test weather API connectivity
    weather_api_status = True
    weather_api_message = "Connected"
    try:
        test_date = datetime.now().strftime("%Y-%m-%d")
        fetch_weather_data(test_date, test_date)
    except Exception as e:
        weather_api_status = False
        weather_api_message = str(e)

    # Overall health status
    all_healthy = (
        rain_model_status and precipitation_model_status and weather_api_status
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "checks": {
            "models": {
                "rain_model": {
                    "loaded": rain_model_status,
                    "threshold": rain_threshold if rain_model_status else None,
                },
                "precipitation_model": {"loaded": precipitation_model_status},
            },
            "external_services": {
                "weather_api": {
                    "status": "healthy" if weather_api_status else "unhealthy",
                    "message": weather_api_message,
                }
            },
            "cache": {
                "size": fetch_weather_data.cache_info().currsize,
                "hits": fetch_weather_data.cache_info().hits,
                "misses": fetch_weather_data.cache_info().misses,
            },
        },
    }


def check_rate_limit():
    """Check if rate limit has been exceeded"""
    current_time = time.time()
    global request_timestamps

    # Remove old timestamps
    request_timestamps = [
        t for t in request_timestamps if current_time - t < RATE_LIMIT_WINDOW
    ]

    # Check rate limit
    if len(request_timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )

    # Add current timestamp
    request_timestamps.append(current_time)


def validate_date_range(date_str: str) -> datetime:
    """Validate date is within acceptable range"""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid date format. Use YYYY-MM-DD",
        )

    # Check date is not too far in the past or future
    today = datetime.now()
    days_diff = abs((date - today).days)

    if days_diff > 365 * 5:  # 5 years
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Date must be within 5 years of today",
        )

    return date


@app.get("/predict/rain/", response_model=Dict[str, Any])
async def predict_rain(
    date: str = Query(
        ..., description="Date in YYYY-MM-DD format", pattern="^\\d{4}-\\d{2}-\\d{2}$"
    )
):
    """
    Predict if it will rain exactly 7 days from the given date

    Returns:
    - input_date: The date provided
    - prediction.date: The date 7 days ahead
    - prediction.will_rain: TRUE or FALSE
    """
    # Check rate limit
    check_rate_limit()

    # Validate models are loaded
    if rain_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rain prediction model is not available. Please try again later.",
        )

    try:
        # Validate and parse input date
        input_date = validate_date_range(date)
        prediction_date = input_date + timedelta(days=7)

        # Log prediction request
        logger.info(f"Rain prediction requested for date: {date}")

        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, _predict_rain_sync, date, prediction_date
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in rain prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction",
        )


def _predict_rain_sync(date: str, prediction_date: datetime) -> Dict[str, Any]:
    """Synchronous rain prediction logic for thread pool execution"""
    # Fetch weather data for the input date
    weather_data = fetch_weather_data(date, date)

    # Prepare features
    features = prepare_features(weather_data, date)

    # Create DataFrame from features
    feature_df = pd.DataFrame([features])

    # Ensure features are in the correct order for LightGBM
    expected_features = rain_info.get("feature_names", [])
    if expected_features:
        missing_features = set(expected_features) - set(feature_df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default values
            for feat in missing_features:
                feature_df[feat] = 0
        feature_df = feature_df[expected_features]

    # Convert categorical features to category dtype for LightGBM
    for cat_col in rain_categorical_features:
        if cat_col in feature_df.columns:
            feature_df[cat_col] = feature_df[cat_col].astype("category")

    # Get prediction probability using LightGBM
    prediction_proba = rain_model.predict(
        feature_df, num_iteration=rain_best_iteration
    )[0]

    # Apply threshold
    will_rain = bool(prediction_proba >= rain_threshold)

    return {
        "input_date": date,
        "prediction": {
            "date": prediction_date.strftime("%Y-%m-%d"),
            "will_rain": "TRUE" if will_rain else "FALSE",
            "confidence": (
                float(prediction_proba) if will_rain else float(1 - prediction_proba)
            ),
        },
        "metadata": {"model_version": "2.0.0", "threshold": rain_threshold},
    }


@app.get("/predict/precipitation/fall", response_model=Dict[str, Any])
async def predict_precipitation_fall(
    date: str = Query(
        ..., description="Date in YYYY-MM-DD format", pattern="^\\d{4}-\\d{2}-\\d{2}$"
    )
):
    """
    Predict cumulated precipitation (in mm) for the next 3 days

    Returns:
    - input_date: The date provided
    - prediction.start_date: Day 1 of the prediction period
    - prediction.end_date: Day 3 of the prediction period
    - prediction.precipitation_fall: Cumulative precipitation in mm
    """
    # Check rate limit
    check_rate_limit()

    # Validate models are loaded
    if precipitation_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Precipitation prediction model is not available. Please try again later.",
        )

    try:
        # Validate and parse input date
        input_date = validate_date_range(date)
        start_date = input_date + timedelta(days=1)
        end_date = input_date + timedelta(days=3)

        # Log prediction request
        logger.info(f"Precipitation prediction requested for date: {date}")

        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, _predict_precipitation_sync, date, start_date, end_date
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error in precipitation prediction: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction",
        )


def _predict_precipitation_sync(
    date: str, start_date: datetime, end_date: datetime
) -> Dict[str, Any]:
    """Synchronous precipitation prediction logic for thread pool execution"""
    # Fetch weather data for the input date
    weather_data = fetch_weather_data(date, date)

    # Prepare features
    features = prepare_features(weather_data, date)

    # Create DataFrame from features
    feature_df = pd.DataFrame([features])

    # Ensure features are in the correct order for LightGBM
    expected_features = precipitation_info.get("feature_names", [])
    if expected_features:
        missing_features = set(expected_features) - set(feature_df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default values
            for feat in missing_features:
                feature_df[feat] = 0
        feature_df = feature_df[expected_features]

    # Convert categorical features to category dtype for LightGBM
    for cat_col in precipitation_categorical_features:
        if cat_col in feature_df.columns:
            feature_df[cat_col] = feature_df[cat_col].astype("category")

    # Get prediction using LightGBM
    prediction = precipitation_model.predict(
        feature_df, num_iteration=precipitation_best_iteration
    )[0]

    # Ensure non-negative prediction with upper bound
    prediction = max(0, min(prediction, 500))  # Cap at 500mm for sanity

    return {
        "input_date": date,
        "prediction": {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "precipitation_fall": f"{prediction:.1f}",
        },
        "metadata": {"model_version": "2.0.0", "unit": "mm", "period_days": 3},
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler with logging"""
    logger.error(f"HTTP {exc.status_code} error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now().isoformat(),
            }
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "status_code": 500,
                "message": "Internal server error occurred",
                "timestamp": datetime.now().isoformat(),
            }
        },
    )


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Weather Prediction API starting up...")
    logger.info(
        f"Models loaded: Rain={rain_model is not None}, Precipitation={precipitation_model is not None}"
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Weather Prediction API shutting down...")
    executor.shutdown(wait=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
