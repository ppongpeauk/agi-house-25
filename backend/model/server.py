from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import torch
from pathlib import Path
import logging
import requests
from datetime import datetime, timedelta
from model import load_model, predict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Disease Outbreak Prediction API",
    description="API for predicting disease outbreaks based on environmental conditions",
    version="1.0.0"
)

# Disease labels from README
DISEASE_LABELS = {
    0: "Cholera",
    1: "Malaria",
    2: "Dengue",
    3: "Chikungunya",
    4: "COVID-19",
    5: "Cutaneous leishmaniasis",
    6: "Dracunculiasis",
    7: "Visceral leishmaniasis",
    8: "Measles",
    9: "Meningitis"
}

# Load the model at startup
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts."""
    global model
    try:
        model = load_model('checkpoints/best_model.pt', device)
        logger.info(f"Model loaded successfully on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

class InferenceRequest(BaseModel):
    """Request model for the inference endpoint."""
    timestamp: int  # Unix timestamp in seconds
    latitude: float = 9.0  # Default to Ethiopia coordinates
    longitude: float = 39.5  # Default to Ethiopia coordinates

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 1743281852,
                "latitude": 9.0,
                "longitude": 39.5
            }
        }

class DiseasePrediction(BaseModel):
    """Output model for disease predictions."""
    disease_name: str
    probability: float

class InferenceResponse(BaseModel):
    """Response model for the inference endpoint."""
    predictions: List[DiseasePrediction]
    weather_data: List[Dict[str, float]]
    location: Dict[str, float]  # Added location info to response

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {"disease_name": "Cholera", "probability": 0.15},
                    {"disease_name": "Malaria", "probability": 0.23}
                ],
                "weather_data": [
                    {
                        "temperature": 25.5,
                        "humidity": 60.0,
                        "precipitation": 0.0,
                        "wind_speed": 10.0
                    }
                ],
                "location": {
                    "latitude": 9.0,
                    "longitude": 39.5
                }
            }
        }

def fetch_weather_data(timestamp: int, latitude: float, longitude: float) -> List[Dict[str, float]]:
    """
    Fetch weather data from Open-Meteo API for the given timestamp and 7 days before.
    Uses historical API for past dates and forecast API for future dates.
    Cannot be before 2016-01-01.

    Args:
        timestamp (int): Unix timestamp in seconds
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate

    Returns:
        List[Dict[str, float]]: List of weather data points
    """
    try:
        # Convert timestamp to datetime
        end_date = datetime.fromtimestamp(timestamp)
        start_date = end_date - timedelta(days=6)  # 7 days including the end date
        current_date = datetime.now()

        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching weather data from {start_date_str} to {end_date_str} for coordinates ({latitude}, {longitude})")

        # Determine which API to use based on whether the date is in the past
        if end_date < current_date:
            url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
            logger.info("Using historical API for past dates")
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            logger.info("Using forecast API for future dates")

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean,precipitation_sum",
            "start_date": start_date_str,
            "end_date": end_date_str,
            "timezone": "auto"
        }

        logger.info(f"Making API request with params: {params}")

        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Log the raw response for debugging
        logger.info(f"Raw API response: {data}")

        # Extract and format weather data
        weather_data = []
        daily_data = data["daily"]

        # Get the indices for our date range
        try:
            start_idx = daily_data["time"].index(start_date_str)
            end_idx = daily_data["time"].index(end_date_str) + 1
            logger.info(f"Found date indices: start={start_idx}, end={end_idx}")
        except ValueError as e:
            logger.error(f"Date not found in API response. Available dates: {daily_data['time']}")
            raise HTTPException(
                status_code=400,
                detail=f"Date not found in API response. Available dates: {daily_data['time']}"
            )

        # Extract data for our date range
        for i in range(start_idx, end_idx):
            # Get the date for this data point
            date = daily_data["time"][i]

            # Extract weather data with better null handling
            weather_point = {
                "temperature": daily_data["temperature_2m_mean"][i],
                "humidity": daily_data["relative_humidity_2m_mean"][i],
                "precipitation": daily_data["precipitation_sum"][i],
                "wind_speed": daily_data["wind_speed_10m_mean"][i]
            }

            # Log raw values before conversion
            logger.info(f"Raw weather values for {date}: {weather_point}")

            # Convert to float, handling None values
            weather_point = {
                k: float(v) if v is not None else 0.0
                for k, v in weather_point.items()
            }

            # Log converted values
            logger.info(f"Converted weather values for {date}: {weather_point}")

            weather_data.append(weather_point)

        logger.info(f"Successfully fetched {len(weather_data)} weather data points")
        return weather_data

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch weather data from API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch weather data: {str(e)}"
        )

@app.post("/inference", response_model=InferenceResponse)
async def make_inference(request: InferenceRequest):
    """
    Make disease outbreak predictions based on weather conditions.

    This endpoint takes a timestamp and coordinates, fetches weather data for that location
    and date range, then returns predicted probabilities for various diseases.

    Args:
        request (InferenceRequest): Contains a Unix timestamp and coordinates

    Returns:
        InferenceResponse: Contains predicted probabilities for each disease, weather data used,
                         and the location coordinates

    Raises:
        HTTPException: If there's an error during weather data fetching or inference
    """
    try:
        # Fetch weather data
        weather_data = fetch_weather_data(request.timestamp, request.latitude, request.longitude)

        if len(weather_data) != 7:
            raise HTTPException(
                status_code=500,
                detail=f"Expected 7 weather data points, got {len(weather_data)}"
            )

        # Convert weather data to tensor
        features = []
        for weather in weather_data:
            features.append([
                weather["temperature"],
                weather["humidity"],
                weather["precipitation"],
                weather["wind_speed"]
            ])

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Make prediction
        predictions = predict(model, features_tensor, device)

        # Format response with disease names
        response_predictions = [
            DiseasePrediction(
                disease_name=DISEASE_LABELS[i],
                probability=float(prob)
            )
            for i, prob in enumerate(predictions[0])
        ]

        return InferenceResponse(
            predictions=response_predictions,
            weather_data=weather_data,
            location={
                "latitude": request.latitude,
                "longitude": request.longitude
            }
        )

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during inference: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
