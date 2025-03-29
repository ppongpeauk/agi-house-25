import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_cholera_data(num_samples=1000, outbreak_probability=0.3):
    """
    Create realistic disease outbreak data based on research parameters.

    Parameters that correlate with disease outbreaks:
    - Temperature: Optimal range 20-30°C
    - Precipitation: Heavy rainfall and flooding
    - Humidity: High humidity (>60%)
    - Wind speed: Can affect disease spread

    Args:
        num_samples (int): Number of samples to generate
        outbreak_probability (float): Probability of an outbreak occurring
    """
    # Generate dates for the past year
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(num_samples + 7)]
    dates.reverse()

    # Initialize data lists
    data = []

    # Parameters for realistic data generation
    base_temp = 25  # Base temperature in Celsius
    temp_variance = 5  # Temperature variation
    base_humidity = 70  # Base humidity percentage
    humidity_variance = 10  # Humidity variation
    base_precip = 5  # Base precipitation in mm
    precip_variance = 15  # Precipitation variation
    base_wind = 10  # Base wind speed in km/h
    wind_variance = 5  # Wind speed variation

    # Generate seasonal patterns
    seasonal_temp = np.sin(np.linspace(0, 4*np.pi, num_samples + 7)) * 3  # Seasonal temperature variation
    seasonal_precip = np.sin(np.linspace(0, 4*np.pi, num_samples + 7)) * 10  # Seasonal precipitation variation

    # Generate data
    for i in range(num_samples + 7):
        # Add seasonal variations
        temp = base_temp + seasonal_temp[i] + np.random.normal(0, temp_variance)
        precip = base_precip + seasonal_precip[i] + np.random.exponential(precip_variance)
        humidity = base_humidity + np.random.normal(0, humidity_variance)
        wind = base_wind + np.random.normal(0, wind_variance)

        # Ensure values are within realistic ranges
        temp = np.clip(temp, 15, 35)
        humidity = np.clip(humidity, 40, 100)
        precip = np.clip(precip, 0, 100)
        wind = np.clip(wind, 0, 50)

        # Generate disease presence (0 or 1) for each disease
        diseases = [0] * 10  # Initialize all diseases as 0
        if random.random() < outbreak_probability:
            # Randomly select 1-3 diseases to be present
            num_diseases = random.randint(1, 3)
            disease_indices = random.sample(range(10), num_diseases)
            for idx in disease_indices:
                diseases[idx] = 1

        data.append({
            'date': dates[i].strftime('%Y-%m-%d'),
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'precipitation': round(precip, 2),
            'wind_speed': round(wind, 2),
            **{f'disease_{j}': diseases[j] for j in range(10)}
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('dummy_data.csv', index=False)
    print(f"Generated {len(df)} samples with disease outbreaks")

if __name__ == "__main__":
    create_cholera_data(num_samples=5000, outbreak_probability=0.3)
