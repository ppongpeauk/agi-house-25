import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_cholera_data(num_samples=1000, outbreak_probability=0.3):
    """
    Create realistic cholera outbreak data based on research parameters.

    Parameters that correlate with cholera outbreaks:
    - Temperature: Optimal range 20-30°C
    - Precipitation: Heavy rainfall and flooding
    - Humidity: High humidity (>60%)
    - Water quality: Can be inferred from precipitation patterns
    - Seasonality: Peak during rainy seasons

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

    # Generate outbreak periods
    outbreak_periods = []
    current_period = []
    for _ in range(num_samples + 7):
        if random.random() < outbreak_probability:
            current_period.append(True)
        else:
            if current_period:
                outbreak_periods.append(current_period)
            current_period = [False]

    if current_period:
        outbreak_periods.append(current_period)

    # Generate data
    for i in range(num_samples + 7):
        # Add seasonal variations
        temp = base_temp + seasonal_temp[i] + np.random.normal(0, temp_variance)
        precip = base_precip + seasonal_precip[i] + np.random.exponential(precip_variance)
        humidity = base_humidity + np.random.normal(0, humidity_variance)
        wind = base_wind + np.random.normal(0, wind_variance)

        # Determine if this is an outbreak period
        is_outbreak = any(i in period for period in outbreak_periods)

        # Adjust parameters during outbreak periods
        if is_outbreak:
            # Increase temperature and humidity during outbreaks
            temp += 2
            humidity += 5
            # Increase precipitation variability
            precip *= 1.5
            # Generate higher incidence rates
            incidence = np.random.poisson(15)  # Higher base rate during outbreaks
        else:
            # Generate lower incidence rates for non-outbreak periods
            incidence = np.random.poisson(2)

        # Ensure values are within realistic ranges
        temp = np.clip(temp, 15, 35)
        humidity = np.clip(humidity, 40, 100)
        precip = np.clip(precip, 0, 100)
        wind = np.clip(wind, 0, 50)
        incidence = np.clip(incidence, 0, 50)

        data.append({
            'date': dates[i].strftime('%Y-%m-%d'),
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'precipitation': round(precip, 2),
            'wind_speed': round(wind, 2),
            'disease_incidence': int(incidence),
            'disease_type': 'cholera',
            'is_outbreak': int(is_outbreak)
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('dummy_data.csv', index=False)
    print(f"Generated {len(df)} samples with {df['is_outbreak'].sum()} outbreak periods")

if __name__ == "__main__":
    create_cholera_data(num_samples=5000, outbreak_probability=0.3)
