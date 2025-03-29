import pandas as pd
from datetime import datetime
import os

def convert_weather_to_disease_format(input_file, output_file):
    """
    Convert weather data format to match disease data format.

    Args:
        input_file (str): Path to input weather CSV file
        output_file (str): Path to output disease format CSV file
    """
    # Read the weather data
    weather_df = pd.read_csv(input_file)

    # Convert timestamp to date
    weather_df['date'] = pd.to_datetime(weather_df['dt'], unit='s').dt.date

    # Create new dataframe with required columns
    disease_df = pd.DataFrame({
        'date': weather_df['date'],
        'temperature': weather_df['temp'],
        'humidity': weather_df['humidity'],
        'precipitation': weather_df['rain_1h'].fillna(0),  # Use rain_1h as precipitation
        'wind_speed': weather_df['wind_speed'],
        'disease_incidence': 0,  # Default value since we don't have this data
        'disease_type': 'unknown',  # Default value since we don't have this data
        'is_outbreak': 0  # Default value since we don't have this data
    })

    # Save to output file
    disease_df.to_csv(output_file, index=False)
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input and output file paths
    input_file = os.path.join(script_dir, "example-weather.csv")
    output_file = os.path.join(script_dir, "converted_disease_data.csv")

    # Convert the file
    convert_weather_to_disease_format(input_file, output_file)
