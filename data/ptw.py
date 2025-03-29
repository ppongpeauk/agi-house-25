import pandas as pd
from datetime import datetime
import os

def merge_weather_and_disease_data(weather_file, disease_file, output_file):
    """
    Merge weather and disease data, averaging weather metrics across days.

    Args:
        weather_file (str): Path to input weather CSV file
        disease_file (str): Path to input disease CSV file
        output_file (str): Path to output merged CSV file
    """
    # Read the weather data
    weather_df = pd.read_csv(weather_file)
    print(f"Weather data shape: {weather_df.shape}")
    print("Weather data sample:\n", weather_df.head())

    # Convert timestamp to date
    weather_df['date'] = pd.to_datetime(weather_df['dt'], unit='s').dt.date

    # Calculate daily averages for weather metrics
    daily_weather = weather_df.groupby('date').agg({
        'temp': 'mean',
        'humidity': 'mean',
        'rain_1h': 'sum',  # Sum precipitation for the day
        'wind_speed': 'mean'
    }).reset_index()

    # Rename columns to match target format
    daily_weather.columns = ['date', 'temperature', 'humidity', 'precipitation', 'wind_speed']

    # Read disease data
    disease_df = pd.read_csv(disease_file)
    print(f"\nDisease data shape: {disease_df.shape}")
    print("Disease data sample:\n", disease_df.head())

    # Convert date column to datetime.date for consistent merging
    disease_df['date'] = pd.to_datetime(disease_df['date']).dt.date

    # Convert disease columns to binary (0 for no outbreak, 1 for outbreak)
    disease_columns = [col for col in disease_df.columns if col.startswith('disease_')]
    for col in disease_columns:
        # Convert to numeric, then to binary (0 or 1)
        disease_df[col] = pd.to_numeric(disease_df[col], errors='coerce').fillna(0)
        disease_df[col] = (disease_df[col] > 0).astype(int)  # Convert to binary: 0 for no outbreak, 1 for outbreak
        print(f"\n{col} value counts:\n{disease_df[col].value_counts()}")

    # Merge weather and disease data
    merged_df = pd.merge(daily_weather, disease_df, on='date', how='left')
    print(f"\nMerged data shape: {merged_df.shape}")
    print("Merged data sample:\n", merged_df.head())

    # Fill missing disease values with 0 (no outbreak)
    merged_df[disease_columns] = merged_df[disease_columns].fillna(0)

    # Ensure all dates from weather data are present
    all_dates = pd.DataFrame({'date': daily_weather['date'].unique()})
    final_df = pd.merge(all_dates, merged_df, on='date', how='left')

    # Sort by date
    final_df = final_df.sort_values('date')

    # Verify all disease columns are binary
    for col in disease_columns:
        if not final_df[col].isin([0, 1]).all():
            raise ValueError(f"Column {col} contains non-binary values")

    # Save to output file
    final_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully merged data and saved to {output_file}")

    # Print some statistics
    print(f"\nData Statistics:")
    print(f"Total number of days: {len(final_df)}")
    print(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
    print(f"Number of days with disease data: {len(disease_df)}")
    print("\nDisease outbreak counts:")
    for col in disease_columns:
        count = final_df[col].sum()
        print(f"{col}: {count} outbreaks")

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input and output file paths
    weather_file = os.path.join(script_dir, "ethiopia_weather.csv")
    disease_file = os.path.join(script_dir, "diseases_by_date.csv")
    output_file = os.path.join(script_dir, "merged_data.csv")

    # Merge the files
    merge_weather_and_disease_data(weather_file, disease_file, output_file)
