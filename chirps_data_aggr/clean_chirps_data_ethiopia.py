#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Load the Ethiopia CHIRPS dataset
input_file = "/Users/arman/Desktop/agi-house-25/data/chirps_csv/ethiopia_precise_chirps_data.csv"
output_file = "/Users/arman/Desktop/agi-house-25/data/chirps_csv/precipitation_ethiopia_chirps_clean.csv"

print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

print(f"Original dataset shape: {df.shape}")

# Check if we already have a date column
if 'date' not in df.columns:
    print("Adding date column...")
    # Create a proper date column using year, month, and day
    df['date'] = df.apply(lambda row: f"{int(row['year'])}-{int(row['month']):02d}-{int(row['day']):02d}", axis=1)
    
    # Convert date strings to datetime objects for better handling
    df['date'] = pd.to_datetime(df['date'])
else:
    # Make sure date is in datetime format
    print("Date column exists, ensuring it's in the correct format...")
    df['date'] = pd.to_datetime(df['date'])

# Make sure all required columns are present and in the right order
if 'day' in df.columns:
    columns_order = ['longitude', 'latitude', 'precipitation', 'year', 'month', 'day', 'date']
else:
    columns_order = ['longitude', 'latitude', 'precipitation', 'year', 'month', 'date']
    
# Only keep columns in the order list
df = df[columns_order]

# Save the updated dataset
df.to_csv(output_file, index=False)
print(f"Updated dataset saved to {output_file}")

# Display basic statistics
print("\nBasic statistics:")
print(df['precipitation'].describe())

# Create a directory for plots if it doesn't exist
plots_dir = "/Users/arman/Desktop/agi-house-25/data/chirps_csv/plots"
os.makedirs(plots_dir, exist_ok=True)

# Plot precipitation distribution
plt.figure(figsize=(10, 6))
plt.hist(df['precipitation'], bins=50)
plt.title('Distribution of Precipitation Values for Ethiopia')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.savefig(f"{plots_dir}/precipitation_distribution.png")
plt.close()

# Plot precipitation by month (averaged)
monthly_avg = df.groupby('month')['precipitation'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(monthly_avg['month'], monthly_avg['precipitation'])
plt.title('Average Precipitation by Month for Ethiopia')
plt.xlabel('Month')
plt.ylabel('Average Precipitation (mm)')
plt.xticks(range(1, 13))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{plots_dir}/precipitation_by_month.png")
plt.close()

# Plot precipitation by year (averaged)
yearly_avg = df.groupby('year')['precipitation'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(yearly_avg['year'], yearly_avg['precipitation'], marker='o', linewidth=2)
plt.title('Average Precipitation by Year for Ethiopia')
plt.xlabel('Year')
plt.ylabel('Average Precipitation (mm)')
plt.grid(True)
plt.savefig(f"{plots_dir}/precipitation_by_year.png")
plt.close()

# Daily time series plot (this is new for daily data)
plt.figure(figsize=(15, 6))
daily_avg = df.groupby('date')['precipitation'].mean().reset_index()
plt.plot(daily_avg['date'], daily_avg['precipitation'], linewidth=1)
plt.title('Daily Precipitation for Ethiopia (2020-March 2025)')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plots_dir}/daily_precipitation.png")
plt.close()

# If date column exists, create a time series plot
if 'date' in df.columns:
    time_series = df.groupby('date')['precipitation'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(time_series['date'], time_series['precipitation'], marker='.', markersize=3, linewidth=1)
    plt.title('Precipitation Time Series for Ethiopia (2020-March 2025)')
    plt.xlabel('Date')
    plt.ylabel('Average Precipitation (mm)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/precipitation_time_series.png")
    plt.close()

# Create a heatmap of precipitation by month and year
pivot_df = df.pivot_table(
    index='year', 
    columns='month', 
    values='precipitation',
    aggfunc='mean'
)

plt.figure(figsize=(12, 8))
heatmap = plt.imshow(pivot_df, cmap='YlGnBu', aspect='auto')
plt.colorbar(heatmap, label='Average Precipitation (mm)')
plt.title('Precipitation Patterns in Ethiopia by Year and Month')
plt.xlabel('Month')
plt.ylabel('Year')
plt.xticks(range(12), range(1, 13))
plt.yticks(range(len(pivot_df.index)), pivot_df.index)
plt.savefig(f"{plots_dir}/precipitation_heatmap.png")
plt.close()

print(f"Plots saved to {plots_dir}")
print("\nData is now ready for model training!") 