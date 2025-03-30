#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the Ethiopia CHIRPS dataset with precise boundaries
input_file = "/Users/arman/Desktop/agi-house-25/data/chirps_csv/ethiopia_precise_chirps_data.csv"
output_file = "/Users/arman/Desktop/agi-house-25/data/chirps_csv/ethiopia_precise_chirps_clean.csv"

print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

print(f"Original dataset shape: {df.shape}")

# Display basic statistics
print("\nBasic statistics before cleaning:")
print(df['precipitation'].describe())

# Remove invalid precipitation values (often stored as -9999)
df = df[df['precipitation'] > -999]  # Keep only values greater than -999

# Additional cleaning if needed
df = df.dropna()  # Remove any rows with NaN values

print(f"\nCleaned dataset shape: {df.shape}")
print("\nBasic statistics after cleaning:")
print(df['precipitation'].describe())

# Save the cleaned dataset
df.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to {output_file}")

# Create a directory for plots if it doesn't exist
plots_dir = "/Users/arman/Desktop/agi-house-25/data/chirps_csv/precise_plots"
os.makedirs(plots_dir, exist_ok=True)

# Plot precipitation distribution
plt.figure(figsize=(10, 6))
plt.hist(df['precipitation'], bins=50)
plt.title('Distribution of Precipitation Values for Ethiopia (Precise Boundaries)')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.savefig(f"{plots_dir}/precipitation_distribution.png")
plt.close()

# Plot precipitation by month (averaged)
monthly_avg = df.groupby('month')['precipitation'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(monthly_avg['month'], monthly_avg['precipitation'])
plt.title('Average Precipitation by Month for Ethiopia (Precise Boundaries)')
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
plt.title('Average Precipitation by Year for Ethiopia (Precise Boundaries)')
plt.xlabel('Year')
plt.ylabel('Average Precipitation (mm)')
plt.grid(True)
plt.savefig(f"{plots_dir}/precipitation_by_year.png")
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
plt.title('Precipitation Patterns in Ethiopia by Year and Month (Precise Boundaries)')
plt.xlabel('Month')
plt.ylabel('Year')
plt.xticks(range(12), range(1, 13))
plt.yticks(range(len(pivot_df.index)), pivot_df.index)
plt.savefig(f"{plots_dir}/precipitation_heatmap.png")
plt.close()

print(f"Plots saved to {plots_dir}")
print("\nData is now ready for model training!") 