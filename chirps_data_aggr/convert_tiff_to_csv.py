#!/usr/bin/env python3
import os
import rasterio
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import random
from rasterio.mask import mask
from shapely.geometry import box, Polygon
import geopandas as gpd

# Input and output directories
input_dir = "/Users/arman/Desktop/agi-house-25/data/chirps_data"
output_dir = "/Users/arman/Desktop/agi-house-25/data/chirps_csv"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all TIFF files
tiff_files = sorted(glob(os.path.join(input_dir, "*.tif")))
print(f"Found {len(tiff_files)} TIFF files to convert")

# Define Ethiopia's precise boundary coordinates
# More precise coordinates for Ethiopia
ETHIOPIA_COORDINATES = [
    # These coordinates define a more accurate outline of Ethiopia
    (33.00, 3.40),   # Southwest corner
    (34.80, 4.60),   # Western border point
    (36.50, 4.40),   # Western border point
    (36.90, 6.20),   # Western border point
    (37.90, 8.40),   # Western border point
    (37.80, 10.60),  # Western border point
    (37.40, 12.40),  # Northwestern corner
    (37.80, 14.10),  # Northern border point
    (38.90, 14.90),  # Northern border point
    (39.80, 14.80),  # Northern border point
    (41.80, 15.00),  # Northeastern corner
    (43.10, 12.70),  # Eastern border point
    (43.30, 11.80),  # Eastern border point
    (43.90, 9.50),   # Eastern border point
    (44.90, 8.00),   # Eastern border point
    (46.90, 7.90),   # Eastern border point
    (47.80, 8.00),   # Eastern border point
    (43.10, 5.00),   # Southeastern corner
    (41.90, 3.80),   # Southern border point
    (40.00, 4.20),   # Southern border point
    (38.00, 3.90),   # Southern border point
    (35.90, 3.40),   # Southern border point
    (33.00, 3.40)    # Back to Southwest corner
]

# Set sampling rate - we'll keep 1/10 of the data points
SAMPLE_RATE = 0.1

print(f"Extracting data for Ethiopia with precise boundaries")
print(f"Sampling rate: {SAMPLE_RATE} (keeping 10% of data points)")

# Create a polygon geometry for Ethiopia
ethiopia_poly = Polygon(ETHIOPIA_COORDINATES)
ethiopia_geom = [ethiopia_poly.__geo_interface__]

# Process each TIFF file
for tiff_file in tqdm(tiff_files):
    # Get base filename without extension
    base_name = os.path.basename(tiff_file).replace('.tif', '')
    output_file = os.path.join(output_dir, f"{base_name}_ethiopia_precise.csv")
    
    # Extract date from filename (format: chirps-v2.0.YYYY.MM.tif)
    parts = base_name.split('.')
    year = parts[2]
    month = parts[3]
    
    try:
        # Open the raster file
        with rasterio.open(tiff_file) as src:
            # Mask the raster data with our Ethiopia polygon
            out_image, out_transform = mask(src, ethiopia_geom, crop=True)
            
            # Get data from the first band
            data = out_image[0]
            
            # Create a meshgrid of coordinates based on the new transform
            height, width = data.shape
            rows, cols = np.indices((height, width))
            
            # Flatten arrays for easier processing
            rows = rows.flatten()
            cols = cols.flatten()
            data_flat = data.flatten()
            
            # Filter out nodata values
            valid_indices = np.where(~np.isnan(data_flat) & (data_flat != src.nodata if src.nodata is not None else True))
            valid_rows = rows[valid_indices]
            valid_cols = cols[valid_indices]
            valid_data = data_flat[valid_indices]
            
            # Sample 10% of the points
            num_points = len(valid_data)
            sample_size = int(num_points * SAMPLE_RATE)
            
            if num_points > 0:
                sample_indices = np.random.choice(num_points, size=sample_size, replace=False)
                sampled_rows = valid_rows[sample_indices]
                sampled_cols = valid_cols[sample_indices]
                sampled_values = valid_data[sample_indices]
                
                # Convert pixel coordinates to geographic coordinates
                xs, ys = rasterio.transform.xy(out_transform, sampled_rows, sampled_cols)
                
                # Create DataFrame
                df = pd.DataFrame({
                    'longitude': xs,
                    'latitude': ys,
                    'precipitation': sampled_values,
                    'year': year,
                    'month': month
                })
                
                # Save to CSV
                df.to_csv(output_file, index=False)
                print(f"Converted {tiff_file} to {output_file} with {len(df)} data points for Ethiopia (10% sample)")
            else:
                print(f"No valid data points found for Ethiopia in {tiff_file}")
    except Exception as e:
        print(f"Error processing {tiff_file}: {e}")
        continue

print("Conversion complete!")

# Create a merged CSV of all Ethiopia data
print("Creating merged dataset for Ethiopia...")
all_csvs = sorted(glob(os.path.join(output_dir, "*_ethiopia_precise.csv")))

if not all_csvs:
    print("No CSV files were created. Check for errors in processing.")
else:
    merged_output = os.path.join(output_dir, "ethiopia_precise_chirps_data.csv")
    
    # Process in chunks to avoid memory issues
    chunk_size = 10  # Process 10 files at a time
    
    # Initialize merged dataframe with first chunk
    for i in range(0, len(all_csvs), chunk_size):
        chunk_files = all_csvs[i:i+chunk_size]
        try:
            chunk_df = pd.concat([pd.read_csv(csv_file) for csv_file in chunk_files])
            
            # Write mode: append if not first chunk, write if first chunk
            mode = 'a' if i > 0 else 'w'
            # Don't write header if appending
            header = i == 0
            
            chunk_df.to_csv(merged_output, index=False, mode=mode, header=header)
            print(f"Processed chunk {i//chunk_size + 1}/{(len(all_csvs) + chunk_size - 1)//chunk_size}")
        except Exception as e:
            print(f"Error processing chunk {i//chunk_size + 1}: {e}")
            continue
    
    print(f"Merged dataset saved to {merged_output}")
    print("Ethiopia precipitation data with precise boundaries (10% sample) is now ready for model training!") 