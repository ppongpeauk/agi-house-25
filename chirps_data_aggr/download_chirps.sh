#!/bin/bash

# Create directory for downloads
mkdir -p /Users/arman/Desktop/agi-house-25/data/chirps_data

# Base URL
BASE_URL="https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_monthly/tifs"

# Download files from January 2020 to March 2025
for year in {2020..2025}; do
  # Determine max month based on year
  if [ "$year" -eq 2025 ]; then
    max_month=3
  else
    max_month=12
  fi
  
  for month in $(seq -f "%02g" 1 "$max_month"); do
    # Skip future months for current year
    if [ "$year" -eq 2025 ] && [ "$month" -gt 3 ]; then
      break
    fi
    
    filename="chirps-v2.0.$year.$month.tif.gz"
    output_file="/Users/arman/Desktop/agi-house-25/data/chirps_data/$filename"
    
    echo "Downloading $filename..."
    wget -c "$BASE_URL/$filename" -O "$output_file"
    
    # Extract the file
    echo "Extracting $filename..."
    gunzip -f "$output_file"
    
    # Small delay to avoid overwhelming the server
    sleep 2
  done
done

echo "Download complete!" 