#!/bin/bash

# Create directory for downloads
mkdir -p /Users/arman/Desktop/agi-house-25/data/chirps_data
cd /Users/arman/Desktop/agi-house-25/data/chirps_data

# Base URL for daily data
BASE_URL="https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs"

# Check if GNU Parallel is installed
if command -v parallel > /dev/null 2>&1; then
  echo "Using GNU Parallel for faster downloads..."
  PARALLEL_AVAILABLE=true
else
  echo "GNU Parallel not found. Using sequential downloads."
  echo "Consider installing GNU Parallel for faster downloads: brew install parallel"
  PARALLEL_AVAILABLE=false
fi

# Create a download function
download_file() {
  local year=$1
  local month=$2
  local day=$3
  local filename="chirps-v2.0.$year.$month.$day.tif.gz"
  local output_file="/Users/arman/Desktop/agi-house-25/data/chirps_data/$filename"
  
  # Skip download if the file already exists (either .gz or extracted)
  if [ -f "${output_file%.gz}" ] || [ -f "$output_file" ]; then
    echo "Skipping $filename - already exists"
    return
  fi
  
  echo "Downloading $filename..."
  wget -q --show-progress --tries=3 --timeout=30 -c "$BASE_URL/$filename" -O "$output_file"
  
  # Only extract if download was successful
  if [ $? -eq 0 ]; then
    echo "Extracting $filename..."
    gunzip -f "$output_file"
  else
    echo "Failed to download $filename"
  fi
}

export -f download_file
export BASE_URL

# Create a list of download jobs
download_list="/Users/arman/Desktop/agi-house-25/data/chirps_data/download_list.txt"
> "$download_list"  # Clear the file

# Generate the download list
for year in {2020..2025}; do
  # Determine max month based on year
  if [ "$year" -eq 2025 ]; then
    max_month=3  # March 2025
  else
    max_month=12
  fi
  
  for month in $(seq -f "%02g" 1 $max_month); do
    # Get the number of days in the month
    if [ "$month" == "02" ]; then
      # Check for leap year
      if [ $(($year % 4)) -eq 0 ] && [ $(($year % 100)) -ne 0 ] || [ $(($year % 400)) -eq 0 ]; then
        days=29
      else
        days=28
      fi
    elif [ "$month" == "04" ] || [ "$month" == "06" ] || [ "$month" == "09" ] || [ "$month" == "11" ]; then
      days=30
    else
      days=31
    fi
    
    # Add to download list
    for day in $(seq -f "%02g" 1 $days); do
      echo "$year $month $day" >> "$download_list"
    done
  done
done

# Download files in parallel or sequentially
if [ "$PARALLEL_AVAILABLE" = true ]; then
  # Number of parallel downloads (adjust based on your connection)
  PARALLEL_JOBS=5
  echo "Starting downloads with $PARALLEL_JOBS parallel processes..."
  
  # Use parallel to download multiple files simultaneously
  cat "$download_list" | parallel -j $PARALLEL_JOBS download_file {1} {2} {3}
else
  # Sequential downloads
  while read -r year month day; do
    download_file "$year" "$month" "$day"
  done < "$download_list"
fi

rm "$download_list"
echo "Daily CHIRPS data download complete (2020-March 2025)!" 