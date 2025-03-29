# CHIRPS Precipitation Data for Ethiopia

This project processes CHIRPS daily rainfall data for Ethiopia (2020-March 2025).

## What it does

Downloads and processes daily climate precipitation data specifically for Ethiopia's geographical boundaries, creating a clean dataset for analysis or modeling.

## How to run

1. Download data:
   ```bash
   ./download_chirps.sh
   ```

2. Convert to CSV:
   ```bash
   python3 convert_tiff_to_csv.py
   ```

3. Clean and visualize:
   ```bash
   python3 clean_chirps_data_ethiopia.py
   ```

## Outputs

- **CSV files**: 
  - `precipitation_ethiopia_chirps_clean.csv` - Clean precipitation data ready for analysis with proper date column
  
- **Visualizations**:
  - Daily precipitation time series
  - Monthly precipitation patterns
  - Yearly trends
  - Precipitation distribution
  - Time series visualization
  - Precipitation heatmap by month and year

## Data source

CHIRPS data from Climate Hazards Group, UC Santa Barbara: https://www.chc.ucsb.edu/data/chirps
