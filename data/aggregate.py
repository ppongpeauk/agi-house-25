import pandas as pd
import os
from datetime import datetime, timedelta
import glob

def generate_daily_timeline(start_year, end_year):
    """
    Generate a DataFrame with daily dates from start_year to end_year.

    Args:
        start_year (int): Starting year
        end_year (int): Ending year

    Returns:
        pd.DataFrame: DataFrame with daily dates
    """
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    return pd.DataFrame({'date': date_range})

def process_outbreak_files(outbreaks_dir):
    """
    Process all outbreak CSV files in the specified directory.

    Args:
        outbreaks_dir (str): Directory containing outbreak CSV files

    Returns:
        pd.DataFrame: Combined DataFrame of all outbreak data
    """
    # Get all CSV files except example and output files
    csv_files = glob.glob(os.path.join(outbreaks_dir, '*.csv'))
    csv_files = [f for f in csv_files if not any(x in f for x in ['example.csv', 'output_example.csv'])]

    if not csv_files:
        print(f"Warning: No CSV files found in {outbreaks_dir}")
        return pd.DataFrame(columns=['date', 'country'] + [f'disease_{i}' for i in range(10)])

    print(f"Found {len(csv_files)} CSV files to process")
    all_outbreaks = []

    for file in csv_files:
        print(f"Processing file: {file}")
        try:
            # Read CSV with header row
            df = pd.read_csv(file)

            # Convert date columns to datetime
            df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d')
            df['end_date'] = pd.to_datetime(df['end_date'], format='%Y-%m-%d')

            # Generate daily rows for each outbreak period
            for _, row in df.iterrows():
                date_range = pd.date_range(start=row['start_date'], end=row['end_date'], freq='D')
                daily_rows = pd.DataFrame({
                    'date': date_range,
                    'country': row['country']
                })

                # Add disease columns
                for i in range(10):
                    daily_rows[f'disease_{i}'] = row[f'disease_{i}']

                all_outbreaks.append(daily_rows)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    if not all_outbreaks:
        print("Warning: No outbreak data was processed")
        return pd.DataFrame(columns=['date', 'country'] + [f'disease_{i}' for i in range(10)])

    return pd.concat(all_outbreaks, ignore_index=True)

def main():
    # Configuration
    START_YEAR = 2015
    END_YEAR = 2025
    OUTBREAKS_DIR = './outbreaks'
    OUTPUT_FILE = './diseases_by_date.csv'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Generate daily timeline
    timeline_df = generate_daily_timeline(START_YEAR, END_YEAR)

    # Process outbreak files
    outbreaks_df = process_outbreak_files(OUTBREAKS_DIR)

    # Merge timeline with outbreaks
    result_df = pd.merge(timeline_df, outbreaks_df, on='date', how='left')

    # Sort by date and country
    result_df = result_df.sort_values(['date', 'country'])

    # Save to CSV
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {OUTPUT_FILE} with data from {START_YEAR} to {END_YEAR}")
    print(f"Total rows: {len(result_df)}")
    print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
    print(f"Countries: {result_df['country'].nunique()}")

if __name__ == "__main__":
    main()
