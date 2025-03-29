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
            print(f"Columns in file: {df.columns.tolist()}")
            print(f"Sample data:\n{df.head()}")

            # Convert date columns to datetime
            df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d')
            df['end_date'] = pd.to_datetime(df['end_date'], format='%Y-%m-%d')

            # Convert disease columns to binary (0 for no outbreak, 1 for outbreak)
            disease_columns = [f'disease_{i}' for i in range(10)]
            for col in disease_columns:
                if col in df.columns:
                    # Convert to numeric, then to binary (0 or 1)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    df[col] = (df[col] > 0).astype(int)  # Convert to binary: 0 for no outbreak, 1 for outbreak
                    print(f"Column {col} values after conversion:\n{df[col].value_counts()}")

            # Generate daily rows for each outbreak period
            for _, row in df.iterrows():
                date_range = pd.date_range(start=row['start_date'], end=row['end_date'], freq='D')
                daily_rows = pd.DataFrame({
                    'date': date_range,
                    'country': row['country']
                })

                # Add disease columns - preserve all disease values for each day
                for col in disease_columns:
                    if col in df.columns:
                        daily_rows[col] = row[col]
                    else:
                        daily_rows[col] = 0  # Default to no outbreak

                all_outbreaks.append(daily_rows)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    if not all_outbreaks:
        print("Warning: No outbreak data was processed")
        return pd.DataFrame(columns=['date', 'country'] + [f'disease_{i}' for i in range(10)])

    # Combine all outbreaks
    combined_df = pd.concat(all_outbreaks, ignore_index=True)
    print(f"\nCombined data shape: {combined_df.shape}")
    print("Sample of combined data:\n", combined_df.head())

    # Group by date and country to handle overlapping outbreaks
    # Use max to ensure we keep 1 if any outbreak had that disease
    grouped_df = combined_df.groupby(['date', 'country']).agg({
        **{f'disease_{i}': 'max' for i in range(10)}
    }).reset_index()

    # Ensure all disease columns are integers
    disease_columns = [f'disease_{i}' for i in range(10)]
    grouped_df[disease_columns] = grouped_df[disease_columns].astype(int)

    print("\nGrouped data statistics:")
    for col in disease_columns:
        print(f"{col}: {grouped_df[col].sum()} outbreaks")

    return grouped_df

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

    # Fill missing values with 0 for disease columns (0 means no outbreak)
    disease_columns = [f'disease_{i}' for i in range(10)]
    result_df[disease_columns] = result_df[disease_columns].fillna(0).astype(int)

    # Save to CSV
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nGenerated {OUTPUT_FILE} with data from {START_YEAR} to {END_YEAR}")
    print(f"Total rows: {len(result_df)}")
    print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
    print(f"Countries: {result_df['country'].nunique()}")

    # Print disease statistics
    print("\nFinal Disease Statistics:")
    for col in disease_columns:
        count = result_df[col].sum()
        print(f"{col}: {count} outbreaks")

    # Print days with multiple diseases
    multiple_diseases = result_df[disease_columns].sum(axis=1)
    days_with_multiple = (multiple_diseases > 1).sum()
    print(f"\nDays with multiple outbreaks: {days_with_multiple}")

if __name__ == "__main__":
    main()
