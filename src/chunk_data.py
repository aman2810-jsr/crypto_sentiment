import pandas as pd
from pathlib import Path

# Define necessary columns
keep_cols = ['date', 'text','is_retweet']
start_date = pd.to_datetime("2021-02-01")
end_date = pd.to_datetime("2021-09-30")

# Output directory
output_dir = Path("dataset/monthly_tweets")
output_dir.mkdir(parents=True, exist_ok=True)

# Dictionary to collect grouped data
monthly_data = {}

# Read the CSV in chunks
chunksize = 100_000
for chunk in pd.read_csv("dataset/Bitcoin_tweets.csv", usecols=keep_cols, chunksize=chunksize,engine='python',on_bad_lines='skip'):
    # Ensure 'date' is datetime type
    chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')

    # Drop rows where 'date' couldn't be parsed
    chunk = chunk.dropna(subset=['date'])

    # Filter by date range
    filtered = chunk[(chunk['date'] >= start_date) & (chunk['date'] <= end_date)]

    # Group and store
    for (year, month), group in filtered.groupby([filtered['date'].dt.year, filtered['date'].dt.month]):
        key = (year, month)
        if key not in monthly_data:
            monthly_data[key] = []
        monthly_data[key].append(group)

# Save each month's data
for (year, month), chunks in monthly_data.items():
    month_name = pd.to_datetime(f'{year}-{month:02d}-01').strftime('%B')
    filename = output_dir / f"tweets_{month_name}_{year}.csv"
    pd.concat(chunks).to_csv(filename, index=False)
    print(f"Saved: {filename}")
