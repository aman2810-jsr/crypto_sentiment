import pandas as pd

# Total rows to load and rows per chunk
total_rows = 2_000_000
chunk_size = 50_000

# Read only the first 20 million rows
df = pd.read_csv('data/Bitcoin_tweets.csv', nrows=total_rows, low_memory=False)

# Split and save in chunks of 1 million rows
for i in range(0, total_rows, chunk_size):
    chunk = df.iloc[i:i + chunk_size]
    chunk.to_csv(f'tweets_part_{(i // chunk_size) + 1}.csv', index=False)
