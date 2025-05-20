import nltk
import os
import emoji
#from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

import os
import nltk

nltk_data_path = r'C:\Users\aj281\AppData\Roaming\nltk_data\tokenizers\punkt'  # use your actual path here, the parent directory of 'tokenizers/punkt'

os.environ['NLTK_DATA'] = nltk_data_path

# Also add this to nltk.data.path list to be safe
nltk.data.path.append(nltk_data_path)


# Download once, at module load time or before multiprocessing starts
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def pre_clean(text):
    return emoji.demojize(text, delimiters=(" ", " ")).lower()

def fast_lemma(text):
    #tokens = word_tokenize(pre_clean(text))
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return " ".join(lemmas)

def process_chunk(chunk):
    return [fast_lemma(text) for text in chunk]

def parallel_fast_preprocess(texts, chunk_size=10000, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))
    return [text for sublist in results for text in sublist]

def run_preprocessing(csv_path="../dataset/monthly_tweets/tweets_with_sentiment_feb_to_july.csv"):
    df = pd.read_csv(csv_path)
    # Your cleaning steps here
    df = df[df['text'].notna()]
    texts = df['text'].tolist()
    df['preprocess_text'] = parallel_fast_preprocess(texts)
    return df

if __name__ == "__main__":
    # This guard is necessary for multiprocessing on Windows
    df = run_preprocessing()
    print(df.head())
