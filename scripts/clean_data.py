import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

# Set up NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """
    Standard Cleaning logic from bootcamp labs:
    - Lowercase
    - Remove punctuation/symbols (regex)
    - Tokenize
    - Remove stopwords
    """
    if pd.isna(text):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and symbols
    # Pattern: remove characters that are in string.punctuation
    pattern = re.compile('[%s]' % re.escape(string.punctuation))
    text = pattern.sub('', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Remove Stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    return " ".join(filtered_tokens)

# Paths (Absolute for reliability)
BASE_PATH = "/Users/basstianlopez/Desktop/it-studies/ironhack/week_10/Project 2 /project-nlp-challenge"
DATA_PATH = os.path.join(BASE_PATH, "dataset/data.csv")
OUTPUT_PATH = os.path.join(BASE_PATH, "dataset/cleaned_data.csv")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print(f"Initial shape: {df.shape}")

# 1. Combine title and text for a richer context
print("Combining 'title' and 'text' columns...")
df['full_text'] = df['title'] + " " + df['text']

# 2. Applying cleaning
print("Cleaning text (this might take a minute)...")
df['cleaned_text'] = df['full_text'].apply(clean_text)

# 3. Handle labels (already 0/1, but ensuring numeric type)
df['label'] = pd.to_numeric(df['label'])

# 4. Save
print(f"Saving cleaned data to: {OUTPUT_PATH}")
# Keep original columns + cleaned_text for transparency
df.to_csv(OUTPUT_PATH, index=False)

print("Done phase 1.2 scrubbing.")
