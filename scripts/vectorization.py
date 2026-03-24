import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Paths
BASE_PATH = "/Users/basstianlopez/Desktop/it-studies/ironhack/week_10/Project 2 /project-nlp-challenge"
DATA_PATH = os.path.join(BASE_PATH, "dataset/cleaned_data.csv")
MODEL_DIR = os.path.join(BASE_PATH, "models")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"Loading cleaned data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Using 'cleaned_text' for vectorization
print("Fitting TfidfVectorizer (Max 5000 features for baseline transparency)...")
# Using max_features to keep the model interpretable and manageable for local RAM
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X = tfidf.fit_transform(df['cleaned_text'].astype(str))

print(f"Vectorized shape: {X.shape}")

print(f"Saving vectorizer to: {VECTORIZER_PATH}")
joblib.dump(tfidf, VECTORIZER_PATH)

print("Baseline Vectorizer (TF-IDF) fitted and saved successfully.")
