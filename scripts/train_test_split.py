import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
BASE_PATH = "/Users/basstianlopez/Desktop/it-studies/ironhack/week_10/Project 2 /project-nlp-challenge"
DATA_PATH = os.path.join(BASE_PATH, "dataset/cleaned_data.csv")
TRAIN_PATH = os.path.join(BASE_PATH, "dataset/train.csv")
TEST_PATH = os.path.join(BASE_PATH, "dataset/test.csv")

print(f"Loading cleaned data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# We use the cleaned_text and the label
# Note: Keeping other columns for context if needed, but mainly cleaned_text/label
print("Splitting data (80% Train, 20% Test) with stratify to keep label balance...")
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])

print(f"Train set size: {train_df.shape}")
print(f"Test set size: {test_df.shape}")

print(f"Saving splits to dataset folder...")
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("Phase 1 complete! Data is cleaned, labels are normalized, and a baseline TF-IDF vectorizer is ready.")
