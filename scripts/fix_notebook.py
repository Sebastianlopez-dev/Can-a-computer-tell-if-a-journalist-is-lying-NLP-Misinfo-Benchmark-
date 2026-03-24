import json

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧪 Phase 1: Data Scouting & Cleaning\n",
    "\n",
    "## 🍱 Overview: The Prep-Work of a Data Scientist\n",
    "In this notebook, we transform raw, noisy news data into a structured numerical format that a computer can understand. \n",
    "\n",
    "### 🥘 The Kitchen Preparation (Analogies)\n",
    "*   **For the 5-year-old**: Imagine you are making a soup. You can't just throw the dirt and the peels of the potatoes into the pot. You have to wash them, peel them, and cut them into small pieces. That is what we are doing with our words!\n",
    "*   **For the Senior Executive**: Data quality is the single greatest determinant of model ROI. We are implementing a 'Garbage In, Garbage Out' guardrail by standardizing the text features to ensure the classifier focuses on meaningful semantic patterns rather than typographical noise."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd \n",
    "import re\n",
    "import string\n",
    "import nltk\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.corpus import stopwords\n",
    "import os\n",
    "\n",
    "# Downloading mandatory dictionaries for text analysis\n",
    "nltk.download('stopwords')\n",
    "nltk.download('punkt')\n",
    "\n",
    "print(\"✅ Kitchen tools (libraries) sanitized and ready for use.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Ingestion\n",
    "We load our primary dataset. It contains nearly 40,000 articles, balanced almost 50/50 between Real (1) and Fake (0) news."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_path = 'dataset/data.csv'\n",
    "df = pd.read_csv(data_path)\n",
    "print(f\"Database Volume: {df.shape[0]} articles loaded.\")\n",
    "df.head(2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Advanced Feature Engineering: Fusion (Title + Text)\n",
    "\n",
    "### 🔎 The Strategy: Why combine Title and Body?\n",
    "**Decision**: We created a new column `full_text` by concatenating `title` and `text`.\n",
    "\n",
    "**Why this is better than keeping them separate (for now):**\n",
    "1.  **Context Density**: In fake news, the 'hook' is often in the title (clickbait) while the body supports it with misinformation. By merging them, we ensure our model doesn't miss the correlation between a sensationalized headline and the actual content.\n",
    "2.  **Handling Sparse Data**: Some articles have very short bodies. Including the title guarantees that every record has a minimum amount of semantic data to be processed.\n",
    "3.  **Baseline Simplicity**: As our first model, we want a 'holistic' view of the article. Separating them would double our feature count (title-features vs body-features), which might lead to overfitting on sparse title words."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['full_text'] = df['title'] + \" \" + df['text']\n",
    "print(\"✅ Feature Fusion Complete: 'full_text' column created.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. The Scrubbing (Cleaning Logic)\n",
    "\n",
    "### 🛠 Why these cleaning steps matter?\n",
    "1.  **Lowercase**: Computers see 'Apple' and 'apple' as different entities. Lowercasing collapses them into the same concept, reducing the 'vocabulary' size without losing meaning.\n",
    "2.  **Punctuation Removal**: Symbols like '!!!' or '???' carry emotional weight but can confuse a baseline classifier. We strip them to focus on the CORE vocabulary.\n",
    "3.  **Tokenization**: This is the act of 'slicing the potato'. We break the string into individual units (tokens) so the computer can count them.\n",
    "4.  **Stopword Removal**: Words like 'the', 'is', 'and' appear in almost every sentence. Because they don't help us distinguish between Real and Fake news (they are neutral), we remove them to boost the 'signal' of unique words like 'conspiracy', 'evidence', or 'verified'."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_text(text):\n",
    "    if pd.isna(text):\n",
    "        return \"\"\n",
    "    \n",
    "    # Standard Cleaning Pipeline\n",
    "    text = text.lower() # Normalization\n",
    "    pattern = re.compile('[%s]' % re.escape(string.punctuation))\n",
    "    text = pattern.sub('', text) # Noise Reduction\n",
    "    \n",
    "    tokens = word_tokenize(text) # Atomization\n",
    "    \n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    filtered_tokens = [w for w in tokens if w not in stop_words] # Signal Boosting\n",
    "    \n",
    "    return \" \".join(filtered_tokens)\n",
    "\n",
    "print(\"✅ Implementation: Cleaning pipeline established.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Vectorization (The Flavor Embeddings)\n",
    "We use **TF-IDF (Term Frequency-Inverse Document Frequency)**.\n",
    "\n",
    "### 🧠 Why TF-IDF instead of simple counting?\n",
    "Simple counting rewards a word just for being frequent. **TF-IDF is smarter**: it rewards a word if it is frequent in *one* article but **rare** in the entire database. This highlights truly unique keywords that define 'Fake News' patterns vs 'Real news' patterns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "import joblib\n",
    "\n",
    "# We limit to 5000 features to maintain transparency and avoid 'learning by heart' (overfitting)\n",
    "tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))\n",
    "print(\"✅ Vectorizer Initialized: TF-IDF with Unigrams and Bigrams.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. The Final Slice (Splitting the Data)\n",
    "\n",
    "### 🧪 Why split? \n",
    "If you want to know if a student actually learned math, you don't give them the same problems they saw in class for the final exam. \n",
    "*   **Train Set (80%)**: The 'Classroom examples'.\n",
    "*   **Test Set (20%)**: The 'Final Exam' (data the model has never seen).\n",
    "\n",
    "We use **Stratification** to ensure that both the Train and Test sets have the same percentage of Real vs Fake news as the original data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Separating for visualization/demo in the next phase\n",
    "train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])\n",
    "\n",
    "print(f\"🎓 Training Set: {len(train_df)} specimens\")\n",
    "print(f\"📝 Testing Set: {len(test_df)} specimens\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('/Users/basstianlopez/Desktop/it-studies/ironhack/week_10/Project 2 /project-nlp-challenge/01_data_cleaning_and_embeddings.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
