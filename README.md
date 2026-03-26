# 📰 Fake News Detector: NLP Challenge

This repository contains a comprehensive NLP pipeline for detecting fake news using multiple vectorization strategies and classification models. Developed as part of the Ironhack Data Analytics Bootcamp.

## 🚀 Project Overview
The goal is to classify news articles as **Real (1)** or **Fake (0)**. We transitioned from a simple baseline to a sophisticated **Matrix Comparison** architecture, contrasting traditional count-based methods with modern semantic embeddings.

## 🏗️ Architecture & Methodology
We follow a 3-phase execution plan:

### Phase 1: Data Preparation & Cleaning
- **Logic**: Unified Title + Text features.
- **Cleaning**: Regex-based scrubbing, NLTK stopword removal, and normalization.
- **Output**: `cleaned_data.csv`, `train.csv`, `test.csv`.
- **The Dictionary**: Persistence of `vectorizer.joblib` to ensure consistent feature mapping across environments.

### Phase 2: Dual Matrix Comparison (Current Status)
- **Representations**: 
  - **TF-IDF**: Sparse vectors (5,000 features) for frequency-based importance.
  - **Word2Vec**: Dense embeddings (100 dimensions) for semantic meaning.
- **Models**: Multinomial Naive Bayes and Logistic Regression.
- **Visualization**: PCA (Principal Component Analysis) for article clustering and side-by-side performance benchmarking (F1-score & Confusion Matrices).

### Phase 3: Advanced Refinement (Next Steps)
- Deep Learning implementation.
- Transformer-based models (Future work).

## 📂 Project Structure
- `01_data_cleaning_and_embeddings.ipynb`: Data ETL and initial vectorization.
- `02_baseline_classifier.ipynb`: TF-IDF Modeling (NB & LR).
- `02.1_word2vec_classifier.ipynb`: Semantic Word2Vec Modeling.
- `02.2_model_comparison.ipynb`: Final Benchmarking and PCA Visuals.
- `models/`: Serialized models and vectorizers.
- `dataset/`: Training and validation data bits.

## 🛠️ Installation & Environment
```bash
# Recommended environment: Python 3.10.16
conda activate ironhack
pip install -r project-nlp-challenge/requirements.txt
```

## 📍 Multi-Environment Compatibility
All notebooks are configured with `BASE_PATH` logic, allowing seamless execution on:
- **Local machine**
- **Google Colab** (via Google Drive mount)

---
*Last updated: 2026-03-26 | Phase 2 COMPLETE*
