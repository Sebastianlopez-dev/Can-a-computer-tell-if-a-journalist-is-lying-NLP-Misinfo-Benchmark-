# Fake News Detection: NLP Representation Evolution

## Objective

This project benchmarks the evolution of Natural Language Processing techniques for binary classification of news articles (Real: 1, Fake: 0). Using a dataset of approximately 40,000 articles, the study contrasts three representation paradigms: frequency-based (TF-IDF), semantic embeddings (Word2Vec), and contextual transformers (DistilBERT), demonstrating the performance gains at each architectural stage.

## Project Overview

The dataset contains news articles with the following structure:
- **label**: 0 if the news is fake, 1 if the news is real
- **title**: The headline of the news article
- **text**: The full content of the article
- **subject**: The category or topic of the news
- **date**: The publication date of the article

Your task: Build a classifier to distinguish between real and fake news, then use it to predict labels for validation data.

## Methodology

The project unfolds across three phases, each advancing the mathematical representation of text:

### Phase 1: Frequency-Based Baseline
- **Text representation**: TF-IDF with bigrams (5,000 features)
- **Data preparation**: Title-text concatenation, regex cleaning, lemmatization, stopword removal
- **Class balancing**: Random undersampling to 50/50 distribution
- **Models**: Multinomial Naive Bayes and Logistic Regression
- **Result**: 94.04% accuracy (performance floor)

### Phase 2: Semantic Representation
- **Text representation**: Word2Vec embeddings (100 dimensions, dense vectors)
- **Model**: Logistic Regression (applied to both sparse and dense representations for fair comparison)
- **Analysis**: Dimensionality reduction via PCA to visualize class separation
- **Result**: 98.20% accuracy; improved robustness against keyword variation

### Phase 3: Contextual Transformers
- **Architecture**: Fine-tuned DistilBERT (distilbert-base-uncased)
- **Mechanism**: Self-attention for contextual word relationships
- **Evaluation**: UMAP visualization of latent space; comparative metrics against Phases 1 and 2
- **Result**: 99.9% accuracy (near-perfect classification with superior generalization)

## Key Results

| Representation | Model | Accuracy | F1-Score | Key Insight |
|---|---|---|---|---|
| Frequency | Naive Bayes (TF-IDF) | 94.04% | 0.94 | Fast baseline; keyword-dependent |
| Semantic | Word2Vec + Logistic Regression | 98.20% | 0.98 | Captures conceptual relationships |
| Contextual | DistilBERT (fine-tuned) | 99.9% | 0.99 | Understands linguistic nuance and context |

PCA and UMAP analyses confirm that semantic and transformer models achieve tighter clustering with clearer decision boundaries compared to frequency-based approaches.

## Repository Structure

```
├── 01_data_cleaning_and_embeddings.ipynb          # ETL and initial vectorization
├── 02_nb_baseline_classifier.ipynb                # TF-IDF baseline with Naive Bayes
├── 02.1_word2vec_classifier.ipynb                 # Word2Vec embeddings and training
├── 02.2_model_comparison_and_test.ipynb           # Comparative metrics and PCA visualization
├── 03_advanced_transformer_refinement.ipynb       # DistilBERT fine-tuning and evaluation
├── 03.1_models_comparison_v2.ipynb                # Advanced model evaluation and UMAP visualization
├── dataset/
│   ├── data.csv                                    # Training dataset (~40,000 articles)
│   └── validation_data.csv                         # Validation set for final predictions
├── models/                                         # Serialized weights (joblib, PyTorch state_dicts)
│   ├── tfidf_vectorizer.joblib
│   ├── word2vec_model.joblib
│   ├── nb_classifier.joblib
│   ├── w2v_logistic_classifier.joblib
│   └── distilbert_classifier/                      # DistilBERT model directory
└── summary/                                        # Phase reports and presentation materials
    ├── phase-1.md
    ├── phase-2.md
    ├── phase-3.md
    ├── presentation_notes.md
    └── NLP_Project_SebastianLopez_ironhack-AI.pptx
```

## Setup and Reproduction

**Environment**: Python 3.10.x with PyTorch and Hugging Face Transformers

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute notebooks sequentially (01 through 03.1)

### Generating Predictions

After training all three phases:
1. Load the trained models from the `models/` directory
2. Apply each model to `dataset/validation_data.csv`
3. Generate predictions in the same format as the original validation file (no extra columns, respect column separators)

**Portability**: The project uses unified path logic to support both local execution and Google Colab (with Google Drive mounting). Serialized vectorizers and models ensure consistent feature encoding across training and inference.

## Model Comparison Summary

The progression demonstrates clear performance improvements as we advance from static to contextual representations:

- **Phase 1** establishes the baseline and verifies the dataset quality
- **Phase 2** quantifies the gain from capturing semantic relationships
- **Phase 3** achieves near-perfect classification through contextual understanding

All phases employ stratified 80/20 train-test splits to maintain class distribution integrity throughout evaluation.

---

**Author**: Basstian Lopez
**Institution**: Ironhack Data Analytics
**Date**: March 2026