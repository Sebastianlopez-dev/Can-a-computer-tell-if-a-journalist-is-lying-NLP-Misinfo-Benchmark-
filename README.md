# Fake News Detection: An Evolutionary NLP Pipeline

This repository contains a comprehensive Natural Language Processing (NLP) framework designed to classify news articles as Real or Fake. Developed as a multi-stage project, it demonstrates an architectural evolution from statistical frequency-based models to state-of-the-art Transformer-based deep learning.

## Project Description

The primary objective is the binary classification of news articles (Real: 1, Fake: 0) using a dataset of approximately 40,000 records. The project implements a side-by-side benchmarking strategy, contrasting traditional "Bag-of-Words" approaches with modern semantic embeddings to determine which representation captures the nuance of misinformation more effectively.

## Methodology and Research Phases

The project is structured into three distinct research phases:

### Phase 1: Data Pre-processing and Statistical Baseline
- Feature Engineering: Concatenation of article titles and text bodies to maximize semantic context.
- Cleaning Pipeline: Regex-based noise reduction, NLTK stopword filtering, and lemmatization.
- Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency) using bigrams with a feature limit of 5,000.
- Baseline Model: Multinomial Naive Bayes and Logistic Regression.
- Outcome: Established a performance "floor" of approximately 94.04% accuracy.

### Phase 2: Semantic Representation Matrix
- Transition: Shifted from word counts (sparse vectors) to word meanings (dense vectors).
- Implementation: Word2Vec embeddings (100 dimensions).
- Evaluation Logic: Logistic Regression was employed as a "Fair Judge" to evaluate both sparse and dense representations under identical conditions.
- Findings: Proved that semantic understanding significantly reduces false positives. Semantic accuracy reached approximately 98.20%.

### Phase 3: Advanced Transformer Refinement
- Implementation: Fine-tuning of the pre-trained DistilBERT model (distilbert-base-uncased).
- Advantage: Utilized self-attention mechanisms to understand contextual word relationships rather than static definitions.
- Visualization: Dimensionality reduction via PCA, t-SNE, and UMAP to visualize the latent space separation between Real and Fake news clusters.

## Key Experimental Results

| Model Representation | Accuracy | F1-Score | Primary Advantage |
| --- | --- | --- | --- |
| TF-IDF (Baseline) | 94.04% | 0.94 | Low computational cost, good for keyword detection. |
| Word2Vec (Semantic) | 98.20% | 0.98 | Captures conceptual meaning; robust against keyword shifts. |
| DistilBERT (Transformer)| 99.9% | 0.99 | Contextual understanding and state-of-the-art precision. |

Side-by-side analysis through Principal Component Analysis (PCA) confirms that semantic models (Phase 2 and 3) achieve tighter clustering and clearer decision boundaries than frequency-based models (Phase 1).

## Repository Structure

- 01_data_cleaning_and_embeddings.ipynb: Data ingestion, ETL, and initial vectorization scripts.
- 02_nb_baseline_classifier.ipynb: Baseline TF-IDF implementation with Naive Bayes.
- 02.1_word2vec_classifier.ipynb: Semantic modeling using Word2Vec embeddings.
- 02.2_model_comparison_and_test.ipynb: Comprehensive benchmarking, metric visualization, and PCA analysis.
- 03_advanced_transformer_refinement.ipynb: Transformer fine-tuning (DistilBERT) and latent space evaluation.
- 03.1_models_comparation_v2.ipynb: Advanced model comparison and statistical validation.
- summary/: Technical reports and presentation notes for each project phase.
- models/: Serialized model weights and vectorizers (.joblib and PyTorch state_dicts).

## Installation and Technical Environment

Recommended Environment: Python 3.10.x

To reproduce the study:
1. Clone the repository.
2. Initialize the environment:
   ```bash
   pip install -r project-nlp-challenge/requirements.txt
   ```
3. Run notebooks sequentially (01 through 03.1).

## Multi-Environment Compatibility

The notebooks utilize a unified BASE_PATH logic to ensure seamless portability between local workstations and Google Colab environments (via Google Drive mounting). All serialized components (vectorizers and dictionaries) are persisted to ensure feature consistency across training and inference stages.

---
Project Author: Basstian Lopez
Institution: Ironhack Data Analytics
Date: March 2026
