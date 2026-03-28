# Phase 1: Data Preprocessing and Frequency-Based Baseline

## Objectives

Establish a performance floor using frequency-based text representation and implement a robust data pipeline for reproducible baseline evaluation.

## Approach

### Data Preparation

**Feature Construction**: Title and text fields were concatenated into a single `full_text` feature to provide comprehensive semantic context and prevent sparsity from short article bodies.

**Text Cleaning Pipeline**:
1. Regex-based noise removal (non-alphanumeric characters, punctuation)
2. Case normalization to unified lowercase tokens
3. Tokenization into individual semantic units
4. Lemmatization to reduce words to root forms (using NLTK WordNetLemmatizer)
5. Stopword removal using NLTK to increase signal-to-noise ratio

**Class Balancing**: The dataset exhibited slight imbalance. Random undersampling was applied to achieve perfect 50/50 distribution (Real/Fake), eliminating bias toward either class.

### Vectorization and Modeling

**TF-IDF Representation**: Text was converted to sparse matrices using Term Frequency-Inverse Document Frequency weighting with the following specifications:
- Maximum 5,000 features (top unigrams and bigrams)
- Bigram range (1,2) to capture contextual word pairs (e.g., "White House") without exponential sparsity
- Sublinear term frequency scaling for stability

**Train/Test Split**: Stratified 80/20 split to maintain label distribution:
- Total balanced dataset: 39,926 samples (50/50 split)
- Training set: 31,940 samples
- Test set: 7,986 samples

**Models Evaluated**:
1. **Multinomial Naive Bayes**: Probabilistic baseline leveraging Bayes' theorem
2. **Logistic Regression**: Linear decision boundary for sparse feature space

## Results

**Baseline Accuracy**: 94.04% (establishes minimum acceptable performance threshold)

**Model Performance**:
- Multinomial Naive Bayes: Strong baseline with fast inference
- Logistic Regression: Comparable accuracy with better generalization properties

**Key Findings**:
- TF-IDF successfully captures frequency patterns distinguishing real from fake news
- Bigram features contribute meaningfully to classification (e.g., "breaking news", "official statement")
- Baseline performance validates dataset quality and problem feasibility

## Technical Implementation

**Serialization Strategy**: Vectorizers and trained models were persisted using joblib to enable:
- Consistent feature encoding between training and inference phases
- Seamless model deployment across environments (local, cloud, web applications)
- Efficient handling of large sparse NumPy arrays

**Artifacts Saved**:
- `tfidf_vectorizer.joblib`: Fitted TF-IDF transformer
- `nb_classifier.joblib`: Trained Multinomial Naive Bayes model
- `lr_classifier.joblib`: Trained Logistic Regression model

All serialized components are stored in the `models/` directory for reproducibility.

## Status

Phase 1 complete, serialized, and verified for downstream semantic and transformer phases. This baseline establishes the performance threshold against which all advanced techniques are measured.