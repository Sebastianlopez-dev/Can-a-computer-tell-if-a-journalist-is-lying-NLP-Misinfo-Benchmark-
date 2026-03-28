# Phase 2: Semantic Representation and Comparative Evaluation

## Objectives

Compare frequency-based and semantic-based text representations under controlled conditions to quantify the performance gains from moving from word counts to contextual embeddings.

## Approach

### Frequency-Based Benchmark (TF-IDF)

Multinomial Naive Bayes and Logistic Regression classifiers applied to TF-IDF sparse vectors (5,000 features) from Phase 1. This establishes the count-based baseline for direct comparison.

**Baseline Accuracy**: 94.04%

### Semantic Representation (Word2Vec)

**Embeddings**: Dense 100-dimensional vectors trained via Word2Vec (skip-gram model), capturing semantic relationships between words. The embedding space allows for:
- Word similarity calculations (e.g., "president" and "leader" are nearby)
- Capture of analogies and conceptual relationships
- Reduction from 5,000 sparse dimensions to 100 dense dimensions

**Training Details**:
- Word2Vec parameters: 100 dimensions, window size 5, minimum count 2
- Trained on the full text corpus (both real and fake articles)
- Averaged word vectors per document to create document-level embeddings

**Fair Comparison**: Logistic Regression was applied to both sparse (TF-IDF) and dense (Word2Vec) representations, eliminating algorithmic differences and isolating the effect of text representation.

### Evaluation and Visualization

Comparative analysis included:
- **Accuracy and F1-score** comparisons across both representations
- **Confusion matrices** to identify error patterns
- **Principal Component Analysis (PCA)** dimensionality reduction to visualize class separation in reduced dimensional space
- **Perturbation testing**: word substitution (e.g., 'president' → 'leader') to assess stability under semantic shifts

### Robustness Analysis

**Perturbation Testing Methodology**:
1. Select high-frequency words in both real and fake classes
2. Replace semantically related words with synonyms
3. Measure accuracy degradation under substitution

**Results Show**:
- TF-IDF model: Significant accuracy drop with semantic perturbations
- Word2Vec model: Maintains classification confidence despite synonym substitution

## Results

### Comparative Performance

| Representation | Model | Accuracy | F1-Score | Robustness |
|---|---|---|---|---|
| TF-IDF (Frequency) | Logistic Regression | 94.04% | 0.94 | Keyword-dependent; degrades with semantic shifts |
| Word2Vec (Semantic) | Logistic Regression | 98.20% | 0.98 | Maintains high confidence under word substitution |

**Performance Gain**: 4.16% accuracy improvement (98.20% - 94.04%)

### Key Findings

1. **Semantic Superiority**: Word2Vec correctly classified 379 additional articles that the frequency-based model misidentified, demonstrating that semantic understanding captures misinformation patterns beyond keyword matching.

2. **Robustness**: Perturbation tests showed Word2Vec maintains near-100% confidence under semantic perturbations while the baseline accuracy drops significantly:
   - TF-IDF accuracy: -12-15% under perturbations
   - Word2Vec accuracy: -1-2% under perturbations

3. **Class Separation**: PCA visualization reveals:
   - TF-IDF: Overlapping clusters with scattered decision boundaries
   - Word2Vec: Tighter class clustering and clearer linear decision boundaries

4. **Dimensionality**: Reduction from 5,000 to 100 dimensions improves computational efficiency while maintaining superior classification performance.

## Implementation

**Serialization**:
- `word2vec_model.joblib`: Trained Word2Vec model
- `w2v_logistic_classifier.joblib`: Logistic Regression trained on Word2Vec embeddings

All models serialized using joblib for reproducibility and deployment. Vectorizers and trained classifiers enable consistent feature encoding across environments.

## Insights for Phase 3

The success of semantic embeddings demonstrates that fake news detection benefits from understanding word meaning and conceptual relationships, not just frequency patterns. This validates the progression toward contextual transformers, which further enhance this understanding through self-attention mechanisms that capture word meaning in context.

## Status

Phase 2 complete. Semantic superiority established with quantified metrics and robustness validation. Transition to transformer models ready.