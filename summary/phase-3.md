# Phase 3: Advanced Transformer Refinement - RAW KNOWLEDGE FEED

## 1. Objectives & Approach
- **Goal:** Elevate the fake news detection performance by fine-tuning a pre-trained Transformer model.
- **Why Transformers?** Previous Word2Vec and Naive Bayes models relied on shallow semantic representations. Transformers utilize self-attention mechanisms to understand contextual word relationships.
- **Choice of Model:** DistilBERT (distilbert-base-uncased). Optimized for memory and compute (Google Colab T4) while maintaining competitive accuracy.

## 2. Technical Infrastructure
- **Frameworks:** Hugging Face Transformers, Datasets API, PyTorch, Scikit-learn, Matplotlib, Seaborn, UMAP.
- **Hardware Management:** Integration of `.to('cuda')` and automatic device detection via `Trainer`.
- **Training Persistence:** Model weights and tokenizer saved to Google Drive: `/models/distilbert_classifier/`.

## 3. Training Logic (Raw Parameters)
- **Tokenization:** Max length 128 (default) with padding and truncation.
- **Optimizer:** AdamW.
- **Weight Decay:** 0.01.
- **Batch Size:** 16 (optimized for Colab RAM).
- **Epochs:** 2 (Initial phase to find base performance).
- **Metric Tracking:** Continuous monitoring of Training Loss vs. Eval Loss via `Trainer.train()`.

## 4. Evaluation Architecture
- **Stratified Split:** 80% train, 20% test.
- **Quantitative Evaluation (Metrics):**
    - MUST use the **Full Test Set (3,996 records)** for statistical validity.
    - Metrics: Accuracy, Precision, Recall, F1-Score (Macro/Weighted), AUC-ROC.
- **Visual Evaluation (Dimensionality Reduction):**
    - MUST use a **1,000-record Sample** to avoid over-plotting (occlusion) and kernel OOM errors.
    - **Modular Function:** `plot_comparison_modular(name, method_obj)` allows 1-line execution for PCA, t-SNE, and UMAP.
    - **Techniques Used:**
        - PCA (Principal Component Analysis): Linear variance maximization.
        - t-SNE (t-Distributed Stochastic Neighbor Embedding): Non-linear local structure focus.
        - UMAP (Uniform Manifold Approximation and Projection): Optimized local-global balance.

## 5. Performance Indicators (Knowledge Gained)
- DistilBERT latent space shows tighter clustering of "Real" vs "Fake" labels compared to the Word2Vec baseline.
- Word2Vec vectors often overlap in the shared semantic space where words like "said" or "told" lack strong class-specific intent.
- Fine-tuned BERT embeddings capture nuances in misinformation formatting and source-specific phrasing.

## 6. Project Architecture Notes
- Directory: `project-nlp-challenge/`
- Current Stage: Section 6 (Evaluation) in `03_advanced_transformer_refinement.ipynb`.
- Final Step Remaining: Execution of `generate_predictions` in Section 7 for competition delivery.

## 7. Lessons Learned (Grammatic/Technical)
- Modular functions reduce code duplication and error potential.
- Sampling for visualizations is a BEST PRACTICE in data science to maintain interpretability without losing the representative distribution of the classes.
- Hugging Face `Trainer` significantly simplifies the training loop while providing robust logging.
