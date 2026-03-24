🍱 Phase 2 Summary: Dual Representation Matrix

We have successfully established a comparative modeling framework to contrast Count-based learning against Semantic-based learning. All benchmarks are optimized for cloud execution in Google Colab.

### 1. TF-IDF Classifier (2.0)
- **Representations**: Count-based sparse vectors (5,000 features).
- **Algorithms**: Multinomial Naive Bayes (Probabilistic) and Logistic Regression (Linear).
- **Rationale**: Establishes the classic NLP benchmark to determine how well simple word frequencies predict labels.

### 2. Word2Vec Classifier (2.1)
- **Representations**: Dense semantic embeddings (100 dimensions).
- **Algorithm**: Logistic Regression.
- **Rationale**: Tests if moving from "word counts" to "word meaning" provides a measurable performance boost.

### 3. Visual Comparison & Dimensionality Reduction (2.2)
- **Viz Tools**: Dimensionality reduction via **PCA (Principal Component Analysis)**.
- **Metrics**: Side-by-side F1-Score comparisons and confusion matrices for all models.
- **Transition**: This dual-competency approach is the perfect final bridge toward Transformers (Phase 3).

📂 Key Artifacts Created:
- [02_baseline_classifier.ipynb](file:///Users/basstianlopez/Desktop/it-studies/it-studies/ironhack/week_10/Project%202%20/project-nlp-challenge/02_baseline_classifier.ipynb) — TF-IDF Matrix (NB & LR).
- [02.1_word2vec_classifier.ipynb](file:///Users/basstianlopez/Desktop/it-studies/it-studies/ironhack/week_10/Project%202%20/project-nlp-challenge/02.1_word2vec_classifier.ipynb) — Word2Vec Matrix (LR).
- [02.2_model_comparison.ipynb](file:///Users/basstianlopez/Desktop/it-studies/it-studies/ironhack/week_10/Project%202%20/project-nlp-challenge/02.2_model_comparison.ipynb) — Visualization & Metrics Comparison.

---

### 💡 Key Decisions & Rationale
1.  **From Single to Matrix**: We shifted from a single baseline model to a **"Matrix Comparison"** of TF-IDF vs. Word2Vec. This provides a clear, scientifically fair representation of NLP evolution in the presentation.
2.  **Removal of "Basstian" Branding**: To align with a professional senior-level presentation, we replaced the name with technical nomenclature: **TF-IDF Classifier** and **Word2Vec Classifier**. 
3.  **The Logistic Regression "Fair Judge"**: We decided to use Logistic Regression in *both* arenas. Since LR can handle both sparse (counts) and dense (meaning) vectors, it acts as the "Fair Judge" that tells us which representation is superior.
4.  **Presentation Storytelling (PCA)**: We explicitly added **PCA dimensionality reduction** to "2.2 Comparison" so the user could *visually* show the class how the data looks to the computer. 
5.  **Data Persistence**: All models are joblib-serialized to `models/` for seamless transition to the Phase 3 final ensemble.

---

### 🏛️ Technical Knowledge: The "Dictionary" Logic
A critical part of the presentation architecture is the **`vectorizer.joblib`**:

*   **The Translator**: In NLP, the computer doesn't see words; it sees column indices. The "Vectorizer" is the dictionary that says "Column #23 is 'Urgent'."
*   **The Freeze Pattern**: We "Serialized" this ruleset into a `.joblib` file. This ensures that when we run our models tomorrow in Google Colab, the "columns" always mean the same thing. Without this exact file, the brain of our model (the classifier) would be trying to read the wrong dictionary.