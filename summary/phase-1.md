# 🍱 Phase 1 Summary: My Technical Pre-Processing & Engineering

## 0. Strategy Exploration: Why I chose a Naive Bayes Baseline
Before processing the data, I established my architectural direction: using **Naive Bayes** as my initial baseline classifier. 
*   **My Baseline Concept**: I use a baseline to establish the "minimum acceptable performance" floor. If the subsequent complex semantic models I build (e.g., Word2Vec) cannot significantly outperform it, I know their added computational cost is unjustified.
*   **Why I chose Naive Bayes**: It calculates text probabilities by counting word frequencies, "naively" assuming each word is independent of the others. Despite this simplistic assumption, I found it exceptionally fast to train, it handles the sparse high-dimensional data (like the TF-IDF matrices I'm creating) effortlessly, requires minimal tuning, and provides remarkably robust accuracy. For me, it acts as the perfect, lightweight "sanity check" and yardstick for this news classifier.

## 1. My Environment Optimization
I executed all processing pipelines within a dedicated **Python 3.10.16 (ironhack)** environment. I chose direct execution to ensure consistency across my serialized models and library dependencies, bypassing potential shell activation issues I wanted to avoid.

## 2. Data Segmentation & Feature Construction (1.2)
To maximize context density, I performed **Document-level Segmentation** by fusing the `title` and `text` columns into a single `full_text` feature. 
*   **My Technical Rationale**: I wanted to ensure the model learns the relationship between sensationalized headlines (clickbait patterns) and the body content while preventing data sparsity for articles with short bodies.

## 3. Pre-Processing Pipeline: The Scrubbing (1.2)
I implemented the `clean_text` module, following a standard NLP pipeline to standardize vocabulary and reduce feature dimensionality:

1.  **Regex Cleaning & Standardizing**: I removed non-alphanumeric noise and punctuation.
2.  **Case Normalization**: I collapsed semantic variants (e.g., 'Apple' and 'apple') into a single token.
3.  **Tokenization**: This is where I segment strings into individual semantic units (tokens). 
4.  **Lemmatization/Stemming (Conceptual Layer)**: I reduced words to their base or root forms. While my current baseline focuses on stopword removal, I view this layer as the "Peeling" phase—removing the skin to reach the core meaning.
5.  **Stopword Removal**: I eliminated high-frequency, low-variance words (e.g., "the", "is") using NLTK to boost the Signal-to-Noise ratio of my corpus.

## 4. Label Normalization & Class Balancing (1.3)
I verified the target distribution across nearly 40k specimens. While the raw data was "almost" 50/50, I identified a slight **Class Imbalance**.
* **My Pro Strategy**: I implemented **Random Undersampling** to prune the majority class until both labels (0/1) reached an identical count. 
* **The Result**: A perfectly balanced dataset (50.0% Real / 50.0% Fake). This ensures my baseline has zero bias and treats both "Fake News" patterns and "Real News" patterns with equal mathematical weight.


## 5. NLP Feature Extraction: My Baseline Vectorization (1.4)
I constructed a sparse matrix representation using a **TfidfVectorizer** (Term Frequency-Inverse Document Frequency).
*   **My Hyperparameters**: I limited this to the top 5,000 features with an `ngram_range=(1,2)` to capture both individual words and key pairs (bigrams).
    *  I chose bigrams to capture essential context (e.g., "White House" vs "white" and "house" separately) without the exponential feature explosion and sparsity that 3+ n-grams would introduce. It's the "sweet spot" for context in a baseline.
*   **The Mechanism**: This converts my text tokens into numerical indices, weighting words by their uniqueness across my entire corpus.

## 6. Dataset Segmentation (1.5)
I performed a **Stratified 80/20 Train/Test Split**.
*   **My Training Set**: 31,953 rows (my development environment).
*   **My Testing Set**: 7,989 rows (my validation "Final Exam").
*   **My Technical Guard**: Using stratification ensures both segments maintain identical label distributions, preventing skewed performance metrics in my results.

---

## 🏛️ My Technical Knowledge: Serialization & Reliability
I have implemented a robust serialization strategy using the **`joblib`** library. For me, this is a critical bridge between training and deployment:

*   **The Translator Engine (Vectorizer Persistence)**: 
    *   In my NLP workflow, I know the computer doesn't see words; it sees column indices. I treat the **`vectorizer.joblib`** as the dictionary that says "Column #23 is 'Urgent'."
    *   **The Freeze Pattern**: By serializing this into a `.joblib` file, I am "freezing" the rules of the game. This ensures that when I run my models tomorrow in Google Colab, my "columns" will always mean exactly the same thing. Without this exact file, the brain of my model would be trying to read the wrong dictionary, leading to catastrophic misinterpretation.

*   **Pipeline Decoupling**: 
    *   I serialize both my Vectorizer and my Trained Classifier (`nb_model.joblib`).
    *   **My Rationale**: This allows me to separate **Training** from **Inference**. I can train my model once on my local machine and then deploy it anywhere (like the cloud or a web app) without needing to re-process the raw 40k articles again. 

*   **Efficiency with Joblib**: 
    *   I chose **Joblib** over standard Python 'pickle' because it is far more efficient at handling the large, sparse NumPy arrays generated during my TF-IDF process. This results in faster loading times and lower memory overhead during my next phases.

**My Status**: Phase 1 COMPLETE, Serialized, & Verified.
