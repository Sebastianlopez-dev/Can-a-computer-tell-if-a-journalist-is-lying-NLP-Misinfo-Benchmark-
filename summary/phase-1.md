🍱 Phase 1 Summary: The Prep Work

Environment Sync: All operations were strictly targeted at the ironhack conda environment (Python 3.10.16) using direct execution to ensure reliability and bypass shell activation issues.

### 🥗 Text Scrubbing (1.2)
Implemented the `clean_text` function (regex cleaning + NLTK stopwords) based on the bootcamp reference logic. I combined the title and text into a single feature column for richer context.

### 🏷️ Label Normalization (1.3)
Verified that our training data is perfectly balanced (approx. 20k Real / 20k Fake) with clean 0/1 labels.

### 📐 Baseline Vectorization (1.4)
Fitted a TfidfVectorizer (limited to 5,000 features for interpretability) on the cleaned corpus. The fitted object is persisted at `project-nlp-challenge/models/vectorizer.joblib`.

### ✂️ Train/Test Split (1.5)
Performed an 80/20 stratified split. You now have `train.csv` (31,953 rows) and `test.csv` (7,989 rows) ready in the `dataset/` folder.

📂 Key Artifacts Created:
- `01_data_cleaning_and_embeddings.ipynb` — Contains the full documented logic with the "Kitchen Metaphor".
- `project-nlp-challenge/dataset/cleaned_data.csv` — The unified cleaned source.
- `project-nlp-challenge/dataset/train.csv` & `test.csv` — The modeling inputs.
- `project-nlp-challenge/models/vectorizer.joblib` — The saved TF-IDF engine.
- `project-nlp-challenge/requirements.txt` — Minimal dependencies required to run the project.

---

### 🏛️ Technical Knowledge: The "Dictionary" Logic
A critical outcome of Phase 1 is the **`vectorizer.joblib`**. For the final presentation, remember this "Senior Peer" analogy:

*   **The Translator**: In NLP, the computer doesn't see words; it sees column indices. The "Vectorizer" is the dictionary that says "Column #23 is 'Urgent'."
*   **The Freeze Pattern**: We "Serialized" (saved) this ruleset into a `.joblib` file. This ensures that when we run our models in Phase 2 (or in Google Colab), the "columns" always mean the same thing. 
*   **Why it Matters**: Without this exact file, the brain of our model (the classifier) would be trying to read the wrong dictionary, leading to catastrophic misinterpretation of the news.

**Status**: Phase 1 COMPLETE & Verified.
