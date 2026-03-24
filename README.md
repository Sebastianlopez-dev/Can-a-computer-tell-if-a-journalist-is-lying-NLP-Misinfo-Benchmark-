# 🧠 Project Memory & Live Sync
This file serves as the local "Mirror" of the Engram persistent memory for the `nlp-challenge` project. It ensures every model and script uses consistent paths and logic.

## 📍 Model & Data Locations
- **Raw Data**: `project-nlp-challenge/dataset/data.csv`
- **Validation Data**: `project-nlp-challenge/dataset/validation_data.csv`
- **Cleaned Data Export**: `project-nlp-challenge/dataset/cleaned_data.csv`
- **Saved Models**: `project-nlp-challenge/models/`
- **Saved Vectorizers**: `project-nlp-challenge/models/vectorizer.joblib`, `models/word2vec_model.joblib`

## 🔄 Live Engram Sync (Last updated: 2026-03-24)
- **Project ID**: `Project 2` (Local Context) / `nlp-challenge` (Memory Context)
- **Status**: Phase 2 Complete (Dual Matrix Benchmarked)
- **Constraint**: Binary Classification (0=Fake, 1=Real).
- **Environment**: `conda activate ironhack` (Python 3.10.16)

## 🛠 Active Technical Decisions
1. **The Representation Matrix**: Transitioned from single baseline to **TF-IDF vs Word2Vec** comparison.
2. **The "Fair Judge" Model**: Using **Logistic Regression** across both representations to isolate feature performance.
3. **Dimensionality Reduction**: Including **PCA** for visual storytelling of article clusters.
4. **Branding Constraint**: Strictly use technical nomenclature (TF-IDF/Word2Vec) in all notebooks and documentation.
5. **Strict Output Requirement**: `validation_results.csv` must strictly maintain the 5-column schema.
6. **The Vectorizer Dictionary (The Dictionary Logic)**: The `vectorizer.joblib` must be treated as a mandatory ruleset. We cannot predict without it because it defines the column-to-word translation.
7. **The Drive Sync Constraint**: Project must be uploaded to Google Drive path `/Project 2/project-nlp-challenge/` for Colab notebooks to access `models/vectorizer.joblib`.

## 💾 Latest Engram Entries
- **Topic Key**: `decision/phase-2-matrix-comparison-methodology-decisions` (Matrix Architecture)
- **Topic Key**: `decision/phase-2-sync-requirement-vectorizer-logic` (Sync & Dictionary Logic)
- **Observation ID**: Captured on 2026-03-24. 
- **Content**: Finalized Matrix comparison, dictionary importance, and sync constraints.

---

## 🚀 Next Session Kick-off (The "First Thing")
1.  **Sync to Cloud**: Upload the `project-nlp-challenge` folder to your Google Drive to path `/Project 2/project-nlp-challenge/`.
2.  **Verify Models**: Ensure `models/vectorizer.joblib` is uploaded (This is the **Dictionary** ruleset for the model).
3.  **Run Benchmarks**: Open `02_...`, `02.1_...`, and `02.2_...` in Google Colab, mount Drive, and run all cells to generate F1-Scores and PCA visualizations.
4.  **Initiate Phase 3**: Deep Learning with Transformers.
