import json
import os

nb_path = '/Users/basstianlopez/Desktop/it-studies/ironhack/week_10/Project 2 /project-nlp-challenge/02_baseline_classifier.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

# Define the new "Save Model" cell
save_model_cell = {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4.5 Save the \"Basstian\" Classifier\n",
    "We persist our trained model into the `models/` folder so it can be reused in future sessions without retraining."
   ]
}

save_model_code_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_path = os.path.join(BASE_PATH, 'models/nb_classifier.joblib')\n",
    "joblib.dump(nb_model, model_path)\n",
    "\n",
    "print(f\"✅ Model saved to: {model_path}\")"
   ]
}

# Insert before Section 5 (Deliverable)
# Section 5 is currently the last markdown cell (cell index 13 if considering markdown+code pairs)
# Let's find it.
for i, cell in enumerate(nb['cells']):
    if "## 5. Final Deliverable" in "".join(cell.get('source', [])):
        nb['cells'].insert(i, save_model_cell)
        nb['cells'].insert(i+1, save_model_code_cell)
        break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
