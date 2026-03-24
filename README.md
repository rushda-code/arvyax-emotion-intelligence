> Built as part of the ArvyaX ML Internship Assignment — Team ArvyaX · RevoltronX
> An end-to-end emotional intelligence pipeline that understands user state,
> predicts intensity, and recommends wellness actions in real time.

# ArvyaX ML Internship Assignment
### Team ArvyaX · RevoltronX
### Theme: From Understanding Humans → To Guiding Them

---

## Overview

This project builds an end-to-end pipeline that takes short, messy user journal
reflections and contextual signals (sleep, stress, energy, time of day) and produces:

- A predicted emotional state (calm, restless, neutral, focused, mixed, overwhelmed)
- A predicted intensity score (1-5)
- A wellness action recommendation (what to do + when to do it)
- A confidence score and uncertainty flag per prediction

The system is designed for real-world noisy data — not clean benchmark datasets.

---

## Project Structure

arvyax_submission/
├── arvyax_pipeline.ipynb       Main end-to-end notebook
├── predictions.csv             Final predictions on 120 test samples
├── README.md                   This file
├── ERROR_ANALYSIS.md           10 failure case deep-dive
└── EDGE_PLAN.md                Mobile/on-device deployment plan

---

## Setup Instructions

### Requirements
Python 3.8 or above

Install dependencies:
    pip install pandas numpy scikit-learn xgboost matplotlib scipy

### Data
Place the following files in the same directory as the notebook:
- Sample_arvyax_reflective_dataset.xlsx - Dataset_120.csv  (training, 1200 rows)
- arvyax_test_inputs_120.xlsx - Sheet1.csv                 (test, 120 rows)

### Running the Notebook
Open arvyax_pipeline.ipynb in Jupyter or Google Colab.
Run all cells top to bottom. No manual steps needed.

The notebook will automatically:
1. Load and explore the data
2. Clean text and handle missing values
3. Engineer features (TF-IDF + metadata)
4. Train XGBoost models for state and intensity
5. Run the decision engine on test data
6. Save predictions.csv and feature_importance.png

---

## Approach

### Problem Framing
This is not a standard classification task. Journal entries are short, vague,
and sometimes contradictory. Labels like "mixed" and "neutral" are inherently
ambiguous. The goal is to build a system that reasons under uncertainty — not
one that chases accuracy on clean data.

### Text Features
We use TF-IDF with unigrams and bigrams (max 300 features, min_df=2,
sublinear_tf=True). TF-IDF was chosen over heavier embeddings intentionally —
it is fast, interpretable, and runs fully offline without any external models
or APIs. For a wellness product running on edge devices, this matters.

### Metadata Features
sleep_hours, energy_level, stress_level, duration_min, time_of_day,
ambience_type, previous_day_mood, face_emotion_hint, reflection_quality.

Engineered features:
- text_length    : character count of journal entry
- word_count     : number of words
- is_very_short  : binary flag for entries with 4 words or fewer

Missing values:
- sleep_hours       : filled with training median (6.0)
- previous_day_mood : filled with 'unknown'
- face_emotion_hint : filled with 'none' (123 missing — 10% of training data)

### Model Choice

Emotional State — XGBoost Classifier
- 6-class classification
- 5-fold stratified cross-validation F1 (macro): 0.583 +/- 0.015
- Robust to noisy labels and mixed feature types
- Fast training, interpretable feature importances

Intensity — XGBoost Regressor
- Treated as regression, not classification
- Intensity is ordinal (3 is between 2 and 4) — regression preserves this
- 5-fold MAE: 1.361 +/- 0.034
- Predictions clipped to [1, 5] and rounded to integer for output

### Decision Engine
Rule-based logic mapping (state + intensity + stress + energy + time_of_day)
to a wellness action and timing. Rule-based was chosen deliberately because:
- Every decision can be explained and justified
- Encodes real wellness domain knowledge
- Handles unseen combinations gracefully
- No training data required for the reasoning layer

### Uncertainty Modeling
Confidence  : max class probability from predict_proba (0-1)
Uncertain flag : set to 1 if confidence < 0.40 OR text is very short (4 words or fewer)
Extra rule  : if face_emotion_hint = none AND previous_day_mood = unknown, flag uncertain

34 out of 120 test predictions are flagged as uncertain.

### Keyword Override Layer
A safety layer that catches high-distress signals the TF-IDF model may miss
due to rare or unseen vocabulary. Words like "yelling", "panic", "cant breathe"
trigger an override that forces the state to overwhelmed or restless regardless
of model output. This is intentional — in a wellness product, missing a distress
signal is more dangerous than over-flagging it.

---

## Feature Importance Finding

TF-IDF (text) : 98.4% of total XGBoost feature importance
Metadata      :  1.6%

Text carries the primary signal. Metadata plays a larger role in the decision
engine and uncertainty modeling than in the classifier itself.

---

## Ablation Study Result

                     F1 (macro)
Text only            0.589 +/- 0.020
Text + Metadata      0.583 +/- 0.015
Metadata gain        -0.006

Text dominates the signal in this dataset. Metadata contribution is marginal
in the classifier but meaningful in the decision engine and uncertainty layer.
In a production system with richer signals (heart rate, calendar, location),
metadata contribution would be significantly higher.

---

## How to Run

1. Install dependencies (see Setup above)
2. Place CSV files in working directory
3. Run all cells in arvyax_pipeline.ipynb
4. Output files saved automatically

Deliverables generated:
- predictions.csv
- feature_importance.png
