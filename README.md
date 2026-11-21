# ReviewGuard AI — Review Verification Using AI

A dark-and-gold, 12-page Streamlit workspace around your existing model
(TF-IDF + Logistic Regression, labels `CG` = computer-generated / fake, `OR` = original / genuine).
Your `model/*.pkl` files are untouched and nothing was retrained.

> Predictions are **text-based estimates** of resemblance to the CG/OR training classes.
> They are not proof of authenticity and cannot verify a purchase or a buyer's identity.

## Run it (Windows PowerShell)

```powershell
cd path\to\ReviewGuard_Project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Always start it from the project folder. If you already have your old virtual environment, just activate it;
no new packages were added (versions in `requirements.txt` match your original `pip freeze`).

## Pages

Overview · Single Review · Batch Analysis · URL Analyzer · Analytics · Sentiment Analysis ·
Suspicious Patterns · History · Export Reports · Model Performance · About Model · Settings

## Project layout

| File | Purpose |
|---|---|
| `app.py` | The interface (12 pages, navigation, filters, downloads) |
| `engine.py` | All logic: model loading, analysis, URL extraction, exports, charts |
| `styles/style.css`, `.streamlit/config.toml` | Dark and gold theme |
| `model/` | Your original `fake_review_model.pkl` and `tfidf_vectorizer.pkl` (unchanged) |
| `fake_reviews_dataset.csv`, `train_model.py`, `main.py` | Your originals (unchanged) |
| `legacy/app_original.py` | Your previous `app.py`, kept for reference |
| `tests/` | `test_engine.py`, `test_app_mock.py` (see "Testing") |

## What changed and why (please read)

While inspecting your code I found problems. Each was fixed minimally and is listed here.

1. **The model never influenced the verdict.** The old code tested `prediction == 1`, but the model returns the
   strings `'CG'`/`'OR'`, so that was never true; only the keyword rules could mark a review FAKE.
   The verdict rule is unchanged in form (FAKE if the model says CG **or** the heuristic score is 2 or more),
   but it now actually uses the model. Expect more reviews to be flagged than before.
2. **Fake/Genuine scores were hard-coded** (90/10 or 15/85). They are now the model's real class probabilities
   (`Fake Probability` = CG, `Genuine Probability` = OR). *Confidence* is still the higher of the two.
3. **A fabricated fallback was removed.** If the model files were missing, the old code silently returned
   "prediction 0, confidence 75%". Now the app shows a clear error and refuses to analyse.
4. **Batch CSV upload, session history and CSV/JSON export did not exist** in the old `app.py`, so they are new,
   not preserved. History is session-only (in memory), as before; nothing is written to disk.
5. URL extraction (DummyJSON, Myntra, Flipkart/Shopsy, generic) is the **original logic, unchanged**; only the
   Streamlit calls were replaced with a returned message list so errors display cleanly.
6. `clean_text`, the keyword list and all heuristic thresholds are unchanged. `styles/style.css` was previously
   unused; it now drives the theme.

**Model Performance page:** the 86.09% is shown as *reported*. Precision, recall, F1 and the confusion matrix
are only shown after you press **Run hold-out evaluation**, which re-creates your 80/20 split
(`random_state=42`) on `fake_reviews_dataset.csv` and scores the existing model. It reproduces 86.09%.
(The vectoriser was fitted on the full dataset, as in your training script.)

## Testing — what was and was not verified

Run from the project folder: `python tests/test_engine.py` and `python tests/test_app_mock.py`.

Verified by execution: syntax (Python 3.11 and 3.13); model loading and label handling; single, batch and
URL pipelines; probabilities; empty-data charts; identical, invalid and spread timestamps; CSV/JSON/TXT exports;
session history, filters and the confirm-before-clear flow; all 12 pages run without errors in empty and
populated states; hold-out metrics.

**Not verified:** the pages were exercised with a recording stand-in for Streamlit, so the real Streamlit
rendering, CSS appearance, and widget behaviour have **not** been seen in a browser. Live scraping of Myntra,
Flipkart, Shopsy and generic sites was not tested (no internet access in the build environment, and those
sites change and block bots). DummyJSON logic was tested against a simulated response. Sentiment tests used a
simple stand-in for TextBlob. If something looks off on your machine, tell me what you see and I will fix it.

Known limits: your model files were saved with scikit-learn 1.8.0; a different version prints a
version warning. Batch runs are capped at 20,000 rows to keep the session responsive.
