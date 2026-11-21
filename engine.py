"""
ReviewGuard AI - analysis engine.

Everything that is not user-interface code lives here so it can be tested
without Streamlit: model loading, the analysis pipeline, review extraction
from URLs, exports and the matplotlib chart builders.

The original logic from the first version of app.py is preserved:
  * model/vectorizer are loaded with pickle from the same files
  * clean_text() is unchanged
  * the rule-based indicators use the same thresholds
  * the verdict is "FAKE" if the model says CG OR the heuristic score >= 2
  * review extraction (DummyJSON / Myntra / Flipkart+Shopsy / generic) is unchanged

Differences are listed in README.md ("What changed and why").
"""

from __future__ import annotations

import json
import pickle
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend (Streamlit renders the figure itself)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

try:  # TextBlob is part of the original project
    from textblob import TextBlob
except ImportError:  # pragma: no cover - only when the package is missing
    TextBlob = None

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model" / "fake_review_model.pkl"
VECTORIZER_PATH = APP_DIR / "model" / "tfidf_vectorizer.pkl"
DATASET_PATH = APP_DIR / "fake_reviews_dataset.csv"

REPORTED_ACCURACY = 86.09  # reported by train_model.py, as supplied by the project owner

FAKE_LABEL = "CG"      # computer-generated
GENUINE_LABEL = "OR"   # original

SUSPICIOUS_WORDS = [
    "best product ever", "must buy", "life changing", "buy now",
    "100% recommended", "perfect product", "unbelievable",
    "fantastic", "amazing", "superb", "excellent",
]
EMOJI_SET = "😀😃😄😁😆😂🤣😍🔥❤️👍"

# Palette (from the design brief)
C_BG = "#0B0F14"
C_BG2 = "#111820"
C_SIDEBAR = "#17212C"
C_CARD = "#1A2430"
C_CARD2 = "#202B38"
C_GOLD = "#F2C45C"
C_GOLD_LIGHT = "#FFE09A"
C_TEXT = "#F4F6F9"
C_TEXT2 = "#98A6B7"
C_BORDER = "#2A3543"
# Chart-only semantic colours (kept muted). Sentiment uses different hues from
# authenticity on purpose, so the two are never confused.
C_FAKE = "#D9695F"
C_GENUINE = C_GOLD
C_POS = "#6FA8DC"
C_NEU = "#98A6B7"
C_NEG = "#B38BD9"

PATTERN_COLUMNS = {
    "Promotional Phrases": "Flag Promotional",
    "Excessive Punctuation": "Flag Punctuation",
    "Repeated Words": "Flag Repeated",
    "Unusual Uppercase": "Flag Uppercase",
    "Excessive Emoji": "Flag Emoji",
    "Very Short Review": "Flag Short",
}

RECORD_COLUMNS = [
    "ID", "Timestamp", "Source", "Source Detail", "Review", "Status",
    "Flagged By", "Model Prediction", "Fake Probability (%)",
    "Genuine Probability (%)", "Confidence (%)", "Sentiment", "Polarity",
    "Word Count", "Character Count", "Suspicious Keywords", "Heuristic Score",
    "Flag Promotional", "Flag Punctuation", "Flag Repeated", "Flag Uppercase",
    "Flag Emoji", "Flag Short", "Flags",
]


# --------------------------------------------------------------------------
# Model loading (same mechanism as the original: pickle + two .pkl files)
# --------------------------------------------------------------------------
def load_models():
    """Return (model, vectorizer, error_message). error_message is None on success."""
    try:
        with open(MODEL_PATH, "rb") as fh:
            model = pickle.load(fh)
        with open(VECTORIZER_PATH, "rb") as fh:
            vectorizer = pickle.load(fh)
        return model, vectorizer, None
    except FileNotFoundError as exc:
        return None, None, f"Model file not found: {exc.filename}"
    except Exception as exc:  # corrupt pickle, version problems, ...
        return None, None, f"Model could not be loaded: {exc}"


def model_info(model, vectorizer) -> dict:
    """Configuration details that can be read from the loaded objects."""
    info = {"loaded": bool(model is not None and vectorizer is not None)}
    if not info["loaded"]:
        return info
    params = model.get_params()
    vparams = vectorizer.get_params()
    info.update(
        {
            "algorithm": type(model).__name__,
            "classes": [str(c) for c in model.classes_],
            "C": params.get("C"),
            "max_iter": params.get("max_iter"),
            "solver": params.get("solver"),
            "n_features_model": int(model.coef_.shape[1]),
            "vectorizer": type(vectorizer).__name__,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "max_features": vparams.get("max_features"),
            "stop_words": vparams.get("stop_words"),
            "lowercase": vparams.get("lowercase"),
            "ngram_range": tuple(vparams.get("ngram_range", (1, 1))),
        }
    )
    return info


def top_model_terms(model, vectorizer, n=12):
    """Highest-weight terms per class, read straight from the model coefficients."""
    if model is None or vectorizer is None:
        return [], []
    names = np.asarray(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    classes = list(model.classes_)
    # binary logistic regression: positive weight -> classes_[1], negative -> classes_[0]
    order = np.argsort(coef)
    neg_terms = [(names[i], float(coef[i])) for i in order[:n]]
    pos_terms = [(names[i], float(coef[i])) for i in order[::-1][:n]]
    return {classes[0]: neg_terms, classes[1]: pos_terms}


# --------------------------------------------------------------------------
# Text pipeline (unchanged from the original)
# --------------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


def suspicious_indicators(review: str) -> dict:
    """Rule-based indicators with the original thresholds. Heuristics only."""
    review_lower = review.lower()
    found_words = [w for w in SUSPICIOUS_WORDS if w in review_lower]
    words = review_lower.split()

    flag_promo = len(found_words) > 0
    flag_punct = review.count("!") > 3
    flag_upper = sum(1 for ch in review if ch.isupper()) > 15
    flag_repeat = (len(words) - len(set(words))) > 5
    flag_emoji = sum(1 for ch in review if ch in EMOJI_SET) > 5
    flag_short = len(words) < 3
    flag_exclaim5 = "!!!!!" in review

    score = len(found_words)
    score += int(flag_punct) + int(flag_upper) + int(flag_repeat)
    score += int(flag_emoji) + int(flag_short) + int(flag_exclaim5)

    labels = []
    if flag_promo:
        labels.append("Promotional phrases")
    if flag_punct or flag_exclaim5:
        labels.append("Excessive punctuation")
    if flag_repeat:
        labels.append("Repeated words")
    if flag_upper:
        labels.append("Unusual uppercase")
    if flag_emoji:
        labels.append("Excessive emoji")
    if flag_short:
        labels.append("Very short review")

    return {
        "found_words": found_words,
        "score": score,
        "flag_promo": flag_promo,
        "flag_punct": flag_punct or flag_exclaim5,
        "flag_repeat": flag_repeat,
        "flag_upper": flag_upper,
        "flag_emoji": flag_emoji,
        "flag_short": flag_short,
        "labels": labels,
        "word_count": len(words),
    }


def sentiment_of(review: str):
    """(label, polarity) using TextBlob with the original thresholds."""
    if TextBlob is None:
        return "Unavailable", None
    polarity = TextBlob(review).sentiment.polarity
    if polarity > 0:
        return "Positive", round(float(polarity), 4)
    if polarity < 0:
        return "Negative", round(float(polarity), 4)
    return "Neutral", round(float(polarity), 4)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def analyze_reviews(reviews, model, vectorizer, source="Single", detail="",
                    start_id=1, progress=None, chunk_size=500):
    """
    Run the full pipeline on a list of review strings and return a list of record dicts.

    The model is applied to the cleaned text exactly as in the original app.
    `progress`, if given, is called with (fraction, message).
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Model files are not loaded - analysis is unavailable.")

    reviews = [str(r) for r in reviews]
    stamp = _now()
    classes = list(model.classes_)
    fake_idx = classes.index(FAKE_LABEL)
    gen_idx = classes.index(GENUINE_LABEL)
    records = []
    total = len(reviews)

    for start in range(0, total, chunk_size):
        chunk = reviews[start:start + chunk_size]
        cleaned = [clean_text(r) for r in chunk]
        vectors = vectorizer.transform(cleaned)
        proba = model.predict_proba(vectors)
        preds = model.predict(vectors)

        for offset, review in enumerate(chunk):
            ind = suspicious_indicators(review)
            pred = str(preds[offset])
            p_fake = float(proba[offset][fake_idx]) * 100
            p_gen = float(proba[offset][gen_idx]) * 100
            confidence = round(max(proba[offset]) * 100, 2)

            by_model = pred == FAKE_LABEL
            by_rules = ind["score"] >= 2
            status = "FAKE" if (by_model or by_rules) else "GENUINE"
            if by_model and by_rules:
                flagged_by = "Model + Heuristics"
            elif by_model:
                flagged_by = "Model"
            elif by_rules:
                flagged_by = "Heuristics"
            else:
                flagged_by = "None"

            sentiment, polarity = sentiment_of(review)
            records.append(
                {
                    "ID": start_id + start + offset,
                    "Timestamp": stamp,
                    "Source": source,
                    "Source Detail": detail,
                    "Review": review,
                    "Status": status,
                    "Flagged By": flagged_by,
                    "Model Prediction": pred,
                    "Fake Probability (%)": round(p_fake, 2),
                    "Genuine Probability (%)": round(p_gen, 2),
                    "Confidence (%)": confidence,
                    "Sentiment": sentiment,
                    "Polarity": polarity,
                    "Word Count": ind["word_count"],
                    "Character Count": len(review),
                    "Suspicious Keywords": ", ".join(ind["found_words"]) if ind["found_words"] else "None",
                    "Heuristic Score": ind["score"],
                    "Flag Promotional": ind["flag_promo"],
                    "Flag Punctuation": ind["flag_punct"],
                    "Flag Repeated": ind["flag_repeat"],
                    "Flag Uppercase": ind["flag_upper"],
                    "Flag Emoji": ind["flag_emoji"],
                    "Flag Short": ind["flag_short"],
                    "Flags": ", ".join(ind["labels"]) if ind["labels"] else "None",
                }
            )
        if progress:
            progress(min(1.0, (start + len(chunk)) / max(total, 1)),
                     f"Analysed {min(start + len(chunk), total)} of {total} reviews")
    return records


def records_to_df(records) -> pd.DataFrame:
    """DataFrame with the stable column order, valid even when empty."""
    if not records:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    return pd.DataFrame(records).reindex(columns=RECORD_COLUMNS)


def example_review() -> str:
    return ("Amazing product!!! Best product ever, must buy! Perfect product, "
            "excellent quality, superb value. Buy now!!!")


# --------------------------------------------------------------------------
# URL review extraction (logic preserved; Streamlit calls replaced by a message list)
# --------------------------------------------------------------------------
EXCLUDED_UI_TEXTS = [
    "see products based on categories", "positive feedback", "negative feedback",
    "verified purchases", "certified buyer", "read more", "rate product",
    "all reviews", "most helpful", "recent reviews", "q&a", "questions and answers",
]


def is_valid_user_review(text):
    """Filters out interface elements, short labels, headers, and UI text blocks."""
    if not text or not isinstance(text, str):
        return False
    clean_lower = text.strip().lower()
    if any(ui_label in clean_lower for ui_label in EXCLUDED_UI_TEXTS):
        return False
    words = clean_lower.split()
    if len(words) < 4:
        return False
    if clean_lower.startswith("http") or "{" in clean_lower:
        return False
    return True


def detect_platform(url: str) -> str:
    u = (url or "").strip().lower()
    if not u:
        return "None"
    if "dummyjson.com" in u:
        return "DummyJSON"
    if "myntra.com" in u:
        return "Myntra"
    if "shopsy.in" in u:
        return "Shopsy"
    if "flipkart.com" in u:
        return "Flipkart"
    return "Generic fallback"


def _http_client():
    try:
        from curl_cffi import requests as cffi_requests  # browser-impersonation layer
        return cffi_requests
    except ImportError:
        return None


def fetch_reviews_from_url(url, max_reviews=50, progress=None):
    """
    Returns (reviews, messages). `messages` is a list of (level, text) with level in
    {"warning", "error", "info"}. Extraction logic is the original, unchanged.
    """
    reviews = []
    messages = []
    url_clean = (url or "").strip().lower()

    if not url_clean:
        messages.append(("warning", "The URL input field is empty."))
        return reviews, messages

    requests = _http_client()
    if requests is None:
        messages.append((
            "error",
            "The 'curl_cffi' package is not installed. Run: pip install -r requirements.txt",
        ))
        return reviews, messages

    def tick(fraction, text):
        if progress:
            progress(min(max(fraction, 0.0), 1.0), text)

    # 1. DummyJSON API Handler
    if "dummyjson.com" in url_clean:
        match = re.search(r'products/(\d+)', url_clean)
        product_id = match.group(1) if match else "1"
        api_url = f"https://dummyjson.com/products/{product_id}"
        tick(0.2, "Contacting DummyJSON API...")
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for r in data.get("reviews", []):
                    comment = r.get("comment", "").strip()
                    if is_valid_user_review(comment) and comment not in reviews:
                        reviews.append(comment)
            else:
                messages.append(("error", f"DummyJSON returned HTTP {response.status_code}."))
        except Exception as e:
            messages.append(("error", f"Connection failed: {e}"))

    # 2. Myntra Architecture
    elif "myntra.com" in url_clean:
        id_match = re.search(r'/([0-9]{5,12})(?:/|$|\?|\.)', url_clean)
        if not id_match:
            messages.append(("error", "Could not parse product ID from link."))
            return reviews, messages

        product_id = id_match.group(1)
        for page in range(1, 11):
            if len(reviews) >= max_reviews:
                break
            tick(page / 10, f"Myntra: fetching page {page} of up to 10...")
            api_url = (f"https://www.myntra.com/gateway/v2/product/{product_id}"
                       f"/reviews-and-ratings?page={page}&pageSize=20")
            try:
                response = requests.get(api_url, impersonate="chrome", timeout=15)
                if response.status_code == 200:
                    batch = response.json().get("reviews", [])
                    if not batch:
                        break
                    for r in batch:
                        text = r.get("reviewText", "").strip()
                        if is_valid_user_review(text) and text not in reviews:
                            reviews.append(text)
                else:
                    messages.append(("warning", f"Myntra returned HTTP {response.status_code} on page {page}."))
                    break
            except Exception:
                break

    # 3. Flipkart / Shopsy Architecture (Paginated Multi-Strategy Extractor)
    elif "shopsy.in" in url_clean or "flipkart.com" in url_clean:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        base_review_url = url
        if "/p/" in url:
            base_review_url = url.replace("/p/", "/product-reviews/")
        elif "/product-reviews/" not in url and "?" in url:
            parts = url.split("?")
            base_review_url = f"{parts[0]}/product-reviews/{parts[1]}"

        page = 1
        max_pages = 15
        consecutive_empty_pages = 0

        while len(reviews) < max_reviews and page <= max_pages:
            if "?" in base_review_url:
                page_url = f"{base_review_url}&page={page}"
            else:
                page_url = f"{base_review_url}?page={page}"

            tick(page / max_pages, f"Fetching review page {page} of up to {max_pages}...")
            try:
                response = requests.get(page_url, impersonate="chrome", headers=headers, timeout=12)
                if response.status_code != 200:
                    messages.append(("warning", f"Page {page} returned HTTP {response.status_code}; stopping."))
                    break

                page_reviews_before = len(reviews)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Strategy 1: Direct HTML parsing for rendered review elements
                review_elements = soup.find_all(
                    ['div', 'p', 'span'],
                    class_=re.compile(r'(Z__43C|t-ZTKy|_27M-vq|_16P22e|_3pLY-c|ZvAfgS)', re.I),
                )
                for el in review_elements:
                    text = el.get_text(separator=' ').strip()
                    text = re.sub(r'READ MORE$', '', text, flags=re.IGNORECASE).strip()
                    if is_valid_user_review(text) and text not in reviews:
                        reviews.append(text)

                # Strategy 2: Structured JSON-LD payload parsing
                for script in soup.find_all('script', type='application/ld+json'):
                    if script.string:
                        try:
                            data = json.loads(script.string)
                            items = data if isinstance(data, list) else [data]
                            for item in items:
                                revs = item.get('review', [])
                                if isinstance(revs, dict):
                                    revs = [revs]
                                for r in revs:
                                    body = r.get('reviewBody') or r.get('description')
                                    if body and isinstance(body, str):
                                        clean_rev = body.strip()
                                        if is_valid_user_review(clean_rev) and clean_rev not in reviews:
                                            reviews.append(clean_rev)
                        except Exception:
                            continue

                # Strategy 3: Regex fallback extraction
                patterns = [
                    r'"reviewText"\s*:\s*"([^"]+)"',
                    r'"text"\s*:\s*"([^"]+)"',
                    r'"body"\s*:\s*"([^"]+)"',
                ]
                for pat in patterns:
                    found = re.findall(pat, response.text)
                    for text in found:
                        try:
                            clean = text.encode().decode('unicode-escape', errors='ignore').strip()
                            clean = re.sub(r'<[^>]+>', '', clean)
                            if is_valid_user_review(clean) and clean not in reviews:
                                reviews.append(clean)
                        except Exception:
                            continue

                if len(reviews) == page_reviews_before:
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= 2:
                        break
                else:
                    consecutive_empty_pages = 0

                page += 1
                time.sleep(0.4)

            except Exception as e:
                messages.append(("error", f"Connection error on page {page}: {e}"))
                break

    # 4. Generic Fallback
    else:
        tick(0.3, "Requesting page (generic fallback)...")
        try:
            response = requests.get(url, impersonate="chrome", timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                found_texts = soup.find_all(['div', 'p', 'span'], class_=re.compile(r'review', re.I))
                for item in found_texts:
                    clean = item.get_text().strip()
                    if is_valid_user_review(clean) and clean not in reviews:
                        reviews.append(clean)
            else:
                messages.append(("error", f"The site returned HTTP {response.status_code}."))
        except Exception as e:
            messages.append(("error", f"Failed to reach host: {e}"))

    tick(1.0, "Extraction finished.")
    return reviews[:max_reviews], messages


# --------------------------------------------------------------------------
# Batch CSV helpers
# --------------------------------------------------------------------------
def guess_text_column(df: pd.DataFrame):
    """Best-guess review text column: well-known names first, then longest average text."""
    if df is None or df.empty or len(df.columns) == 0:
        return None
    lowered = {str(c).lower().strip(): c for c in df.columns}
    for name in ("text_", "review", "reviews", "review_text", "reviewtext", "text", "comment", "content", "body"):
        if name in lowered:
            return lowered[name]
    best, best_len = None, -1.0
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith(("str", "string")):
            avg = df[col].dropna().astype(str).str.len().mean()
            if avg == avg and avg > best_len:  # NaN-safe
                best, best_len = col, avg
    return best


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    # utf-8 with BOM so Excel on Windows opens it correctly
    return df.to_csv(index=False).encode("utf-8-sig")


def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    records = json.loads(df.to_json(orient="records", force_ascii=False))
    payload = {
        "application": "ReviewGuard AI",
        "exported_at": _now(),
        "record_count": len(records),
        "note": ("Predictions are text-based estimates of resemblance to the CG/OR training "
                 "classes; they are not proof of authenticity."),
        "records": records,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def single_review_report(rec: dict) -> str:
    """Text report - same sections as the original download, plus the new fields."""
    return f"""ReviewGuard AI - Single Review Report
Generated: {rec.get('Timestamp', '')}

Review:
{rec['Review']}

Prediction:
{rec['Status']}  (model class: {rec['Model Prediction']}; flagged by: {rec['Flagged By']})

Model probabilities:
Fake (CG): {rec['Fake Probability (%)']}%
Genuine (OR): {rec['Genuine Probability (%)']}%

Confidence:
{rec['Confidence (%)']}%

Sentiment:
{rec['Sentiment']}

Suspicious Keywords:
{rec['Suspicious Keywords']}

Heuristic indicators (score {rec['Heuristic Score']}):
{rec['Flags']}

Note: this is a text-based estimate of resemblance to the CG/OR training classes.
It is not proof of authenticity and cannot verify a purchase or a buyer's identity.
"""


# --------------------------------------------------------------------------
# Held-out evaluation (real metrics, computed on demand from the labelled dataset)
# --------------------------------------------------------------------------
def evaluate_holdout(model, vectorizer, dataset_path=DATASET_PATH):
    """
    Re-creates the 80/20 split used in train_model.py (random_state=42) and scores the
    existing model on the 20% hold-out. Returns None if the dataset is unavailable.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    from sklearn.model_selection import train_test_split

    path = Path(dataset_path)
    if model is None or vectorizer is None or not path.exists():
        return None
    data = pd.read_csv(path)
    X = vectorizer.transform(data["text_"].astype(str))
    y = data["label"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = model.predict(X_test)
    labels = [FAKE_LABEL, GENUINE_LABEL]
    prec, rec, f1, sup = precision_recall_fscore_support(y_test, pred, labels=labels, zero_division=0)
    return {
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)) * 100,
        "labels": labels,
        "precision": [float(x) * 100 for x in prec],
        "recall": [float(x) * 100 for x in rec],
        "f1": [float(x) * 100 for x in f1],
        "support": [int(x) for x in sup],
        "confusion": confusion_matrix(y_test, pred, labels=labels).tolist(),
    }


# --------------------------------------------------------------------------
# Charts (matplotlib, dark/gold theme). Every builder returns None if there is
# nothing meaningful to draw, so the UI can show an empty state instead of an error.
# --------------------------------------------------------------------------
def _fig(width=6.0, height=3.4):
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor(C_CARD)
    ax.set_facecolor(C_CARD)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_BORDER)
    ax.tick_params(colors=C_TEXT2, labelsize=8)
    ax.xaxis.label.set_color(C_TEXT2)
    ax.yaxis.label.set_color(C_TEXT2)
    ax.grid(axis="y", color=C_BORDER, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, text):
    ax.set_title(text, color=C_TEXT, fontsize=10, fontweight="bold", loc="left", pad=10)


def _clean_df(df):
    return df if isinstance(df, pd.DataFrame) and not df.empty else None


def chart_activity(df):
    """Reviews analysed over time, stacked by verdict, with adaptive time buckets.
    Handles identical timestamps (single bucket) and drops unparsable ones."""
    df = _clean_df(df)
    if df is None:
        return None
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    valid = df.assign(_ts=ts).dropna(subset=["_ts"])
    if valid.empty:
        return None

    span = valid["_ts"].max() - valid["_ts"].min()
    if span <= pd.Timedelta(0):
        freq, unit, fmt = None, "single moment", "%H:%M:%S"
    elif span <= pd.Timedelta(hours=2):
        freq, unit, fmt = "min", "minute", "%H:%M"
    elif span <= pd.Timedelta(days=3):
        freq, unit, fmt = "h", "hour", "%d %b %H:00"
    else:
        freq, unit, fmt = "D", "day", "%d %b"

    bucket = valid["_ts"] if freq is None else valid["_ts"].dt.floor(freq)
    grouped = (valid.assign(_b=bucket)
               .groupby(["_b", "Status"]).size().unstack(fill_value=0))
    for col in ("FAKE", "GENUINE"):
        if col not in grouped.columns:
            grouped[col] = 0
    grouped = grouped.sort_index().tail(40)

    fig, ax = _fig(6.6, 3.4)
    x = np.arange(len(grouped))
    ax.bar(x, grouped["GENUINE"], color=C_GENUINE, width=0.55, label="Genuine")
    ax.bar(x, grouped["FAKE"], bottom=grouped["GENUINE"], color=C_FAKE, width=0.55, label="Fake")
    ax.set_xticks(x)
    ax.set_xticklabels([t.strftime(fmt) for t in grouped.index], rotation=0 if len(x) <= 6 else 45,
                       ha="center" if len(x) <= 6 else "right")
    if len(x) == 1:
        ax.set_xlim(-1.2, 1.2)
    ax.set_ylabel("Reviews analysed")
    ax.yaxis.get_major_locator().set_params(integer=True)
    _title(ax, f"Verification activity (grouped by {unit})")
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for t in leg.get_texts():
        t.set_color(C_TEXT2)
    fig.tight_layout()
    return fig


def chart_donut(fake, genuine, title="Authenticity distribution"):
    if (fake + genuine) <= 0:
        return None
    fig, ax = plt.subplots(figsize=(4.0, 3.4), dpi=110)
    fig.patch.set_facecolor(C_CARD)
    ax.set_facecolor(C_CARD)
    sizes = [genuine, fake]
    colors = [C_GENUINE, C_FAKE]
    labels = ["Genuine", "Fake"]
    keep = [i for i, s in enumerate(sizes) if s > 0]
    wedges, _ = ax.pie([sizes[i] for i in keep], colors=[colors[i] for i in keep], startangle=90,
                       counterclock=False, wedgeprops=dict(width=0.38, edgecolor=C_CARD, linewidth=2))
    total = fake + genuine
    ax.text(0, 0.06, f"{total}", ha="center", va="center", color=C_TEXT, fontsize=18, fontweight="bold")
    ax.text(0, -0.18, "reviews", ha="center", va="center", color=C_TEXT2, fontsize=8)
    leg = ax.legend(wedges, [f"{labels[i]}  {sizes[i]} ({sizes[i] / total * 100:.1f}%)" for i in keep],
                    loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8, ncol=2)
    for t in leg.get_texts():
        t.set_color(C_TEXT2)
    ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", loc="left", pad=6)
    fig.tight_layout()
    return fig


def chart_sentiment(counts: dict, title="Sentiment distribution"):
    order = ["Positive", "Neutral", "Negative"]
    values = [int(counts.get(k, 0)) for k in order]
    if sum(values) == 0:
        return None
    fig, ax = _fig(4.6, 3.2)
    bars = ax.bar(order, values, color=[C_POS, C_NEU, C_NEG], width=0.55)
    total = sum(values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n{v / total * 100:.0f}%", ha="center",
                va="bottom", color=C_TEXT, fontsize=8)
    ax.set_ylim(0, max(values) * 1.3)
    ax.yaxis.get_major_locator().set_params(integer=True)
    _title(ax, title)
    fig.tight_layout()
    return fig


def chart_sentiment_by_status(df):
    df = _clean_df(df)
    if df is None:
        return None
    sub = df[df["Sentiment"].isin(["Positive", "Neutral", "Negative"])]
    if sub.empty:
        return None
    table = pd.crosstab(sub["Sentiment"], sub["Status"]).reindex(
        index=["Positive", "Neutral", "Negative"], columns=["GENUINE", "FAKE"], fill_value=0)
    fig, ax = _fig(5.4, 3.2)
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, table["GENUINE"], w, color=C_GENUINE, label="Genuine")
    ax.bar(x + w / 2, table["FAKE"], w, color=C_FAKE, label="Fake")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.yaxis.get_major_locator().set_params(integer=True)
    _title(ax, "Sentiment vs authenticity verdict")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(C_TEXT2)
    fig.tight_layout()
    return fig


def chart_confidence_hist(df):
    df = _clean_df(df)
    if df is None:
        return None
    conf = pd.to_numeric(df["Confidence (%)"], errors="coerce")
    if conf.dropna().empty:
        return None
    fig, ax = _fig(5.4, 3.2)
    bins = np.linspace(50, 100, 11)
    for status, color, label in (("GENUINE", C_GENUINE, "Genuine"), ("FAKE", C_FAKE, "Fake")):
        vals = conf[df["Status"] == status].dropna()
        if len(vals):
            ax.hist(vals, bins=bins, alpha=0.8, color=color, label=label, edgecolor=C_CARD)
    ax.set_xlabel("Model confidence (%)")
    ax.set_ylabel("Reviews")
    ax.yaxis.get_major_locator().set_params(integer=True)
    _title(ax, "Confidence distribution")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(C_TEXT2)
    fig.tight_layout()
    return fig


def chart_fake_probability_hist(df):
    df = _clean_df(df)
    if df is None:
        return None
    p = pd.to_numeric(df["Fake Probability (%)"], errors="coerce").dropna()
    if p.empty:
        return None
    fig, ax = _fig(5.4, 3.2)
    ax.hist(p, bins=np.linspace(0, 100, 21), color=C_GOLD, edgecolor=C_CARD, alpha=0.9)
    ax.axvline(50, color=C_TEXT2, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Model fake (CG) probability (%)")
    ax.set_ylabel("Reviews")
    ax.yaxis.get_major_locator().set_params(integer=True)
    _title(ax, "Fake-probability spread")
    fig.tight_layout()
    return fig


def chart_trend(df, window=10):
    """Running fake share by analysis order (not by clock time, so identical timestamps are fine)."""
    df = _clean_df(df)
    if df is None or len(df) < 2:
        return None
    ordered = df.sort_values("ID")
    is_fake = (ordered["Status"] == "FAKE").astype(float).to_numpy()
    w = min(window, len(is_fake))
    rolling = pd.Series(is_fake).rolling(w, min_periods=1).mean() * 100
    fig, ax = _fig(6.6, 3.2)
    ax.plot(np.arange(1, len(rolling) + 1), rolling, color=C_GOLD, linewidth=2)
    ax.fill_between(np.arange(1, len(rolling) + 1), rolling, color=C_GOLD, alpha=0.12)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Review analysed (in order)")
    ax.set_ylabel("Fake share (%)")
    _title(ax, f"Fake share trend (rolling window of {w})")
    fig.tight_layout()
    return fig


def chart_source_compare(df):
    df = _clean_df(df)
    if df is None:
        return None
    table = pd.crosstab(df["Source"], df["Status"]).reindex(columns=["GENUINE", "FAKE"], fill_value=0)
    fig, ax = _fig(5.4, 3.2)
    x = np.arange(len(table))
    w = 0.36
    ax.bar(x - w / 2, table["GENUINE"], w, color=C_GENUINE, label="Genuine")
    ax.bar(x + w / 2, table["FAKE"], w, color=C_FAKE, label="Fake")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.yaxis.get_major_locator().set_params(integer=True)
    _title(ax, "Fake vs genuine by source")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(C_TEXT2)
    fig.tight_layout()
    return fig


def chart_probability_bar(fake_pct, genuine_pct):
    """Horizontal probability indicator for one review."""
    fig, ax = plt.subplots(figsize=(6.0, 1.5), dpi=110)
    fig.patch.set_facecolor(C_CARD)
    ax.set_facecolor(C_CARD)
    ax.barh([0], [genuine_pct], color=C_GENUINE, height=0.5)
    ax.barh([0], [fake_pct], left=[genuine_pct], color=C_FAKE, height=0.5)
    ax.text(max(genuine_pct / 2, 3), 0, f"Genuine (OR) {genuine_pct:.1f}%", va="center", ha="center",
            color="#1A1405", fontsize=8, fontweight="bold") if genuine_pct > 22 else None
    ax.text(genuine_pct + fake_pct / 2, 0, f"Fake (CG) {fake_pct:.1f}%", va="center", ha="center",
            color="#1F0A08", fontsize=8, fontweight="bold") if fake_pct > 22 else None
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    return fig


def chart_patterns(df):
    df = _clean_df(df)
    if df is None:
        return None
    counts = {label: int(df[col].fillna(False).astype(bool).sum()) for label, col in PATTERN_COLUMNS.items()}
    if sum(counts.values()) == 0:
        return None
    labels = list(counts)[::-1]
    values = [counts[k] for k in labels]
    fig, ax = _fig(5.8, 3.3)
    ax.grid(axis="x", color=C_BORDER, linewidth=0.6, alpha=0.7)
    ax.grid(axis="y", visible=False)
    bars = ax.barh(labels, values, color=C_GOLD, height=0.55)
    for b, v in zip(bars, values):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v}", va="center", color=C_TEXT, fontsize=8)
    ax.set_xlim(0, max(values) * 1.2)
    ax.xaxis.get_major_locator().set_params(integer=True)
    _title(ax, "Heuristic indicators triggered")
    fig.tight_layout()
    return fig


def chart_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=110)
    fig.patch.set_facecolor(C_CARD)
    ax.set_facecolor(C_CARD)
    arr = np.array(cm)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("gold", [C_CARD2, C_GOLD])
    ax.imshow(arr, cmap=cmap)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            dark = arr[i, j] > arr.max() * 0.5
            ax.text(j, i, f"{arr[i, j]:,}", ha="center", va="center",
                    color="#1A1405" if dark else C_TEXT, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color=C_TEXT2)
    ax.set_yticklabels(labels, color=C_TEXT2)
    ax.set_xlabel("Predicted", color=C_TEXT2)
    ax.set_ylabel("Actual", color=C_TEXT2)
    for s in ax.spines.values():
        s.set_color(C_BORDER)
    ax.set_title("Confusion matrix (hold-out set)", color=C_TEXT, fontsize=10, fontweight="bold", loc="left", pad=10)
    fig.tight_layout()
    return fig


def chart_terms(terms, title, color):
    if not terms:
        return None
    names = [t for t, _ in terms][::-1]
    values = [abs(v) for _, v in terms][::-1]
    fig, ax = _fig(4.6, 3.6)
    ax.grid(axis="x", color=C_BORDER, linewidth=0.6, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.barh(names, values, color=color, height=0.6)
    ax.set_xlabel("Absolute coefficient")
    _title(ax, title)
    fig.tight_layout()
    return fig
