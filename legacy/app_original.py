# ===============================
# IMPORT LIBRARIES
# ===============================
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from textblob import TextBlob
import pandas as pd
import os
import re
import time
import json
from bs4 import BeautifulSoup

# Browser impersonation layer to handle protected API requests
from curl_cffi import requests  

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="ReviewGuard AI",
    page_icon="🔍",
    layout="wide"
)

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("model/fake_review_model.pkl", "rb"))
        vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_models()

# ===============================
# TEXT CLEANING FUNCTION
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# ===============================
# CORE ANALYSIS PIPELINE
# ===============================
SUSPICIOUS_WORDS = [
    "best product ever", "must buy", "life changing", "buy now",
    "100% recommended", "perfect product", "unbelievable",
    "fantastic", "amazing", "superb", "excellent"
]

def analyze_single_review(review):
    """Processes a single text string through ML + Rule-Based NLP + Sentiment."""
    cleaned_review = clean_text(review)
    
    # 1. ML Model Inference
    if model and vectorizer:
        review_vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vector)[0]
        probability = model.predict_proba(review_vector)
        confidence = round(max(probability[0]) * 100, 2)
    else:
        prediction = 0
        confidence = 75.0

    # 2. Rule-Based Scoring
    review_lower = review.lower()
    rule_fake_score = 0
    found_words = [word for word in SUSPICIOUS_WORDS if word in review_lower]
    rule_fake_score += len(found_words)

    if review.count("!") > 3:
        rule_fake_score += 1

    capital_letters = sum(1 for char in review if char.isupper())
    if capital_letters > 15:
        rule_fake_score += 1

    words = review_lower.split()
    if (len(words) - len(set(words))) > 5:
        rule_fake_score += 1

    emoji_count = sum(1 for char in review if char in "😀😃😄😁😆😂🤣😍🔥❤️👍")
    if emoji_count > 5:
        rule_fake_score += 1

    if len(words) < 3:
        rule_fake_score += 1

    if "!!!!!" in review:
        rule_fake_score += 1

    # 3. Decision Logic
    if prediction == 1 or rule_fake_score >= 2:
        status = "FAKE"
        fake_score = 90
        genuine_score = 10
    else:
        status = "GENUINE"
        fake_score = 15
        genuine_score = 85

    # 4. Sentiment Polarity
    polarity = TextBlob(review).sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "Review": review,
        "Status": status,
        "Confidence (%)": confidence,
        "Sentiment": sentiment,
        "Word Count": len(words),
        "Suspicious Keywords": ", ".join(found_words) if found_words else "None",
        "Fake Score": fake_score,
        "Genuine Score": genuine_score
    }

# ===============================
# STRICT UI FILTER & REVIEW ROUTER
# ===============================
EXCLUDED_UI_TEXTS = [
    "see products based on categories",
    "positive feedback",
    "negative feedback",
    "verified purchases",
    "certified buyer",
    "read more",
    "rate product",
    "all reviews",
    "most helpful",
    "recent reviews",
    "q&a",
    "questions and answers"
]

def is_valid_user_review(text):
    """Filters out interface elements, short labels, headers, and UI text blocks."""
    if not text or not isinstance(text, str):
        return False
        
    clean_lower = text.strip().lower()
    
    # Check against known UI boilerplate text
    if any(ui_label in clean_lower for ui_label in EXCLUDED_UI_TEXTS):
        return False
        
    # Reviews must be at least 4 words long to avoid catching short UI labels
    words = clean_lower.split()
    if len(words) < 4:
        return False
        
    # Ignore standalone URLs or generic code-like fragments
    if clean_lower.startswith("http") or "{" in clean_lower:
        return False
        
    return True

def fetch_reviews_from_url(url, max_reviews=50):
    reviews = []
    url_clean = url.strip().lower()
    
    if not url_clean:
        st.warning("⚠️ The URL input field is empty.")
        return reviews

    # 1. DummyJSON API Handler
    if "dummyjson.com" in url_clean:
        match = re.search(r'products/(\d+)', url_clean)
        product_id = match.group(1) if match else "1"
        api_url = f"https://dummyjson.com/products/{product_id}"
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for r in data.get("reviews", []):
                    comment = r.get("comment", "").strip()
                    if is_valid_user_review(comment) and comment not in reviews:
                        reviews.append(comment)
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")

    # 2. Myntra Architecture
    elif "myntra.com" in url_clean:
        id_match = re.search(r'/([0-9]{5,12})(?:/|$|\?|\.)', url_clean)
        if not id_match:
            st.error("❌ Could not parse product ID from link.")
            return reviews
        
        product_id = id_match.group(1)
        
        for page in range(1, 11): 
            if len(reviews) >= max_reviews:
                break
            api_url = f"https://www.myntra.com/gateway/v2/product/{product_id}/reviews-and-ratings?page={page}&pageSize=20"
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

        # Step A: Convert product URL to dedicated review route
        base_review_url = url
        if "/p/" in url:
            base_review_url = url.replace("/p/", "/product-reviews/")
        elif "/product-reviews/" not in url and "?" in url:
            parts = url.split("?")
            base_review_url = f"{parts[0]}/product-reviews/{parts[1]}"

        page = 1
        max_pages = 15  # Fetch through up to 15 pages
        consecutive_empty_pages = 0

        while len(reviews) < max_reviews and page <= max_pages:
            # Build URL with correct pagination separator
            if "?" in base_review_url:
                page_url = f"{base_review_url}&page={page}"
            else:
                page_url = f"{base_review_url}?page={page}"

            try:
                response = requests.get(page_url, impersonate="chrome", headers=headers, timeout=12)
                if response.status_code != 200:
                    break

                page_reviews_before = len(reviews)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Strategy 1: Direct HTML parsing for rendered review elements
                review_elements = soup.find_all(
                    ['div', 'p', 'span'], 
                    class_=re.compile(r'(Z__43C|t-ZTKy|_27M-vq|_16P22e|_3pLY-c|ZvAfgS)', re.I)
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
                    r'"body"\s*:\s*"([^"]+)"'
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

                # Buffer check: Stop only if 2 consecutive pages yield zero new reviews
                if len(reviews) == page_reviews_before:
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= 2:
                        break
                else:
                    consecutive_empty_pages = 0

                page += 1
                time.sleep(0.4)

            except Exception as e:
                st.error(f"❌ Connection error on page {page}: {e}")
                break

    # 4. Generic Fallback
    else:
        try:
            response = requests.get(url, impersonate="chrome", timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                found_texts = soup.find_all(['div', 'p', 'span'], class_=re.compile(r'review', re.I))
                for item in found_texts:
                    clean = item.get_text().strip()
                    if is_valid_user_review(clean) and clean not in reviews:
                        reviews.append(clean)
        except Exception as e:
            st.error(f"❌ Failed to reach host: {e}")

    return reviews[:max_reviews]

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("📌 About Project")
st.sidebar.info("""
This AI system detects whether product reviews are Fake or Genuine.

Algorithms Used:
✅ TF-IDF Vectorizer
✅ Logistic Regression
✅ Rule-Based NLP
""")

st.sidebar.metric(label="🎯 Model Accuracy", value="86.09%")

theme = st.sidebar.selectbox("🎨 Select Theme", ["Light Mode", "Dark Mode"])

if theme == "Dark Mode":
    bg_color = "#0E1117"
    text_color = "white"
    box_color = "rgba(30, 30, 30, 0.75)"
    input_bg = "rgba(15, 15, 15, 0.6)"
else:
    bg_color = "#f5f7fa"
    text_color = "#222222"
    box_color = "rgba(255, 255, 255, 0.8)"
    input_bg = "rgba(255, 255, 255, 0.9)"

# ===============================
# CUSTOM STYLING
# ===============================
st.markdown(f"""
<style>
@keyframes fluidGradient {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

.stApp {{
    background: {"linear-gradient(-45deg, #090d16, #13122b, #1b0c3a, #0b0f19)" if theme == "Dark Mode" else "linear-gradient(-45deg, #ff9a9e, #fecfef, #a1c4fd, #c2e9fb)"};
    background-size: 400% 400%;
    animation: fluidGradient 18s ease infinite;
    color: {text_color};
}}

.main .block-container {{
    background: {box_color};
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    margin-top: 15px;
}}

/* FORCED BUTTON STYLING */
div.stButton > button, 
button[data-testid="baseButton-secondary"],
button[kind="secondary"] {{
    background: linear-gradient(90deg, #11998e, #38ef7d) !important;
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    min-height: 52px !important;
    width: 100% !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    margin-top: 15px !important;
    margin-bottom: 15px !important;
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}}

textarea, input {{
    background-color: {input_bg} !important;
    border-radius: 12px !important;
    border: 1px solid rgba(128, 128, 128, 0.2) !important;
    color: {text_color} !important;
}}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.title("🔍 ReviewGuard AI")
st.write("Batch analyze user-posted e-commerce reviews for authentic patterns vs. fake manipulation.")

tab1, tab2 = st.tabs(["🌐 Automated Batch URL Analyzer", "✍ Single Review Testing Workspace"])

# ===============================
# TAB 1: BATCH URL ANALYZER
# ===============================
with tab1:
    st.subheader("Fetch User Reviews")
    
    product_url = st.text_input(
        "Paste E-Commerce Product Link", 
        placeholder="Enter URL from Myntra, Shopsy, Flipkart, DummyJSON, etc."
    )
    
    if st.button("🚀 Fetch & Analyze All Reviews"):
        if not product_url.strip():
            st.warning("Please provide a product URL to proceed.")
        else:
            with st.spinner("📥 Extracting user comments and running batch evaluation..."):
                extracted_reviews = fetch_reviews_from_url(product_url, max_reviews=50)
                
                if extracted_reviews:
                    results = [analyze_single_review(rev) for rev in extracted_reviews]
                    df_results = pd.DataFrame(results)
                    
                    st.success(f"Analysis Complete! Extracted and processed {len(df_results)} customer reviews.")
                    st.markdown("---")
                    
                    # High-Level Metrics Row
                    total_reviews = len(df_results)
                    fake_count = sum(df_results["Status"] == "FAKE")
                    genuine_count = sum(df_results["Status"] == "GENUINE")
                    avg_confidence = df_results["Confidence (%)"].mean()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total User Reviews", total_reviews)
                    col2.metric("Genuine Reviews", f"{genuine_count} ({round(genuine_count/total_reviews*100)}%)")
                    col3.metric("Fake Reviews Detected", f"{fake_count} ({round(fake_count/total_reviews*100)}%)")
                    col4.metric("Avg. Model Confidence", f"{round(avg_confidence, 1)}%")
                    
                    st.markdown("---")
                    
                    # Chart & Summary Section
                    c1, c2 = st.columns([1, 1])
                    
                    with c1:
                        st.subheader("📊 Authenticity Distribution")
                        fig, ax = plt.subplots(figsize=(5, 4))
                        ax.pie(
                            [fake_count, genuine_count], 
                            labels=['Fake', 'Genuine'], 
                            autopct='%1.1f%%', 
                            colors=['#ef5350', '#66bb6a'],
                            startangle=90
                        )
                        st.pyplot(fig)
                    
                    with c2:
                        st.subheader("💡 Automated Insights")
                        if fake_count > genuine_count:
                            st.error("⚠️ High Risk: A significant portion of reviews show markers of fake manipulation.")
                        else:
                            st.success("✅ Low Risk: Customer feedback appears mostly organic and authentic.")
                            
                        st.write(f"- **Positive Sentiments:** {sum(df_results['Sentiment'] == 'Positive')}")
                        st.write(f"- **Neutral Sentiments:** {sum(df_results['Sentiment'] == 'Neutral')}")
                        st.write(f"- **Negative Sentiments:** {sum(df_results['Sentiment'] == 'Negative')}")
                    
                    st.markdown("---")
                    
                    # Batch Results Data Table
                    st.subheader("📋 User Review Analysis Breakdown")
                    
                    def highlight_status(val):
                        color = '#ffcdd2' if val == 'FAKE' else '#c8e6c9'
                        text_color = '#b71c1c' if val == 'FAKE' else '#1b5e20'
                        return f'background-color: {color}; color: {text_color}; font-weight: bold;'

                    try:
                        styled_df = df_results[["Status", "Confidence (%)", "Sentiment", "Suspicious Keywords", "Review"]].style.map(
                            highlight_status, subset=["Status"]
                        )
                    except AttributeError:
                        styled_df = df_results[["Status", "Confidence (%)", "Sentiment", "Suspicious Keywords", "Review"]].style.applymap(
                            highlight_status, subset=["Status"]
                        )

                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Detailed Review Inspection Accordion
                    with st.expander("🔎 Inspect Individual Written Reviews"):
                        for idx, row in df_results.iterrows():
                            status_icon = "❌" if row["Status"] == "FAKE" else "✅"
                            st.markdown(f"**{status_icon} User Review #{idx+1} — Status: {row['Status']}** ({row['Confidence (%)']}% Confidence)")
                            st.write(f"*\"{row['Review']}\"*")
                            st.caption(f"Sentiment: {row['Sentiment']} | Length: {row['Word Count']} words | Suspicious Keywords: {row['Suspicious Keywords']}")
                            st.markdown("---")
                else:
                    st.error("Could not find or extract written user comments from the provided link.")

# ===============================
# TAB 2: MANUAL SINGLE WORKSPACE
# ===============================
with tab2:
    st.subheader("✍ Manual Review Testing")

    manual_review = st.text_area(
        "Paste a review string to evaluate individually:",
        height=180
    )

    if st.button("🔍 Analyze Review"):

        if manual_review.strip() == "":
            st.warning("Please enter a review.")
        else:
            result = analyze_single_review(manual_review)

            st.success("Analysis Completed Successfully")
            st.markdown("---")

            # Prediction Output
            if result["Status"] == "FAKE":
                st.error("❌ Fake Review Detected")
            else:
                st.success("✅ Genuine Review")

            # Key Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{result['Confidence (%)']}%")
                st.metric("Fake Score", f"{result['Fake Score']}%")
            with col2:
                st.metric("Sentiment", result["Sentiment"])
                st.metric("Genuine Score", f"{result['Genuine Score']}%")

            st.write("### Suspicious Keywords")
            st.info(result["Suspicious Keywords"])

            st.write("### Review Content")
            st.write(result["Review"])

            # Pie Chart Visualization
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                [result["Fake Score"], result["Genuine Score"]],
                labels=["Fake", "Genuine"],
                autopct="%1.1f%%",
                colors=["#ef5350", "#66bb6a"],
                startangle=90
            )
            st.pyplot(fig)

            # Downloadable Report Generation
            report = f"""Review:
{result['Review']}

Prediction:
{result['Status']}

Confidence:
{result['Confidence (%)']}%

Sentiment:
{result['Sentiment']}

Suspicious Keywords:
{result['Suspicious Keywords']}
"""

            st.download_button(
                "📥 Download Report",
                report,
                "manual_review_report.txt",
                "text/plain"
            )