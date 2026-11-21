# ==========================================================================
# ReviewGuard AI - Review Verification Using AI
# Premium interactive intelligence workspace (Streamlit)
#
# Run with:  python -m streamlit run app.py
#
# All analysis logic lives in engine.py; this file is the interface only.
# ==========================================================================
import html
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import engine as E

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ReviewGuard AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "2.0"

# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
P_OVERVIEW = "◈  Overview"
P_SINGLE = "✎  Single Review"
P_BATCH = "▤  Batch Analysis"
P_URL = "⌘  URL Analyzer"
P_ANALYTICS = "▦  Analytics"
P_SENTIMENT = "☺  Sentiment Analysis"
P_PATTERNS = "⚑  Suspicious Patterns"
P_HISTORY = "◷  History"
P_EXPORT = "⇩  Export Reports"
P_PERF = "◎  Model Performance"
P_ABOUT = "ⓘ  About Model"
P_SETTINGS = "⚙  Settings"
PAGES = [P_OVERVIEW, P_SINGLE, P_BATCH, P_URL, P_ANALYTICS, P_SENTIMENT,
         P_PATTERNS, P_HISTORY, P_EXPORT, P_PERF, P_ABOUT, P_SETTINGS]

DISCLAIMER = ("Predictions are <b>text-based estimates</b> of how closely a review resembles the "
              "computer-generated (CG) or original (OR) reviews the model was trained on. They are "
              "<b>not proof of authenticity</b> and cannot verify a purchase or a buyer's identity.")

TABLE_COLUMNS = ["ID", "Timestamp", "Source", "Status", "Flagged By", "Fake Probability (%)",
                 "Confidence (%)", "Sentiment", "Flags", "Review"]
COLUMN_CONFIG = {
    "Fake Probability (%)": st.column_config.ProgressColumn(
        "Fake probability (%)", min_value=0, max_value=100, format="%.1f"),
    "Confidence (%)": st.column_config.NumberColumn("Confidence (%)", format="%.1f"),
    "Review": st.column_config.TextColumn("Review", width="large"),
}


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_css():
    css_path = Path(__file__).resolve().parent / "styles" / "style.css"
    try:
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    except OSError:
        st.warning("styles/style.css could not be read; the default theme is being used.")


# --------------------------------------------------------------------------
# Model (cached; same pickle mechanism as the original app)
# --------------------------------------------------------------------------
@st.cache_resource
def get_models():
    return E.load_models()


MODEL, VECTORIZER, MODEL_ERROR = get_models()
MODEL_OK = MODEL is not None and VECTORIZER is not None


@st.cache_data(show_spinner=False)
def read_csv_bytes(data: bytes):
    """Parse an uploaded CSV. Returns (DataFrame, error_message)."""
    import io
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding), None
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "The file is empty."
        except pd.errors.ParserError as exc:
            return None, f"The file could not be parsed as CSV: {exc}"
        except Exception as exc:  # pragma: no cover
            return None, f"Could not read the file: {exc}"
    return None, "The file's text encoding is not supported."


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("nav", P_OVERVIEW)
    ss.setdefault("history", [])            # list of record dicts (session only)
    ss.setdefault("next_id", 1)
    ss.setdefault("latest_single", None)
    ss.setdefault("batch_df", None)
    ss.setdefault("batch_meta", None)
    ss.setdefault("url_df", None)
    ss.setdefault("url_meta", None)
    ss.setdefault("single_text", "")
    ss.setdefault("confirm_clear", False)
    ss.setdefault("holdout", None)
    ss.setdefault("session_started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ss.setdefault("prefs", {"table_rows": 25, "url_max": 50})
    ss.setdefault("_df_cache", (None, None))


def goto(page):
    st.session_state["nav"] = page


def add_records(records):
    ss = st.session_state
    ss["history"].extend(records)
    ss["next_id"] = records[-1]["ID"] + 1 if records else ss["next_id"]


def history_df() -> pd.DataFrame:
    """History as a DataFrame (cached until the number of records changes)."""
    ss = st.session_state
    n = len(ss["history"])
    cached_n, cached_df = ss["_df_cache"]
    if cached_n != n or cached_df is None:
        cached_df = E.records_to_df(ss["history"])
        ss["_df_cache"] = (n, cached_df)
    return cached_df


def clear_all_data():
    ss = st.session_state
    ss["history"] = []
    ss["next_id"] = 1
    ss["latest_single"] = None
    ss["batch_df"] = None
    ss["batch_meta"] = None
    ss["url_df"] = None
    ss["url_meta"] = None
    ss["confirm_clear"] = False
    ss["_df_cache"] = (None, None)


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def esc(value) -> str:
    return html.escape(str(value))


def page_header(eyebrow, title, subtitle):
    st.markdown(
        f"<div class='rg-eyebrow'>{esc(eyebrow)}</div>"
        f"<div class='rg-title'>{esc(title)}</div>"
        f"<div class='rg-subtitle'>{subtitle}</div>"
        f"<div class='rg-rule'></div>",
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f"<div class='rg-section'>{esc(title)}</div>", unsafe_allow_html=True)


def stat_cards(cards):
    """cards: list of (label, value, sub, tone) where tone in {'', 'gold', 'fake'}."""
    parts = []
    for label, value, sub, tone in cards:
        parts.append(
            f"<div class='rg-stat'><div class='rg-stat-label'>{esc(label)}</div>"
            f"<div class='rg-stat-value {tone}'>{esc(value)}</div>"
            f"<div class='rg-stat-sub'>{esc(sub)}</div></div>"
        )
    st.markdown(f"<div class='rg-grid'>{''.join(parts)}</div>", unsafe_allow_html=True)


def callout(text_html):
    st.markdown(f"<div class='rg-callout'>{text_html}</div>", unsafe_allow_html=True)


def empty_state(title, text, button_label=None, target=None, key=None):
    st.markdown(f"<div class='rg-empty'><h4>{esc(title)}</h4><p>{esc(text)}</p></div>",
                unsafe_allow_html=True)
    if button_label and target:
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.button(button_label, key=key or f"empty_{target}", on_click=goto, args=(target,),
                      type="primary")


def show_fig(fig, empty_title="No data to chart yet", empty_text="Analyse some reviews to see this chart."):
    if fig is None:
        st.markdown(f"<div class='rg-empty'><h4>{esc(empty_title)}</h4><p>{esc(empty_text)}</p></div>",
                    unsafe_allow_html=True)
        return
    try:
        st.pyplot(fig)
    finally:
        plt.close(fig)


def chips(items, kind=""):
    if not items:
        return ""
    return "".join(f"<span class='rg-chip {kind}'>{esc(i)}</span>" for i in items)


def show_table(df, columns=None, height=420, key=None):
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    cols = [c for c in (columns or TABLE_COLUMNS) if c in df.columns]
    cfg = {k: v for k, v in COLUMN_CONFIG.items() if k in cols}
    st.dataframe(df[cols], hide_index=True, height=height, column_config=cfg, key=key)


def require_model():
    if not MODEL_OK:
        st.error(f"The model could not be loaded, so analysis is unavailable. {MODEL_ERROR or ''} "
                 "Check that the 'model' folder contains fake_review_model.pkl and tfidf_vectorizer.pkl.")
    return MODEL_OK


def search_filter(df, query):
    if not query or not query.strip() or df.empty:
        return df
    return df[df["Review"].astype(str).str.contains(query.strip(), case=False, regex=False, na=False)]


def pct(part, total):
    return f"{(part / total * 100):.1f}%" if total else "0.0%"


def counts(df):
    total = len(df)
    fake = int((df["Status"] == "FAKE").sum()) if total else 0
    return total, fake, total - fake


def sentiment_counts(df):
    if df.empty:
        return {}
    return df["Sentiment"].value_counts().to_dict()


def review_box(text):
    st.markdown(f"<div class='rg-review-box'>{esc(text)}</div>", unsafe_allow_html=True)


def disclaimer_footer():
    st.markdown(f"<div class='rg-footer'>ReviewGuard AI v{APP_VERSION} · {DISCLAIMER}</div>",
                unsafe_allow_html=True)


def record_detail(rec):
    """Expanded view of one record (used by History)."""
    review_box(rec["Review"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", rec["Status"])
    c2.metric("Fake probability", f"{rec['Fake Probability (%)']}%")
    c3.metric("Genuine probability", f"{rec['Genuine Probability (%)']}%")
    c4.metric("Confidence", f"{rec['Confidence (%)']}%")
    info_chips = [
        "Sentiment: " + str(rec["Sentiment"]),
        "Source: " + str(rec["Source"]),
        "Model class: " + str(rec["Model Prediction"]),
        "Flagged by: " + str(rec["Flagged By"]),
        str(rec["Word Count"]) + " words",
    ]
    if rec["Flags"] != "None":
        flag_html = chips(rec["Flags"].split(", "), "fake")
    else:
        flag_html = chips(["No heuristic indicators"])
    st.markdown(f"<div style='margin-top:10px'>{chips(info_chips, 'gold')}{flag_html}</div>",
                unsafe_allow_html=True)
    st.caption(f"Analysed {rec['Timestamp']} · Suspicious keywords: {rec['Suspicious Keywords']}"
               + (f" · {rec['Source Detail']}" if rec.get("Source Detail") else ""))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div class='rg-brand'><div class='rg-logo'>R</div><div>"
            "<div class='rg-brand-name'>ReviewGuard AI</div>"
            "<div class='rg-brand-sub'>Review Intelligence</div></div></div>"
            "<div class='rg-side-label'>Workspace</div>",
            unsafe_allow_html=True,
        )
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")

        status_dot = "ok" if MODEL_OK else "bad"
        status_txt = "Model loaded" if MODEL_OK else "Model unavailable"
        n = len(st.session_state["history"])
        st.markdown(
            f"<div class='rg-side-card'><span class='rg-dot {status_dot}'></span><b>{status_txt}</b><br>"
            f"Reported accuracy: <b>{E.REPORTED_ACCURACY}%</b><br>"
            f"Records this session: <b>{n}</b></div>"
            "<div class='rg-side-card' style='font-size:12px'>Text-based estimates only - "
            "not proof of authenticity.</div>",
            unsafe_allow_html=True,
        )


# ==========================================================================
# PAGE 1 - OVERVIEW
# ==========================================================================
def page_overview():
    page_header("Command Centre", "Overview",
                "Live view of everything analysed in this session: verdicts, sentiment and recent activity.")

    section("Quick actions")
    q1, q2, q3, q4 = st.columns(4)
    q1.button("✎  Analyse Single Review", key="qa_single", on_click=goto, args=(P_SINGLE,))
    q2.button("▤  Batch Analysis", key="qa_batch", on_click=goto, args=(P_BATCH,))
    q3.button("⌘  URL Analyzer", key="qa_url", on_click=goto, args=(P_URL,))
    q4.button("⇩  Export Reports", key="qa_export", on_click=goto, args=(P_EXPORT,))

    df_all = history_df()
    if df_all.empty:
        stat_cards([
            ("Reviews analysed", "0", "this session", ""),
            ("Genuine detected", "0", "-", "gold"),
            ("Fake detected", "0", "-", "fake"),
            ("Fake percentage", "-", "no data yet", ""),
            ("Reported model accuracy", f"{E.REPORTED_ACCURACY}%", "from model training", "gold"),
        ])
        empty_state("No reviews analysed yet",
                    "Analyse a review, upload a CSV or fetch reviews from a product URL and this "
                    "dashboard will fill with real results. Nothing here is sample data.",
                    "Analyse your first review", P_SINGLE, key="ov_empty")
        callout(DISCLAIMER)
        return

    section("Filters")
    f1, f2, f3 = st.columns([2, 1.2, 2])
    sources = sorted(df_all["Source"].unique())
    sel_sources = f1.multiselect("Source", sources, default=sources, key="ov_sources")
    sel_status = f2.selectbox("Verdict", ["All", "FAKE", "GENUINE"], key="ov_status")
    query = f3.text_input("Search review text", key="ov_query", placeholder="Type a word or phrase...")

    df = df_all[df_all["Source"].isin(sel_sources)]
    if sel_status != "All":
        df = df[df["Status"] == sel_status]
    df = search_filter(df, query)

    total, fake, genuine = counts(df)
    stat_cards([
        ("Reviews analysed", f"{total:,}", f"{len(df_all):,} in session", ""),
        ("Genuine detected", f"{genuine:,}", pct(genuine, total), "gold"),
        ("Fake detected", f"{fake:,}", pct(fake, total), "fake"),
        ("Fake percentage", pct(fake, total), "of filtered reviews", ""),
        ("Reported model accuracy", f"{E.REPORTED_ACCURACY}%", "from model training", "gold"),
    ])
    if df.empty:
        st.info("No reviews match the current filters. Adjust the filters or search text above.")
    else:
        left, right = st.columns([1.7, 1])
        with left:
            show_fig(E.chart_activity(df))
        with right:
            show_fig(E.chart_donut(fake, genuine))
        c1, c2 = st.columns([1, 1.7])
        with c1:
            show_fig(E.chart_sentiment(sentiment_counts(df)))
        with c2:
            section("Recent analysis activity")
            recent = df.sort_values("ID", ascending=False).head(st.session_state["prefs"]["table_rows"])
            show_table(recent, ["ID", "Timestamp", "Source", "Status", "Confidence (%)", "Sentiment", "Review"],
                       height=330)
    callout(DISCLAIMER)


# ==========================================================================
# PAGE 2 - SINGLE REVIEW
# ==========================================================================
def _save_single_text():
    st.session_state["single_text"] = st.session_state.get("single_text_w", "")


def _clear_single():
    st.session_state["single_text"] = ""
    st.session_state["single_text_w"] = ""
    st.session_state["latest_single"] = None


def _example_single():
    st.session_state["single_text"] = E.example_review()
    st.session_state["single_text_w"] = E.example_review()


def page_single():
    page_header("Intelligence Analyzer", "Single Review",
                "Paste one review to see the model's probabilities, sentiment and suspicious writing indicators.")
    ss = st.session_state
    if "single_text_w" not in ss:
        ss["single_text_w"] = ss.get("single_text", "")

    st.text_area("Review text", key="single_text_w", height=210, on_change=_save_single_text,
                 placeholder="Paste or type a product review here...")
    text = ss["single_text_w"]
    words = len(text.split())
    st.caption(f"{len(text):,} characters · {words:,} words  (counts update when you click outside the box "
               "or press Ctrl+Enter)")

    b1, b2, b3, _ = st.columns([1.2, 0.8, 1.1, 2])
    analyse = b1.button("🔍  Analyse review", type="primary", key="single_go", disabled=not MODEL_OK)
    b2.button("Clear", key="single_clear", on_click=_clear_single)
    b3.button("Use example review", key="single_example", on_click=_example_single)
    if not MODEL_OK:
        require_model()

    if analyse:
        if not text.strip():
            st.warning("Please enter a review before analysing.")
        else:
            try:
                rec = E.analyze_reviews([text], MODEL, VECTORIZER, "Single", "", ss["next_id"])[0]
                add_records([rec])
                ss["latest_single"] = rec
                st.toast("Analysis complete")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    rec = ss["latest_single"]
    if rec is None:
        callout(DISCLAIMER)
        empty_state("No result yet", "Enter a review above and press Analyse review.")
        return

    st.markdown("<div class='rg-rule' style='margin-top:18px'></div>", unsafe_allow_html=True)
    is_fake = rec["Status"] == "FAKE"
    headline = "Likely fake (CG-style)" if is_fake else "Likely genuine (OR-style)"
    basis = {
        "Model + Heuristics": "Both the trained model and the rule-based indicators flagged this review.",
        "Model": "The trained model classified this review as computer-generated (CG).",
        "Heuristics": "The model leaned original (OR), but two or more heuristic indicators were triggered.",
        "None": "The model leaned original (OR) and no verdict-changing heuristic indicators were triggered.",
    }[rec["Flagged By"]]
    st.markdown(
        f"<div class='rg-verdict {'fake' if is_fake else 'genuine'}'>"
        f"<div class='rg-verdict-kicker'>Prediction result · #{rec['ID']}</div>"
        f"<div class='rg-verdict-main'>{headline}</div>"
        f"<div class='rg-verdict-note'>{esc(basis)}</div></div>",
        unsafe_allow_html=True,
    )
    stat_cards([
        ("Fake probability", f"{rec['Fake Probability (%)']}%", "model estimate (CG class)", "fake"),
        ("Genuine probability", f"{rec['Genuine Probability (%)']}%", "model estimate (OR class)", "gold"),
        ("Prediction confidence", f"{rec['Confidence (%)']}%", "highest class probability", ""),
        ("Sentiment", rec["Sentiment"], f"polarity {rec['Polarity']}", ""),
        ("Heuristic score", str(rec["Heuristic Score"]), "2 or more flags the review", ""),
    ])
    show_fig(E.chart_probability_bar(rec["Fake Probability (%)"], rec["Genuine Probability (%)"]))

    left, right = st.columns(2)
    with left:
        if rec["Flags"] == "None":
            body = "<p>No heuristic indicators were triggered.</p>"
        else:
            body = chips(rec["Flags"].split(", "), "fake")
        st.markdown(f"<div class='rg-panel'><h4>Suspicious writing indicators</h4>{body}</div>",
                    unsafe_allow_html=True)
        st.caption("Heuristics (promotional phrases, punctuation, repetition, capitals, emoji, very short "
                   "text) are simple signals, not proof of anything.")
    with right:
        st.markdown("<div class='rg-panel'><h4>Suspicious keywords found</h4>"
                    f"<p>{esc(rec['Suspicious Keywords'])}</p></div>", unsafe_allow_html=True)
        st.caption(f"{rec['Word Count']} words · {rec['Character Count']} characters · model class "
                   f"{rec['Model Prediction']}")
    st.download_button("⇩  Download report (.txt)", E.single_review_report(rec),
                       file_name="reviewguard_single_review_report.txt", mime="text/plain", key="single_dl")
    callout(DISCLAIMER)


# ==========================================================================
# PAGE 3 - BATCH ANALYSIS
# ==========================================================================
def page_batch():
    page_header("Bulk Processing Workspace", "Batch Analysis",
                "Upload a CSV of reviews, choose the text column and analyse every row with the same model.")
    ss = st.session_state
    up = st.file_uploader("Upload a CSV file", type=["csv"], key="batch_upload")

    if up is not None:
        data = up.getvalue()
        df, err = read_csv_bytes(data)
        if err:
            st.error(err)
        elif df is None or df.empty or len(df.columns) == 0:
            st.error("The CSV has no data rows to analyse.")
        else:
            section("File information")
            stat_cards([
                ("File", up.name, f"{len(data) / 1024:,.1f} KB", ""),
                ("Rows", f"{len(df):,}", "in the file", ""),
                ("Columns", f"{len(df.columns)}", ", ".join(map(str, df.columns[:4])) +
                 ("..." if len(df.columns) > 4 else ""), ""),
            ])
            section("Dataset preview")
            st.dataframe(df.head(10), hide_index=True)

            cols = list(df.columns)
            guess = E.guess_text_column(df)
            idx = cols.index(guess) if guess in cols else 0
            c1, c2 = st.columns(2)
            text_col = c1.selectbox("Review text column", cols, index=idx, key="batch_col")
            max_rows_cap = min(len(df), 20000)
            max_rows = c2.number_input("Maximum rows to analyse", min_value=1, max_value=max_rows_cap,
                                       value=min(max_rows_cap, 2000), step=100, key="batch_max")
            if len(df) > 20000:
                st.caption("Large files are capped at 20,000 rows per run to keep the session responsive.")

            if st.button("🚀  Analyse all", type="primary", key="batch_go", disabled=not MODEL_OK):
                texts = df[text_col].dropna().astype(str).str.strip()
                texts = texts[texts != ""].head(int(max_rows)).tolist()
                if not texts:
                    st.error(f"Column '{text_col}' has no non-empty text to analyse. "
                             "Choose a different column.")
                else:
                    bar = st.progress(0.0, text="Starting...")
                    try:
                        recs = E.analyze_reviews(
                            texts, MODEL, VECTORIZER, "Batch", up.name, ss["next_id"],
                            progress=lambda f, t: bar.progress(f, text=t))
                        add_records(recs)
                        ss["batch_df"] = E.records_to_df(recs)
                        ss["batch_meta"] = {"file": up.name, "column": text_col, "rows": len(recs),
                                            "skipped": int(len(df) - len(texts)),
                                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        bar.progress(1.0, text="Done")
                        st.success(f"Analysed {len(recs):,} reviews from '{up.name}'.")
                    except Exception as exc:
                        bar.empty()
                        st.error(f"Batch analysis failed: {exc}")
    if not MODEL_OK:
        require_model()

    res = ss["batch_df"]
    if res is None or res.empty:
        if up is None:
            empty_state("No batch results yet",
                        "Upload a CSV above to begin. Results, counts and a CSV download will appear here.")
        return

    meta = ss["batch_meta"] or {}
    st.markdown("<div class='rg-rule' style='margin-top:18px'></div>", unsafe_allow_html=True)
    section(f"Results summary · {meta.get('file', '')}")
    total, fake, genuine = counts(res)
    stat_cards([
        ("Reviews analysed", f"{total:,}", f"column '{meta.get('column', '')}'", ""),
        ("Genuine", f"{genuine:,}", pct(genuine, total), "gold"),
        ("Fake", f"{fake:,}", pct(fake, total), "fake"),
        ("Avg. confidence", f"{res['Confidence (%)'].mean():.1f}%", "highest class probability", ""),
    ])
    c1, c2 = st.columns([1, 1.4])
    with c1:
        show_fig(E.chart_donut(fake, genuine))
    with c2:
        show_fig(E.chart_confidence_hist(res))
    section("Results table")
    f1, f2 = st.columns([1, 2])
    status = f1.selectbox("Verdict", ["All", "FAKE", "GENUINE"], key="batch_status")
    q = f2.text_input("Search review text", key="batch_q")
    view = res if status == "All" else res[res["Status"] == status]
    view = search_filter(view, q)
    st.caption(f"Showing {len(view):,} of {len(res):,} rows")
    show_table(view)
    st.download_button("⇩  Download results (CSV)", E.df_to_csv_bytes(res),
                       file_name="reviewguard_batch_results.csv", mime="text/csv", key="batch_dl")
    callout(DISCLAIMER)


# ==========================================================================
# PAGE 4 - URL ANALYZER
# ==========================================================================
def page_url():
    page_header("Product Review Intelligence", "URL Analyzer",
                "Fetch public reviews from a supported product page and run them through the model.")
    ss = st.session_state
    callout("<b>Supported sources:</b> DummyJSON, Myntra, Flipkart and Shopsy, plus a generic fallback that "
            "looks for review-like page elements. Websites change and often block automated access, so "
            "<b>not every product page will work</b> - if nothing can be extracted you will see why.")

    c1, c2 = st.columns([3, 1])
    url = c1.text_input("Product URL", key="url_input",
                        placeholder="https://dummyjson.com/products/1  ·  Myntra / Flipkart / Shopsy link")
    max_reviews = c2.number_input("Maximum reviews", min_value=1, max_value=200,
                                  value=int(ss["prefs"]["url_max"]), step=5, key="url_max_w")
    if url.strip():
        st.markdown(f"Detected source: {chips([E.detect_platform(url)], 'gold')}", unsafe_allow_html=True)

    if st.button("🌐  Fetch reviews", type="primary", key="url_go", disabled=not MODEL_OK):
        if not url.strip():
            st.warning("Please paste a product URL first.")
        else:
            bar = st.progress(0.0, text="Starting extraction...")
            try:
                reviews, messages = E.fetch_reviews_from_url(
                    url, int(max_reviews), progress=lambda f, t: bar.progress(f, text=t))
            except Exception as exc:  # defensive: extraction must never crash the page
                reviews, messages = [], [("error", f"Unexpected extraction error: {exc}")]
            bar.empty()
            for level, msg in messages:
                {"error": st.error, "warning": st.warning}.get(level, st.info)(msg)
            if reviews:
                try:
                    recs = E.analyze_reviews(reviews, MODEL, VECTORIZER, "URL", url.strip(), ss["next_id"])
                    add_records(recs)
                    ss["url_df"] = E.records_to_df(recs)
                    ss["url_meta"] = {"url": url.strip(), "platform": E.detect_platform(url),
                                      "extracted": len(reviews)}
                    st.success(f"Extracted and analysed {len(recs)} reviews.")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
            elif not any(level == "error" for level, _ in messages):
                st.error("No written user reviews could be extracted from that link. The page may be "
                         "unsupported, blocked, or may load its reviews with JavaScript.")
    if not MODEL_OK:
        require_model()

    res = ss["url_df"]
    if res is None or res.empty:
        empty_state("No URL results yet",
                    "Paste a product link and press Fetch reviews. Try https://dummyjson.com/products/1 to "
                    "see how it works.")
        return
    meta = ss["url_meta"] or {}
    section(f"Analysis summary · {meta.get('platform', '')}")
    total, fake, genuine = counts(res)
    stat_cards([
        ("Reviews extracted", f"{meta.get('extracted', total)}", meta.get("platform", ""), ""),
        ("Genuine", f"{genuine}", pct(genuine, total), "gold"),
        ("Fake", f"{fake}", pct(fake, total), "fake"),
        ("Avg. confidence", f"{res['Confidence (%)'].mean():.1f}%", "highest class probability", ""),
    ])
    left, right = st.columns([1, 1.3])
    with left:
        show_fig(E.chart_donut(fake, genuine))
    with right:
        if fake > genuine:
            callout("<b>Higher-risk pattern:</b> more reviews resemble the computer-generated class than "
                    "the original class. Treat this as a prompt to look closer, not a conclusion.")
        else:
            callout("<b>Lower-risk pattern:</b> most reviews resemble the original class. This does not "
                    "guarantee they are authentic.")
        st.caption(f"Source: {meta.get('url', '')}")
    section("Results table")
    show_table(res)
    with st.expander("🔎  Inspect individual reviews"):
        for _, r in res.head(50).iterrows():
            st.markdown(f"**#{r['ID']} · {r['Status']}** · {r['Confidence (%)']}% confidence · {r['Sentiment']}")
            review_box(r["Review"])
    st.download_button("⇩  Download URL results (CSV)", E.df_to_csv_bytes(res),
                       file_name="reviewguard_url_results.csv", mime="text/csv", key="url_dl")
    callout(DISCLAIMER)


# ==========================================================================
# PAGE 5 - ANALYTICS
# ==========================================================================
def page_analytics():
    page_header("Data Intelligence", "Analytics",
                "Distributions, comparisons and trends across every review analysed this session.")
    df_all = history_df()
    if df_all.empty:
        empty_state("No data to analyse yet",
                    "Run a single, batch or URL analysis and the charts below will be built from your real results.",
                    "Go to Batch Analysis", P_BATCH, key="an_empty")
        return

    section("Filters")
    f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1.4])
    sources = sorted(df_all["Source"].unique())
    sel_sources = f1.multiselect("Source", sources, default=sources, key="an_sources")
    sel_status = f2.selectbox("Verdict", ["All", "FAKE", "GENUINE"], key="an_status")
    sel_sent = f3.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"], key="an_sent")
    conf = f4.slider("Confidence (%)", 50, 100, (50, 100), key="an_conf")
    query = st.text_input("Search review text", key="an_query")

    df = df_all[df_all["Source"].isin(sel_sources)]
    if sel_status != "All":
        df = df[df["Status"] == sel_status]
    if sel_sent != "All":
        df = df[df["Sentiment"] == sel_sent]
    df = df[(df["Confidence (%)"] >= conf[0]) & (df["Confidence (%)"] <= conf[1])]
    df = search_filter(df, query)

    total, fake, genuine = counts(df)
    stat_cards([
        ("Matching reviews", f"{total:,}", f"of {len(df_all):,}", ""),
        ("Genuine", f"{genuine:,}", pct(genuine, total), "gold"),
        ("Fake", f"{fake:,}", pct(fake, total), "fake"),
        ("Avg. fake probability", f"{df['Fake Probability (%)'].mean():.1f}%" if total else "-", "model estimate", ""),
    ])
    if df.empty:
        st.info("No reviews match these filters.")
        return
    a, b = st.columns(2)
    with a:
        show_fig(E.chart_donut(fake, genuine))
    with b:
        show_fig(E.chart_confidence_hist(df))
    c, d = st.columns(2)
    with c:
        show_fig(E.chart_source_compare(df))
    with d:
        show_fig(E.chart_fake_probability_hist(df))
    show_fig(E.chart_trend(df), "Not enough data for a trend", "At least two analysed reviews are needed.")
    section("Searchable data table")
    st.caption(f"{len(df):,} rows")
    show_table(df, height=460)


# ==========================================================================
# PAGE 6 - SENTIMENT
# ==========================================================================
def page_sentiment():
    page_header("Sentiment Analysis", "Sentiment",
                "How positive, negative or neutral the wording is - measured with TextBlob.")
    callout("<b>Sentiment is not authenticity.</b> Sentiment describes the tone of the wording. A fake review "
            "can be negative and a genuine one can be glowing, so the two are reported separately.")
    df_all = history_df()
    if df_all.empty:
        empty_state("No sentiment data yet", "Analyse some reviews to see sentiment results.",
                    "Analyse a review", P_SINGLE, key="se_empty")
        return
    if (df_all["Sentiment"] == "Unavailable").all():
        st.error("TextBlob is not installed, so sentiment could not be computed. "
                 "Run: pip install -r requirements.txt")
        return
    total = len(df_all)
    sc = sentiment_counts(df_all)
    pos, neu, neg = sc.get("Positive", 0), sc.get("Neutral", 0), sc.get("Negative", 0)
    avg_pol = pd.to_numeric(df_all["Polarity"], errors="coerce").mean()
    stat_cards([
        ("Positive", f"{pos:,}", pct(pos, total), ""),
        ("Neutral", f"{neu:,}", pct(neu, total), ""),
        ("Negative", f"{neg:,}", pct(neg, total), ""),
        ("Average polarity", f"{avg_pol:.2f}" if avg_pol == avg_pol else "-", "-1 negative to +1 positive", "gold"),
    ])
    a, b = st.columns(2)
    with a:
        show_fig(E.chart_sentiment(sc))
    with b:
        show_fig(E.chart_sentiment_by_status(df_all))
    section("Review-level sentiment")
    f1, f2 = st.columns([1, 2])
    pick = f1.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"], key="se_pick")
    q = f2.text_input("Search review text", key="se_q")
    view = df_all if pick == "All" else df_all[df_all["Sentiment"] == pick]
    view = search_filter(view, q)
    st.caption(f"{len(view):,} rows")
    show_table(view, ["ID", "Source", "Sentiment", "Polarity", "Status", "Fake Probability (%)", "Review"])


# ==========================================================================
# PAGE 7 - SUSPICIOUS PATTERNS
# ==========================================================================
def page_patterns():
    page_header("Risk Monitor", "Suspicious Patterns",
                "Rule-based writing indicators found in the analysed reviews.")
    callout("<b>Heuristic indicators, not proof.</b> Real customers also write short, excited or repetitive "
            "reviews. These flags only point to text worth a closer look.")
    df_all = history_df()
    if df_all.empty:
        empty_state("No reviews to inspect yet", "Analyse reviews to see which writing indicators appear.",
                    "Analyse a review", P_SINGLE, key="pa_empty")
        return
    flag_cols = list(E.PATTERN_COLUMNS.values())
    any_flag = df_all[flag_cols].fillna(False).astype(bool).any(axis=1)
    n_flagged = int(any_flag.sum())
    cards = [("Reviews with any indicator", f"{n_flagged:,}", pct(n_flagged, len(df_all)), "fake")]
    for label, col in E.PATTERN_COLUMNS.items():
        cards.append((label, f"{int(df_all[col].fillna(False).astype(bool).sum()):,}", "reviews", ""))
    stat_cards(cards)

    a, b = st.columns([1.3, 1])
    with a:
        show_fig(E.chart_patterns(df_all), "No indicators triggered",
                 "None of the analysed reviews triggered a heuristic indicator.")
    with b:
        st.markdown("<div class='rg-panel'><h4>How the indicators are defined</h4><p>"
                    "Promotional phrases: any of a fixed list (e.g. “must buy”, “amazing”).<br>"
                    "Excessive punctuation: more than 3 exclamation marks.<br>"
                    "Repeated words: more than 5 repeated words.<br>"
                    "Unusual uppercase: more than 15 capital letters.<br>"
                    "Excessive emoji: more than 5 from the tracked set.<br>"
                    "Very short: fewer than 3 words.</p></div>", unsafe_allow_html=True)

    section("Flagged review table")
    pick = st.multiselect("Show reviews with these indicators (any)", list(E.PATTERN_COLUMNS),
                          key="pa_pick", placeholder="All flagged reviews")
    if pick:
        mask = df_all[[E.PATTERN_COLUMNS[p] for p in pick]].fillna(False).astype(bool).any(axis=1)
    else:
        mask = any_flag
    view = search_filter(df_all[mask], st.text_input("Search review text", key="pa_q"))
    if view.empty:
        st.info("No reviews match.")
    else:
        st.caption(f"{len(view):,} flagged reviews")
        show_table(view, ["ID", "Source", "Status", "Heuristic Score", "Flags", "Suspicious Keywords", "Review"])


# ==========================================================================
# PAGE 8 - HISTORY
# ==========================================================================
def page_history():
    page_header("Activity Records", "History",
                "Every review analysed in this session. History lives in memory only and is lost when the "
                "browser session ends.")
    ss = st.session_state
    df_all = history_df()
    if df_all.empty:
        empty_state("History is empty", "Analyse a review, batch or URL to start building session history.",
                    "Analyse a review", P_SINGLE, key="hi_empty")
        return

    f1, f2, f3, f4 = st.columns([2, 1, 1, 1.2])
    q = f1.text_input("Search reviews", key="hi_q", placeholder="Search review text...")
    pred = f2.selectbox("Prediction", ["All", "FAKE", "GENUINE"], key="hi_pred")
    sent = f3.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"], key="hi_sent")
    sort = f4.selectbox("Sort by", ["Newest first", "Oldest first", "Fake probability (high)",
                                    "Fake probability (low)", "Confidence (high)", "Confidence (low)"],
                        key="hi_sort")
    df = df_all
    if pred != "All":
        df = df[df["Status"] == pred]
    if sent != "All":
        df = df[df["Sentiment"] == sent]
    df = search_filter(df, q)
    sorters = {
        "Newest first": ("ID", False), "Oldest first": ("ID", True),
        "Fake probability (high)": ("Fake Probability (%)", False),
        "Fake probability (low)": ("Fake Probability (%)", True),
        "Confidence (high)": ("Confidence (%)", False), "Confidence (low)": ("Confidence (%)", True),
    }
    col, asc = sorters[sort]
    df = df.sort_values(col, ascending=asc, kind="stable")

    st.caption(f"{len(df):,} of {len(df_all):,} records")
    show_table(df, height=380)

    section("Review details")
    if df.empty:
        st.info("No records match the filters.")
    else:
        size = ss["prefs"]["table_rows"]
        pages = max(1, -(-len(df) // size))
        page_no = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="hi_page") if pages > 1 else 1
        chunk = df.iloc[(page_no - 1) * size: page_no * size]
        st.caption(f"Showing {len(chunk)} records (page {page_no} of {pages}). "
                   "Change records per page in Settings.")
        for _, r in chunk.iterrows():
            preview = (r["Review"][:70] + "...") if len(r["Review"]) > 70 else r["Review"]
            with st.expander(f"#{r['ID']} · {r['Status']} · {r['Source']} · {preview}"):
                record_detail(r.to_dict())

    section("Clear history")
    if not ss["confirm_clear"]:
        st.button("🗑  Clear history…", key="hi_clear", on_click=lambda: ss.update(confirm_clear=True))
    else:
        st.warning(f"This permanently removes all {len(df_all):,} records, the latest single report and the "
                   "batch and URL result tables from this session. This cannot be undone.")
        c1, c2, _ = st.columns([1, 1, 3])
        c1.button("Yes, clear everything", key="hi_clear_yes", type="primary", on_click=clear_all_data)
        c2.button("Cancel", key="hi_clear_no", on_click=lambda: ss.update(confirm_clear=False))


# ==========================================================================
# PAGE 9 - EXPORT
# ==========================================================================
def page_export():
    page_header("Reporting Centre", "Export Reports",
                "Download your real analysis records. Files contain exactly what you see in the workspace.")
    ss = st.session_state
    df_all = history_df()
    batch, urls, latest = ss["batch_df"], ss["url_df"], ss["latest_single"]
    total, fake, genuine = counts(df_all)
    section("Report summary")
    stat_cards([
        ("Total records", f"{total:,}", "this session", ""),
        ("Genuine", f"{genuine:,}", pct(genuine, total), "gold"),
        ("Fake", f"{fake:,}", pct(fake, total), "fake"),
        ("Batch results", f"{0 if batch is None else len(batch):,}", "latest batch run", ""),
        ("URL results", f"{0 if urls is None else len(urls):,}", "latest URL run", ""),
    ])
    if df_all.empty:
        empty_state("Nothing to export yet", "Analyse some reviews first - exports are built from real results only.",
                    "Analyse a review", P_SINGLE, key="ex_empty")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='rg-panel'><h4>All records</h4><p>Every analysed review in the session.</p></div>",
                    unsafe_allow_html=True)
        st.download_button("⇩  All records (CSV)", E.df_to_csv_bytes(df_all), "reviewguard_all_records.csv",
                           "text/csv", key="ex_csv")
        st.download_button("⇩  All records (JSON)", E.df_to_json_bytes(df_all), "reviewguard_all_records.json",
                           "application/json", key="ex_json")
    with c2:
        st.markdown("<div class='rg-panel'><h4>Batch, URL and single reports</h4>"
                    "<p>Results of your latest runs.</p></div>", unsafe_allow_html=True)
        st.download_button("⇩  Batch results (CSV)",
                           E.df_to_csv_bytes(batch) if batch is not None else b"", "reviewguard_batch_results.csv",
                           "text/csv", key="ex_batch", disabled=batch is None or batch.empty)
        st.download_button("⇩  URL results (CSV)",
                           E.df_to_csv_bytes(urls) if urls is not None else b"", "reviewguard_url_results.csv",
                           "text/csv", key="ex_url", disabled=urls is None or urls.empty)
        st.download_button("⇩  Latest single-review report (.txt)",
                           E.single_review_report(latest) if latest else "", "reviewguard_single_review_report.txt",
                           "text/plain", key="ex_single", disabled=latest is None)
    if batch is None:
        st.caption("Batch export unlocks after you run a batch analysis.")
    if latest is None:
        st.caption("The single-review report unlocks after you analyse a single review.")
    callout(DISCLAIMER)


# ==========================================================================
# PAGE 10 - MODEL PERFORMANCE
# ==========================================================================
def page_performance():
    page_header("ML Monitoring", "Model Performance",
                "What is known about the trained model, and metrics that can be verified on labelled data.")
    info = E.model_info(MODEL, VECTORIZER)
    stat_cards([
        ("Reported accuracy", f"{E.REPORTED_ACCURACY}%", "from model training", "gold"),
        ("Algorithm", "Logistic Regression", "scikit-learn", ""),
        ("Feature extraction", "TF-IDF", "unigram term weights", ""),
        ("Model status", "Loaded" if MODEL_OK else "Unavailable", MODEL_ERROR or "pickle files read OK", "" if MODEL_OK else "fake"),
    ])
    if not MODEL_OK:
        require_model()
        return

    left, right = st.columns(2)
    with left:
        section("Model classes")
        st.markdown(chips([f"{info['classes'][0]} = Computer-generated / Fake",
                           f"{info['classes'][1]} = Original / Genuine"], "gold"), unsafe_allow_html=True)
        section("Model configuration")
        st.dataframe(pd.DataFrame([
            ("Algorithm", info["algorithm"]), ("Regularisation C", info["C"]), ("Solver", info["solver"]),
            ("Max iterations", info["max_iter"]), ("Input features", info["n_features_model"]),
        ], columns=["Setting", "Value"]).astype(str), hide_index=True)
    with right:
        section("Vectoriser configuration")
        st.dataframe(pd.DataFrame([
            ("Vectoriser", info["vectorizer"]), ("Vocabulary size", info["vocabulary_size"]),
            ("Max features", info["max_features"]), ("Stop words", info["stop_words"]),
            ("Lower-casing", info["lowercase"]), ("N-gram range", info["ngram_range"]),
        ], columns=["Setting", "Value"]).astype(str), hide_index=True)

    section("Verified evaluation on the hold-out set")
    callout("Precision, recall, F1 and the confusion matrix are <b>not stored in the model files</b>, so they "
            "are not assumed. The button below re-creates the 80/20 split from <code>train_model.py</code> "
            "(random_state=42) on <code>fake_reviews_dataset.csv</code> and scores the existing model on the "
            "20% hold-out. Note: the saved vectoriser was fitted on the full dataset, as in the training script.")
    dataset_ok = E.DATASET_PATH.exists()
    if not dataset_ok:
        st.info("fake_reviews_dataset.csv was not found next to app.py, so no verified metrics can be computed.")
    if st.button("Run hold-out evaluation", key="perf_run", disabled=not dataset_ok):
        with st.spinner("Scoring the hold-out set..."):
            try:
                st.session_state["holdout"] = E.evaluate_holdout(MODEL, VECTORIZER)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
    ev = st.session_state["holdout"]
    if ev:
        stat_cards([
            ("Hold-out accuracy", f"{ev['accuracy']:.2f}%", f"{ev['n_test']:,} test reviews", "gold"),
            ("Precision (CG / OR)", f"{ev['precision'][0]:.1f}% / {ev['precision'][1]:.1f}%", "per class", ""),
            ("Recall (CG / OR)", f"{ev['recall'][0]:.1f}% / {ev['recall'][1]:.1f}%", "per class", ""),
            ("F1 (CG / OR)", f"{ev['f1'][0]:.1f}% / {ev['f1'][1]:.1f}%", "per class", ""),
        ])
        a, b = st.columns([1, 1.4])
        with a:
            show_fig(E.chart_confusion(ev["confusion"], ev["labels"]))
        with b:
            st.dataframe(pd.DataFrame({
                "Class": ev["labels"],
                "Precision (%)": [round(x, 2) for x in ev["precision"]],
                "Recall (%)": [round(x, 2) for x in ev["recall"]],
                "F1 (%)": [round(x, 2) for x in ev["f1"]],
                "Support": ev["support"],
            }), hide_index=True)

    section("Highest-weight terms (from the model coefficients)")
    terms = E.top_model_terms(MODEL, VECTORIZER, 12)
    a, b = st.columns(2)
    with a:
        show_fig(E.chart_terms(terms.get("CG"), "Terms pushing toward CG (fake)", E.C_FAKE))
    with b:
        show_fig(E.chart_terms(terms.get("OR"), "Terms pushing toward OR (genuine)", E.C_GOLD))
    st.caption("These terms describe the training data's statistical patterns, not a rule for what makes a review fake.")


# ==========================================================================
# PAGE 11 - ABOUT MODEL
# ==========================================================================
def page_about():
    page_header("Technical Information", "About the Model",
                "How a review travels from raw text to a prediction in ReviewGuard AI.")
    steps = [
        ("Text preprocessing", "Lower-case, remove URLs, strip anything that is not a letter or space."),
        ("TF-IDF features", "Words become weighted numeric features (5,000 terms, English stop words removed)."),
        ("Logistic Regression", "Outputs a probability for CG (fake) and OR (original)."),
        ("Rule-based indicators", "Counts promotional phrases, punctuation, repetition, capitals, emoji, length."),
        ("TextBlob sentiment", "Polarity score labels the tone Positive, Neutral or Negative."),
        ("Verdict", "FAKE if the model says CG <i>or</i> two or more indicators fire; otherwise GENUINE."),
    ]
    parts = []
    for i, (title, text) in enumerate(steps, 1):
        parts.append(f"<div class='rg-step'><div class='rg-step-n'>{i}</div><h5>{title}</h5><p>{text}</p></div>")
        if i < len(steps):
            parts.append("<div class='rg-arrow'>→</div>")
    st.markdown(f"<div class='rg-flow'>{''.join(parts)}</div>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["1 · Preprocessing", "2 · TF-IDF", "3 · Logistic Regression",
                                  "4 · Rule-based indicators", "5 · Sentiment"])
    with t1:
        st.markdown("Each review is passed through `clean_text()`: lower-cased, links removed, and every "
                    "character that is not a letter or whitespace dropped. The **original** review text is kept "
                    "for display, sentiment and the rule-based indicators.")
    with t2:
        st.markdown("**TF-IDF** (term frequency × inverse document frequency) scores how characteristic a word "
                    "is of a review relative to the whole dataset. The saved vectoriser keeps the 5,000 "
                    "highest-frequency terms and ignores English stop words such as “the” and “and”.")
    with t3:
        st.markdown("**Logistic Regression** learns a weight per term from the labelled dataset "
                    "(`fake_reviews_dataset.csv`, 40,432 reviews, 50% CG / 50% OR) and converts the weighted sum "
                    "to class probabilities. The *Fake probability* shown in the app is the model's CG "
                    "probability; *Confidence* is the larger of the two probabilities.")
    with t4:
        st.markdown("Simple, transparent rules count: promotional phrases from a fixed list, more than 3 "
                    "exclamation marks, more than 5 repeated words, more than 15 capital letters, more than 5 "
                    "tracked emoji, and fewer than 3 words. They are **heuristics** and are never shown as proof.")
    with t5:
        st.markdown("**TextBlob** returns a polarity from -1 to +1. Above 0 is Positive, below 0 Negative, "
                    "exactly 0 Neutral. Sentiment describes tone only and is independent of the CG/OR prediction.")
    st.markdown("<div class='rg-panel'><h4>Limitations</h4><p>The model learns statistical patterns of one "
                "dataset. It cannot verify purchases, identify authors or establish authenticity, and its "
                f"reported accuracy ({E.REPORTED_ACCURACY}%) applies to that dataset, not to every website or "
                "writing style.</p></div>", unsafe_allow_html=True)


# ==========================================================================
# PAGE 12 - SETTINGS
# ==========================================================================
def _set_pref(key, widget_key):
    st.session_state["prefs"][key] = st.session_state[widget_key]


def _reset_prefs():
    st.session_state["prefs"] = {"table_rows": 25, "url_max": 50}
    st.session_state["w_rows"] = 25
    st.session_state["w_urlmax"] = 50


def page_settings():
    page_header("Workspace Configuration", "Settings",
                "Information about this workspace and the few preferences that change its behaviour.")
    ss = st.session_state
    a, b = st.columns(2)
    with a:
        section("Workspace preferences")
        st.session_state.setdefault("w_rows", ss["prefs"]["table_rows"])
        st.session_state.setdefault("w_urlmax", ss["prefs"]["url_max"])
        st.selectbox("Records per page (Overview recent activity, History details)", [10, 25, 50, 100],
                     key="w_rows", on_change=_set_pref, args=("table_rows", "w_rows"))
        st.number_input("Default maximum reviews for URL Analyzer", 1, 200, step=5, key="w_urlmax",
                        on_change=_set_pref, args=("url_max", "w_urlmax"))
        st.button("Reset preferences", key="set_reset", on_click=_reset_prefs)
        section("Theme")
        st.markdown(
            "<div class='rg-panel'><p>Fixed premium dark theme (no toggle).<br>"
            + "".join(f"<span class='rg-swatch' style='background:{c}'></span>{n} {c}<br>" for n, c in [
                ("Background", E.C_BG), ("Cards", E.C_CARD), ("Gold", E.C_GOLD), ("Text", E.C_TEXT)])
            + "</p></div>", unsafe_allow_html=True)
    with b:
        section("Model status")
        st.markdown(f"<div class='rg-panel'><p><span class='rg-dot {'ok' if MODEL_OK else 'bad'}'></span>"
                    f"{'Model and vectoriser loaded' if MODEL_OK else esc(MODEL_ERROR)}<br>"
                    f"Model: {esc(E.MODEL_PATH.name)}<br>Vectoriser: {esc(E.VECTORIZER_PATH.name)}<br>"
                    f"Reported accuracy: {E.REPORTED_ACCURACY}%</p></div>", unsafe_allow_html=True)
        section("Application information")
        try:
            import sklearn
            sk = sklearn.__version__
        except Exception:
            sk = "unknown"
        st.markdown(f"<div class='rg-panel'><p>ReviewGuard AI v{APP_VERSION}<br>"
                    f"Project: Review Verification Using AI<br>Streamlit {st.__version__} · scikit-learn {sk}<br>"
                    f"Python {sys.version.split()[0]} · {esc(platform.system())}<br>"
                    f"TextBlob: {'available' if E.TextBlob else 'not installed'}</p></div>", unsafe_allow_html=True)
        section("Session information")
        df = history_df()
        by_source = df["Source"].value_counts().to_dict() if not df.empty else {}
        st.markdown(f"<div class='rg-panel'><p>Session started: {esc(ss['session_started'])}<br>"
                    f"Records: {len(df):,} "
                    f"({', '.join(f'{k}: {v}' for k, v in by_source.items()) or 'none'})<br>"
                    "Storage: in memory only; nothing is written to disk.</p></div>", unsafe_allow_html=True)


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    init_state()
    inject_css()
    render_sidebar()
    router = {
        P_OVERVIEW: page_overview, P_SINGLE: page_single, P_BATCH: page_batch, P_URL: page_url,
        P_ANALYTICS: page_analytics, P_SENTIMENT: page_sentiment, P_PATTERNS: page_patterns,
        P_HISTORY: page_history, P_EXPORT: page_export, P_PERF: page_performance,
        P_ABOUT: page_about, P_SETTINGS: page_settings,
    }
    page = st.session_state.get("nav", P_OVERVIEW)
    try:
        router.get(page, page_overview)()
    except Exception as exc:  # last-resort guard so one bad page never blanks the app
        st.error(f"Something went wrong while rendering this page: {exc}")
    disclaimer_footer()


main()
