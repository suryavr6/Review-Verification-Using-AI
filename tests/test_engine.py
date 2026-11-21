"""
Engine tests. Run from the project folder:  python tests/test_engine.py
(If TextBlob is not installed, a tiny stand-in is used so the rest can still be checked.)
"""
import json, os, sys, types, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

stubbed = False
try:
    import textblob  # noqa
except ImportError:
    stubbed = True
    m = types.ModuleType("textblob")
    class _S:  # naive polarity: only for testing without the real package
        def __init__(self, t):
            t = t.lower(); self.sentiment = types.SimpleNamespace(
                polarity=0.5 if "love" in t or "great" in t else (-0.5 if "broke" in t else 0.0))
    m.TextBlob = _S; sys.modules["textblob"] = m

import pandas as pd
import engine as E

ok = 0
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    assert cond, name
    ok += 1

model, vec, err = E.load_models()
check("model loads", model is not None and err is None, err or "")
check("classes are CG/OR", list(model.classes_) == ["CG", "OR"])

# single review
recs = E.analyze_reviews([E.example_review()], model, vec, "Single", start_id=1)
r = recs[0]
check("single record has all columns", set(E.RECORD_COLUMNS) <= set(r))
check("probabilities sum to ~100", abs(r["Fake Probability (%)"] + r["Genuine Probability (%)"] - 100) < 0.05)
check("example review flagged FAKE by heuristics", r["Status"] == "FAKE" and "Heuristics" in r["Flagged By"], r["Flagged By"])
genuine = E.analyze_reviews(["The zipper broke after two weeks and support never replied."], model, vec)[0]
check("plain review not flagged by heuristics", genuine["Heuristic Score"] == 0)

# model label fix: CG predictions must now mark FAKE
d = pd.read_csv("fake_reviews_dataset.csv").sample(400, random_state=3)
recs = E.analyze_reviews(d["text_"].tolist(), model, vec, "Batch", "sample.csv", start_id=1)
df = E.records_to_df(recs)
cg = df[df["Model Prediction"] == "CG"]
check("every model-CG review is FAKE", (cg["Status"] == "FAKE").all(), f"{len(cg)} CG rows")
check("batch ids unique", df["ID"].is_unique)
share = (df["Model Prediction"] == d["label"].values).mean()
check("batch model agreement with labels is plausible (>0.75)", share > 0.75, f"{share:.2f}")

# progress callback
calls = []
E.analyze_reviews(d["text_"].tolist()[:120], model, vec, progress=lambda f, t: calls.append(f), chunk_size=50)
check("progress reaches 1.0", calls and calls[-1] == 1.0, str(calls))

# empty model guard
try:
    E.analyze_reviews(["x"], None, None); check("analysis without model raises", False)
except RuntimeError:
    check("analysis without model raises", True)

# charts: empty + identical timestamps + invalid timestamps
empty = E.records_to_df([])
for fn in (E.chart_activity, E.chart_confidence_hist, E.chart_fake_probability_hist, E.chart_trend,
           E.chart_source_compare, E.chart_patterns, E.chart_sentiment_by_status):
    check(f"{fn.__name__}(empty) -> None", fn(empty) is None)
check("donut(0,0) -> None", E.chart_donut(0, 0) is None)
check("sentiment({}) -> None", E.chart_sentiment({}) is None)
check("identical timestamps -> figure", E.chart_activity(df) is not None and df["Timestamp"].nunique() == 1)
bad = df.copy(); bad["Timestamp"] = "not a date"
check("all-invalid timestamps -> None (no crash)", E.chart_activity(bad) is None)
mixed = df.copy(); mixed.loc[mixed.index[:5], "Timestamp"] = "garbage"
check("some invalid timestamps tolerated", E.chart_activity(mixed) is not None)
spread = df.copy()
spread["Timestamp"] = pd.date_range("2026-11-01", periods=len(df), freq="37h").strftime("%Y-%m-%d %H:%M:%S")
check("spread timestamps -> figure", E.chart_activity(spread) is not None)
check("one-row df charts", E.chart_activity(df.head(1)) is not None and E.chart_trend(df.head(1)) is None)

# exports
csv = E.df_to_csv_bytes(df); js = json.loads(E.df_to_json_bytes(df))
check("csv has data rows", csv.count(b"\n") > len(df))
check("json has all records", js["record_count"] == len(df) == len(js["records"]))
check("report text contains review", E.example_review() in E.single_review_report(r))
check("json handles empty", json.loads(E.df_to_json_bytes(empty))["record_count"] == 0)

# URL error handling (no network / missing curl_cffi must never crash)
rv, msgs = E.fetch_reviews_from_url("", 10)
check("empty URL -> warning", rv == [] and msgs and msgs[0][0] == "warning")
rv, msgs = E.fetch_reviews_from_url("https://example.invalid/product", 10)
check("unreachable/unsupported URL -> clean error", rv == [] and len(msgs) >= 1, str(msgs))
check("platform detection", [E.detect_platform(u) for u in (
    "https://dummyjson.com/products/1", "https://www.myntra.com/x/123456/buy", "https://www.flipkart.com/a/p/b",
    "https://www.shopsy.in/a", "https://example.com", "")] == ["DummyJSON", "Myntra", "Flipkart", "Shopsy", "Generic fallback", "None"])

# URL extraction with a fake HTTP client (verifies the preserved DummyJSON logic path)
class _Resp:
    status_code = 200
    text = ""
    def json(self): return {"reviews": [{"comment": "Great quality and fast delivery overall"}, {"comment": "ok"}]}
fake = types.SimpleNamespace(get=lambda *a, **k: _Resp())
E._http_client = lambda: fake
rv, msgs = E.fetch_reviews_from_url("https://dummyjson.com/products/3", 10)
check("DummyJSON logic keeps valid reviews, drops short ones", rv == ["Great quality and fast delivery overall"], str(rv))

# CSV helper
check("guess_text_column prefers text_", E.guess_text_column(pd.DataFrame({"a": [1], "text_": ["hello there friend"]})) == "text_")
check("guess_text_column on empty", E.guess_text_column(pd.DataFrame()) is None)

# model info / terms / holdout
info = E.model_info(model, vec)
check("model info", info["loaded"] and info["vocabulary_size"] == 5000, str(info["max_features"]))
terms = E.top_model_terms(model, vec, 5)
check("top terms for both classes", set(terms) == {"CG", "OR"} and len(terms["CG"]) == 5)
ev = E.evaluate_holdout(model, vec)
check("hold-out accuracy reproduces reported 86.09%", abs(ev["accuracy"] - 86.09) < 0.01, f"{ev['accuracy']:.2f}")
check("hold-out confusion sums to n_test", sum(map(sum, ev["confusion"])) == ev["n_test"])
check("evaluate without dataset -> None", E.evaluate_holdout(model, vec, "nope.csv") is None)

# render all charts once (visual check)
os.makedirs("/tmp/rg_charts", exist_ok=True)
for name, fig in {
    "activity": E.chart_activity(df), "activity_spread": E.chart_activity(spread),
    "donut": E.chart_donut(int((df.Status=="FAKE").sum()), int((df.Status=="GENUINE").sum())),
    "sent": E.chart_sentiment({"Positive": 5, "Neutral": 3, "Negative": 2}),
    "sentstatus": E.chart_sentiment_by_status(df), "conf": E.chart_confidence_hist(df),
    "fakep": E.chart_fake_probability_hist(df), "trend": E.chart_trend(df), "source": E.chart_source_compare(df),
    "prob": E.chart_probability_bar(63.2, 36.8), "patterns": E.chart_patterns(df),
    "cm": E.chart_confusion(ev["confusion"], ev["labels"]), "terms": E.chart_terms(terms["CG"], "CG terms", E.C_FAKE),
}.items():
    check(f"chart '{name}' renders", fig is not None)
    fig.savefig(f"/tmp/rg_charts/{name}.png", facecolor=fig.get_facecolor())
print(f"\n{ok} checks passed" + ("  (TextBlob stand-in used)" if stubbed else ""))
