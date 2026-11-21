"""
Smoke test of app.py's page logic WITHOUT Streamlit installed.

A recording stand-in for the `streamlit` module is used, so this verifies that every page's
Python code runs (empty state and with data) and that button flows work. It does NOT verify
Streamlit's real API behaviour or visual rendering - run the app for that.

Run from the project folder:  python tests/test_app_mock.py
"""
import io, os, sys, types, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

try:
    import textblob  # noqa
except ImportError:
    m = types.ModuleType("textblob")
    class _S:
        def __init__(self, t):
            t = t.lower(); self.sentiment = types.SimpleNamespace(
                polarity=0.5 if ("love" in t or "great" in t or "amazing" in t) else (-0.4 if "broke" in t else 0.0))
    m.TextBlob = _S; sys.modules["textblob"] = m

# ---------------------------------------------------------------- fake streamlit
class Dummy:
    def __call__(self, *a, **k): return Dummy()
    def __getattr__(self, n): return Dummy()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter(())
    def __bool__(self): return False

class Col(Dummy):
    """Column/tab/container stand-in: widget calls on it go to the recording st module."""
    def __getattr__(self, n):
        return getattr(sys.modules["streamlit"], n)

class State(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v

LOG = {"error": [], "warning": [], "success": [], "info": [], "downloads": [], "pyplot": 0, "tables": []}
PRESSED = set(); UPLOAD = {"obj": None}

class Bar:
    def progress(self, *a, **k): pass
    def empty(self): pass

def _opts(options): return list(options)
def make_st():
    st = types.ModuleType("streamlit"); st.__version__ = "mock"
    st.session_state = State()
    st.sidebar = Col(); st.column_config = Dummy()
    st.cache_resource = lambda f=None, **k: (f if f else (lambda g: g))
    st.cache_data = lambda f=None, **k: (f if f else (lambda g: g))
    st.set_page_config = lambda **k: None
    st.markdown = lambda *a, **k: None
    for n in ("caption", "write", "toast", "subheader", "title", "code", "text"):
        setattr(st, n, lambda *a, **k: None)
    for n in ("error", "warning", "success", "info"):
        setattr(st, n, (lambda nn: lambda msg="", *a, **k: LOG[nn].append(str(msg)))(n))
    st.columns = lambda spec, **k: [Col() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels, **k: [Col() for _ in labels]
    st.expander = lambda *a, **k: Col()
    st.container = lambda *a, **k: Col()
    st.spinner = lambda *a, **k: Dummy()
    st.progress = lambda *a, **k: Bar()
    st.metric = lambda *a, **k: None
    def pyplot(fig, **k): LOG["pyplot"] += 1
    st.pyplot = pyplot
    def dataframe(df, **k):
        assert hasattr(df, "columns"); LOG["tables"].append(len(df))
    st.dataframe = dataframe
    def button(label, key=None, on_click=None, args=(), disabled=False, **k):
        hit = (key in PRESSED) and not disabled
        if hit and on_click: on_click(*args)
        return hit
    st.button = button
    def download_button(label, data, file_name=None, mime=None, key=None, disabled=False, **k):
        if not disabled:
            assert len(data) > 0, f"empty download: {label}"
            LOG["downloads"].append((label, len(data)))
        return False
    st.download_button = download_button
    def _store(key, default):
        if key is not None:
            st.session_state.setdefault(key, default); return st.session_state[key]
        return default
    st.radio = lambda label, options, key=None, **k: _store(key, options[0])
    st.selectbox = lambda label, options, index=0, key=None, **k: _store(key, list(options)[index])
    st.multiselect = lambda label, options, default=None, key=None, **k: _store(key, default or [])
    st.text_input = lambda label, value="", key=None, **k: _store(key, value)
    st.text_area = lambda label, value="", key=None, **k: _store(key, value)
    st.number_input = lambda label, min_value=None, max_value=None, value=None, key=None, **k: _store(key, value if value is not None else min_value)
    st.slider = lambda label, a=0, b=100, value=None, key=None, **k: _store(key, value if value is not None else a)
    st.file_uploader = lambda *a, **k: UPLOAD["obj"]
    return st

st = make_st(); sys.modules["streamlit"] = st
import pandas as pd
import engine as E
import app  # runs main() once (empty state)

PAGES = app.PAGES
def render(page, pressed=()):
    PRESSED.clear(); PRESSED.update(pressed)
    st.session_state["nav"] = page
    before = len(LOG["error"])
    app.main()
    new = [e for e in LOG["error"][before:] if "Something went wrong" in e]
    assert not new, f"{page}: {new}"
fails = 0
def check(name, cond, extra=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    fails += (not cond)

# 1. all 12 pages, empty state
for p in PAGES: render(p)
check("12 pages render in empty state", len(PAGES) == 12)

# 2. quick-action buttons switch pages
for key, target in (("qa_single", app.P_SINGLE), ("qa_batch", app.P_BATCH), ("qa_url", app.P_URL), ("qa_export", app.P_EXPORT)):
    st.session_state["nav"] = app.P_OVERVIEW
    render(app.P_OVERVIEW, {key})
    check(f"quick action {key} -> {target.strip()}", st.session_state["nav"] == target)

# 3. single review: empty warning, then analysis
st.session_state["single_text_w"] = "   "; LOG["warning"].clear()
render(app.P_SINGLE, {"single_go"})
check("empty single review warns", any("enter a review" in w for w in LOG["warning"]))
render(app.P_SINGLE, {"single_example"})
check("example button fills text", st.session_state["single_text_w"] == E.example_review())
render(app.P_SINGLE, {"single_go"})
check("single analysis stored", st.session_state["latest_single"] is not None and len(st.session_state["history"]) == 1)
render(app.P_SINGLE, {"single_clear"})
check("clear button empties text and result", st.session_state["single_text_w"] == "" and st.session_state["latest_single"] is None)
st.session_state["single_text_w"] = "Great sturdy pan, heats evenly and cleans easily."; render(app.P_SINGLE, {"single_go"})

# 4. batch CSV
class Up:
    def __init__(s, name, data): s.name, s.data = name, data
    def getvalue(s): return s.data
df = pd.read_csv("fake_reviews_dataset.csv").sample(300, random_state=5)
UPLOAD["obj"] = Up("sample.csv", df.to_csv(index=False).encode())
st.session_state["batch_max"] = 300
render(app.P_BATCH, {"batch_go"})
check("batch results stored", st.session_state["batch_df"] is not None and len(st.session_state["batch_df"]) == 300)
UPLOAD["obj"] = Up("bad.csv", b"")
render(app.P_BATCH); check("empty CSV handled", any("empty" in e.lower() for e in LOG["error"]))
UPLOAD["obj"] = Up("nocol.csv", b"a,b\n,\n,\n"); st.session_state["batch_col"] = "a"
LOG["error"].clear(); render(app.P_BATCH, {"batch_go"})
check("CSV with no text in column errors cleanly", any("no non-empty text" in e for e in LOG["error"]))
UPLOAD["obj"] = None; render(app.P_BATCH)

# 5. URL analyzer (fake HTTP client)
class R:
    status_code = 200; text = ""
    def json(s): return {"reviews": [{"comment": "Really great value and quick delivery for the price"}, {"comment": "Item broke after a week of normal use"}]}
E._http_client = lambda: types.SimpleNamespace(get=lambda *a, **k: R())
st.session_state["url_input"] = "https://dummyjson.com/products/1"
render(app.P_URL, {"url_go"})
check("URL results stored", st.session_state["url_df"] is not None and len(st.session_state["url_df"]) == 2)
E._http_client = lambda: None; LOG["error"].clear()
st.session_state["url_input"] = "https://www.flipkart.com/x/p/y"
render(app.P_URL, {"url_go"}); check("URL failure shows a clear error", len(LOG["error"]) >= 1, LOG["error"][0][:60] if LOG["error"] else "")
st.session_state["url_input"] = ""; LOG["warning"].clear(); render(app.P_URL, {"url_go"})
check("empty URL warns", any("paste a product URL" in w for w in LOG["warning"]))

# 6. every page with data
for p in PAGES: render(p)
check("12 pages render with data", True, f"{len(st.session_state['history'])} records, {LOG['pyplot']} figures")

# 7. filters / history behaviours
st.session_state.update(hi_pred="FAKE", hi_sort="Fake probability (high)", hi_q="a"); render(app.P_HISTORY)
st.session_state.update(hi_pred="All", hi_sort="Newest first", hi_q="zzzzqqq"); render(app.P_HISTORY)
st.session_state.update(ov_status="FAKE", ov_query="", an_sent="Positive"); render(app.P_OVERVIEW); render(app.P_ANALYTICS)
st.session_state.update(ov_status="All", an_sent="All", pa_pick=["Repeated Words"]); render(app.P_PATTERNS)
st.session_state["ov_sources"] = []; render(app.P_OVERVIEW)   # no sources selected -> empty filter result
check("no-match filters render", True)

# 8. exports
LOG["downloads"].clear(); render(app.P_EXPORT)
labels = [d[0] for d in LOG["downloads"]]
check("all five exports produce data", len(labels) == 5, str(labels))

# 9. model performance + hold-out evaluation
render(app.P_PERF, {"perf_run"})
ev = st.session_state["holdout"]
check("hold-out evaluation ran", ev is not None and abs(ev["accuracy"] - 86.09) < 0.01)
render(app.P_PERF)

# 10. clear history with confirmation
n = len(st.session_state["history"])
render(app.P_HISTORY, {"hi_clear"})
check("clear asks for confirmation first", st.session_state["confirm_clear"] and len(st.session_state["history"]) == n)
render(app.P_HISTORY, {"hi_clear_no"})
check("cancel keeps history", not st.session_state["confirm_clear"] and len(st.session_state["history"]) == n)
render(app.P_HISTORY, {"hi_clear"}); render(app.P_HISTORY, {"hi_clear_yes"})
check("confirmed clear empties everything", len(st.session_state["history"]) == 0 and st.session_state["batch_df"] is None)
for p in PAGES: render(p)
check("12 pages render after clearing", True)

print("\nFAILURES:", fails); sys.exit(1 if fails else 0)
