"""
Movie Review Mood Checker - Streamlit app
NVIDIA DLI mini project: BERT (Transformer NLP) + data analysis (Accelerated Data Science)
"""
import re
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# cuDF (GPU pandas) is used automatically if this machine has an NVIDIA GPU + RAPIDS.
# Free Hugging Face Spaces run on CPU, so the app falls back to normal pandas.
try:
    import cudf
    DF_ENGINE = "cuDF (GPU)"
except Exception:
    cudf = None
    DF_ENGINE = "pandas (CPU)"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256

STOP_WORDS = set("""the a and of to is in it i this that was as for with movie film but on are you not his have he be one
its at all by an they from who so like her or just about has if out what some there can more when very she would up my
even only no time had which their were see than been do me will we into how also story much get them other could
because then any these him did does most well being made make really""".split())

st.set_page_config(page_title="Movie Review Mood Checker", page_icon="🎬", layout="wide")


# ---------- Model (Course 1: Transformer NLP) ----------
@st.cache_resource(show_spinner="Loading the BERT model (first time takes ~30 s)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()  # FP16 = faster on GPU
    return tokenizer, model


def predict(texts, tokenizer, model, batch_size=32, progress=None):
    """Text -> tokens -> BERT -> logits -> softmax. Returns P(positive) for each text."""
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits.float()
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        if progress is not None:
            progress.progress(min((i + batch_size) / len(texts), 1.0), text=f"BERT is reading... {min(i + batch_size, len(texts)):,}/{len(texts):,}")
    return np.concatenate(probs)


# ---------- Data (Course 2: Accelerated Data Science) ----------
@st.cache_data(show_spinner="Downloading IMDB reviews...")
def load_imdb_sample(n):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=42).select(range(n))
    return ds.to_pandas()  # columns: text, label (0 = negative, 1 = positive)


def analyse(df):
    """Summary stats per mood. Runs on cuDF if available, otherwise pandas - same code."""
    frame = cudf.from_pandas(df[["text", "mood", "confidence"]]) if cudf is not None else df[["text", "mood", "confidence"]].copy()
    start = time.time()
    frame["words"] = frame["text"].str.count(" ") + 1
    summary = frame.groupby("mood").agg({"text": "count", "words": "mean", "confidence": "mean"})
    summary = summary.rename(columns={"text": "reviews", "words": "avg_words", "confidence": "avg_confidence"})
    secs = time.time() - start
    if cudf is not None:
        summary = summary.to_pandas()
    return summary.sort_index(), secs


def top_words(texts, n=10):
    words = (pd.Series(texts).str.lower()
             .str.replace("<br />", " ", regex=False)
             .str.replace(r"[^a-z ]", " ", regex=True)
             .str.split().explode())
    words = words[(words.str.len() > 2) & (~words.isin(STOP_WORDS))]
    return words.value_counts().head(n)


def label_of(p):
    return ("Positive", "😊") if p > 0.5 else ("Negative", "😞")


# ---------- Page ----------
tokenizer, model = load_model()

st.title("🎬 Movie Review Mood Checker")
st.caption("A BERT-family model reads movie reviews and decides if they are Positive or Negative. "
           "NVIDIA DLI mini project: Transformer NLP + Accelerated Data Science.")

with st.sidebar:
    st.header("⚙️ Running on")
    st.write(f"**Model device:** {'GPU (' + torch.cuda.get_device_name(0) + ')' if DEVICE == 'cuda' else 'CPU'}")
    st.write(f"**Data engine:** {DF_ENGINE}")
    st.write(f"**Model:** `{MODEL_NAME}`")
    st.write(f"**Max tokens per review:** {MAX_LENGTH}")
    st.info("Free Hugging Face Spaces use a CPU. The GPU speed tests (cuDF vs pandas, FP16 batching) are in the Colab notebook.")

tab1, tab2, tab3 = st.tabs(["✍️ Single review", "📊 Many reviews", "🧠 How it works"])

# ----- Tab 1: one review -----
with tab1:
    text = st.text_area("Type or paste a movie review:", height=140,
                        value="The plot was predictable, but the acting saved the movie and I left the cinema smiling.")
    if st.button("Check mood", type="primary") and text.strip():
        p = float(predict([text], tokenizer, model)[0])
        mood, emoji = label_of(p)
        confidence = max(p, 1 - p)

        c1, c2 = st.columns(2)
        c1.metric("Mood", f"{emoji} {mood}")
        c2.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(confidence)
        st.bar_chart(pd.DataFrame({"probability": [1 - p, p]}, index=["Negative", "Positive"]))

        with st.expander("🔍 See how BERT reads it (tokens and IDs)"):
            ids = tokenizer(text, truncation=True, max_length=MAX_LENGTH)["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(ids)
            st.write(f"{len(tokens)} tokens. `##` means the piece joins onto the previous one; [CLS] and [SEP] are special start/end tokens.")
            st.dataframe(pd.DataFrame({"token": tokens, "id": ids}).T, use_container_width=True)

# ----- Tab 2: many reviews -----
with tab2:
    source = st.radio("Where do the reviews come from?", ["Sample of real IMDB reviews", "Upload my own CSV"], horizontal=True)

    df = None
    if source == "Sample of real IMDB reviews":
        n = st.slider("How many reviews?", 50, 2000, 300, step=50,
                      help="On a free CPU Space, about 10-40 reviews per second.")
        if st.button("Load and analyse", type="primary"):
            df = load_imdb_sample(n).copy()
    else:
        st.write("CSV needs a column called **text**. Optional column **label** (0 = negative, 1 = positive) to check accuracy.")
        file = st.file_uploader("Upload CSV", type="csv")
        if file is not None and st.button("Analyse", type="primary"):
            df = pd.read_csv(file)
            if "text" not in df.columns:
                st.error("No column called 'text' found.")
                df = None
            else:
                df["text"] = df["text"].astype(str)

    if df is not None:
        bar = st.progress(0.0, text="BERT is reading...")
        start = time.time()
        probs = predict(df["text"].tolist(), tokenizer, model, progress=bar)
        secs = time.time() - start
        bar.empty()

        df["positive_prob"] = probs.round(4)
        df["mood"] = np.where(probs > 0.5, "Positive", "Negative")
        df["confidence"] = np.maximum(probs, 1 - probs).round(4)

        summary, analyse_secs = analyse(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reviews read", f"{len(df):,}")
        c2.metric("Speed", f"{len(df) / secs:.0f} reviews/s")
        c3.metric("Positive", f"{(df['mood'] == 'Positive').mean() * 100:.0f}%")
        if "label" in df.columns:
            accuracy = ((df["mood"] == "Positive").astype(int) == df["label"]).mean()
            c4.metric("Accuracy vs real labels", f"{accuracy * 100:.1f}%")
        else:
            c4.metric("Avg confidence", f"{df['confidence'].mean() * 100:.1f}%")

        left, right = st.columns(2)
        with left:
            st.subheader("Reviews by mood")
            st.bar_chart(summary["reviews"])
        with right:
            st.subheader("Summary table")
            st.dataframe(summary.round(2), use_container_width=True)
            st.caption(f"Analysed with {DF_ENGINE} in {analyse_secs * 1000:.1f} ms")

        w1, w2 = st.columns(2)
        with w1:
            st.subheader("😊 Top words in Positive reviews")
            st.bar_chart(top_words(df.loc[df["mood"] == "Positive", "text"]), color="#76b900")
        with w2:
            st.subheader("😞 Top words in Negative reviews")
            st.bar_chart(top_words(df.loc[df["mood"] == "Negative", "text"]), color="#d9534f")

        st.subheader("All results")
        st.dataframe(df[["mood", "confidence", "text"] + (["label"] if "label" in df.columns else [])],
                     use_container_width=True, height=300)
        st.download_button("⬇️ Download results as CSV", df.to_csv(index=False), "mood_results.csv", "text/csv")

# ----- Tab 3: explanation -----
with tab3:
    st.markdown("""
### What happens when you click the button
1. **Tokenizer** chops your review into subword tokens (WordPiece) and turns them into ID numbers.
2. **DistilBERT** (a smaller, faster BERT) uses **self-attention** to read every word in the context of every other word.
3. A small **classifier head** outputs two raw scores (**logits**): Negative and Positive.
4. **Softmax** turns the scores into percentages that add up to 100%.

### Which NVIDIA course each part comes from
| Part of the app | Course |
| --- | --- |
| Tokenizer, BERT, softmax, batching, FP16 on GPU | Building Transformer-Based NLP Applications |
| Grouping, string analysis, cuDF vs pandas | Fundamentals of Accelerated Data Science |

### Limits
- Reviews longer than 256 tokens are cut off.
- The model was trained on short movie sentences (SST-2), so sarcasm and mixed reviews can fool it.
""")
