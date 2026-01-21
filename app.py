import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import chromadb
import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq

# ------------------------------------------------
# API KEY CHECK
# ------------------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("❌ GROQ_API_KEY is missing in Streamlit Secrets")
    st.stop()

# ------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AI Review & Data Analyst",
    layout="wide"
)

st.title("🧠 AI Review & Data Analyst (RAG + Fraud Detection)")

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])

if not uploaded:
    st.info("⬅️ Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(uploaded)

# ------------------------------------------------
# LLM INIT
# ------------------------------------------------
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=st.secrets["GROQ_API_KEY"]
)

# ------------------------------------------------
# COLUMN AUTO-DETECTION
# ------------------------------------------------
def auto_detect_column(keywords):
    for col in df.columns:
        for kw in keywords:
            if kw in col.lower():
                return col
    return None

product_col = auto_detect_column(["product", "item", "name"])
review_col  = auto_detect_column(["review", "comment", "feedback", "text"])
rating_col  = auto_detect_column(["rating", "stars", "score"])
date_col    = auto_detect_column(["date", "time", "created"])

# ------------------------------------------------
# MANUAL COLUMN MAPPING
# ------------------------------------------------
st.sidebar.markdown("### 🔧 Column Mapping")

def col_selector(label, detected):
    return st.sidebar.selectbox(
        label,
        [None] + list(df.columns),
        index=(list(df.columns).index(detected) + 1)
        if detected in df.columns else 0
    )

product_col = col_selector("Product", product_col)
review_col  = col_selector("Review", review_col)
rating_col  = col_selector("Rating", rating_col)
date_col    = col_selector("Date", date_col)

# ------------------------------------------------
# DATE PARSING
# ------------------------------------------------
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ------------------------------------------------
# FAKE REVIEW DETECTION (WEAK SUPERVISION)
# ------------------------------------------------
if review_col and "is_suspicious" not in df.columns:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    weak_labels = (
        df[review_col].astype(str)
        .str.lower()
        .str.count(r"\b(best|excellent|awesome|fake|scam|worst)\b") > 2
    )

    model.fit(df[review_col].astype(str), weak_labels.astype(int))
    df["is_suspicious"] = model.predict(df[review_col].astype(str))

# ------------------------------------------------
# DASHBOARD METRICS
# ------------------------------------------------
total = len(df)
suspicious = int(df["is_suspicious"].sum()) if "is_suspicious" in df.columns else 0
integrity = int((1 - suspicious / total) * 100) if total else 100

c1, c2, c3 = st.columns(3)

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=integrity,
    title={"text": "Integrity Score"},
    gauge={"axis": {"range": [0, 100]}}
))

c1.plotly_chart(gauge, use_container_width=True)
c2.metric("Total Records", total)
c3.metric("Suspicious Records", suspicious)

# ------------------------------------------------
# PRODUCT RISK SUMMARY
# ------------------------------------------------
if product_col and "is_suspicious" in df.columns:
    st.subheader("📦 Product Risk Summary")

    summary = (
        df.groupby(product_col)["is_suspicious"]
        .agg(["count", "sum"])
        .reset_index()
    )

    summary["risk_%"] = (summary["sum"] / summary["count"] * 100).round(1)
    st.dataframe(summary, use_container_width=True)

# ------------------------------------------------
# BUILD RAG INDEX (CHROMA SAFE)
# ------------------------------------------------
text_cols = df.select_dtypes(include="object").columns.tolist()

dataset_hash = hashlib.md5(uploaded.getvalue()).hexdigest()[:8]
collection_name = f"dataset_rag_{dataset_hash}"

client = chromadb.Client()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

if "rag_collection" not in st.session_state:
    with st.spinner("📚 Building vector index..."):
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_fn
            )

            docs, meta = [], []
            for i, row in df.iterrows():
                docs.append(" | ".join(
                    f"{c}: {row[c]}" for c in text_cols if pd.notna(row[c])
                ))
                meta.append(
                    {product_col: str(row[product_col])}
                    if product_col and pd.notna(row[product_col])
                    else {"row": str(i)}
                )

            collection.add(
                documents=docs,
                ids=[str(i) for i in range(len(docs))],
                metadatas=meta
            )

        st.session_state.rag_collection = collection

# ------------------------------------------------
# RAG CHAT
# ------------------------------------------------
st.subheader("🧠 Ask Your Dataset")

question = st.chat_input("Ask anything about your data")

if question:
    results = st.session_state.rag_collection.query(
        query_texts=[question],
        n_results=5
    )

    context = "\n".join(results["documents"][0])
    response = llm.invoke(context + "\n\nQuestion: " + question)

    st.markdown(response.content)

# ------------------------------------------------
# RAW DATA
# ------------------------------------------------
with st.expander("📁 View Raw Data"):
    st.dataframe(df, use_container_width=True)

