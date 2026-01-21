import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import chromadb
import hashlib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is not set in Streamlit Secrets")
    st.stop()
# ------------------------------------------------
# CHROMA-COMPATIBLE HUGGINGFACE EMBEDDING WRAPPER
# ------------------------------------------------
class ChromaHuggingFaceEmbedding:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = model
        self.embedder = HuggingFaceEmbeddings(model_name=model)

    def __call__(self, input):
        # Chroma expects: List[str] -> List[List[float]]
        return self.embedder.embed_documents(input)

    def name(self):
        return f"huggingface-{self.model}"

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
st.set_page_config("AI Review & Data Analyst", layout="wide")

st.title("🧠 AI Review & Data Analyst (Flexible + Cloud)")

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded = st.sidebar.file_uploader("Upload ANY CSV file", type=["csv"])

if not uploaded:
    st.info("⬅️ Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(uploaded)

# ------------------------------------------------
# API KEY INPUT




# Initialize LLM after API key is set
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=st.secrets["GROQ_API_KEY"],
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

product_col = st.sidebar.selectbox("Product", [None] + list(df.columns),
    index=(list(df.columns).index(product_col) + 1) if product_col in df.columns else 0)

review_col = st.sidebar.selectbox("Review", [None] + list(df.columns),
    index=(list(df.columns).index(review_col) + 1) if review_col in df.columns else 0)

rating_col = st.sidebar.selectbox("Rating", [None] + list(df.columns),
    index=(list(df.columns).index(rating_col) + 1) if rating_col in df.columns else 0)

date_col = st.sidebar.selectbox("Date", [None] + list(df.columns),
    index=(list(df.columns).index(date_col) + 1) if date_col in df.columns else 0)

# ------------------------------------------------
# DATE CONVERSION
# ------------------------------------------------
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ------------------------------------------------
# FAKE REVIEW DETECTION
# ------------------------------------------------
if review_col and "is_suspicious" not in df.columns:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    weak_label = (
        df[review_col].astype(str)
        .str.lower()
        .str.count(r"\b(best|excellent|awesome|fake|scam|worst)\b") > 2
    )

    model.fit(df[review_col].astype(str), weak_label.astype(int))
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

c1.plotly_chart(gauge, width="stretch")
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
    st.dataframe(summary, width="stretch")

# ------------------------------------------------
# BUILD RAG INDEX (100% SAFE)
# ------------------------------------------------
text_cols = df.select_dtypes(include="object").columns.tolist()

dataset_hash = hashlib.md5(uploaded.getvalue()).hexdigest()[:8]
collection_name = f"dataset_rag_{dataset_hash}"

client = chromadb.Client()
embedding_fn = ChromaHuggingFaceEmbedding()

if "rag_collection" not in st.session_state:
    with st.spinner("📚 Building vector index..."):
        try:
            collection = client.get_collection(name=collection_name)

        except Exception:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_fn
            )

            docs, meta = [], []
            for idx, row in df.iterrows():
                docs.append(" | ".join(
                    f"{c}: {row[c]}" for c in text_cols if pd.notna(row[c])
                ))
                if product_col and pd.notna(row[product_col]):
                    meta.append({product_col: str(row[product_col])})
                else:
                    meta.append({"row_index": str(idx)})

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
    answer = llm.invoke(context + "\n\n" + question)
    st.markdown(answer)

# ------------------------------------------------
# RAW DATA
# ------------------------------------------------
with st.expander("📁 View Raw Data"):
    st.dataframe(df)



