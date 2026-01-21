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
    page_title="Fraud Detection Chatbot",
    layout="wide"
)

st.title("🤖 Fraud Detection Chatbot")
st.caption("Upload a CSV → Detect fraud → Ask questions")

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
# FRAUD / FAKE REVIEW DETECTION (SAFE)
# ------------------------------------------------
if review_col and "is_suspicious" not in df.columns:

    # Clean review text
    df[review_col] = (
        df[review_col]
        .astype(str)
        .str.strip()
    )

    # Remove empty / invalid rows
    df = df[df[review_col].str.len() > 3]

    if df[review_col].nunique() < 2:
        st.warning("⚠️ Not enough meaningful reviews to train fraud model.")
        df["is_suspicious"] = 0
    else:
        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                min_df=2,
                token_pattern=r"(?u)\b\w+\b"
            )),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ])

        # Weak supervision
        weak_labels = (
            df[review_col]
            .str.lower()
            .str.count(r"\b(fake|scam|fraud|refund|cheat|worst)\b") > 1
        )

        model.fit(df[review_col], weak_labels.astype(int))
        df["is_suspicious"] = model.predict(df[review_col])

# ------------------------------------------------
# METRICS
# ------------------------------------------------
total = len(df)
suspicious = int(df["is_suspicious"].sum()) if "is_suspicious" in df else 0
integrity = int((1 - suspicious / total) * 100) if total else 100

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", total)
c2.metric("Suspicious Records", suspicious)
c3.metric("Integrity Score", f"{integrity}%")

# ------------------------------------------------
# BUILD RAG INDEX (CHROMA)
# ------------------------------------------------
text_cols = df.select_dtypes(include="object").columns.tolist()

dataset_hash = hashlib.md5(uploaded.getvalue()).hexdigest()[:8]
collection_name = f"fraud_rag_{dataset_hash}"

client = chromadb.Client()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

if "rag_collection" not in st.session_state:
    with st.spinner("📚 Indexing dataset for chatbot..."):

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )

        docs, meta = [], []

        for i, row in df.iterrows():
            text = " | ".join(
                f"{c}: {row[c]}"
                for c in text_cols
                if pd.notna(row[c]) and str(row[c]).strip()
            )

            if text.strip():
                docs.append(text)
                meta.append({
                    "row": int(i),
                    "suspicious": bool(row.get("is_suspicious", False))
                })

        if docs:
            collection.add(
                documents=docs,
                ids=[str(i) for i in range(len(docs))],
                metadatas=meta
            )

        st.session_state.rag_collection = collection

# ------------------------------------------------
# CHAT MEMORY
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------
# CHAT INPUT
# ------------------------------------------------
question = st.chat_input("Ask about fraud, risks, trends, or specific products")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    results = st.session_state.rag_collection.query(
        query_texts=[question],
        n_results=5
    )

    context = "\n".join(results["documents"][0]) if results["documents"] else ""

    system_prompt = f"""
You are a fraud detection assistant.

Dataset summary:
- Total records: {total}
- Suspicious records: {suspicious}
- Integrity score: {integrity}%

Rules:
- Use ONLY the dataset context.
- Explain fraud risks clearly.
- Provide patterns & recommendations.
- If unsure, say so honestly.
"""

    final_prompt = f"""
{system_prompt}

Context:
{context}

User Question:
{question}
"""

    answer = llm.invoke(final_prompt)

    with st.chat_message("assistant"):
        st.markdown(answer.content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.content
    })

# ------------------------------------------------
# RAW DATA VIEW
# ------------------------------------------------
with st.expander("📁 View Raw Data"):
    st.dataframe(df, use_container_width=True)
