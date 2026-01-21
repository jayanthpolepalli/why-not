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
st.caption("Upload a file → Detect fraud → Ask questions")

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
# FAKE / FRAUD REVIEW DETECTION
# ------------------------------------------------
if review_col and "is_suspicious" not in df.columns:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    weak_labels = (
        df[review_col].astype(str)
        .str.lower()
        .str.count(r"\b(fake|scam|worst|refund|fraud|cheat)\b") > 1
    )

    model.fit(df[review_col].astype(str), weak_labels.astype(int))
    df["is_suspicious"] = model.predict(df[review_col].astype(str))

# ------------------------------------------------
# METRICS
# ------------------------------------------------
total = len(df)
suspicious = int(df["is_suspicious"].sum())
integrity = int((1 - suspicious / total) * 100) if total else 100

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", total)
c2.metric("Suspicious Records", suspicious)
c3.metric("Integrity Score", f"{integrity}%")

# ------------------------------------------------
# BUILD RAG INDEX
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
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )

        docs, meta = [], []
        for i, row in df.iterrows():
            docs.append(" | ".join(
                f"{c}: {row[c]}" for c in text_cols if pd.notna(row[c])
            ))
            meta.append({
                "row": i,
                "suspicious": bool(row.get("is_suspicious", False))
            })

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

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------
# CHATBOT INPUT
# ------------------------------------------------
question = st.chat_input("Ask about fraud, risks, trends, or specific products")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant rows
    results = st.session_state.rag_collection.query(
        query_texts=[question],
        n_results=5
    )

    context = "\n".join(results["documents"][0])

    # System prompt (VERY IMPORTANT)
    system_prompt = f"""
You are a fraud detection assistant.

Dataset summary:
- Total records: {total}
- Suspicious records: {suspicious}
- Integrity score: {integrity}%

Rules:
- Use ONLY the dataset context.
- Explain fraud risks clearly.
- Give insights, patterns, and recommendations.
- If user asks numbers, calculate from context.
- If unsure, say so honestly.
"""

    final_prompt = f"""
{system_prompt}

Context from dataset:
{context}

User question:
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
# RAW DATA
# ------------------------------------------------
with st.expander("📁 View Raw Data"):
    st.dataframe(df, use_container_width=True)


