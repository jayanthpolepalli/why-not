# 🧠 AI Review & Data Analyst (RAG + Fraud Detection)

An end-to-end **Streamlit-based AI dashboard** that performs:

* Fake / suspicious review detection
* Product-level risk analysis
* Retrieval-Augmented Generation (RAG) over tabular data
* Natural language Q&A using **Groq LLM (LLaMA 3.3)**

This tool is designed for **review integrity analysis**, **fraud detection**, and **AI-assisted data exploration**.

---

## 🚀 Features

### 📂 Data Ingestion

* Upload any CSV file
* Automatic column detection (product, review, rating, date)
* Manual column override via sidebar

### 🕵️ Fake Review Detection

* Weakly supervised ML pipeline
* TF-IDF + RandomForest classifier
* Auto-generates `is_suspicious` column

### 📊 Analytics Dashboard

* Integrity score (gauge)
* Total vs suspicious records
* Product-level risk percentage table

### 🧠 AI Chat (RAG)

* Builds a vector index using **ChromaDB**
* SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
* Ask natural language questions about your dataset
* Responses generated via **Groq LLaMA 3.3 (70B)**

### 🔐 Safe Vector Storage

* Dataset-based hashing to avoid collisions
* Session-persistent vector collections

---

## 🧰 Tech Stack

| Layer         | Technology           |
| ------------- | -------------------- |
| UI            | Streamlit            |
| ML            | Scikit-learn         |
| Vector DB     | ChromaDB             |
| Embeddings    | SentenceTransformers |
| LLM           | Groq (LLaMA 3.3 70B) |
| Visualization | Plotly               |

---

## 📦 Installation

```bash
pip install streamlit pandas plotly scikit-learn chromadb sentence-transformers langchain-groq
```

---

## 🔑 Environment Setup

### Streamlit Secrets

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

### 1️⃣ Column Auto-Detection

Uses keyword matching to detect:

* Product name
* Review text
* Rating
* Date

### 2️⃣ Weak Supervision

Reviews are weakly labeled using keyword frequency:

* "best", "excellent", "awesome", "fake", "scam", "worst"

These labels train a RandomForest classifier to detect suspicious reviews.

### 3️⃣ Risk Metrics

* **Integrity Score** = % of non-suspicious reviews
* Product-level risk % computed via group-by

### 4️⃣ RAG Pipeline

* Text columns are embedded
* Stored in ChromaDB
* Top-K relevant rows retrieved
* Context passed to Groq LLM for reasoning

---

## 💬 Example Questions

* "Which products have the highest fraud risk?"
* "Summarize customer sentiment"
* "Are there spikes in suspicious reviews?"
* "Which product has the worst integrity score?"

---

## 📁 Project Structure

```text
app.py
.streamlit/
  └── secrets.toml
```

---

## ⚠️ Limitations

* Weak supervision ≠ ground truth
* Best suited for exploratory analysis
* Not a replacement for human moderation

---

## 🔮 Future Improvements

* Advanced NLP fraud models
* Time-series anomaly detection
* Review sentiment + emotion analysis
* User authentication
* Persistent vector storage (DuckDB / S3)

---

## 👨‍💻 Author

Built for **AI-driven review integrity & data intelligence** use cases.

---

## 📜 License

MIT License
