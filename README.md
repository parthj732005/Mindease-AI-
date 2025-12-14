# 🧘‍♂️ MindEase AI

**MindEase AI** is a Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to **ask questions about YouTube videos** using AI.
Instead of watching full videos, users can get **concise answers, summaries, and timestamped insights** directly from video transcripts.

---

## ✨ Features

* 🔍 Ask natural language questions about video content
* ⏱️ Jump directly to relevant moments using timestamps
* 🧠 AI-generated answers using retrieved transcript context only
* 📝 Automatic summaries for each video
* 📺 Video-specific Q&A and global multi-video chat
* ⚡ Fast semantic search using FAISS embeddings

---

## 🧠 How It Works (RAG Pipeline)

1. Video IDs, titles, and transcripts are stored in `youtube_transcripts.csv`
2. Transcripts are parsed and split into **30-second time-based chunks**
3. Each chunk is embedded using **OpenAI `text-embedding-3-small`**
4. Embeddings are indexed using **FAISS**
5. User questions retrieve the most relevant chunks
6. **GPT-4o-mini** generates answers using *only* the retrieved context

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Embeddings**: OpenAI `text-embedding-3-small`
* **LLM**: OpenAI `gpt-4o-mini`
* **Vector Store**: FAISS
* **Language**: Python

---

## 📂 Project Structure

```text
MindEase-AI/
│
├── streamlit_rag_app.py
├── youtube_transcripts.csv
├── requirements.txt
└── README.md
```

---

## 🧪 Installation & Setup

It is **strongly recommended** to use a virtual environment.

---

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/MindEase-AI.git
cd MindEase-AI
```

---

### 🔹 Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

---

### 🔹 Step 3: Activate the Virtual Environment

**Windows (PowerShell):**

```bash
venv\Scripts\activate
```

**Windows (CMD):**

```bash
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

You should now see `(venv)` in your terminal.

---

### 🔹 Step 4: Install Dependencies

Using `requirements.txt` (recommended):

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit openai numpy faiss-cpu
```

---

### 🔹 Step 5: Run the App

```bash
streamlit run streamlit_rag_app.py
```

---

## 🔑 OpenAI API Key

* The app will prompt you to enter your **OpenAI API key** in the sidebar
* The key is stored **only in session memory**
* **Do not commit your API key**

---

## 📄 Required CSV Format

The app expects a file named:

```text
youtube_transcripts.csv
```

### Required columns:

| Column Name  | Description                     |
| ------------ | ------------------------------- |
| `video_id`   | YouTube video ID                |
| `title`      | Video title                     |
| `transcript` | Full transcript with timestamps |

---





## 🚀 Future Improvements

* Transcript auto-fetch from YouTube
* Multi-language support
* Intent-aware video filtering

---

## 📜 License

This project is for **educational and research purposes**.

---

## 🙌 Acknowledgements

* OpenAI
* Streamlit
* FAISS
* YouTube Transcript API

---
