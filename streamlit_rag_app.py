import streamlit as st
from openai import OpenAI
import numpy as np
import faiss
import csv
import re

# ==========================
# App Config
# ==========================
st.set_page_config(
    page_title="MindEase AI",
    page_icon=None,
    layout="wide",
)

CSV_FILE = "youtube_transcripts.csv"

# ==========================
# API Key and Client in Session State
# ==========================
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if "client" not in st.session_state:
    st.session_state.client = None

# ==========================
# Ask for API key (only until set)
# ==========================
if not st.session_state.api_key_set:
    with st.sidebar:
        st.markdown("## Enter OpenAI API Key")
        api_key_raw = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
        )

        if api_key_raw:
            # Strip non-ASCII characters (to avoid Unicode header issues)
            clean_key = api_key_raw.encode("ascii", "ignore").decode("ascii").strip()
            try:
                client = OpenAI(api_key=clean_key)
                # Simple test call to verify key (very cheap)
                _ = client.models.list()
                st.session_state.client = client
                st.session_state.api_key_set = True
                st.success("API key saved. Reloading app...")
                st.rerun()
            except Exception as e:
                st.error(f"Invalid API key or connection error: {e}")

# If client is still not set, stop the app here
if st.session_state.client is None:
    st.title("MindEase AI")
    st.info("Please enter a valid OpenAI API key in the sidebar to start.")
    st.stop()

client = st.session_state.client  

# ==========================
# Main Session State
# ==========================
for key, default in {
    "page": "home",
    "index": None,
    "chunks": [],
    "videos": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
CHUNK_SIZE = 30 
# ==========================
# Sidebar – Settings and Navigation
# ==========================
with st.sidebar:
    st.markdown("## Settings")
    TOP_K = st.slider("Number of Top Segments", 1, 5, 3)
    

    st.markdown("---")
    # Navigation buttons
    if st.button("Home"):
        st.session_state.page = "home"
    if st.button("Video Insight Chat"):
        st.session_state.page = "chat"
    st.markdown("---")
    st.markdown("### Videos")
    for v in st.session_state.videos:
        if st.button(v["title"], key=f"nav_{v['video_id']}"):
            st.session_state.page = v["video_id"]

# ==========================
# Helpers
# ==========================
def load_videos():
    videos = []
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append({
                "video_id": row["video_id"],
                "title": row["title"],
                "transcript": row["transcript"],
            })
    return videos


def parse_transcript(raw: str):
    """
    Parses transcripts of the form:

    0:00
    [Music]
    0:14
    hi everyone do you live in a city or
    0:17
    town do you commute to school or work

    or: "0:00 some text" on one line.

    It returns a list of {"start": seconds, "text": "..."} entries.
    """
    lines = raw.splitlines()
    parsed = []
    pattern = re.compile(r"^(\d+):(\d+)\s*(.*)$")
    last_time = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            m, s, text = match.groups()
            last_time = int(m) * 60 + int(s)
            # If there is text on the same line and it is not like [Music]
            if text and not text.startswith("["):
                parsed.append({"start": last_time, "text": text})
        else:
            # Continuation line for the last timestamp
            if last_time is not None and not line.startswith("["):
                parsed.append({"start": last_time, "text": line})

    return parsed


def build_chunks(parsed, size):
    """
    Groups parsed lines into time-based chunks of at most `size` seconds.
    Each chunk has keys: start, end, text.
    """
    if not parsed:
        return []

    chunks = []
    current = []
    start = parsed[0]["start"]

    for p in parsed:
        if p["start"] - start <= size:
            current.append(p)
        else:
            chunks.append({
                "start": start,
                "end": current[-1]["start"],
                "text": " ".join(x["text"] for x in current),
            })
            current = [p]
            start = p["start"]

    if current:
        chunks.append({
            "start": start,
            "end": current[-1]["start"],
            "text": " ".join(x["text"] for x in current),
        })

    return chunks


def embed(texts):
    try:
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return np.array([r.embedding for r in res.data]).astype("float32")
    except Exception as e:
        st.error(f"Error while creating embeddings: {e}")
        st.stop()


def auto_index():
    """
    Loads all videos from CSV, parses and chunks transcripts,
    builds a single FAISS index over all chunks.
    """
    videos = load_videos()
    st.session_state.videos = videos

    all_chunks = []
    texts = []

    for v in videos:
        parsed = parse_transcript(v["transcript"])
        if not parsed:
            continue

        chunks = build_chunks(parsed, CHUNK_SIZE)
        for c in chunks:
            c["video_id"] = v["video_id"]
            c["title"] = v["title"]
            all_chunks.append(c)
            texts.append(c["text"])

    if not texts:
        st.error("No transcript text available to index.")
        st.stop()

    embeddings = embed(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    st.session_state.index = index
    st.session_state.chunks = all_chunks


def rag(question, video_id=None):
    """
    RAG over all chunks.
    If video_id is provided, we filter hits to that video after search.
    """
    if st.session_state.index is None or not st.session_state.chunks:
        st.error("Index not built yet.")
        st.stop()

    q_emb = embed([question])
    distances, idx = st.session_state.index.search(q_emb, TOP_K)

    # Initial hits
    hits = [st.session_state.chunks[i] for i in idx[0]]

    # If restricted to a specific video, filter by video_id
    if video_id is not None:
        hits = [h for h in hits if h["video_id"] == video_id]

        # If filtering removed everything, just use the top chunk from that video
        if not hits:
            hits = [
                c for c in st.session_state.chunks
                if c["video_id"] == video_id
            ][:TOP_K]

    context = "\n".join(h["text"] for h in hits)

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    try:
        ans = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        answer_text = ans.output_text
    except Exception as e:
        st.error(f"Error while generating answer: {e}")
        st.stop()

    return answer_text, hits


def summarize(video):
    prompt = (
        f"Summarize the video titled '{video['title']}' "
        f"in 5 bullet points.\n\n"
        f"{video['transcript'][:4000]}"
    )
    try:
        res = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return res.output_text
    except Exception as e:
        st.error(f"Error while generating summary: {e}")
        st.stop()

# ==========================
# AUTO-INDEX ON STARTUP (after client is ready)
# ==========================
if st.session_state.index is None:
    auto_index()

# ==========================
# UI – HOME
# ==========================
if st.session_state.page == "home":
    st.title("MindEase AI")

    st.markdown("""
### What is this app?

This app lets you chat with YouTube videos using AI.

Instead of watching full videos, you can:

- Ask questions in natural language.
- Jump directly to relevant moments.
- See exact timestamps and spoken text.
- Get clean summaries for each video.

### How it works

1. Video IDS, title and transcripts are stored in a CSV file.
2. Transcripts are parsed into time-based chunks (for example, 30 seconds).
3. AI builds embeddings for all chunks and indexes them.
4. Your question retrieves the most relevant chunks and timestamps.

### How to use

- Use "Chat with all videos" in the sidebar to ask questions across all videos.
- Click a video title in the sidebar to open its dedicated page.
- On each video page, use:
  - "Generate Summary" to get a quick overview.
  - The question box to ask about that specific video.
- Click on timestamps to open the YouTube video at the right moment.
""")

    st.info("Select a video from the sidebar or use 'Chat with all videos' to start.")

# ==========================
# UI – GLOBAL CHAT
# ==========================
if st.session_state.page == "chat":
    st.title("Video Insight Chat")

    q = st.text_area("Ask a question about any of the videos")
    if st.button("Ask"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            answer, refs = rag(q)
            st.markdown("### Answer")
            st.markdown(answer)

            st.markdown("### Top Segments")
            for r in refs:
                url = f"https://www.youtube.com/watch?v={r['video_id']}&t={r['start']}s"
                st.markdown(f"**{r['title']}** — [{r['start']}s–{r['end']}s]({url})")
                st.caption(r["text"])

# ==========================
# UI – VIDEO PAGES
# ==========================
for v in st.session_state.videos:
    if st.session_state.page == v["video_id"]:
        st.title(v["title"])
        st.video(f"https://www.youtube.com/watch?v={v['video_id']}")

        if st.button("Generate Summary", key=f"summary_{v['video_id']}"):
            with st.spinner("Summarizing..."):
                summary_text = summarize(v)
            st.markdown("### Summary")
            st.markdown(summary_text)

        st.markdown("### Ask about this video")
        q = st.text_area("Your question", key=f"q_{v['video_id']}")
        if st.button("Ask", key=f"a_{v['video_id']}"):
            if not q.strip():
                st.warning("Please enter a question.")
            else:
                answer, refs = rag(q, video_id=v["video_id"])
                st.markdown("### Answer")
                st.markdown(answer)

                st.markdown("### Top Segments in this Video")
                for r in refs:
                    if r["video_id"] != v["video_id"]:
                        continue
                    url = f"https://www.youtube.com/watch?v={r['video_id']}&t={r['start']}s"
                    st.markdown(f"[{r['start']}s–{r['end']}s]({url})")
                    st.caption(r["text"])
