#Mindease-AI-MindEase AI: A Streamlit app for question answering over YouTube video transcripts using OpenAI embeddings and FAISS, with timestamped video insights.

##OverviewThis application, MindEase AI, lets you chat with the content of YouTube videos using AI. Instead of watching full videos, you can get quick answers and jump directly to relevant moments.

##FeaturesThe application provides the following features:

* Ask questions in natural language about the video content.
* Jump directly to relevant moments in the video using timestamps.
* View exact timestamps and spoken text for context.
* Generate clean summaries for each video.

##How It WorksThe system operates using a Retrieval-Augmented Generation (RAG) process:

1. Video IDs, titles, and transcripts are stored in a `youtube_transcripts.csv` file.
2. Transcripts are parsed into time-based chunks (e.g., 30 seconds).
3. An AI model (OpenAI's `text-embedding-3-small`) builds embeddings for all chunks, and these are indexed using FAISS.
4. Your question is embedded and used to retrieve the most relevant chunks and timestamps from the index.
5. A language model (OpenAI's `gpt-4o-mini`) is used to generate an answer based *only* on the retrieved context.

##Setup and Installation###PrerequisitesYou will need:

* Python 3.x
* An [OpenAI API Key](https://platform.openai.com/account/api-keys).
* A CSV file named `youtube_transcripts.csv` in the root directory containing video metadata (`video_id`, `title`, `transcript`).

###Installation1. Clone the repository (or save the `streamlit_rag_app.py` file).
2. Install the required Python packages:
```bash
pip install streamlit openai numpy faiss-cpu

```



###Running the App1. Run the application:
```bash
streamlit run streamlit_rag_app.py

```



##Usage1. **API Key Setup**: The application will first prompt you to enter a valid OpenAI API key in the sidebar.
2. **Navigation**: Select a video from the list in the sidebar or choose **Video Insight Chat**.
3. **Global Chat**: Use the **Video Insight Chat** option to ask questions across all indexed videos.
4. **Video-Specific Actions**:
* Click a video title in the sidebar to open its dedicated page.
* On the video page, use **Generate Summary** to get a quick overview.
* Use the question box on the video page to ask about that specific video's content.
* Click on the generated timestamps/links in the output to open the YouTube video at the correct moment.
