# Self-RAG Chatbot

A Streamlit chatbot that uses **Self-RAG** (Self-Reflective Retrieval-Augmented Generation)
to answer questions about your uploaded PDF files.

## Stack
| Component | Library |
|---|---|
| UI | Streamlit |
| LLM + Graders | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, no API key) |
| Vector store | Chroma (in-memory) |
| Orchestration | LangGraph |

## Self-RAG Flow
```
Upload PDFs → chunk & embed → Chroma vectorstore
                                      ↓
User question → retrieve top-4 chunks
                      ↓
              Grade each chunk (relevant / irrelevant)
                      ↓
           [all irrelevant] → rewrite query → retrieve again
           [some relevant]  → generate answer with Groq
                                      ↓
                        Check hallucination (grounded?)
                              ↓             ↓
                           not grounded   grounded
                              ↓             ↓
                          retry gen    check answer quality
                                          ↓         ↓
                                       useful    not useful
                                          ↓         ↓
                                         END    rewrite query
```

## Setup

1. **Clone / download** this folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your Groq API key**:
   ```bash
   cp .env.example .env
   # Edit .env and paste your key
   ```
   Get a free key at https://console.groq.com

5. **Run the app**:
   ```bash
   streamlit run app.py
   ```

6. Open http://localhost:8501 in your browser.

## Project Structure
```
self_rag_project/
├── app.py            # Streamlit UI
├── vectorstore.py    # PDF ingestion + Chroma + HuggingFace embeddings
├── graders.py        # 4 Self-RAG graders + RAG chain (Groq)
├── graph.py          # LangGraph workflow
├── requirements.txt
├── .env.example
└── README.md
```

## Notes
- The vectorstore is **in-memory** — it resets when you clear/restart.
- The HuggingFace embedding model downloads (~90 MB) on first run.
- Groq's free tier is fast and generous for this use case.
