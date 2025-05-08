# MediSynQ: Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) medical chatbot with:
- **Streamlit UI** for chat
- **PySpur** for tool orchestration
- **BioBERT** for clinical language embeddings
- **LLaMA/DeepSeek** for LLM answers
- **ChromaDB** for vector storage and retrieval

## Project Structure
```
medisynq-chatbot/
├── app.py                # Streamlit UI
├── rag_tools.py          # All tool logic, LLM completions, and web search (PySpur)
├── embedding.py          # BioBERT embedding utilities
├── vector_store.py       # ChromaDB interface
├── ingest_data.py        # Data ingestion script
├── requirements.txt
└── README.md
```

## Setup
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Set up API keys and models as needed (see TODOs in code).
3. Run the app:
   ```sh
   streamlit run app.py
   ```

## TODO
- Integrate BioBERT for embeddings
- Connect ChromaDB for retrieval
- Integrate LLaMA/DeepSeek for answer generation
- Implement SerpAPI web search
- Fill in tool logic in `rag_tools.py`

---
For more details, see comments in each file.
