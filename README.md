# MediSynQ: Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) medical chatbot with:
- **Streamlit UI** for chat
- **PySpur** for tool orchestration
- **BioBERT** for clinical language embeddings
- **LLaMA/DeepSeek** (via Groq API) for LLM answers
- **ChromaDB** for vector storage and retrieval
- **SerpAPI** for trusted medical web search

---

## Project Structure
```
medisynq-chatbot/
├── app.py           # Streamlit UI (main entrypoint)
├── rag_tools.py     # All tool logic, LLM completions, and web search (PySpur)
├── embedding.py     # BioBERT embedding utilities
├── vector_store.py  # ChromaDB interface
├── ingest_data.py   # Data ingestion script (CSV → ChromaDB)
├── requirements.txt # Python dependencies
├── .env             # (Not tracked) API keys for Groq and SerpAPI
├── .gitignore       # Git ignore rules
└── README.md        # This file
```

---

## Features
- **Conversational medical chatbot** with context-aware answers
- **Retrieval-Augmented Generation (RAG):** pulls from discharge notes and clinical trials
- **Trusted web search** for symptom causes (Mayo Clinic, WebMD, NIH, etc)
- **Customizable LLM model selection** (Groq API)
- **Embeddings via BioBERT** and HuggingFace transformers
- **Fast vector search with ChromaDB**
- **Progress bars** for data ingestion

---

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/atharv2802/medisynq-chatbot.git
cd medisynq-chatbot
```

### 2. Install Python Dependencies
```sh
pip install -r requirements.txt

> ⚠️ **Important (ONNXRuntime Fix for NumPy Compatibility)**  
> If you're using **Python 3.11.5** and encounter an error related to `onnxruntime` or `numpy`, run:
> pip uninstall onnxruntime
> pip install onnxruntime==1.15.1
> ```

```

### 3. Create a `.env` File
Create a `.env` file in the project root with:
```
GROQ_API_KEY=your-groq-api-key-here
SERPAPI_KEY=your-serpapi-key-here
```

### 4. Prepare Data Files
- Place your CSVs in a `Data/` directory:
    - `Data/filtered_notes.csv`
    - `Data/filtered_trials.csv`

---

## Data Ingestion (One-Time)
This step embeds your CSV data and builds the vector store.
```sh
python ingest_data.py
```
You will see progress bars for embedding generation.

---

## Running the Chatbot
Start the Streamlit app:
```sh
streamlit run app.py
```
- Open the provided local URL in your browser.
- Ask medical questions, toggle RAG, and select LLM model in the sidebar.

---

## Usage Notes
- **API keys**: Required for Groq (LLM) and SerpAPI (web search).
- **Data ingestion**: Only needed when your data changes.
- **.env and data files** are not tracked by git (see `.gitignore`).
- **All tool logic** is in `rag_tools.py` for easy extension.

---

## License
MIT License

---

For questions or contributions, open an issue or PR on GitHub.
