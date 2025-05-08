# vector_store.py
# ChromaDB vector store interface
import chromadb

chroma_client = chromadb.Client(
    chromadb.config.Settings(persist_directory="chromadb_store")
)

# --- Discharge Notes Collection ---
def add_discharge_notes(documents, embeddings, metadatas=None):
    collection = chroma_client.get_or_create_collection("discharge_notes")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas or [{} for _ in documents],
        ids=[f"note_{i}" for i in range(len(documents))]
    )

def query_discharge_notes(query_embedding, n_results=3):
    collection = chroma_client.get_or_create_collection("discharge_notes")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results.get("documents", [[]])[0] if results.get("documents") else []

# --- Clinical Trials Collection ---
def add_clinical_trials(documents, embeddings, metadatas=None):
    collection = chroma_client.get_or_create_collection("clinical_trials")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas or [{} for _ in documents],
        ids=[f"trial_{i}" for i in range(len(documents))]
    )

def query_clinical_trials(query_embedding, n_results=3):
    collection = chroma_client.get_or_create_collection("clinical_trials")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results.get("documents", [[]])[0] if results.get("documents") else []
