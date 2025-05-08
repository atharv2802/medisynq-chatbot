# ingest_data.py
# Script to ingest discharge notes and clinical trials CSVs into separate ChromaDB collections
import pandas as pd
from tqdm import tqdm
from embedding import embed_text
import chromadb

chroma_client = chromadb.Client(
    chromadb.config.Settings(persist_directory="chromadb_store")
)

# --- Discharge Notes Ingestion ---
def ingest_discharge_notes(csv_path):
    df = pd.read_csv(csv_path)
    collection = chroma_client.get_or_create_collection("discharge_notes")
    documents = df["text_chunk"].astype(str).tolist()
    metadatas = df[["subject_id", "hadm_id"]].to_dict(orient="records")
    print(f"Embedding and adding {len(documents)} discharge notes to ChromaDB...")
    embeddings = [embed_text(text) for text in tqdm(documents, desc="Embedding discharge notes")]
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"note_{i}" for i in range(len(documents))]
    )
    print("Discharge notes ingestion complete.")

# --- Clinical Trials Ingestion ---
def ingest_clinical_trials(csv_path):
    df = pd.read_csv(csv_path)
    collection = chroma_client.get_or_create_collection("clinical_trials")
    # Concatenate relevant fields for context
    def row_to_text(row):
        fields = [
            str(row.get("Study Title", "")),
            str(row.get("Conditions", "")),
            str(row.get("Inclusion Criteria", "")),
            str(row.get("Exclusion Criteria", "")),
            str(row.get("Interventions", "")),
            str(row.get("Study Type", "")),
            str(row.get("Overall Status", ""))
        ]
        return " | ".join(fields)
    documents = df.apply(row_to_text, axis=1).tolist()
    metadatas = df[["NCT ID"]].rename(columns={"NCT ID": "nct_id"}).to_dict(orient="records")
    print(f"Embedding and adding {len(documents)} clinical trials to ChromaDB...")
    embeddings = [embed_text(text) for text in tqdm(documents, desc="Embedding clinical trials")]
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"trial_{i}" for i in range(len(documents))]
    )
    print("Clinical trials ingestion complete.")

if __name__ == "__main__":
    ingest_discharge_notes("Data/filtered_notes.csv")
    ingest_clinical_trials("Data/filtered_trials.csv")
