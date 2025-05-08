# embedding.py
# BioBERT embedding utilities (HuggingFace transformers)

from transformers import AutoTokenizer, AutoModel
import torch

# TODO: Download and cache BioBERT weights (e.g., dmis-lab/biobert-base-cased-v1.1)
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
_tokenizer = None
_model = None

def load_biobert():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
        _model = AutoModel.from_pretrained(BIOBERT_MODEL_NAME)

def embed_text(text: str):
    """Embed text using BioBERT (returns mean pooled vector)."""
    load_biobert()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = _model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings
