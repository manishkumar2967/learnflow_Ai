from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]):
    return get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
