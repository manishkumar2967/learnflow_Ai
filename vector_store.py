import faiss
import numpy as np
from .embeddings import embed_texts
from .text_splitter import Chunk

class StudyVectorStore:
    def __init__(self, chunks: list[Chunk]):
        if not chunks: raise ValueError("No usable text was found in the uploaded PDFs.")
        self.chunks = chunks
        vectors = np.asarray(embed_texts([c.text for c in chunks]), dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        vector = np.asarray(embed_texts([query]), dtype="float32")
        scores, ids = self.index.search(vector, min(k, len(self.chunks)))
        return [(self.chunks[i], float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0]
