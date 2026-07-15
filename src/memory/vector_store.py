"""
Visual Memory AI — FAISS Vector Store
Manages storage and retrieval of visual embeddings using FAISS.
"""

import json
import os
from typing import List, Tuple
import numpy as np
import faiss


class VectorStore:
    """FAISS-based vector store for visual memory embeddings."""

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self._id_map: List[int] = []
        print(f"[VectorStore] Initialized FAISS IndexFlatIP | Dim: {dimension}")

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, embedding: np.ndarray, memory_id: int) -> None:
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)
        self.index.add(embedding)
        self._id_map.append(memory_id)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.size == 0:
            return []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        k = min(top_k, self.size)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._id_map):
                results.append((self._id_map[idx], float(score)))
        return results

    def save(self, index_path: str, id_map_path: str) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        os.makedirs(os.path.dirname(id_map_path), exist_ok=True)
        with open(id_map_path, "w") as f:
            json.dump(self._id_map, f)
        print(f"[VectorStore] Saved {self.size} embeddings to {index_path}")

    def load(self, index_path: str, id_map_path: str) -> bool:
        if not os.path.exists(index_path) or not os.path.exists(id_map_path):
            print("[VectorStore] No existing index found, starting fresh")
            return False
        self.index = faiss.read_index(index_path)
        with open(id_map_path, "r") as f:
            self._id_map = json.load(f)
        print(f"[VectorStore] Loaded {self.size} embeddings from {index_path}")
        return True

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self._id_map.clear()
        print("[VectorStore] Cleared all embeddings")
