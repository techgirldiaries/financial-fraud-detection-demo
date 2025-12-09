"""
Simple in-memory vector store for cosine similarity retrieval.
"""

import numpy as np
from typing import Any, Dict, List

class SimpleVectorStore:
    def __init__(self, fraud_vectors: np.ndarray, fraud_texts: List[str]):
        self.fraud_vectors = np.asarray(fraud_vectors) if fraud_vectors is not None else np.zeros((0,))
        self.fraud_texts = list(fraud_texts or [])

    @staticmethod
    def embed_transaction(transaction_features: np.ndarray) -> np.ndarray:
        transaction = np.asarray(transaction_features).astype(float)
        norm = np.linalg.norm(transaction) + 1e-8
        return transaction / norm

    def retrieve(self, transaction_features: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        if len(self.fraud_vectors) == 0:
            return []
        query_vec = self.embed_transaction(transaction_features)
        F = np.asarray(self.fraud_vectors)
        F_norm = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
        similarities = F_norm.dot(query_vec)
        idx = np.argsort(similarities)[-top_k:][::-1]
        out = []
        for i in idx:
            sim = float(similarities[i])
            txt = self.fraud_texts[i] if i < len(self.fraud_texts) else ''
            out.append({'similarity': sim, 'pattern': txt})
        return out