"""
Multi-Reference Embedding Storage
==================================
Stores multiple reference embeddings per model, with support for updates.
Combines average distance + min distance for better robustness against appearance changes over time.
"""

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Model configurations for ensemble
MODELS = {
    "ArcFace": {"weight": 0.5, "threshold": 0.68},
    "Facenet512": {"weight": 0.3, "threshold": 0.40},
}

DATA_DIR = "data"


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two normalised vectors (0 = identical, 2 = opposite)."""
    return float(1.0 - np.dot(a, b))


class ReferenceStore:
    """Stores multiple reference embeddings per model, with support for updates."""

    def __init__(self, store_path: str | None = None):
        self.store_path = store_path or os.path.join(DATA_DIR, "references.json")
        # {model_name: [list of embedding arrays]}
        self.references: dict[str, list[np.ndarray]] = {m: [] for m in MODELS}

    def add(self, model_name: str, embedding: np.ndarray):
        self.references[model_name].append(embedding)

    def get_avg_vector(self, model_name: str) -> np.ndarray:
        """Average (mean) vector for one model."""
        embs = self.references[model_name]
        if not embs:
            raise ValueError(f"No embeddings found for {model_name}")
        avg = np.mean(embs, axis=0)
        return avg / np.linalg.norm(avg)

    def get_min_distance(self, model_name: str, test_emb: np.ndarray) -> float:
        """Minimum distance between test_emb and all reference embeddings."""
        embs = self.references[model_name]
        if not embs:
            raise ValueError(f"No embeddings found for {model_name}")
        return min(cosine_distance(ref, test_emb) for ref in embs)

    def compute_distance(self, model_name: str, test_emb: np.ndarray) -> float:
        """
        Combines average distance + min distance.
        Uses min distance to catch cases where the face has changed but still
        matches one specific reference.
        """
        avg_dist = cosine_distance(self.get_avg_vector(model_name), test_emb)
        min_dist = self.get_min_distance(model_name, test_emb)
        # Favour min_distance (60%) — more robust against appearance changes over time
        return 0.4 * avg_dist + 0.6 * min_dist

    def count(self) -> int:
        first_model = list(MODELS.keys())[0]
        return len(self.references[first_model])

    def get_all_concatenated(self) -> np.ndarray:
        """Concatenate all embeddings from every model into a large feature vector for the classifier."""
        first_model = list(MODELS.keys())[0]
        n = len(self.references[first_model])
        rows = []
        for i in range(n):
            concat = np.concatenate([self.references[m][i] for m in MODELS])
            rows.append(concat)
        return np.array(rows)

    def enroll(self, img_path: str, get_multi_model_embeddings_fn):
        """Add a new image to the reference store (online enrollment)."""
        embs = get_multi_model_embeddings_fn(img_path)
        for model_name, emb in embs.items():
            self.add(model_name, emb)
        return embs

    def save(self):
        data = {}
        for model_name, embs in self.references.items():
            data[model_name] = [e.tolist() for e in embs]
        with open(self.store_path, "w") as f:
            json.dump(data, f)
        print(f"  => Reference embeddings saved → {self.store_path}")

    def load(self) -> bool:
        if not os.path.exists(self.store_path):
            return False
        with open(self.store_path) as f:
            data = json.load(f)
        for model_name, embs in data.items():
            if model_name in self.references:
                self.references[model_name] = [np.array(e) for e in embs]
        print(f"  => Reference embeddings loaded from {self.store_path}")
        return True
