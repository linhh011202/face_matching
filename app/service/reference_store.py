"""
Multi-Reference Embedding Storage
==================================
Lưu trữ nhiều reference embeddings cho mỗi model, hỗ trợ cập nhật.
Kết hợp average distance + min distance để robust hơn với biến đổi thời gian.
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
    """Cosine distance giữa 2 vector đã chuẩn hoá (0 = giống, 2 = khác)."""
    return float(1.0 - np.dot(a, b))


class ReferenceStore:
    """Lưu trữ nhiều reference embeddings cho mỗi model, hỗ trợ cập nhật."""

    def __init__(self, store_path: str | None = None):
        self.store_path = store_path or os.path.join(DATA_DIR, "references.json")
        # {model_name: [list of embedding arrays]}
        self.references: dict[str, list[np.ndarray]] = {m: [] for m in MODELS}

    def add(self, model_name: str, embedding: np.ndarray):
        self.references[model_name].append(embedding)

    def get_avg_vector(self, model_name: str) -> np.ndarray:
        """Vector trung bình cho 1 model."""
        embs = self.references[model_name]
        if not embs:
            raise ValueError(f"Không có embedding nào cho {model_name}")
        avg = np.mean(embs, axis=0)
        return avg / np.linalg.norm(avg)

    def get_min_distance(self, model_name: str, test_emb: np.ndarray) -> float:
        """Khoảng cách nhỏ nhất giữa test_emb và mọi reference embedding."""
        embs = self.references[model_name]
        if not embs:
            raise ValueError(f"Không có embedding nào cho {model_name}")
        return min(cosine_distance(ref, test_emb) for ref in embs)

    def compute_distance(self, model_name: str, test_emb: np.ndarray) -> float:
        """
        Kết hợp average distance + min distance.
        Dùng min distance để bắt trường hợp khuôn mặt thay đổi nhưng vẫn
        khớp với 1 reference cụ thể.
        """
        avg_dist = cosine_distance(self.get_avg_vector(model_name), test_emb)
        min_dist = self.get_min_distance(model_name, test_emb)
        # Ưu tiên min_distance (60%) vì nó robust hơn với biến đổi thời gian
        return 0.4 * avg_dist + 0.6 * min_dist

    def count(self) -> int:
        first_model = list(MODELS.keys())[0]
        return len(self.references[first_model])

    def get_all_concatenated(self) -> np.ndarray:
        """Nối tất cả embeddings từ mọi model thành feature vector lớn cho classifier."""
        first_model = list(MODELS.keys())[0]
        n = len(self.references[first_model])
        rows = []
        for i in range(n):
            concat = np.concatenate([self.references[m][i] for m in MODELS])
            rows.append(concat)
        return np.array(rows)

    def enroll(self, img_path: str, get_multi_model_embeddings_fn):
        """Thêm 1 ảnh mới vào reference store (online enrollment)."""
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
        print(f"  => Đã lưu reference embeddings → {self.store_path}")

    def load(self) -> bool:
        if not os.path.exists(self.store_path):
            return False
        with open(self.store_path) as f:
            data = json.load(f)
        for model_name, embs in data.items():
            if model_name in self.references:
                self.references[model_name] = [np.array(e) for e in embs]
        print(f"  => Đã load reference embeddings từ {self.store_path}")
        return True
