"""
Personalized Classifier
========================
Lightweight binary classifier (SGDClassifier) học personalized decision boundary.

- Input: concatenated embeddings từ tất cả models
- Output: probability of match (0.0 → 1.0)
- Dùng SGD để có thể incremental update (partial_fit) khi có data mới

Tạo negative samples bằng cách perturbation (thêm noise) vào positive embeddings.
"""

import logging
import os
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from app.service.reference_store import DATA_DIR

logger = logging.getLogger(__name__)


class PersonalizedClassifier:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.path.join(DATA_DIR, "classifier.pkl")
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(
            loss="modified_huber",  # cho phép predict_proba
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )
        self.is_trained = False

    def _generate_negative_samples(
        self, positive_features: np.ndarray, n_negatives: int = 50
    ) -> np.ndarray:
        """
        Tạo negative samples bằng perturbation mạnh.
        Thêm noise lớn để mô phỏng người khác.
        """
        rng = np.random.default_rng(42)
        negatives = []
        for _ in range(n_negatives):
            idx = rng.integers(0, len(positive_features))
            base = positive_features[idx].copy()
            # Noise mạnh để tạo "người khác"
            noise = rng.normal(0, 0.3, size=base.shape)
            neg = base + noise
            neg = neg / np.linalg.norm(neg)
            negatives.append(neg)
        return np.array(negatives)

    def train(self, positive_features: np.ndarray):
        """Train classifier từ positive embeddings (reference images)."""
        if len(positive_features) < 2:
            print("  [classifier] Cần ít nhất 2 reference images để train.")
            return

        neg_features = self._generate_negative_samples(positive_features)

        features = np.vstack([positive_features, neg_features])
        labels = np.array([1] * len(positive_features) + [0] * len(neg_features))

        # Chuẩn hoá features
        features_scaled = self.scaler.fit_transform(features)

        self.clf.fit(features_scaled, labels)
        self.is_trained = True

        # Training accuracy
        acc = self.clf.score(features_scaled, labels)
        print(f"  [classifier] Đã train — accuracy: {acc:.2%}")

    def partial_update(self, positive_features: np.ndarray):
        """
        Online update: incremental learning với ảnh mới (partial_fit).
        Không cần retrain toàn bộ — chỉ cập nhật weights.
        """
        if not self.is_trained:
            print("  [classifier] Chưa có model, train đầy đủ trước.")
            self.train(positive_features)
            return

        neg_features = self._generate_negative_samples(positive_features)
        features = np.vstack([positive_features, neg_features])
        labels = np.array([1] * len(positive_features) + [0] * len(neg_features))

        features_scaled = self.scaler.transform(features)
        self.clf.partial_fit(features_scaled, labels)
        print(f"  [classifier] Online update với {len(positive_features)} ảnh mới")

    def predict_score(self, features: np.ndarray) -> float:
        """
        Trả về confidence score (0.0 → 1.0).
        Score cao → khả năng match cao.
        Trả về 1 - confidence để dùng như distance (thấp = match).
        """
        if not self.is_trained:
            return 0.5  # neutral nếu chưa train

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.clf.predict_proba(features_scaled)[0]
        # proba[1] = probability of class 1 (match)
        match_prob = proba[1] if len(proba) > 1 else proba[0]
        # Chuyển thành distance-like score (0 = match, 1 = not match)
        return float(1.0 - match_prob)

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {"scaler": self.scaler, "clf": self.clf, "trained": self.is_trained}, f
            )
        print(f"  => Đã lưu classifier → {self.model_path}")

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.clf = data["clf"]
        self.is_trained = data["trained"]
        print(f"  => Đã load classifier từ {self.model_path}")
        return True
