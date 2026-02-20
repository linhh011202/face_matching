"""
Personalized Classifier
========================
Lightweight binary classifier (SGDClassifier) that learns a personalized decision boundary.

- Input: concatenated embeddings from all models
- Output: probability of match (0.0 → 1.0)
- Uses SGD for incremental updates (partial_fit) when new data arrives

Generates negative samples via perturbation (adding noise) to positive embeddings.
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
            loss="modified_huber",  # enables predict_proba
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )
        self.is_trained = False

    def _generate_negative_samples(
        self, positive_features: np.ndarray, n_negatives: int = 50
    ) -> np.ndarray:
        """
        Generate negative samples via strong perturbation.
        Adds large noise to simulate a different person.
        """
        rng = np.random.default_rng(42)
        negatives = []
        for _ in range(n_negatives):
            idx = rng.integers(0, len(positive_features))
            base = positive_features[idx].copy()
            # Strong noise to simulate "a different person"
            noise = rng.normal(0, 0.3, size=base.shape)
            neg = base + noise
            neg = neg / np.linalg.norm(neg)
            negatives.append(neg)
        return np.array(negatives)

    def train(self, positive_features: np.ndarray):
        """Train the classifier from positive embeddings (reference images)."""
        if len(positive_features) < 2:
            print("  [classifier] At least 2 reference images are required to train.")
            return

        neg_features = self._generate_negative_samples(positive_features)

        features = np.vstack([positive_features, neg_features])
        labels = np.array([1] * len(positive_features) + [0] * len(neg_features))

        # Normalise features
        features_scaled = self.scaler.fit_transform(features)

        self.clf.fit(features_scaled, labels)
        self.is_trained = True

        # Training accuracy
        acc = self.clf.score(features_scaled, labels)
        print(f"  [classifier] Trained — accuracy: {acc:.2%}")

    def partial_update(self, positive_features: np.ndarray):
        """
        Online update: incremental learning with new images (partial_fit).
        No full retrain needed — only weights are updated.
        """
        if not self.is_trained:
            print("  [classifier] No model found, run full training first.")
            self.train(positive_features)
            return

        neg_features = self._generate_negative_samples(positive_features)
        features = np.vstack([positive_features, neg_features])
        labels = np.array([1] * len(positive_features) + [0] * len(neg_features))

        features_scaled = self.scaler.transform(features)
        self.clf.partial_fit(features_scaled, labels)
        print(
            f"  [classifier] Online update with {len(positive_features)} new image(s)"
        )

    def predict_score(self, features: np.ndarray) -> float:
        """
        Returns a confidence score (0.0 → 1.0).
        High score → higher match probability.
        Returns 1 - confidence to use as distance (low = match).
        """
        if not self.is_trained:
            return 0.5  # neutral if not yet trained

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        proba = self.clf.predict_proba(features_scaled)[0]
        # proba[1] = probability of class 1 (match)
        match_prob = proba[1] if len(proba) > 1 else proba[0]
        # Convert to distance-like score (0 = match, 1 = not match)
        return float(1.0 - match_prob)

    def save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {"scaler": self.scaler, "clf": self.clf, "trained": self.is_trained}, f
            )
        print(f"  => Classifier saved → {self.model_path}")

    def load(self) -> bool:
        if not os.path.exists(self.model_path):
            return False
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.clf = data["clf"]
        self.is_trained = data["trained"]
        print(f"  => Classifier loaded from {self.model_path}")
        return True
