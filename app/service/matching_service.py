"""
Face Matching - Enhanced Ensemble Approach
==========================================
Combines multiple techniques for more accurate matching, especially as face appearance
changes over time:

1. Dual-Model Ensemble: ArcFace + Facenet512
2. Multi-Reference: Stores individual embeddings instead of just an average vector
3. Lightweight Classifier: sklearn SGDClassifier for personalized decision boundary
4. Adaptive Update: Reference embeddings can be updated over time

Final Score = 0.5 * arcface_score + 0.3 * facenet_score + 0.2 * classifier_score
"""

import logging
import os
from pathlib import Path

import numpy as np
from deepface import DeepFace

from app.service.reference_store import (
    MODELS,
    DATA_DIR,
    ReferenceStore,
)
from app.service.classifier_service import PersonalizedClassifier

logger = logging.getLogger(__name__)

CLASSIFIER_WEIGHT = 0.2  # weight cho sklearn classifier score
DETECTOR_BACKEND = "retinaface"
FINAL_THRESHOLD = 0.50  # final decision threshold (score < threshold → MATCH)

VECTOR_DIR = "image/vector"
TEST_DIR = "image/test"


# ── Helpers ──────────────────────────────────────────────────


def ensure_dirs():
    """Create the data directory if it does not exist."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_embedding(img_path: str, model_name: str) -> np.ndarray:
    """Extract face embedding from an image using the specified model."""
    result = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
    )
    emb = np.array(result[0]["embedding"])
    # L2 normalise
    return emb / np.linalg.norm(emb)


def get_multi_model_embeddings(img_path: str) -> dict[str, np.ndarray]:
    """Extract embeddings from all models for a single image."""
    embeddings = {}
    for model_name in MODELS:
        embeddings[model_name] = get_embedding(img_path, model_name)
    return embeddings


# ── Ensemble Scoring ─────────────────────────────────────────


def compute_final_score(
    ref_store: ReferenceStore,
    classifier: PersonalizedClassifier,
    test_embeddings: dict[str, np.ndarray],
) -> dict:
    """
    Compute the ensemble final score.

    Returns a dict with per-model score details + final score.
    """
    details = {}
    weighted_sum = 0.0

    # Distance score from each model
    for model_name, config in MODELS.items():
        dist = ref_store.compute_distance(model_name, test_embeddings[model_name])
        weighted_sum += config["weight"] * dist
        details[model_name] = {
            "distance": dist,
            "threshold": config["threshold"],
            "weight": config["weight"],
            "match": dist < config["threshold"],
        }

    # Classifier score
    concat_emb = np.concatenate([test_embeddings[m] for m in MODELS])
    clf_score = classifier.predict_score(concat_emb)
    weighted_sum += CLASSIFIER_WEIGHT * clf_score
    details["classifier"] = {
        "score": clf_score,
        "weight": CLASSIFIER_WEIGHT,
        "match": clf_score < 0.5,
    }

    details["final_score"] = weighted_sum
    details["final_match"] = weighted_sum < FINAL_THRESHOLD
    return details


# ── Pipeline Steps ───────────────────────────────────────────


def build_references(ref_store: ReferenceStore):
    """Step 1: Build reference embeddings from vector images."""
    if ref_store.load():
        print(f"\n[1] Loaded {ref_store.count()} reference embeddings from cache")
        return

    print(f"\n[1] Extracting reference embeddings from '{VECTOR_DIR}':")
    for filename in sorted(os.listdir(VECTOR_DIR)):
        filepath = os.path.join(VECTOR_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        print(f"  [ref] Processing: {filename}")
        try:
            embs = get_multi_model_embeddings(filepath)
            for model_name, emb in embs.items():
                ref_store.add(model_name, emb)
        except Exception as e:
            print(f"  [ref] Error with {filename}: {e}")

    if ref_store.count() == 0:
        raise RuntimeError("No embeddings could be extracted!")

    print(
        f"  => Extracted embeddings from {ref_store.count()} image(s)"
        f" × {len(MODELS)} models"
    )
    ref_store.save()


def build_classifier(classifier: PersonalizedClassifier, ref_store: ReferenceStore):
    """Step 2: Train or load the personalized classifier."""
    if classifier.load():
        print("\n[2] Loaded classifier from cache")
        return

    print("\n[2] Train personalized classifier:")
    positive_features = ref_store.get_all_concatenated()
    classifier.train(positive_features)
    classifier.save()


def print_result(result: dict):
    """Print detailed matching result."""
    for model_name in MODELS:
        d = result[model_name]
        m = "✓" if d["match"] else "✗"
        print(
            f"    {model_name:12s}: dist={d['distance']:.4f} "
            f"(threshold={d['threshold']}) [{m}]"
        )

    cd = result["classifier"]
    cm = "✓" if cd["match"] else "✗"
    print(f"    {'Classifier':12s}: score={cd['score']:.4f} [{cm}]")

    status = "MATCH ✓" if result["final_match"] else "NOT MATCH ✗"
    print(f"    {'─' * 40}")
    print(
        f"    Final Score : {result['final_score']:.4f} (threshold: {FINAL_THRESHOLD})"
    )
    print(f"    Result       : {status}")


def run_tests(ref_store: ReferenceStore, classifier: PersonalizedClassifier):
    """Step 3: Test matching."""
    print(f"\n[3] Testing face matching in '{TEST_DIR}':")
    print("-" * 60)

    for filename in sorted(os.listdir(TEST_DIR)):
        filepath = os.path.join(TEST_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        print(f"\n  📷 {filename}")
        try:
            test_embs = get_multi_model_embeddings(filepath)
            result = compute_final_score(ref_store, classifier, test_embs)
            print_result(result)
        except Exception as e:
            print(f"    Error: {e}")


# ── CLI Commands ─────────────────────────────────────────────


def cmd_match():
    """Default command: build references, train classifier, test matching."""
    ref_store = ReferenceStore()
    classifier = PersonalizedClassifier()

    build_references(ref_store)
    build_classifier(classifier, ref_store)
    run_tests(ref_store, classifier)


def cmd_retrain():
    """
    Train your own model: clears the old cache, re-extracts embeddings from
    image/vector/, and trains the classifier from scratch.
    Use when adding/removing many reference images or when a full reset is needed.
    """
    print("\n[retrain] Clearing old cache and retraining from scratch...")
    for fname in ("references.json", "classifier.pkl"):
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"  Removed {fpath}")

    ref_store = ReferenceStore()
    classifier = PersonalizedClassifier()

    build_references(ref_store)
    build_classifier(classifier, ref_store)
    print("\n[retrain] Done — model has been retrained.")


def cmd_enroll(image_paths: list[str]):
    """
    Online model updating: adds new images to the reference store and
    incrementally updates the classifier (partial_fit) without a full retrain.

    Use when a user enrolls 1-2 new images (e.g. a photo taken today) so the
    model adapts to gradual appearance changes over time.
    """
    ref_store = ReferenceStore()
    classifier = PersonalizedClassifier()

    if not ref_store.load():
        print("[enroll] No reference store found. Run 'match' or 'retrain' first.")
        return
    if not classifier.load():
        print("[enroll] No classifier found. Run 'match' or 'retrain' first.")
        return

    old_count = ref_store.count()
    new_features = []

    for img_path in image_paths:
        if not os.path.isfile(img_path):
            print(f"  [enroll] File not found: {img_path}")
            continue
        print(f"  [enroll] Processing: {img_path}")
        try:
            ref_store.enroll(img_path, get_multi_model_embeddings)
            # Get concatenated embedding for classifier update
            embs = {m: ref_store.references[m][-1] for m in MODELS}
            concat = np.concatenate([embs[m] for m in MODELS])
            new_features.append(concat)
        except Exception as e:
            print(f"  [enroll] Error: {e}")

    if not new_features:
        print("[enroll] No images were added.")
        return

    # Incremental update classifier (partial_fit)
    new_features_arr = np.array(new_features)
    classifier.partial_update(new_features_arr)

    ref_store.save()
    classifier.save()

    print(f"\n[enroll] Added {ref_store.count() - old_count} new image(s)")
    print(f"  References: {old_count} → {ref_store.count()}")
    print("  Classifier updated (online update).")
