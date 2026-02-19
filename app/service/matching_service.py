"""
Face Matching - Enhanced Ensemble Approach
==========================================
Kết hợp nhiều kỹ thuật để matching chính xác hơn, đặc biệt khi khuôn mặt
thay đổi theo thời gian:

1. Dual-Model Ensemble: ArcFace + Facenet512
2. Multi-Reference: Lưu từng embedding thay vì chỉ average vector
3. Lightweight Classifier: sklearn SGDClassifier cho personalized decision boundary
4. Adaptive Update: Có thể cập nhật reference embeddings theo thời gian

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
FINAL_THRESHOLD = 0.50  # ngưỡng quyết định cuối cùng (score < threshold → MATCH)

VECTOR_DIR = "image/vector"
TEST_DIR = "image/test"


# ── Helpers ──────────────────────────────────────────────────


def ensure_dirs():
    """Tạo thư mục data nếu chưa có."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def get_embedding(img_path: str, model_name: str) -> np.ndarray:
    """Trích xuất face embedding từ ảnh bằng model chỉ định."""
    result = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
    )
    emb = np.array(result[0]["embedding"])
    # Chuẩn hoá L2
    return emb / np.linalg.norm(emb)


def get_multi_model_embeddings(img_path: str) -> dict[str, np.ndarray]:
    """Trích xuất embeddings từ tất cả models cho 1 ảnh."""
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
    Tính điểm tổng hợp từ ensemble.

    Returns dict với chi tiết từng model score + final score.
    """
    details = {}
    weighted_sum = 0.0

    # Distance scores từ từng model
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
    """Bước 1: Xây dựng reference embeddings từ ảnh vector."""
    if ref_store.load():
        print(f"\n[1] Đã load {ref_store.count()} reference embeddings từ cache")
        return

    print(f"\n[1] Trích xuất reference embeddings từ '{VECTOR_DIR}':")
    for filename in sorted(os.listdir(VECTOR_DIR)):
        filepath = os.path.join(VECTOR_DIR, filename)
        if not os.path.isfile(filepath):
            continue
        print(f"  [ref] Đang xử lý: {filename}")
        try:
            embs = get_multi_model_embeddings(filepath)
            for model_name, emb in embs.items():
                ref_store.add(model_name, emb)
        except Exception as e:
            print(f"  [ref] Lỗi với {filename}: {e}")

    if ref_store.count() == 0:
        raise RuntimeError("Không trích xuất được embedding nào!")

    print(
        f"  => Đã trích xuất embeddings từ {ref_store.count()} ảnh "
        f"× {len(MODELS)} models"
    )
    ref_store.save()


def build_classifier(classifier: PersonalizedClassifier, ref_store: ReferenceStore):
    """Bước 2: Train hoặc load personalized classifier."""
    if classifier.load():
        print("\n[2] Đã load classifier từ cache")
        return

    print("\n[2] Train personalized classifier:")
    positive_features = ref_store.get_all_concatenated()
    classifier.train(positive_features)
    classifier.save()


def print_result(result: dict):
    """In chi tiết kết quả matching."""
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
    print(f"    Kết quả     : {status}")


def run_tests(ref_store: ReferenceStore, classifier: PersonalizedClassifier):
    """Bước 3: Test matching."""
    print(f"\n[3] Kiểm tra face matching trong '{TEST_DIR}':")
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
            print(f"    Lỗi: {e}")


# ── CLI Commands ─────────────────────────────────────────────


def cmd_match():
    """Lệnh mặc định: build references, train classifier, test matching."""
    ref_store = ReferenceStore()
    classifier = PersonalizedClassifier()

    build_references(ref_store)
    build_classifier(classifier, ref_store)
    run_tests(ref_store, classifier)


def cmd_retrain():
    """
    Train your own model: Xoá cache cũ, trích xuất lại embeddings từ
    image/vector/ và train classifier từ đầu.
    Dùng khi thêm/xoá nhiều ảnh reference hoặc muốn reset.
    """
    print("\n[retrain] Xoá cache cũ và train lại từ đầu...")
    for fname in ("references.json", "classifier.pkl"):
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"  Đã xoá {fpath}")

    ref_store = ReferenceStore()
    classifier = PersonalizedClassifier()

    build_references(ref_store)
    build_classifier(classifier, ref_store)
    print("\n[retrain] Hoàn tất — model đã được train lại.")


def cmd_enroll(image_paths: list[str]):
    """
    Online model updating: Thêm ảnh mới vào reference store và
    incremental update classifier (partial_fit) mà không cần retrain toàn bộ.

    Dùng khi user thêm 1-2 ảnh mới (ví dụ: ảnh chụp hôm nay) để model
    thích nghi với thay đổi khuôn mặt theo thời gian.
    """
    ref_store = ReferenceStore()
    classifier = PersonalizedClassifier()

    if not ref_store.load():
        print("[enroll] Chưa có reference store. Chạy 'match' hoặc 'retrain' trước.")
        return
    if not classifier.load():
        print("[enroll] Chưa có classifier. Chạy 'match' hoặc 'retrain' trước.")
        return

    old_count = ref_store.count()
    new_features = []

    for img_path in image_paths:
        if not os.path.isfile(img_path):
            print(f"  [enroll] File không tồn tại: {img_path}")
            continue
        print(f"  [enroll] Đang xử lý: {img_path}")
        try:
            ref_store.enroll(img_path, get_multi_model_embeddings)
            # Lấy concatenated embedding cho classifier update
            embs = {m: ref_store.references[m][-1] for m in MODELS}
            concat = np.concatenate([embs[m] for m in MODELS])
            new_features.append(concat)
        except Exception as e:
            print(f"  [enroll] Lỗi: {e}")

    if not new_features:
        print("[enroll] Không có ảnh nào được thêm.")
        return

    # Incremental update classifier (partial_fit)
    new_features_arr = np.array(new_features)
    classifier.partial_update(new_features_arr)

    ref_store.save()
    classifier.save()

    print(f"\n[enroll] Đã thêm {ref_store.count() - old_count} ảnh mới ")
    print(f"  References: {old_count} → {ref_store.count()}")
    print("  Classifier đã được cập nhật (online update).")
