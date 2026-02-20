import logging
import uuid

import numpy as np

from app.repository.user_face_repository import UserFaceRepository
from app.service.classifier_service import PersonalizedClassifier
from app.service.matching_service import (
    compute_final_score,
    get_multi_model_embeddings,
    print_result,
    FINAL_THRESHOLD,
    ensure_dirs,
)
from app.service.reference_store import ReferenceStore, MODELS
from app.service.storage_service import StorageService

logger = logging.getLogger(__name__)


class FaceVerificationService:
    """
    Verify a user's identity at sign-in using the Enhanced Ensemble approach
    (ArcFace + Facenet512 + SGDClassifier).

    Flow:
      1. Get registered face images (pose: straight/left/right) from DB
         → build ReferenceStore + train PersonalizedClassifier
      2. Get all login face images (3 most recent) → download
         → extract multi-model embeddings for each
      3. Compute ensemble final score per image → pick the best match
      4. If matched, retrain SGDClassifier with the new login embeddings
    """

    def __init__(
        self,
        user_face_repository: UserFaceRepository,
        storage_service: StorageService,
    ) -> None:
        self._face_repo = user_face_repository
        self._storage_svc = storage_service
        logger.info("FaceVerificationService initialized")

    def _build_reference_store(
        self, local_paths: list[str]
    ) -> tuple[ReferenceStore, PersonalizedClassifier]:
        """Build a ReferenceStore and train classifier from registered face images."""
        ensure_dirs()
        ref_store = ReferenceStore()
        classifier = PersonalizedClassifier()

        for path in local_paths:
            try:
                embs = get_multi_model_embeddings(path)
                for model_name, emb in embs.items():
                    ref_store.add(model_name, emb)
            except Exception as e:
                logger.warning(f"Failed to extract embeddings from {path}: {e}")

        if ref_store.count() == 0:
            return ref_store, classifier

        # Train classifier on registered embeddings
        positive_features = ref_store.get_all_concatenated()
        if len(positive_features) >= 2:
            classifier.train(positive_features)

        return ref_store, classifier

    @staticmethod
    def _retrain_classifier(
        ref_store: ReferenceStore,
        classifier: PersonalizedClassifier,
        login_embeddings_list: list[dict[str, np.ndarray]],
    ) -> None:
        """
        Retrain SGDClassifier by adding matched login embeddings to
        the reference store and performing an incremental partial_fit.
        """
        new_features = []
        for embs in login_embeddings_list:
            for model_name, emb in embs.items():
                ref_store.add(model_name, emb)
            concat = np.concatenate([embs[m] for m in MODELS])
            new_features.append(concat)

        if new_features:
            new_features_arr = np.array(new_features)
            classifier.partial_update(new_features_arr)
            logger.info(
                f"Retrained classifier with {len(new_features)} new login embeddings"
            )

    def _download_registered_images(self, user_id: uuid.UUID) -> list[str] | None:
        """Download registered face images for a user. Returns local paths or None."""
        registered_faces = self._face_repo.get_registered_faces_by_user_id(user_id)
        if not registered_faces:
            logger.error(f"No registered face records found for user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — NO REGISTERED FACES")
            return None

        registered_paths: list[str] = []
        for face in registered_faces:
            source_images = face.source_images or []
            if source_images:
                paths = self._storage_svc.download_images(source_images)
                registered_paths.extend(paths)

        if not registered_paths:
            logger.error(f"Failed to download registered images for user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — REGISTERED IMAGES DOWNLOAD FAILED")
            return None

        return registered_paths

    def _download_login_images(self, user_id: uuid.UUID) -> list[str] | None:
        """Download the most recent login face images. Returns local paths or None."""
        login_face = self._face_repo.get_login_face_by_user_id(user_id)
        if login_face is None:
            logger.error(f"No login face found for user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — NO LOGIN FACE FOUND")
            return None

        login_source_images = login_face.source_images or []
        if not login_source_images:
            logger.error(f"No source images for login face of user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — NO LOGIN IMAGE")
            return None

        login_paths = self._storage_svc.download_images(login_source_images)
        if not login_paths:
            logger.error(f"Failed to download login images for user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — LOGIN IMAGE DOWNLOAD FAILED")
            return None

        return login_paths

    def _match_login_images(
        self,
        login_paths: list[str],
        ref_store: ReferenceStore,
        classifier: PersonalizedClassifier,
        user_id: uuid.UUID,
    ) -> list[tuple[int, dict, dict[str, np.ndarray]]]:
        """Extract embeddings from each login image and compute scores."""
        results: list[tuple[int, dict, dict[str, np.ndarray]]] = []
        for idx, path in enumerate(login_paths):
            try:
                embs = get_multi_model_embeddings(path)
                result = compute_final_score(ref_store, classifier, embs)
                results.append((idx, result, embs))
            except Exception as e:
                logger.warning(
                    f"Failed to extract/match login image #{idx + 1} "
                    f"for user_id={user_id}: {e}"
                )
        return results

    @staticmethod
    def _print_verification_report(
        user_id: uuid.UUID,
        session_id: str | None,
        per_image_results: list[tuple[int, dict, dict[str, np.ndarray]]],
        total_images: int,
        best_idx: int,
        best_result: dict,
        status: str,
    ) -> None:
        """Print detailed per-image verification report."""
        print("=" * 60)
        print(f"  [VERIFY] User ID : {user_id}")
        print(f"  [VERIFY] Session : {session_id}")
        print(
            f"  [VERIFY] Login images evaluated: {len(per_image_results)}/{total_images}"
        )
        print("-" * 60)

        for idx, result, _ in per_image_results:
            best_marker = " ★ BEST" if idx == best_idx else ""
            img_status = "MATCH" if result["final_match"] else "NOT MATCH"
            print(
                f"\n  Image #{idx + 1} — score={result['final_score']:.4f} "
                f"[{img_status}]{best_marker}"
            )
            print_result(result)

        print("-" * 60)
        print(
            f"  [VERIFY] Best image : #{best_idx + 1} "
            f"(score={best_result['final_score']:.4f})"
        )
        print(f"  [VERIFY] Result     : {status}")
        print("=" * 60)

    def verify_user(self, user_id: uuid.UUID, session_id: str | None = None) -> bool:
        """
        Verify a user's identity by comparing all 3 login face images against
        registered faces using the Enhanced Ensemble.

        Picks the best-scoring image out of the 3, retrains the classifier
        with the new login embeddings if the result is a MATCH.

        Args:
            user_id: User's UUID.
            session_id: Session ID from the sign-in event.

        Returns:
            True if face matches (MATCH), False otherwise (NOT MATCH).
        """
        logger.info(f"Verifying face for user_id: {user_id} (session_id={session_id})")

        # ── Step 1: Build reference store from registered faces ──
        registered_paths = self._download_registered_images(user_id)
        if not registered_paths:
            return False

        ref_store, classifier = self._build_reference_store(registered_paths)
        self._storage_svc.cleanup_files(registered_paths)

        if ref_store.count() == 0:
            logger.error(
                f"No embeddings extracted from registered images: user_id={user_id}"
            )
            print(f"[VERIFY] user_id={user_id} — NO REGISTERED EMBEDDINGS")
            return False

        # ── Step 2: Download all login images (3 most recent) ────
        login_paths = self._download_login_images(user_id)
        if not login_paths:
            return False

        # ── Step 3: Match each login image, collect results ──────
        per_image_results = self._match_login_images(
            login_paths, ref_store, classifier, user_id
        )
        self._storage_svc.cleanup_files(login_paths)

        if not per_image_results:
            logger.error(f"All login image extractions failed for user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — ALL EMBEDDING EXTRACTIONS FAILED")
            return False

        # ── Step 4: Pick best image (lowest final_score) ─────────
        best_idx, best_result, _ = min(
            per_image_results, key=lambda t: t[1]["final_score"]
        )
        is_match = best_result["final_match"]
        status = "MATCH ✓" if is_match else "NOT MATCH ✗"

        self._print_verification_report(
            user_id,
            session_id,
            per_image_results,
            len(login_paths),
            best_idx,
            best_result,
            status,
        )

        # ── Step 5: If match, retrain classifier with login imgs ─
        if is_match:
            login_embs_list = [embs for _, _, embs in per_image_results]
            self._retrain_classifier(ref_store, classifier, login_embs_list)
            logger.info(
                f"Classifier retrained with {len(login_embs_list)} login images "
                f"for user_id={user_id}"
            )

        logger.info(
            f"Face verification for user_id={user_id}: {status} "
            f"(best_image=#{best_idx + 1}, "
            f"final_score={best_result['final_score']:.4f}, "
            f"threshold={FINAL_THRESHOLD})"
        )
        return is_match
