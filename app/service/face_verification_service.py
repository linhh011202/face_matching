import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.core.config import configs
from app.repository.user_face_repository import UserFaceRepository
from app.service.classifier_service import PersonalizedClassifier
from app.service.embedding_service import EmbeddingService
from app.service.storage_service import StorageService

logger = logging.getLogger(__name__)


class FaceVerificationService:
    """
    Fast sign-in verification using ArcFace embeddings.

    Pipeline (optimized for latency):
      Phase 1 — Parallel data loading:
        • DB: registered embeddings
        • DB: recent login embeddings (personalization)
        • Network: download login image(s) from Firebase Storage
      Phase 2 — Embedding extraction with early-exit:
        • Extract login embeddings sequentially
        • If a confident match is found → skip remaining images
      Phase 3 — Post-match updates:
        • Persist best login embedding for future personalization
        • Retrain per-user classifier (incremental SGD)

    Fallback path:
      If DB has no registered embeddings yet, extract on-demand from
      registered source images and persist embeddings for future sign-ins.
    """

    def __init__(
        self,
        user_face_repository: UserFaceRepository,
        storage_service: StorageService,
        embedding_service: EmbeddingService,
    ) -> None:
        self._face_repo = user_face_repository
        self._storage_svc = storage_service
        self._embedding_svc = embedding_service
        # Per-user classifier cache (in-memory, survives across requests)
        self._classifiers: dict[str, PersonalizedClassifier] = {}
        logger.info("FaceVerificationService initialized")

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    # ── Data loading helpers ──────────────────────────────────

    def _download_login_images(
        self, user_id: uuid.UUID
    ) -> tuple[int | None, list[str]]:
        """Download the latest login image(s). Returns (login_face_id, local_paths)."""
        login_face = self._face_repo.get_login_face_by_user_id(user_id)
        if login_face is None:
            logger.error(f"No login face found for user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — NO LOGIN FACE FOUND")
            return None, []

        source_images = login_face.source_images or []
        if not source_images:
            logger.error(f"No source images for login face of user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — NO LOGIN IMAGE")
            return login_face.id, []

        max_images = min(len(source_images), max(3, configs.SIGNIN_MAX_LOGIN_IMAGES))
        selected_urls = source_images[-max_images:]
        logger.info(
            f"Login images for user_id={user_id}: "
            f"total_in_db={len(source_images)}, "
            f"selected={len(selected_urls)}/{len(source_images)} "
            f"(max_images={max_images})"
        )

        paths = self._storage_svc.download_images(
            selected_urls, max_workers=configs.DOWNLOAD_WORKERS
        )
        if not paths:
            logger.error(f"Failed to download login images for user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — LOGIN IMAGE DOWNLOAD FAILED")
            return login_face.id, []

        logger.info(
            f"Downloaded {len(paths)}/{len(selected_urls)} login images "
            f"for user_id={user_id}"
        )
        return login_face.id, paths

    def _extract_registered_embeddings_on_demand(
        self, user_id: uuid.UUID
    ) -> list[np.ndarray]:
        """
        Fallback for users whose registered embeddings were not generated yet.
        Extract embeddings from registered source images and store into DB.
        """
        faces = self._face_repo.get_registered_faces_by_user_id(user_id)
        if not faces:
            return []

        extracted: list[np.ndarray] = []
        downloaded_paths: list[str] = []

        for face in faces:
            if face.id is None:
                continue
            source_images = face.source_images or []
            if not source_images:
                continue

            paths = self._storage_svc.download_images(
                source_images, max_workers=configs.DOWNLOAD_WORKERS
            )
            downloaded_paths.extend(paths)
            if not paths:
                continue

            emb = self._embedding_svc.compute_average_embedding(
                paths, max_workers=configs.EMBEDDING_WORKERS
            )
            if emb is None:
                continue

            if self._face_repo.update_embedding(face.id, emb.tolist()):
                extracted.append(emb)

        self._storage_svc.cleanup_files(downloaded_paths)
        return extracted

    # ── Classifier helpers ────────────────────────────────────

    def _get_or_create_classifier(self, user_id: uuid.UUID) -> PersonalizedClassifier:
        """Get or create a per-user classifier (in-memory cache)."""
        key = str(user_id)
        if key not in self._classifiers:
            self._classifiers[key] = PersonalizedClassifier()
        return self._classifiers[key]

    def _retrain_classifier(
        self,
        user_id: uuid.UUID,
        reference_embeddings: list[np.ndarray],
        new_embedding: np.ndarray,
    ) -> None:
        """Retrain per-user classifier after successful match."""
        try:
            clf = self._get_or_create_classifier(user_id)
            if clf.is_trained:
                # Incremental update with the newly verified embedding
                clf.partial_update(np.array([new_embedding]))
            else:
                # First-time training from all reference embeddings + new login
                all_positives = np.vstack(
                    [np.array(reference_embeddings), new_embedding.reshape(1, -1)]
                )
                clf.train(all_positives)
            logger.info(f"Classifier retrained for user_id={user_id}")
        except Exception as e:
            logger.warning(
                f"Classifier retrain failed for user_id={user_id}: {e}",
                exc_info=True,
            )

    # ── Main verification flow ────────────────────────────────

    def _load_data_parallel(
        self, user_id: uuid.UUID
    ) -> tuple[list[np.ndarray], int | None, list[str]]:
        """
        Phase 1: Load all required data in parallel.
        Returns (reference_embeddings, login_face_id, login_paths).
        """
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures: dict[str, object] = {}

            futures["login_images"] = pool.submit(self._download_login_images, user_id)
            if configs.SIGNIN_USE_DB_EMBEDDINGS:
                futures["registered"] = pool.submit(
                    self._face_repo.get_registered_embeddings_by_user_id, user_id
                )
            if configs.SIGNIN_PERSONALIZATION_ENABLED:
                futures["login_history"] = pool.submit(
                    self._face_repo.get_recent_login_embeddings_by_user_id,
                    user_id,
                    configs.SIGNIN_PERSONALIZATION_MAX_EMBEDDINGS,
                )

            registered = (
                futures["registered"].result() if "registered" in futures else []
            )
            login_history = (
                futures["login_history"].result() if "login_history" in futures else []
            )
            login_face_id, login_paths = futures["login_images"].result()

        # Fallback: extract on-demand if no precomputed embeddings
        if not registered and configs.SIGNIN_ALLOW_ON_DEMAND_REGISTERED_EMBEDDINGS:
            logger.warning(
                f"No precomputed registered embeddings for user_id={user_id}; "
                "running on-demand extraction fallback"
            )
            registered = self._extract_registered_embeddings_on_demand(user_id)

        reference_embeddings = registered + login_history
        if reference_embeddings:
            logger.info(
                f"Loaded references for user_id={user_id}: "
                f"registered={len(registered)}, "
                f"personalized={len(login_history)}, "
                f"total={len(reference_embeddings)}"
            )
        return reference_embeddings, login_face_id, login_paths

    def _score_login_images(
        self,
        login_paths: list[str],
        reference_embeddings: list[np.ndarray],
    ) -> list[tuple[int, float, np.ndarray]]:
        """
        Phase 2: Extract embeddings from all login images, compare with references.
        Returns all scored results so the caller can pick the best one.
        """
        scored: list[tuple[int, float, np.ndarray]] = []
        try:
            for idx, path in enumerate(login_paths):
                emb = self._embedding_svc.extract_embedding(path)
                if emb is None:
                    continue
                best_dist = min(
                    self._cosine_distance(ref, emb) for ref in reference_embeddings
                )
                scored.append((idx, best_dist, emb))
        finally:
            self._storage_svc.cleanup_files(login_paths)
        return scored

    def _post_match_updates(
        self,
        user_id: uuid.UUID,
        login_face_id: int,
        best_emb: np.ndarray,
        reference_embeddings: list[np.ndarray],
    ) -> None:
        """Phase 3: Persist login embedding + retrain classifier."""
        if configs.SIGNIN_UPDATE_LOGIN_EMBEDDING:
            if self._face_repo.update_embedding(login_face_id, best_emb.tolist()):
                logger.info(
                    f"Updated login embedding for personalization: "
                    f"user_id={user_id}, face_id={login_face_id}"
                )
        self._retrain_classifier(user_id, reference_embeddings, best_emb)

    def verify_user(self, user_id: uuid.UUID, session_id: str | None = None) -> bool:
        """
        Verify user by comparing login embedding(s) to registered embeddings.
        Returns True when best distance is below SIGNIN_DISTANCE_THRESHOLD.
        """
        started_at = time.perf_counter()
        threshold = configs.SIGNIN_DISTANCE_THRESHOLD
        logger.info(f"Verifying face for user_id: {user_id} (session_id={session_id})")

        reference_embeddings, login_face_id, login_paths = self._load_data_parallel(
            user_id
        )

        if not reference_embeddings:
            logger.error(f"No registered embeddings available for user_id={user_id}")
            print(f"[VERIFY] user_id={user_id} — NO REGISTERED EMBEDDINGS")
            return False

        if not login_paths:
            return False

        scored_results = self._score_login_images(login_paths, reference_embeddings)

        if not scored_results:
            logger.error(
                f"All login embedding extractions failed for user_id={user_id}"
            )
            print(f"[VERIFY] user_id={user_id} — ALL EMBEDDING EXTRACTIONS FAILED")
            return False

        best_idx, best_dist, best_emb = min(scored_results, key=lambda x: x[1])
        is_match = best_dist < threshold
        status = "MATCH ✓" if is_match else "NOT MATCH ✗"

        if is_match and login_face_id is not None:
            self._post_match_updates(
                user_id, login_face_id, best_emb, reference_embeddings
            )

        print("=" * 60)
        print(f"  [VERIFY] User ID : {user_id}")
        print(f"  [VERIFY] Session : {session_id}")
        print(
            f"  [VERIFY] Evaluated: {len(scored_results)}/{len(login_paths)} image(s)"
        )
        print(f"  [VERIFY] Distance: {best_dist:.4f} (threshold={threshold})")
        print(f"  [VERIFY] Best img : #{best_idx + 1}")
        print(f"  [VERIFY] Result   : {status}")
        print("=" * 60)

        elapsed = time.perf_counter() - started_at
        logger.info(
            f"Face verification for user_id={user_id}: {status} "
            f"(distance={best_dist:.4f}, threshold={threshold}, elapsed={elapsed:.2f}s)"
        )
        return is_match
