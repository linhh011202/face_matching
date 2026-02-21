import logging
import time
import uuid

import numpy as np

from app.core.config import configs
from app.repository.user_face_repository import UserFaceRepository
from app.service.embedding_service import EmbeddingService
from app.service.storage_service import StorageService

logger = logging.getLogger(__name__)


class FaceVerificationService:
    """
    Fast sign-in verification using ArcFace embeddings.

    Primary path:
      1. Read precomputed registered embeddings from DB.
      2. Download most recent login image(s).
      3. Extract login embeddings and compare with cosine distance.

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
        logger.info("FaceVerificationService initialized (fast DB-embedding mode)")

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

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

        max_images = max(1, configs.SIGNIN_MAX_LOGIN_IMAGES)
        selected_urls = source_images[-max_images:]

        paths = self._storage_svc.download_images(
            selected_urls, max_workers=configs.DOWNLOAD_WORKERS
        )
        if not paths:
            logger.error(f"Failed to download login images for user_id: {user_id}")
            print(f"[VERIFY] user_id={user_id} — LOGIN IMAGE DOWNLOAD FAILED")
            return login_face.id, []
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

    def _get_reference_embeddings(self, user_id: uuid.UUID) -> list[np.ndarray]:
        """
        Load reference embeddings for matching.
        Includes:
          - registered embeddings (mandatory)
          - recent successful login embeddings (optional personalization)
        """
        registered_embeddings: list[np.ndarray] = []
        if configs.SIGNIN_USE_DB_EMBEDDINGS:
            registered_embeddings = (
                self._face_repo.get_registered_embeddings_by_user_id(user_id)
            )

        if (
            not registered_embeddings
            and configs.SIGNIN_ALLOW_ON_DEMAND_REGISTERED_EMBEDDINGS
        ):
            logger.warning(
                f"No precomputed registered embeddings for user_id={user_id}; "
                "running on-demand extraction fallback"
            )
            registered_embeddings = self._extract_registered_embeddings_on_demand(
                user_id
            )

        if not registered_embeddings:
            return []

        if not configs.SIGNIN_PERSONALIZATION_ENABLED:
            return registered_embeddings

        login_history_embeddings = (
            self._face_repo.get_recent_login_embeddings_by_user_id(
                user_id=user_id,
                limit=configs.SIGNIN_PERSONALIZATION_MAX_EMBEDDINGS,
            )
        )
        refs = registered_embeddings + login_history_embeddings
        logger.info(
            f"Loaded references for user_id={user_id}: "
            f"registered={len(registered_embeddings)}, "
            f"personalized={len(login_history_embeddings)}, total={len(refs)}"
        )
        return refs

    def verify_user(self, user_id: uuid.UUID, session_id: str | None = None) -> bool:
        """
        Verify user by comparing login embedding(s) to registered embeddings.
        Returns True when best distance is below SIGNIN_DISTANCE_THRESHOLD.
        """
        started_at = time.perf_counter()
        threshold = configs.SIGNIN_DISTANCE_THRESHOLD
        logger.info(f"Verifying face for user_id: {user_id} (session_id={session_id})")

        reference_embeddings = self._get_reference_embeddings(user_id)
        if not reference_embeddings:
            logger.error(f"No registered embeddings available for user_id={user_id}")
            print(f"[VERIFY] user_id={user_id} — NO REGISTERED EMBEDDINGS")
            return False

        login_face_id, login_paths = self._download_login_images(user_id)
        if not login_paths:
            return False

        scored_results: list[tuple[int, float, np.ndarray]] = []
        try:
            for idx, path in enumerate(login_paths):
                emb = self._embedding_svc.extract_embedding(path)
                if emb is None:
                    continue
                best_dist = min(
                    self._cosine_distance(ref_emb, emb)
                    for ref_emb in reference_embeddings
                )
                scored_results.append((idx, best_dist, emb))
        finally:
            self._storage_svc.cleanup_files(login_paths)

        if not scored_results:
            logger.error(
                f"All login embedding extractions failed for user_id={user_id}"
            )
            print(f"[VERIFY] user_id={user_id} — ALL EMBEDDING EXTRACTIONS FAILED")
            return False

        best_idx, best_dist, best_emb = min(scored_results, key=lambda x: x[1])
        is_match = best_dist < threshold
        status = "MATCH ✓" if is_match else "NOT MATCH ✗"

        # Online personalization update (cheap, non-blocking accuracy boost for future logins).
        if (
            is_match
            and configs.SIGNIN_UPDATE_LOGIN_EMBEDDING
            and login_face_id is not None
        ):
            updated = self._face_repo.update_embedding(login_face_id, best_emb.tolist())
            if updated:
                logger.info(
                    f"Updated login embedding for personalization: "
                    f"user_id={user_id}, face_id={login_face_id}"
                )

        print("=" * 60)
        print(f"  [VERIFY] User ID : {user_id}")
        print(f"  [VERIFY] Session : {session_id}")
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
