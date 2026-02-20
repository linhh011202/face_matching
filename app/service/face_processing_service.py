import logging
import uuid

from app.repository.user_face_repository import UserFaceRepository
from app.service.embedding_service import EmbeddingService
from app.service.storage_service import StorageService

logger = logging.getLogger(__name__)


class FaceProcessingService:
    """
    Orchestrate the face processing pipeline:
      1. Query DB for user's face records (source_images per pose)
      2. Download images from Firebase Storage
      3. Extract embeddings using DeepFace (ArcFace)
      4. Store the averaged embedding vector back into tb_user_faces
    """

    def __init__(
        self,
        user_face_repository: UserFaceRepository,
        embedding_service: EmbeddingService,
        storage_service: StorageService,
    ) -> None:
        self._face_repo = user_face_repository
        self._embedding_svc = embedding_service
        self._storage_svc = storage_service
        logger.info("FaceProcessingService initialized")

    def process_user(self, user_id: uuid.UUID) -> bool:
        """
        Process all face records for a user — download images, extract
        embeddings, and store them in the database.

        Args:
            user_id: User's UUID (from PubSub message).

        Returns:
            True if at least one embedding was stored successfully.
        """
        logger.info(f"Processing face embeddings for user_id: {user_id}")

        # Check if embeddings already exist
        if self._face_repo.check_embeddings_exist(user_id):
            logger.info(f"Embeddings already exist for user_id: {user_id}, skipping")
            return True

        # Get face records from DB
        faces = self._face_repo.get_faces_by_user_id(user_id)
        if not faces:
            logger.warning(f"No face records found for user_id: {user_id}")
            return False

        success_count = 0
        total_downloaded = []

        for face in faces:
            face_id = face.id
            pose = face.pose
            source_images = face.source_images or []

            if not source_images:
                logger.warning(f"No source images for face_id={face_id} pose={pose}")
                continue

            logger.info(
                f"Processing face_id={face_id} pose={pose} — "
                f"{len(source_images)} source image(s)"
            )

            # Step 1: Download images from Firebase Storage
            local_paths = self._storage_svc.download_images(source_images)
            total_downloaded.extend(local_paths)

            if not local_paths:
                logger.error(
                    f"Failed to download any images for face_id={face_id} pose={pose}"
                )
                continue

            # Step 2: Extract average embedding
            embedding = self._embedding_svc.compute_average_embedding(local_paths)
            if embedding is None:
                logger.error(
                    f"Failed to extract embedding for face_id={face_id} pose={pose}"
                )
                continue

            # Step 3: Store embedding in DB
            embedding_list = embedding.tolist()
            if self._face_repo.update_embedding(face_id, embedding_list):
                success_count += 1
                logger.info(
                    f"Stored embedding for face_id={face_id} pose={pose} "
                    f"(dim={len(embedding_list)})"
                )
            else:
                logger.error(
                    f"Failed to store embedding for face_id={face_id} pose={pose}"
                )

        # Cleanup temp files
        self._storage_svc.cleanup_files(total_downloaded)

        logger.info(
            f"Completed processing for user_id {user_id}: "
            f"{success_count}/{len(faces)} embeddings stored"
        )
        return success_count > 0
