import logging
import uuid
from contextlib import AbstractContextManager
from typing import Callable

from sqlalchemy.orm import Session

from app.model.user_face_model import UserFaceModel

logger = logging.getLogger(__name__)

_REGISTERED_POSES = ["left", "right", "straight"]


class UserFaceRepository:
    """Repository for tb_user_faces — all queries by user_id, no JOIN needed."""

    def __init__(self, session_factory: Callable[..., AbstractContextManager[Session]]):
        self._session_factory = session_factory
        logger.info("UserFaceRepository initialized")

    # ── read methods ──────────────────────────────────────────

    def get_faces_by_user_id(self, user_id: uuid.UUID) -> list[UserFaceModel] | None:
        """Get all face records (pose: left/right/straight) for a user."""
        logger.debug(f"Querying faces for user_id: {user_id}")
        try:
            with self._session_factory() as session:
                faces = (
                    session.query(UserFaceModel)
                    .filter(
                        UserFaceModel.user_id == user_id,
                        UserFaceModel.pose.in_(_REGISTERED_POSES),
                    )
                    .order_by(UserFaceModel.id)
                    .all()
                )
                if not faces:
                    logger.warning(f"No face records found for user_id: {user_id}")
                    return None

                logger.info(f"Found {len(faces)} face records for user_id: {user_id}")
                return faces

        except Exception as e:
            logger.error(
                f"Database error querying faces for user_id={user_id}: {e}",
                exc_info=True,
            )
            return None

    def get_registered_faces_by_user_id(
        self, user_id: uuid.UUID
    ) -> list[UserFaceModel] | None:
        """Alias for get_faces_by_user_id — semantic clarity."""
        return self.get_faces_by_user_id(user_id)

    def get_login_face_by_user_id(self, user_id: uuid.UUID) -> UserFaceModel | None:
        """Get the most recent face record with pose='login' for a user."""
        logger.debug(f"Querying login face for user_id: {user_id}")
        try:
            with self._session_factory() as session:
                face = (
                    session.query(UserFaceModel)
                    .filter(
                        UserFaceModel.user_id == user_id,
                        UserFaceModel.pose == "login",
                    )
                    .order_by(UserFaceModel.id.desc())
                    .first()
                )
                if face is None:
                    logger.warning(f"No login face record found for user_id: {user_id}")
                    return None

                logger.info(f"Found login face record for user_id: {user_id}")
                return face

        except Exception as e:
            logger.error(
                f"Database error querying login face for user_id={user_id}: {e}",
                exc_info=True,
            )
            return None

    def check_embeddings_exist(self, user_id: uuid.UUID) -> bool:
        """Check if all registered face records already have embeddings."""
        try:
            with self._session_factory() as session:
                faces = (
                    session.query(UserFaceModel.embedding)
                    .filter(
                        UserFaceModel.user_id == user_id,
                        UserFaceModel.pose.in_(_REGISTERED_POSES),
                    )
                    .all()
                )
                if not faces:
                    return False

                return all(f.embedding is not None for f in faces)

        except Exception as e:
            logger.error(
                f"Database error checking embeddings for user_id={user_id}: {e}",
                exc_info=True,
            )
            return False

    # ── write methods ─────────────────────────────────────────

    def update_embedding(self, face_id: int, embedding: list[float]) -> bool:
        """Update the embedding vector for a specific face record."""
        logger.debug(f"Updating embedding for face_id={face_id}")
        try:
            with self._session_factory() as session:
                face = (
                    session.query(UserFaceModel)
                    .filter(UserFaceModel.id == face_id)
                    .first()
                )
                if face is None:
                    logger.warning(f"Face record not found: face_id={face_id}")
                    return False

                face.embedding = embedding
                session.commit()
                logger.info(f"Updated embedding for face_id={face_id}")
                return True

        except Exception as e:
            logger.error(
                f"Database error updating embedding for face_id={face_id}: {e}",
                exc_info=True,
            )
            return False
