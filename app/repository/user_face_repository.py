import logging
from contextlib import AbstractContextManager
from typing import Callable

from sqlalchemy.orm import Session

from app.model.user_model import UserModel
from app.model.user_face_model import UserFaceModel

logger = logging.getLogger(__name__)


class UserFaceRepository:
    """Repository for tb_user_faces — query faces by email, update embeddings."""

    def __init__(self, session_factory: Callable[..., AbstractContextManager[Session]]):
        self._session_factory = session_factory
        logger.info("UserFaceRepository initialized")

    def get_faces_by_email(self, email: str) -> list[UserFaceModel] | None:
        """
        Get all face records (pose: left/right/straight) for a user by email.
        Uses a single JOIN query instead of two separate queries.
        Returns None if user not found or no faces.
        """
        logger.debug(f"Querying faces for user: {email}")
        try:
            with self._session_factory() as session:
                faces = (
                    session.query(UserFaceModel)
                    .join(UserModel, UserFaceModel.user_id == UserModel.id)
                    .filter(
                        UserModel.email == email,
                        UserFaceModel.pose.in_(["left", "right", "straight"]),
                    )
                    .order_by(UserFaceModel.id)
                    .all()
                )

                if not faces:
                    logger.warning(f"No face records found for user: {email}")
                    return None

                logger.info(f"Found {len(faces)} face records for user: {email}")
                return faces

        except Exception as e:
            logger.error(
                f"Database error querying faces for '{email}': {e}", exc_info=True
            )
            return None

    def update_embedding(self, face_id: int, embedding: list[float]) -> bool:
        """
        Update the embedding vector for a specific face record.

        Args:
            face_id: The ID of the tb_user_faces row.
            embedding: The 512-dim embedding vector as a list of floats.

        Returns:
            True if updated successfully, False otherwise.
        """
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

    def check_embeddings_exist(self, email: str) -> bool:
        """Check if all face records for a user already have embeddings.
        Uses a single JOIN query instead of two separate queries.
        """
        try:
            with self._session_factory() as session:
                faces = (
                    session.query(UserFaceModel.embedding)
                    .join(UserModel, UserFaceModel.user_id == UserModel.id)
                    .filter(
                        UserModel.email == email,
                        UserFaceModel.pose.in_(["left", "right", "straight"]),
                    )
                    .all()
                )

                if not faces:
                    return False

                return all(f.embedding is not None for f in faces)

        except Exception as e:
            logger.error(
                f"Database error checking embeddings for '{email}': {e}", exc_info=True
            )
            return False
