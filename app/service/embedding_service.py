import logging

import numpy as np
from deepface import DeepFace

from app.core.config import configs

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Extract face embeddings using DeepFace (ArcFace model → 512-dim vector)."""

    def __init__(self) -> None:
        self._model_name = configs.EMBEDDING_MODEL
        self._detector_backend = configs.DETECTOR_BACKEND
        self._vector_dim = configs.VECTOR_DIMENSION
        logger.info(
            f"EmbeddingService initialized — model={self._model_name}, "
            f"detector={self._detector_backend}, dim={self._vector_dim}"
        )

    def extract_embedding(self, img_path: str) -> np.ndarray | None:
        """
        Extract a face embedding from an image file.

        Args:
            img_path: Path to the image file on disk.

        Returns:
            L2-normalized embedding vector (np.ndarray of shape (512,)),
            or None if extraction fails.
        """
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name=self._model_name,
                detector_backend=self._detector_backend,
                enforce_detection=True,
            )
            emb = np.array(result[0]["embedding"])
            # L2 normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            logger.debug(f"Extracted embedding from {img_path} — dim={emb.shape[0]}")
            return emb

        except Exception as e:
            logger.error(f"Failed to extract embedding from {img_path}: {e}")
            return None

    def compute_average_embedding(self, img_paths: list[str]) -> np.ndarray | None:
        """
        Extract embeddings from multiple images and return the L2-normalized average.

        This is used when a single pose has multiple source images — we average
        the embeddings to get a more robust representation.

        Args:
            img_paths: List of local file paths to images.

        Returns:
            Averaged & L2-normalized embedding, or None if no embeddings extracted.
        """
        embeddings = []
        for path in img_paths:
            emb = self.extract_embedding(path)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.warning("No embeddings extracted from any image")
            return None

        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        logger.info(
            f"Computed average embedding from {len(embeddings)}/{len(img_paths)} images"
        )
        return avg
