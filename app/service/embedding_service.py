import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        self._enforce_detection = configs.EMBEDDING_ENFORCE_DETECTION
        logger.info(
            f"EmbeddingService initialized — model={self._model_name}, "
            f"detector={self._detector_backend}, dim={self._vector_dim}"
        )
        if configs.EMBEDDING_WARMUP:
            self.warmup()

    def warmup(self) -> None:
        """
        Pre-load model weights once per worker process to avoid cold-start latency
        on the first real request.
        """
        try:
            DeepFace.build_model(self._model_name)
            logger.info(f"DeepFace model warmup completed: {self._model_name}")
        except Exception as e:
            logger.warning(f"Model warmup failed for {self._model_name}: {e}")

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
                enforce_detection=self._enforce_detection,
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

    @staticmethod
    def _normalize(emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def compute_average_embedding(
        self, img_paths: list[str], max_workers: int = 1
    ) -> np.ndarray | None:
        """
        Extract embeddings from multiple images and return the L2-normalized average.

        This is used when a single pose has multiple source images — we average
        the embeddings to get a more robust representation.

        Args:
            img_paths: List of local file paths to images.

        Returns:
            Averaged & L2-normalized embedding, or None if no embeddings extracted.
        """
        if not img_paths:
            return None

        workers = max(1, min(max_workers, len(img_paths)))
        embeddings: list[np.ndarray] = []

        if workers == 1:
            for path in img_paths:
                emb = self.extract_embedding(path)
                if emb is not None:
                    embeddings.append(emb)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.extract_embedding, path): path
                    for path in img_paths
                }
                for future in as_completed(futures):
                    emb = future.result()
                    if emb is not None:
                        embeddings.append(emb)

        if not embeddings:
            logger.warning("No embeddings extracted from any image")
            return None

        avg = self._normalize(np.mean(embeddings, axis=0))

        logger.info(
            f"Computed average embedding from {len(embeddings)}/{len(img_paths)} images "
            f"(workers={workers})"
        )
        return avg
