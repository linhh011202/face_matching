import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import firebase_admin
from firebase_admin import credentials, storage

from app.core.config import configs

logger = logging.getLogger(__name__)

_firebase_app: firebase_admin.App | None = None


def _get_firebase_app() -> firebase_admin.App:
    """Initialize Firebase Admin SDK (singleton)."""
    global _firebase_app
    if _firebase_app is None:
        cred_path = Path(configs.FIREBASE_CREDENTIALS_PATH)
        if not cred_path.is_absolute():
            cred_path = Path(__file__).resolve().parents[2] / cred_path
        cred = credentials.Certificate(str(cred_path))
        _firebase_app = firebase_admin.initialize_app(
            cred,
            {
                "storageBucket": configs.GCS_BUCKET_NAME,
                "databaseURL": configs.FIREBASE_RTDB_URL,
            },
        )
        logger.info("Firebase Admin app initialized")
    return _firebase_app


class StorageService:
    """Download images from Firebase Storage (GCS) to local temp files."""

    def __init__(self) -> None:
        self._temp_dir = configs.TEMP_DIR
        os.makedirs(self._temp_dir, exist_ok=True)
        logger.info(f"StorageService initialized — temp_dir={self._temp_dir}")

    def _get_bucket(self):
        _get_firebase_app()
        return storage.bucket()

    @staticmethod
    def _extract_blob_path(url: str) -> str | None:
        """
        Extract the GCS blob path from a Firebase Storage public URL.

        Handles URLs like:
          https://storage.googleapis.com/bucket-name/uploads/session/file.jpg
          https://firebasestorage.googleapis.com/v0/b/bucket/o/path%2Fto%2Ffile?...
        """
        parsed = urlparse(url)

        # Format: https://storage.googleapis.com/BUCKET/OBJECT_PATH
        if "storage.googleapis.com" in parsed.hostname:
            # Path is /BUCKET/OBJECT_PATH — strip the bucket prefix
            parts = parsed.path.lstrip("/").split("/", 1)
            if len(parts) >= 2:
                return unquote(parts[1])

        # Format: https://firebasestorage.googleapis.com/v0/b/BUCKET/o/ENCODED_PATH
        if "firebasestorage.googleapis.com" in parsed.hostname:
            path = parsed.path
            if "/o/" in path:
                obj_path = path.split("/o/", 1)[1]
                return unquote(obj_path)

        logger.warning(f"Could not extract blob path from URL: {url}")
        return None

    def download_image(self, image_url: str) -> str | None:
        """
        Download an image from Firebase Storage to a local temp file.

        Args:
            image_url: Public URL of the image in Firebase Storage.

        Returns:
            Local file path of the downloaded image, or None on failure.
        """
        blob_path = self._extract_blob_path(image_url)
        if not blob_path:
            logger.error(f"Cannot parse blob path from URL: {image_url}")
            return None

        try:
            bucket = self._get_bucket()
            blob = bucket.blob(blob_path)

            # Determine extension from blob path
            ext = Path(blob_path).suffix or ".jpg"
            fd, local_path = tempfile.mkstemp(suffix=ext, dir=self._temp_dir)
            os.close(fd)

            blob.download_to_filename(local_path)
            logger.debug(f"Downloaded {blob_path} → {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download image {image_url}: {e}")
            return None

    def download_images(self, image_urls: list[str]) -> list[str]:
        """
        Download multiple images. Returns list of local file paths
        (only successfully downloaded images).
        """
        local_paths = []
        for url in image_urls:
            path = self.download_image(url)
            if path:
                local_paths.append(path)
        logger.info(f"Downloaded {len(local_paths)}/{len(image_urls)} images")
        return local_paths

    @staticmethod
    def cleanup_files(file_paths: list[str]) -> None:
        """Remove temporary downloaded files."""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                logger.warning(f"Failed to cleanup {path}: {e}")
