"""
Face Matching
=============
Two modes of operation:

1. PubSub Worker (default):
   Subscribes to 'banking-ekyc-sign-up' topic. When identity_service publishes
   a 'user_ekyc_completed' event → downloads images → extracts 512-dim ArcFace
   embeddings → stores into tb_user_faces.embedding (pgvector).

2. Local Matching (Enhanced Ensemble):
   Dual-Model ensemble (ArcFace + Facenet512) + SGDClassifier.
   Build references from local images, train classifier, test matching.

Usage:
    python -m app.main                             # Start PubSub subscriber worker
    python -m app.main worker                      # Same — start subscriber
    python -m app.main process <email>             # Manually process a single user
    python -m app.main match                       # Local ensemble matching
    python -m app.main retrain                     # Retrain local model from scratch
    python -m app.main enroll img1.jpg img2.jpg    # Add images + online update
"""

# ── CRITICAL: Pre-load system OpenSSL ─────────────────────────
# TensorFlow bundles its own OpenSSL which conflicts with libpq's
# (used by psycopg2). Loading the system OpenSSL into the global
# symbol table FIRST ensures libpq always uses the correct version,
# preventing segfaults when making SSL connections after TF import.
import ctypes
import ctypes.util

_crypto_path = ctypes.util.find_library("crypto")
_ssl_path = ctypes.util.find_library("ssl")
if _crypto_path:
    ctypes.CDLL(_crypto_path, mode=ctypes.RTLD_GLOBAL)
if _ssl_path:
    ctypes.CDLL(_ssl_path, mode=ctypes.RTLD_GLOBAL)
# ──────────────────────────────────────────────────────────────

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402

from app.core.config import configs  # noqa: E402
from app.core.database import Database  # noqa: E402
from app.repository.user_face_repository import UserFaceRepository  # noqa: E402
from app.service.embedding_service import EmbeddingService  # noqa: E402
from app.service.storage_service import StorageService  # noqa: E402
from app.service.face_processing_service import FaceProcessingService  # noqa: E402
from app.service.matching_service import (  # noqa: E402
    MODELS,
    CLASSIFIER_WEIGHT,
    ensure_dirs,
    cmd_match,
    cmd_retrain,
    cmd_enroll,
)
from app.subscriber.pubsub_subscriber import PubSubSubscriber  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── PubSub Worker Commands ───────────────────────────────────


def _build_services() -> tuple[FaceProcessingService, Database]:
    """Wire up all dependencies for PubSub worker."""
    db = Database(db_url=configs.DATABASE_URL)
    repo = UserFaceRepository(session_factory=db.session)
    embedding_svc = EmbeddingService()
    storage_svc = StorageService()
    face_processing_svc = FaceProcessingService(
        user_face_repository=repo,
        embedding_service=embedding_svc,
        storage_service=storage_svc,
    )
    return face_processing_svc, db


def cmd_worker():
    """Start PubSub subscriber — listens for eKYC completed events."""
    face_processing_svc, _ = _build_services()
    subscriber = PubSubSubscriber(face_processing_service=face_processing_svc)
    subscriber.start()


def cmd_process(email: str):
    """Manually trigger face processing for a user by email."""
    face_processing_svc, _ = _build_services()
    logger.info(f"Manually processing user: {email}")
    success = face_processing_svc.process_user(email)
    if success:
        logger.info(f"Successfully processed embeddings for: {email}")
    else:
        logger.error(f"Failed to process embeddings for: {email}")
    return success


# ── CLI ──────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face Matching — PubSub worker & local ensemble matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.main                             # Start PubSub subscriber worker
  python -m app.main worker                      # Same as above
  python -m app.main process user@email.com      # Manually process a single user
  python -m app.main match                       # Local ensemble matching
  python -m app.main retrain                     # Retrain local model from scratch
  python -m app.main enroll img1.jpg img2.jpg    # Add images + online update
""",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("worker", help="Start PubSub subscriber (default)")

    process_p = sub.add_parser("process", help="Manually process a user by email")
    process_p.add_argument("email", help="User email to process")

    sub.add_parser("match", help="Local ensemble matching (ArcFace + Facenet512)")
    sub.add_parser("retrain", help="Retrain local model from scratch")

    enroll_p = sub.add_parser(
        "enroll", help="Add images + incremental update classifier"
    )
    enroll_p.add_argument(
        "images", nargs="+", help="Image paths to add to reference store"
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    command = args.command or "worker"

    # Local matching commands
    if command in ("match", "retrain", "enroll"):
        weights_str = " + ".join(f"{v['weight']}×{k}" for k, v in MODELS.items())
        print("=" * 60)
        print("FACE MATCHING — Enhanced Ensemble")
        print(f"  Models: {', '.join(MODELS.keys())} + Personalized Classifier")
        print(f"  Weights: {weights_str} + {CLASSIFIER_WEIGHT}×Classifier")
        print("=" * 60)

        ensure_dirs()

        if command == "match":
            cmd_match()
        elif command == "retrain":
            cmd_retrain()
        elif command == "enroll":
            cmd_enroll(args.images)

        print(f"\n{'=' * 60}")
        print("Hoàn tất!")
        return

    # PubSub worker commands
    print("=" * 60)
    print(f"  Face Matching Worker — {configs.PROJECT_NAME}")
    print(f"  Model: {configs.EMBEDDING_MODEL} ({configs.VECTOR_DIMENSION}-dim)")
    print(f"  Database: {configs.POSTGRES_HOST}/{configs.POSTGRES_DB}")
    print("=" * 60)

    if command == "worker":
        cmd_worker()
    elif command == "process":
        cmd_process(args.email)


if __name__ == "__main__":
    main()
