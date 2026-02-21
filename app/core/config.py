import os
from dataclasses import dataclass
from pathlib import Path

import yaml


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_yaml_config() -> dict:
    config_path = Path(
        os.environ.get("CONFIG_PATH")
        or Path(__file__).resolve().parents[2] / "config.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


_raw = _load_yaml_config()


@dataclass(frozen=True)
class Configs:
    PROJECT_NAME: str = _raw.get("project_name", "face_matching")

    # Database
    POSTGRES_USER: str = os.environ.get("POSTGRES_USER") or _raw.get(
        "database", {}
    ).get("user", "")
    POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD") or _raw.get(
        "database", {}
    ).get("password", "")
    POSTGRES_DB: str = os.environ.get("POSTGRES_DB") or _raw.get("database", {}).get(
        "db", ""
    )
    POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST") or _raw.get(
        "database", {}
    ).get("host", "")
    POSTGRES_PORT: int = int(
        os.environ.get("POSTGRES_PORT") or _raw.get("database", {}).get("port", 5432)
    )

    # GCP
    GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID") or _raw.get("gcp", {}).get(
        "project_id", ""
    )

    # PubSub
    PUBSUB_SIGNUP_SUBSCRIPTION: str = os.environ.get(
        "PUBSUB_SIGNUP_SUBSCRIPTION"
    ) or _raw.get("pubsub", {}).get("signup_subscription", "")
    PUBSUB_SIGNIN_SUBSCRIPTION: str = os.environ.get(
        "PUBSUB_SIGNIN_SUBSCRIPTION"
    ) or _raw.get("pubsub", {}).get("signin_subscription", "")
    PUBSUB_MAX_MESSAGES: int = int(
        os.environ.get("PUBSUB_MAX_MESSAGES")
        or _raw.get("pubsub", {}).get("max_messages", 10)
    )
    PUBSUB_ACK_DEADLINE: int = int(
        os.environ.get("PUBSUB_ACK_DEADLINE")
        or _raw.get("pubsub", {}).get("ack_deadline_seconds", 600)
    )

    # Firebase / GCS
    FIREBASE_CREDENTIALS_PATH: str = os.environ.get(
        "FIREBASE_CREDENTIALS_PATH"
    ) or _raw.get("firebase", {}).get("credentials_path", "firebase_credentials.json")
    GCS_BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME") or _raw.get(
        "firebase", {}
    ).get("bucket_name", "")
    FIREBASE_RTDB_URL: str = os.environ.get("FIREBASE_RTDB_URL") or _raw.get(
        "firebase", {}
    ).get("rtdb_url", "")

    # Embedding model
    EMBEDDING_MODEL: str = _raw.get("embedding", {}).get("model_name", "ArcFace")
    DETECTOR_BACKEND: str = _raw.get("embedding", {}).get(
        "detector_backend", "retinaface"
    )
    VECTOR_DIMENSION: int = int(_raw.get("embedding", {}).get("vector_dimension", 512))
    EMBEDDING_ENFORCE_DETECTION: bool = _as_bool(
        os.environ.get("EMBEDDING_ENFORCE_DETECTION"),
        _raw.get("embedding", {}).get("enforce_detection", True),
    )
    EMBEDDING_WARMUP: bool = _as_bool(
        os.environ.get("EMBEDDING_WARMUP"),
        _raw.get("embedding", {}).get("warmup", True),
    )
    EMBEDDING_WORKERS: int = int(
        os.environ.get("EMBEDDING_WORKERS")
        or _raw.get("embedding", {}).get("workers", 2)
    )

    # Temp directory for downloaded images
    TEMP_DIR: str = _raw.get("temp_dir", "/tmp/face_matching")
    DOWNLOAD_WORKERS: int = int(
        os.environ.get("DOWNLOAD_WORKERS") or _raw.get("download_workers", 6)
    )

    # Sign-up processing
    SIGNUP_EMBEDDING_WORKERS: int = int(
        os.environ.get("SIGNUP_EMBEDDING_WORKERS")
        or _raw.get("signup", {}).get("embedding_workers", EMBEDDING_WORKERS)
    )
    SIGNUP_MAX_IMAGES_PER_POSE: int = int(
        os.environ.get("SIGNUP_MAX_IMAGES_PER_POSE")
        or _raw.get("signup", {}).get("max_images_per_pose", 1)
    )

    # Sign-in fast path
    SIGNIN_USE_DB_EMBEDDINGS: bool = _as_bool(
        os.environ.get("SIGNIN_USE_DB_EMBEDDINGS"),
        _raw.get("signin", {}).get("use_db_embeddings", True),
    )
    SIGNIN_MAX_LOGIN_IMAGES: int = int(
        os.environ.get("SIGNIN_MAX_LOGIN_IMAGES")
        or _raw.get("signin", {}).get("max_login_images", 3)
    )
    SIGNIN_DISTANCE_THRESHOLD: float = float(
        os.environ.get("SIGNIN_DISTANCE_THRESHOLD")
        or _raw.get("signin", {}).get("distance_threshold", 0.68)
    )
    SIGNIN_ALLOW_ON_DEMAND_REGISTERED_EMBEDDINGS: bool = _as_bool(
        os.environ.get("SIGNIN_ALLOW_ON_DEMAND_REGISTERED_EMBEDDINGS"),
        _raw.get("signin", {}).get("allow_on_demand_registered_embeddings", False),
    )
    SIGNIN_PERSONALIZATION_ENABLED: bool = _as_bool(
        os.environ.get("SIGNIN_PERSONALIZATION_ENABLED"),
        _raw.get("signin", {}).get("personalization_enabled", True),
    )
    SIGNIN_PERSONALIZATION_MAX_EMBEDDINGS: int = int(
        os.environ.get("SIGNIN_PERSONALIZATION_MAX_EMBEDDINGS")
        or _raw.get("signin", {}).get("personalization_max_embeddings", 5)
    )
    SIGNIN_UPDATE_LOGIN_EMBEDDING: bool = _as_bool(
        os.environ.get("SIGNIN_UPDATE_LOGIN_EMBEDDING"),
        _raw.get("signin", {}).get("update_login_embedding", True),
    )

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            f"?sslmode=require"
        )

    @property
    def SIGNUP_SUBSCRIPTION_PATH(self) -> str:
        return f"projects/{self.GCP_PROJECT_ID}/subscriptions/{self.PUBSUB_SIGNUP_SUBSCRIPTION}"

    @property
    def SIGNIN_SUBSCRIPTION_PATH(self) -> str:
        return f"projects/{self.GCP_PROJECT_ID}/subscriptions/{self.PUBSUB_SIGNIN_SUBSCRIPTION}"


configs = Configs()
