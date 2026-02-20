import os
from dataclasses import dataclass
from pathlib import Path

import yaml


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

    # Temp directory for downloaded images
    TEMP_DIR: str = _raw.get("temp_dir", "/tmp/face_matching")

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
