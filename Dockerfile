# ------------------------------------------------------------------
# Stage 1 – Build dependencies with uv
# ------------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code and install the project itself
COPY . .
RUN uv sync --frozen --no-dev

# ------------------------------------------------------------------
# Stage 2 – Minimal runtime image
# ------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Runtime system dependencies:
#   libpq5      - required by psycopg2-binary
#   libssl3     - required by ctypes OpenSSL preload (fix TF/libpq SSL conflict)
#   libgomp1    - OpenMP, required by TensorFlow / numpy operations
#   libgl1      - OpenCV (used internally by DeepFace)
#   libglib2.0-0 - GLib, required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
   libpq5 \
   libssl3 \
   libgomp1 \
   libgl1 \
   libglib2.0-0 \
   && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment and app source
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app ./app

# Config and Firebase credentials are mounted at runtime via K8s secret volume
# Local dev:
#   docker run -e CONFIG_PATH=... -e FIREBASE_CREDENTIALS_PATH=... \
#              -v $(pwd)/config.yaml:/app/secrets/config.yaml \
#              -v $(pwd)/firebase_credentials.json:/app/secrets/firebase_credentials.json \
#              face-matching
ENV CONFIG_PATH="/app/secrets/config.yaml"
ENV FIREBASE_CREDENTIALS_PATH="/app/secrets/firebase_credentials.json"
ENV PATH="/app/.venv/bin:$PATH"

# PubSub worker — no HTTP port exposed
CMD ["python", "-m", "app.main", "worker"]
