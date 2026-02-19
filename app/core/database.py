import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, orm
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_url: str) -> None:
        # Neon serverless uses a PgBouncer pooler (*-pooler.*.neon.tech),
        # so we use NullPool to avoid double-pooling and stale connections.
        self._engine = create_engine(
            db_url,
            echo=False,
            poolclass=NullPool,  # Let Neon handle pooling
            connect_args={
                "connect_timeout": 15,
                # TCP keepalive — detects dead connections (Neon cold start / drop)
                "keepalives": 1,
                "keepalives_idle": 10,  # Start probes after 10s idle
                "keepalives_interval": 5,  # Probe every 5s
                "keepalives_count": 3,  # Drop after 3 failed probes (25s max)
            },
        )

        # Neon pooler rejects session params in connect 'options',
        # so we set statement_timeout after connection is established.
        @event.listens_for(self._engine, "connect")
        def _on_connect(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("SET statement_timeout = '30s'")
            cursor.close()

        self._session_factory = orm.scoped_session(
            orm.sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine,
            ),
        )

    @property
    def engine(self):
        return self._engine

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        session: Session = self._session_factory()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
            self._session_factory.remove()
