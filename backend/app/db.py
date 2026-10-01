"""SQLAlchemy engine / session management."""
from __future__ import annotations

import time
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

ENGINE_VERSION = "2.0-calendar-grid"
LEGACY_ENGINE_VERSION = "1.0-observed-array"


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def wait_for_db(retries: int = 30, delay: float = 1.0) -> None:
    for i in range(retries):
        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            return
        except OperationalError:
            if i == retries - 1:
                raise
            time.sleep(delay)


def _column_names(conn, table: str):
    return {c["name"] for c in inspect(conn).get_columns(table)}


def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    if column not in _column_names(conn, table):
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))


def migrate() -> None:
    """Idempotent startup migration.

    New columns are added to pre-existing databases, and every result row
    written by the old observed-array kernel is stamped
    ``1.0-observed-array`` so the UI can flag it as gap-unsafe.  Rows created
    by the new kernel carry ``2.0-calendar-grid``.
    """
    with engine.begin() as conn:
        existing = set(inspect(conn).get_table_names())
        if "fits" in existing:
            # engine_version is added WITHOUT a default so pre-existing rows
            # stay NULL and can be reliably backfilled as legacy; the ORM
            # (and freshly created tables) supplies the new default itself.
            _ensure_column(conn, "fits", "engine_version",
                           "engine_version VARCHAR(32)")
            _ensure_column(conn, "fits", "n_effective",
                           "n_effective INTEGER DEFAULT 0")
            _ensure_column(conn, "fits", "n_calendar",
                           "n_calendar INTEGER DEFAULT 0")
            _ensure_column(conn, "fits", "missing_count",
                           "missing_count INTEGER DEFAULT 0")
            _ensure_column(conn, "fits", "grid_dates",
                           "grid_dates JSON DEFAULT '[]'")
            conn.execute(text(
                "UPDATE fits SET engine_version = :legacy "
                "WHERE engine_version IS NULL OR engine_version = ''"
            ), {"legacy": LEGACY_ENGINE_VERSION})
        if "backtests" in existing:
            _ensure_column(conn, "backtests", "engine_version",
                           "engine_version VARCHAR(32)")
            conn.execute(text(
                "UPDATE backtests SET engine_version = :legacy "
                "WHERE engine_version IS NULL OR engine_version = ''"
            ), {"legacy": LEGACY_ENGINE_VERSION})


def init_db() -> None:
    from . import models  # noqa: F401  ensure metadata populated
    Base.metadata.create_all(bind=engine)
    migrate()


@contextmanager
def session_scope():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
