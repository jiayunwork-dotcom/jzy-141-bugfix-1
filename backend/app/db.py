"""SQLAlchemy engine / session management."""
from __future__ import annotations

import time
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


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


def init_db() -> None:
    from . import models  # noqa: F401  ensure metadata populated
    Base.metadata.create_all(bind=engine)
    # Lightweight idempotent migration for databases created before
    # gap-aware dense-grid support (create_all only makes missing tables).
    _add_columns_if_missing(engine)


def _add_columns_if_missing(sync_engine) -> None:
    from sqlalchemy import inspect as sa_inspect

    additions = {
        "fits": {
            "n_effective": "INTEGER",
            "grid_dates": "JSON",
            "missing_indices": "JSON",
            "results_version": "INTEGER",
        },
    }
    inspector = sa_inspect(sync_engine)
    existing_tables = set(inspector.get_table_names())
    with sync_engine.begin() as conn:
        for table, columns in additions.items():
            if table not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table)}
            for name, coltype in columns.items():
                if name in present:
                    continue
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN {name} {coltype}"
                )


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
