"""Application configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _database_url() -> str:
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit
    user = os.getenv("POSTGRES_USER", "planner")
    password = os.getenv("POSTGRES_PASSWORD", "planner")
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "replenishment")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


@dataclass(frozen=True)
class Settings:
    database_url: str = _database_url()
    # Default seasonal period (weekly data -> 52 weeks per year).
    default_period: int = int(os.getenv("DEFAULT_SEASONAL_PERIOD", "52"))
    # Number of worker threads used for fit/backtest jobs.
    job_workers: int = int(os.getenv("JOB_WORKERS", "4"))


settings = Settings()
