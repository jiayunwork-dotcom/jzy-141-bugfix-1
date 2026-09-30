"""ORM models: series, fitting results, backtest results."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Series(Base):
    __tablename__ = "series"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    period: Mapped[int] = mapped_column(Integer, default=52)
    dates: Mapped[list] = mapped_column(JSON, nullable=False)
    values: Mapped[list] = mapped_column(JSON, nullable=False)
    missing_dates: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    fits: Mapped[list["Fit"]] = relationship(
        back_populates="series", cascade="all, delete-orphan"
    )
    backtests: Mapped[list["Backtest"]] = relationship(
        back_populates="series", cascade="all, delete-orphan"
    )


class Fit(Base):
    __tablename__ = "fits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("series.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    label: Mapped[str] = mapped_column(String(255), default="")
    auto: Mapped[bool] = mapped_column(Boolean, default=True)
    trend_kind: Mapped[str] = mapped_column(String(20))
    seasonal_kind: Mapped[str] = mapped_column(String(20))
    period: Mapped[int] = mapped_column(Integer)
    params: Mapped[dict] = mapped_column(JSON)
    locks: Mapped[dict] = mapped_column(JSON, default=dict)
    sse: Mapped[float] = mapped_column(Float)
    aic: Mapped[float] = mapped_column(Float)
    residuals: Mapped[list] = mapped_column(JSON)
    fitted: Mapped[list] = mapped_column(JSON)
    forecast: Mapped[dict] = mapped_column(JSON)
    initial_state: Mapped[dict] = mapped_column(JSON)
    final_state: Mapped[dict] = mapped_column(JSON)
    scores: Mapped[list] = mapped_column(JSON, default=list)

    series: Mapped[Series] = relationship(back_populates="fits")


class Backtest(Base):
    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    series_id: Mapped[int] = mapped_column(
        ForeignKey("series.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    label: Mapped[str] = mapped_column(String(255), default="")
    origin_start: Mapped[int] = mapped_column(Integer)
    horizon: Mapped[int] = mapped_column(Integer)
    stride: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.95)
    interval_method: Mapped[str] = mapped_column(String(20), default="analytic")
    auto: Mapped[bool] = mapped_column(Boolean, default=True)
    trend_kind: Mapped[str | None] = mapped_column(String(20), nullable=True)
    seasonal_kind: Mapped[str | None] = mapped_column(String(20), nullable=True)
    locks: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON)

    series: Mapped[Series] = relationship(back_populates="backtests")
