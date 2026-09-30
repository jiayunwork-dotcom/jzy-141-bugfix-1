"""Pydantic request/response schemas."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SeriesCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    dates: List[str]
    values: List[float]
    period: int = 52


class SeriesOut(BaseModel):
    id: int
    name: str
    period: int
    dates: List[str]
    values: List[float]
    missing_dates: List[str] = []
    created_at: str


class FitRequest(BaseModel):
    series_id: int
    horizon: int = 12
    confidence: float = 0.95
    interval_method: str = "analytic"  # analytic | simulate
    auto: bool = True
    trend_kind: Optional[str] = None
    seasonal_kind: Optional[str] = None
    locks: Dict[str, float] = Field(default_factory=dict)
    label: str = ""


class BacktestRequest(BaseModel):
    series_id: int
    origin_start: int
    horizon: int = 8
    stride: int = 1
    confidence: float = 0.95
    interval_method: str = "analytic"
    auto: bool = True
    trend_kind: Optional[str] = None
    seasonal_kind: Optional[str] = None
    locks: Dict[str, float] = Field(default_factory=dict)
    label: str = ""


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    stage: str
    result: Optional[dict] = None
    error: Optional[str] = None
    series_id: Optional[int] = None
    created_id: Optional[int] = None
