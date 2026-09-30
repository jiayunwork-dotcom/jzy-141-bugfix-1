"""Weekly date / CSV helpers shared by routers and tests."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, List, Sequence, Tuple


class SeriesValidationError(ValueError):
    pass


@dataclass
class ParsedSeries:
    dates: List[date]
    values: List[float]


EXPECTED_HEADERS = ("周起始日期", "销量")
HEADER_ALIASES = {
    "date": "date", "week": "date", "日期": "date", "周": "date",
    "周起始日期": "date", "week_start": "date", "week_start_date": "date",
    "sales": "value", "value": "value", "y": "value", "销量": "value",
    "qty": "value", "quantity": "value", "units": "value",
}


def _parse_date(s: str) -> date:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
                "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # ISO fallback
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise SeriesValidationError(f"无法解析日期：{s!r}") from exc


def parse_csv(content: str) -> ParsedSeries:
    """Parse a two-column CSV (week start date, sales)."""
    text = content.lstrip("﻿")
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise SeriesValidationError("CSV 至少需要表头和一行数据。")

    header = [c.strip().lower() for c in rows[0]]
    has_header = any(HEADER_ALIASES.get(c) == "date" for c in header)
    data_rows = rows[1:] if has_header else rows
    if len(data_rows[0]) < 2:
        raise SeriesValidationError("CSV 需要两列：周起始日期、销量。")

    parsed: List[Tuple[date, float]] = []
    for i, row in enumerate(data_rows):
        if len(row) < 2:
            raise SeriesValidationError(f"第 {i+1} 数据行列数不足。")
        d = _parse_date(row[0])
        try:
            v = float(row[1])
        except ValueError as exc:
            raise SeriesValidationError(
                f"第 {i+1} 行销量不是数字：{row[1]!r}"
            ) from exc
        parsed.append((d, v))
    return ParsedSeries([p[0] for p in parsed], [p[1] for p in parsed])


def check_weekly_continuity(dates: Sequence[date]) -> List[date]:
    """Return the list of missing week-start dates (7-day step assumed)."""
    if len(dates) < 2:
        return []
    ordered = sorted(dates)
    missing: List[date] = []
    for prev, cur in zip(ordered, ordered[1:]):
        gap = (cur - prev).days
        if gap == 7:
            continue
        if gap % 7 != 0 or gap < 7:
            raise SeriesValidationError(
                f"相邻日期间隔不是整周：{prev} 与 {cur} 相差 {gap} 天。"
            )
        step = prev + timedelta(days=7)
        while step < cur:
            missing.append(step)
            step += timedelta(days=7)
    return missing


def validate_series_rows(dates: Iterable[date], values: Iterable[float]
                         ) -> Tuple[List[date], List[float], List[date]]:
    dates = list(dates)
    values = list(values)
    if len(dates) != len(values):
        raise SeriesValidationError("日期与销量数量不一致。")
    pairs = sorted(zip(dates, values), key=lambda p: p[0])
    dates_s = [p[0] for p in pairs]
    values_s = [p[1] for p in pairs]
    missing = check_weekly_continuity(dates_s)
    return dates_s, values_s, missing
