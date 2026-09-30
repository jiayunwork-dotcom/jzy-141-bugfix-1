/**
 * CSV parsing and weekly-continuity checks.
 * Kept dependency-free so the same rules are easy to unit-test.
 */

export interface SeriesRow {
  date: string; // ISO yyyy-mm-dd
  value: number;
}

export interface ParseResult {
  rows: SeriesRow[];
  hasHeader: boolean;
}

export class CsvParseError extends Error {}

const DATE_KEYS = new Set([
  "date",
  "week",
  "week_start",
  "week_start_date",
  "周",
  "日期",
  "周起始日期",
]);
const VALUE_KEYS = new Set([
  "sales",
  "value",
  "y",
  "qty",
  "quantity",
  "units",
  "销量",
]);

/** Parse a loose ISO-ish date to yyyy-mm-dd; throws CsvParseError. */
export function parseDate(raw: string): string {
  const s = raw.trim();
  const patterns: Array<[RegExp, (m: RegExpMatchArray) => string]> = [
    [/^(\d{4})-(\d{1,2})-(\d{1,2})$/, (m) => pad(m[1], m[2], m[3])],
    [/^(\d{4})\/(\d{1,2})\/(\d{1,2})$/, (m) => pad(m[1], m[2], m[3])],
    [/^(\d{4})\.(\d{1,2})\.(\d{1,2})$/, (m) => pad(m[1], m[2], m[3])],
    [/^(\d{4})(\d{2})(\d{2})$/, (m) => pad(m[1], m[2], m[3])],
    [/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/, (m) => pad(m[3], m[1], m[2])],
  ];
  for (const [re, build] of patterns) {
    const m = s.match(re);
    if (m) {
      const iso = build(m);
      if (isValidIsoDate(iso)) return iso;
    }
  }
  throw new CsvParseError(`无法解析日期："${raw}"`);
}

function pad(y: string, mo: string, d: string): string {
  const yi = Number(y);
  const mi = Number(mo);
  const di = Number(d);
  return `${String(yi).padStart(4, "0")}-${String(mi).padStart(2, "0")}-${String(
    di
  ).padStart(2, "0")}`;
}

export function isValidIsoDate(iso: string): boolean {
  const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return false;
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  return (
    d.getUTCFullYear() === Number(m[1]) &&
    d.getUTCMonth() === Number(m[2]) - 1 &&
    d.getUTCDate() === Number(m[3])
  );
}

/** Minimal RFC-4180-ish CSV line tokeniser. */
export function tokenizeCsv(text: string): string[][] {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let inQuotes = false;
  const src = text.replace(/^﻿/, "");
  for (let i = 0; i < src.length; i++) {
    const ch = src[i];
    if (inQuotes) {
      if (ch === '"') {
        if (src[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && src[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.some((c) => c.trim() !== ""));
}

export function parseCsv(text: string): ParseResult {
  const rows = tokenizeCsv(text.replace(/^﻿/, ""));
  if (rows.length < 2) {
    throw new CsvParseError("CSV 至少需要表头和一行数据。");
  }
  const header = rows[0].map((c) => c.trim().toLowerCase());
  const hasHeader =
    header.length >= 2 &&
    (DATE_KEYS.has(header[0]) || DATE_KEYS.has(header[1]));

  const dataRows = hasHeader ? rows.slice(1) : rows;
  if (dataRows.length === 0) {
    throw new CsvParseError("CSV 没有数据行。");
  }

  const parsed: SeriesRow[] = [];
  dataRows.forEach((cells, idx) => {
    if (cells.length < 2) {
      throw new CsvParseError(`第 ${idx + 1} 数据行列数不足（需要两列）。`);
    }
    const iso = parseDate(cells[0]);
    const value = Number(cells[1].trim());
    if (!Number.isFinite(value)) {
      throw new CsvParseError(
        `第 ${idx + 1} 行销量不是数字："${cells[1]}"`
      );
    }
    parsed.push({ date: iso, value });
  });

  if (parsed.length === 0) {
    throw new CsvParseError("CSV 没有有效数据。");
  }
  return { rows: parsed, hasHeader };
}

const MS_WEEK = 7 * 24 * 3600 * 1000;

function toUtc(iso: string): number {
  return Date.parse(iso + "T00:00:00Z");
}

function addWeeks(iso: string, k: number): string {
  return new Date(toUtc(iso) + k * MS_WEEK).toISOString().slice(0, 10);
}

export interface ContinuityResult {
  /** Sorted rows; duplicates are rejected by the caller if needed. */
  sorted: SeriesRow[];
  /** Missing week-start dates (in ascending order). */
  missingWeeks: string[];
  /** True when any adjacent gap is not a whole number of weeks. */
  irregularGap: boolean;
  irregularMessage: string | null;
}

/**
 * Examine weekly continuity of the given rows.
 * - duplicate week-start dates are an error.
 * - gaps must be whole weeks; otherwise irregularGap is set.
 * - missing weeks inside a gap are enumerated.
 */
export function checkWeeklyContinuity(rows: SeriesRow[]): ContinuityResult {
  const sorted = [...rows].sort((a, b) => a.date.localeCompare(b.date));
  const seen = new Set<string>();
  for (const r of sorted) {
    if (seen.has(r.date)) {
      throw new CsvParseError(`检测到重复周起始日期：${r.date}`);
    }
    seen.add(r.date);
  }
  const missingWeeks: string[] = [];
  let irregularGap = false;
  let irregularMessage: string | null = null;

  for (let i = 1; i < sorted.length; i++) {
    const diffMs = toUtc(sorted[i].date) - toUtc(sorted[i - 1].date);
    const diffDays = Math.round(diffMs / (24 * 3600 * 1000));
    if (diffDays === 7) continue;
    if (diffDays % 7 !== 0 || diffDays < 7) {
      irregularGap = true;
      irregularMessage = `相邻日期 ${sorted[i - 1].date} 与 ${sorted[i].date} 相差 ${diffDays} 天，不是整周。`;
      continue;
    }
    let cur = addWeeks(sorted[i - 1].date, 1);
    while (cur < sorted[i].date) {
      missingWeeks.push(cur);
      cur = addWeeks(cur, 1);
    }
  }

  return { sorted, missingWeeks, irregularGap, irregularMessage };
}
