import { describe, expect, it } from "vitest";
import {
  checkWeeklyContinuity,
  CsvParseError,
  parseCsv,
  parseDate,
  tokenizeCsv,
} from "../utils/csv";

describe("parseDate", () => {
  it("accepts common date formats", () => {
    expect(parseDate("2026-01-05")).toBe("2026-01-05");
    expect(parseDate("2026/1/5")).toBe("2026-01-05");
    expect(parseDate("20260105")).toBe("2026-01-05");
    expect(parseDate("1/5/2026")).toBe("2026-01-05");
  });

  it("rejects invalid calendar dates", () => {
    expect(() => parseDate("2026-13-01")).toThrow(CsvParseError);
    expect(() => parseDate("not-a-date")).toThrow(CsvParseError);
  });
});

describe("parseCsv", () => {
  it("parses a header + rows and strips BOM", () => {
    const text =
      "﻿周起始日期,销量\n2026-01-05,120\n2026-01-12,135.5\n";
    const { rows, hasHeader } = parseCsv(text);
    expect(hasHeader).toBe(true);
    expect(rows).toEqual([
      { date: "2026-01-05", value: 120 },
      { date: "2026-01-12", value: 135.5 },
    ]);
  });

  it("parses without a header", () => {
    const text = "2026-01-05,10\n2026-01-12,11";
    const { rows, hasHeader } = parseCsv(text);
    expect(hasHeader).toBe(false);
    expect(rows).toHaveLength(2);
  });

  it("handles quoted fields with commas", () => {
    const tokens = tokenizeCsv('"hello, world",12\n"x""y",3');
    expect(tokens).toEqual([
      ["hello, world", "12"],
      ['x"y', "3"],
    ]);
  });

  it("errors on non-numeric sales", () => {
    expect(() =>
      parseCsv("date,sales\n2026-01-05,abc")
    ).toThrow(/销量不是数字/);
  });

  it("errors with fewer than two rows", () => {
    expect(() => parseCsv("date,sales\n")).toThrow();
  });

  it("ignores trailing blank lines", () => {
    const { rows } = parseCsv(
      "date,value\n2026-01-05,1\n2026-01-12,2\n\n"
    );
    expect(rows).toHaveLength(2);
  });
});

describe("checkWeeklyContinuity", () => {
  const rows = (dates: string[]) =>
    dates.map((d) => ({ date: d, value: 1 }));

  it("reports no missing weeks for a continuous series", () => {
    const r = checkWeeklyContinuity(
      rows(["2026-01-05", "2026-01-12", "2026-01-19"])
    );
    expect(r.missingWeeks).toEqual([]);
    expect(r.irregularGap).toBe(false);
  });

  it("flags each missing week in a 3-week gap", () => {
    const r = checkWeeklyContinuity(
      rows(["2026-01-05", "2026-01-26"])
    );
    expect(r.missingWeeks).toEqual(["2026-01-12", "2026-01-19"]);
  });

  it("sorts unsorted input and still detects gaps", () => {
    const r = checkWeeklyContinuity(
      rows(["2026-01-19", "2026-01-05"])
    );
    expect(r.sorted.map((x) => x.date)).toEqual([
      "2026-01-05",
      "2026-01-19",
    ]);
    expect(r.missingWeeks).toEqual(["2026-01-12"]);
  });

  it("marks non-week-aligned gaps as irregular", () => {
    const r = checkWeeklyContinuity(
      rows(["2026-01-05", "2026-01-13"])
    );
    expect(r.irregularGap).toBe(true);
    expect(r.irregularMessage).toMatch(/不是整周/);
  });

  it("rejects duplicate week-start dates", () => {
    expect(() =>
      checkWeeklyContinuity(
        rows(["2026-01-05", "2026-01-05"])
      )
    ).toThrow(/重复/);
  });
});
