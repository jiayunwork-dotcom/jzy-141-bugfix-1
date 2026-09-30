import { useState } from "react";
import { apiClient } from "../api/client";
import type { Series } from "../api/types";
import {
  checkWeeklyContinuity,
  CsvParseError,
  parseCsv,
} from "../utils/csv";

interface Props {
  onCreated: (s: Series) => void;
}

export default function UploadPanel({ onCreated }: Props) {
  const [name, setName] = useState("");
  const [period, setPeriod] = useState(52);
  const [preview, setPreview] = useState<{
    dates: string[];
    values: number[];
    missing: string[];
    irregular: string | null;
  } | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(f: File) {
    setFile(f);
    setError(null);
    try {
      const text = await f.text();
      const { rows } = parseCsv(text);
      const cont = checkWeeklyContinuity(rows);
      setPreview({
        dates: cont.sorted.map((r) => r.date),
        values: cont.sorted.map((r) => r.value),
        missing: cont.missingWeeks,
        irregular: cont.irregularMessage,
      });
    } catch (e) {
      setPreview(null);
      if (e instanceof CsvParseError) setError(e.message);
      else setError(String(e));
    }
  }

  async function submit() {
    if (!file || !preview) return;
    setBusy(true);
    setError(null);
    try {
      const created = await apiClient.uploadSeries(
        file,
        name || file.name.replace(/\.csv$/i, ""),
        period
      );
      onCreated(created);
      setFile(null);
      setPreview(null);
      setName("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <h2>上传 CSV（两列：周起始日期、销量）</h2>
      <div className="row">
        <div className="field">
          <label>序列名称（可选）</label>
          <input
            type="text"
            value={name}
            placeholder="使用文件名"
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div className="field">
          <label>季节周期（周）</label>
          <input
            type="number"
            min={2}
            value={period}
            onChange={(e) => setPeriod(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>CSV 文件</label>
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
        </div>
        <button
          onClick={submit}
          disabled={!preview || busy || !!preview.irregular}
        >
          {busy ? "上传中…" : "导入序列"}
        </button>
      </div>

      {error && <div className="alert error" style={{ marginTop: 12 }}>{error}</div>}
      {preview && (
        <div style={{ marginTop: 12 }}>
          <div className="muted">
            解析到 {preview.dates.length} 条周数据，
            {preview.dates[0]} ~ {preview.dates[preview.dates.length - 1]}。
          </div>
          {preview.irregular && (
            <div className="alert error" style={{ marginTop: 8 }}>
              {preview.irregular} 日期不是严格周对齐，已阻止导入，请修正 CSV。
            </div>
          )}
          {preview.missing.length > 0 && (
            <div className="alert warn" style={{ marginTop: 8 }}>
              检测到 {preview.missing.length} 个缺周：
              {preview.missing.slice(0, 8).join("、")}
              {preview.missing.length > 8 ? " 等" : ""}
              。序列仍可导入并在列表中标记，但请确认是否需要补数。
            </div>
          )}
        </div>
      )}
    </div>
  );
}
