import type { Job } from "../api/types";

export default function JobProgress({ job }: { job: Job | null }) {
  if (!job) return null;
  const pct = Math.round((job.status === "done" ? 1 : job.progress) * 100);
  const label =
    job.status === "done"
      ? "完成"
      : job.status === "error"
      ? "失败"
      : job.status === "pending"
      ? "排队中"
      : `计算中 · ${job.stage}（${pct}%）`;
  return (
    <div className="panel">
      <h2>后台任务</h2>
      <div className="progress">
        <div style={{ width: `${pct}%` }} />
      </div>
      <div style={{ marginTop: 8 }} className="muted">
        {label}
      </div>
      {job.status === "error" && (
        <div className="alert error" style={{ marginTop: 10 }}>
          {job.error}
        </div>
      )}
    </div>
  );
}
