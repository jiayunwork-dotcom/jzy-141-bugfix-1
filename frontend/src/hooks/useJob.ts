import { useEffect, useRef, useState } from "react";
import { apiClient } from "../api/client";
import type { Job } from "../api/types";

interface UseJobOptions {
  intervalMs?: number;
  onDone?: (job: Job) => void;
}

/** Polls a job until it reaches done/error. */
export function useJob(jobId: string | null, opts: UseJobOptions = {}) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const doneFired = useRef(false);

  useEffect(() => {
    doneFired.current = false;
    if (!jobId) {
      setJob(null);
      return;
    }
    let cancelled = false;
    let timer: number | undefined;

    async function tick() {
      try {
        const j = await apiClient.getJob(jobId as string);
        if (cancelled) return;
        setJob(j);
        if (j.status === "done" || j.status === "error") {
          if (!doneFired.current) {
            doneFired.current = true;
            opts.onDone?.(j);
          }
          return;
        }
        timer = window.setTimeout(tick, opts.intervalMs ?? 700);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    }
    tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  return { job, error };
}
