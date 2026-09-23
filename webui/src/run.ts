/** Pure helpers behind RunButton's job polling (no DOM, no Svelte). */

import type { JobStatus } from "./api";

/** What a run button shows for the job it queued. */
export type RunPhase = "idle" | "queued" | "running" | "done" | "failed";

/** A run button's full state: the phase, plus the error (only when the phase is "failed"). */
export interface RunState {
  phase: RunPhase;
  error: string | null; // set only when phase is "failed"
}

/**
 * `Job.status` -> the `RunPhase` a poll of it should show. "paused"/"cancelled" both map to
 * "failed" (from this button's point of view the job did not produce the data — show the retry
 * button either way; there is no owner-facing distinction worth making here).
 */
export function phaseOf(status: JobStatus): RunPhase {
  switch (status) {
    case "queued":
      return "queued";
    case "running":
      return "running";
    case "done":
      return "done";
    default: // "paused" | "cancelled"
      return "failed";
  }
}

/** How often a queued job is polled for progress. */
export const POLL_INTERVAL_MS = 1500;