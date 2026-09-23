import { describe, expect, it } from "vitest";

import type { JobStatus } from "./api";
import { POLL_INTERVAL_MS, phaseOf } from "./run";
import type { RunPhase } from "./run";

// Every JobStatus -> the RunPhase a poll of it should show: "paused"/"cancelled" both read as
// "the job did not produce the data", i.e. failed, from a run button's point of view.
const PHASE_OF: [JobStatus, RunPhase][] = [
  ["queued", "queued"],
  ["running", "running"],
  ["paused", "failed"],
  ["done", "done"],
  ["failed", "failed"],
  ["cancelled", "failed"],
];

describe("phaseOf", () => {
  it("maps every JobStatus to its documented RunPhase", () => {
    for (const [status, expected] of PHASE_OF) {
      expect(phaseOf(status)).toBe(expected);
    }
  });
});

describe("POLL_INTERVAL_MS", () => {
  it("is 1500", () => {
    expect(POLL_INTERVAL_MS).toBe(1500);
  });
});