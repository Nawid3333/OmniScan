import { describe, expect, it } from "vitest";

import type { ChapterProgress } from "./api";
import { exportState, isDone, percent, pipelineState, totals } from "./progress";

function chapter(extra: Partial<ChapterProgress> = {}): ChapterProgress {
  return {
    chapter: "Chapter 1",
    lines: 10,
    translated: 10,
    checked: 10,
    problems: 0,
    last_stage: "export",
    failed: null,
    exported: true,
    outdated: false,
    ...extra,
  };
}

describe("progress view helpers", () => {
  it("says how far the pipeline got", () => {
    expect(pipelineState(chapter())).toBe("through export");
    expect(pipelineState(chapter({ last_stage: "slice", failed: "detect" }))).toBe("failed at detect");
    expect(pipelineState(chapter({ last_stage: null }))).toBe("not started");
  });

  it("says whether the pages are exported and current", () => {
    expect(exportState(chapter())).toBe("exported");
    expect(exportState(chapter({ outdated: true }))).toBe("outdated");
    expect(exportState(chapter({ exported: false }))).toBe("not exported");
  });

  it("rounds percentages and counts an empty chapter as complete", () => {
    expect(percent(2, 3)).toBe(67);
    expect(percent(0, 4)).toBe(0);
    expect(percent(0, 0)).toBe(100);
  });

  it("knows a chapter ready to release", () => {
    expect(isDone(chapter())).toBe(true);
    expect(isDone(chapter({ translated: 9 }))).toBe(false);
    expect(isDone(chapter({ checked: 9 }))).toBe(false);
    expect(isDone(chapter({ problems: 1 }))).toBe(false);
    expect(isDone(chapter({ exported: false }))).toBe(false);
    expect(isDone(chapter({ outdated: true }))).toBe(false);
    expect(isDone(chapter({ lines: 0, translated: 0, checked: 0 }))).toBe(false);
  });

  it("adds the chapters up", () => {
    const chapters = [chapter(), chapter({ chapter: "Chapter 2", translated: 4, checked: 1, problems: 3 })];
    expect(totals(chapters)).toEqual({ lines: 20, translated: 14, checked: 11, problems: 3, done: 1 });
    expect(totals([])).toEqual({ lines: 0, translated: 0, checked: 0, problems: 0, done: 0 });
  });
});
