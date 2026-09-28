import { describe, expect, it } from "vitest";

import type { Problem } from "./api";
import { filterProblems, kindCounts, kindLabel, regionCount } from "./problems";

function problem(region_id: string, kind: string, extra: Partial<Problem> = {}): Problem {
  return { region_id, kind, message: "", status: "todo", finished_page: false, word: "", read: "", ...extra };
}

const PROBLEMS = [
  problem("r0001", "overflow"),
  problem("r0001", "uncertain"),
  problem("r0003", "source_left", { status: "checked" }),
  problem("r0003", "source_left", { status: "checked", finished_page: true }),
  problem("r0004", "typo", { word: "sayy" }),
];

describe("problems view helpers", () => {
  it("filters by kind and leaves out checked lines on request", () => {
    expect(filterProblems(PROBLEMS, "", false)).toHaveLength(5);
    expect(filterProblems(PROBLEMS, "source_left", false).map((p) => p.finished_page)).toEqual([false, true]);
    expect(filterProblems(PROBLEMS, "", true).map((p) => p.region_id)).toEqual(["r0001", "r0001", "r0004"]);
    expect(filterProblems(PROBLEMS, "source_left", true)).toEqual([]);
  });

  it("counts the kinds in the order they first appear", () => {
    expect(kindCounts(PROBLEMS)).toEqual([
      { kind: "overflow", count: 1 },
      { kind: "uncertain", count: 1 },
      { kind: "source_left", count: 2 },
      { kind: "typo", count: 1 },
    ]);
    expect(kindCounts([])).toEqual([]);
  });

  it("counts the regions with a problem", () => {
    expect(regionCount(PROBLEMS)).toBe(3);
    expect(regionCount([])).toBe(0);
  });

  it("names the kinds, and keeps an unknown one as it is", () => {
    expect(kindLabel("untranslated")).toBe("no English");
    expect(kindLabel("typo")).toBe("possible typo");
    expect(kindLabel("something_new")).toBe("something_new");
  });
});
