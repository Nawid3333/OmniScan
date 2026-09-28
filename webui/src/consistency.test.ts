import { describe, expect, it } from "vitest";

import type { ConsistencyReport } from "./api";
import { filterReport, placeCount } from "./consistency";

const REPORT: ConsistencyReport = {
  divergences: [
    {
      source: "진우야",
      renderings: [
        { english: "Jinwoo!", places: [["Chapter 1", "r0001"], ["Chapter 2", "r0004"]] },
        { english: "Hey, Jinwoo!", places: [["Chapter 3", "r0002"]] },
      ],
    },
    { source: "가자", renderings: [{ english: "Let's go", places: [["Chapter 1", "r0002"]] }, { english: "Go", places: [["Chapter 2", "r0001"]] }] },
  ],
  term_misses: [{ chapter: "Chapter 2", region_id: "r0003", term: "헌터", target: "Hunter", english: "Hey you" }],
  typos: [
    { chapter: "Chapter 3", region_id: "r0007", word: "teh", suggestions: ["the", "ten"], english: "Grab teh sword" },
  ],
};

describe("filterReport", () => {
  it("keeps everything for a blank query", () => {
    expect(filterReport(REPORT, "  ")).toBe(REPORT);
  });

  it("matches sources, renderings and terms in any case", () => {
    const hey = filterReport(REPORT, "HEY");
    expect(hey.divergences.map((d) => d.source)).toEqual(["진우야"]);
    expect(hey.term_misses).toHaveLength(1);
    expect(filterReport(REPORT, "가자").divergences.map((d) => d.source)).toEqual(["가자"]);
    expect(filterReport(REPORT, "hunter").divergences).toEqual([]);
  });

  it("matches typos by word or English line", () => {
    expect(filterReport(REPORT, "TEH").typos).toHaveLength(1);
    expect(filterReport(REPORT, "sword").typos).toHaveLength(1);
    expect(filterReport(REPORT, "Jinwoo").typos).toEqual([]);
  });
});

describe("placeCount", () => {
  it("counts the regions over every rendering", () => {
    expect(placeCount(REPORT.divergences[0])).toBe(3);
  });
});
