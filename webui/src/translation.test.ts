import { describe, expect, it } from "vitest";

import type { CandidateRun, FinalArtifact, FinalLine, GlossaryHit, Region, RegionsArtifact } from "./api";
import {
  buildRows,
  candidatesDisagree,
  filterRows,
  hitColor,
  highlightSource,
  sortRegions,
  summarizeReview,
} from "./translation";

let nextId = 0;

function region(overrides: Partial<Region>): Region {
  nextId += 1;
  return {
    id: `r${String(nextId).padStart(4, "0")}`,
    slice_index: 0,
    kind: "bubble_text",
    bbox: { x0: 0, y0: 0, x1: 10, y1: 10 },
    bubble_bbox: null,
    polygon: null,
    reading_order: nextId,
    lang: "ko",
    orientation: "h",
    lines: [],
    text: "",
    confidence: 1,
    ocr_alt: null,
    text_color: null,
    stroke_color: null,
    mask_ref: null,
    ...overrides,
  };
}

function run(runId: string, candidates: [string, string, string?][]): CandidateRun {
  return {
    schema_version: 1,
    run_id: runId,
    profile: "fast",
    model: "qwen3:8b",
    created_at: "2026-09-18T12:00:00Z",
    candidates: candidates.map(([regionId, text, notes]) => ({ region_id: regionId, text, notes: notes ?? null })),
    usage: {},
  };
}

function hit(overrides: Partial<GlossaryHit>): GlossaryHit {
  return {
    entry_id: 1,
    source: "철수",
    target: "Cheolsu",
    status: "locked",
    start: 0,
    end: 3,
    particle: null,
    target_in_final: true,
    ...overrides,
  };
}

const FINAL_LINE: FinalLine = {
  region_id: "r0001",
  text: "Cheolsu came",
  decision: "pick",
  sources: ["runA"],
  rationale: "",
  flags: [],
};

describe("sortRegions", () => {
  it("sorts by slice_index, then reading_order, then numeric-aware id; input not mutated", () => {
    const rLate = region({ id: "r10", slice_index: 1, reading_order: 1 });
    const rMid = region({ id: "r2", slice_index: 0, reading_order: 2 });
    const rEarly = region({ id: "r2", slice_index: 0, reading_order: 1 });
    const input = [rLate, rMid, rEarly];
    const sorted = sortRegions(input);
    expect(sorted.map((r) => r.id)).toEqual([rEarly.id, rMid.id, rLate.id]);
    expect(input.map((r) => r.id)).toEqual([rLate.id, rMid.id, rEarly.id]);
  });
});

describe("candidatesDisagree", () => {
  it("ignores case and surrounding whitespace", () => {
    expect(candidatesDisagree({ a: "Hi", b: " hi " })).toBe(false);
  });

  it("is true when two non-null texts differ", () => {
    expect(candidatesDisagree({ a: "Hi", b: "Hello" })).toBe(true);
  });

  it("needs at least two non-null texts", () => {
    expect(candidatesDisagree({ a: "Hi", b: null })).toBe(false);
    expect(candidatesDisagree({})).toBe(false);
  });
});

describe("buildRows", () => {
  const ocr: RegionsArtifact = {
    schema_version: 1,
    regions: [
      region({ id: "r0001", slice_index: 0, reading_order: 1, text: "철수가 왔다" }),
      region({ id: "r0002", slice_index: 0, reading_order: 2, text: "hi" }),
    ],
  };
  const runs = [
    run("runA", [
      ["r0001", "Hello"],
      ["r0002", "World"],
    ]),
    run("runB", [["r0001", "hello"]]),
  ];

  it("maps candidates per run (null where a run lacks the region) and keeps sortRegions order", () => {
    const rows = buildRows(ocr, runs, null, null);
    expect(rows.map((row) => row.region.id)).toEqual(["r0001", "r0002"]);
    expect(rows[0].candidates).toEqual({ runA: "Hello", runB: "hello" });
    expect(rows[0].disagree).toBe(false);
    expect(rows[0].needsAttention).toBe(false);
    expect(rows[1].candidates).toEqual({ runA: "World", runB: null });
    expect(rows[1].disagree).toBe(false);
    expect(rows[1].final).toBeNull();
  });

  it("flags a final line with flags", () => {
    const final: FinalArtifact = {
      schema_version: 1,
      judge_model: "judge:8b",
      created_at: "2026-09-18T12:00:00Z",
      lines: [{ ...FINAL_LINE, flags: ["uncertain"] }],
    };
    const rows = buildRows(ocr, runs, final, null);
    expect(rows[0].final?.flags).toEqual(["uncertain"]);
    expect(rows[0].needsAttention).toBe(true);
    expect(rows[1].needsAttention).toBe(false);
  });

  it("collects locked-term violations from hits", () => {
    const hits = { regions: { r0001: [hit({ target_in_final: false })] } };
    const rows = buildRows(ocr, runs, null, hits);
    expect(rows[0].violations).toHaveLength(1);
    expect(rows[0].needsAttention).toBe(true);
    expect(rows[1].violations).toHaveLength(0);
  });

  it("flags runs with no candidate text and no final line", () => {
    const emptyRuns = [run("runA", []), run("runB", [])];
    const rows = buildRows(ocr, emptyRuns, null, null);
    expect(rows.every((row) => row.needsAttention)).toBe(true);
  });
});

describe("highlightSource", () => {
  it("splits text around one hit", () => {
    const segments = highlightSource("철수가 왔다", [hit({ start: 0, end: 3 })]);
    expect(segments).toEqual([
      { text: "철수가", hit: expect.objectContaining({ start: 0 }) },
      { text: " 왔다", hit: null },
    ]);
    expect(segments.map((s) => s.text).join("")).toBe("철수가 왔다");
  });

  it("ignores an overlapping second hit", () => {
    const segments = highlightSource("abcdef", [
      hit({ source: "a", start: 1, end: 4 }),
      hit({ source: "b", start: 3, end: 6 }),
    ]);
    expect(segments.map((s) => s.text)).toEqual(["a", "bcd", "ef"]);
    expect(segments.map((s) => s.hit !== null)).toEqual([false, true, false]);
  });

  it("ignores out-of-range hits", () => {
    expect(highlightSource("abc", [hit({ start: 5, end: 9 })])).toEqual([
      { text: "abc", hit: null },
    ]);
    expect(highlightSource("abc", [hit({ start: -2, end: 1 })])).toEqual([
      { text: "abc", hit: null },
    ]);
  });

  it("clamps an end past the text", () => {
    const segments = highlightSource("abc", [hit({ start: 1, end: 99 })]);
    expect(segments.map((s) => s.text)).toEqual(["a", "bc"]);
    expect(segments[1].hit?.end).toBe(3);
  });

  it("returns [] for empty text", () => {
    expect(highlightSource("", [hit({ start: 0, end: 2 })])).toEqual([]);
  });
});

describe("summarizeReview", () => {
  it("counts totals over hand-built rows", () => {
    const ocr: RegionsArtifact = {
      schema_version: 1,
      regions: [
        region({ id: "r0001", reading_order: 1 }),
        region({ id: "r0002", reading_order: 2 }),
        region({ id: "r0003", reading_order: 3, text: "clean" }),
      ],
    };
    const runs = [run("runA", [["r0001", "Hi"], ["r0003", "clean"]])];
    const final: FinalArtifact = {
      schema_version: 1,
      judge_model: "judge:8b",
      created_at: "2026-09-18T12:00:00Z",
      lines: [{ ...FINAL_LINE, flags: ["glossary_violation"] }],
    };
    const rows = buildRows(ocr, runs, final, { regions: { r0002: [hit({ target_in_final: false })] } });
    expect(summarizeReview(rows)).toEqual({
      total: 3,
      withFinal: 1,
      flagged: 2,
      violations: 1,
      disagreements: 0,
    });
    expect(filterRows(rows, true).map((row) => row.region.id)).toEqual(["r0001", "r0002"]);
    expect(filterRows(rows, false)).toHaveLength(3);
    expect(filterRows([], true)).toEqual([]);
  });
});

describe("hitColor", () => {
  it("returns the four documented colours", () => {
    expect(hitColor(hit({ target_in_final: false }))).toBe("#dc2626");
    expect(hitColor(hit({ target_in_final: true }))).toBe("#16a34a");
    expect(hitColor(hit({ status: "proposed" }))).toBe("#d97706");
    expect(hitColor(hit({ status: "rejected" }))).toBe("#6b7280");
  });
});