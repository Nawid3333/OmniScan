import { describe, expect, it } from "vitest";

import type { Region } from "./api";
import {
  LOW_CONFIDENCE,
  disagrees,
  filterRegions,
  isLowConfidence,
  kindColor,
  summarize,
} from "./ocr";

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

describe("kindColor", () => {
  it("returns four distinct strings", () => {
    const colors = (["bubble_text", "free_text", "sfx", "watermark"] as const).map(kindColor);
    expect(new Set(colors).size).toBe(4);
  });
});

describe("disagrees", () => {
  it("is false when ocr_alt is null", () => {
    expect(disagrees(region({ text: "a", ocr_alt: null }))).toBe(false);
  });

  it("is false when ocr_alt equals text", () => {
    expect(disagrees(region({ text: "a", ocr_alt: "a" }))).toBe(false);
  });

  it("is true when ocr_alt differs from text", () => {
    expect(disagrees(region({ text: "a", ocr_alt: "b" }))).toBe(true);
  });
});

describe("isLowConfidence", () => {
  it("is strictly below the threshold", () => {
    expect(isLowConfidence(region({ confidence: 0.49 }))).toBe(true);
    expect(isLowConfidence(region({ confidence: 0.5 }))).toBe(false);
    expect(LOW_CONFIDENCE).toBe(0.5);
  });

  it("respects a custom threshold", () => {
    expect(isLowConfidence(region({ confidence: 0.8 }), 0.9)).toBe(true);
    expect(isLowConfidence(region({ confidence: 0.8 }), 0.7)).toBe(false);
  });
});

describe("filterRegions", () => {
  const list = [
    region({ kind: "bubble_text" }),
    region({ kind: "sfx" }),
    region({ kind: "bubble_text" }),
    region({ kind: "watermark" }),
  ];

  it("keeps only visible kinds and preserves input order", () => {
    const visible = filterRegions(list, new Set(["bubble_text", "watermark"]));
    expect(visible).toEqual([list[0], list[2], list[3]]);
  });

  it("returns [] for an empty set", () => {
    expect(filterRegions(list, new Set())).toEqual([]);
  });
});

describe("summarize", () => {
  it("counts total, low-confidence and disagreements", () => {
    const list = [
      region({ confidence: 0.9, ocr_alt: null }),
      region({ confidence: 0.4, ocr_alt: "x", text: "y" }),
      region({ confidence: 0.5, ocr_alt: "same", text: "same" }),
      region({ confidence: 0.1, ocr_alt: null }),
    ];
    expect(summarize(list)).toEqual({ total: 4, lowConfidence: 2, disagreements: 1 });
  });
});