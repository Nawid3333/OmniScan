import { describe, expect, it } from "vitest";

import type { FinalLine, LayoutItem } from "./api";
import { overlayStyle, textByRegion } from "./edit";

let nextId = 0;

function line(overrides: Partial<FinalLine>): FinalLine {
  nextId += 1;
  return {
    region_id: `r${String(nextId).padStart(4, "0")}`,
    text: "",
    decision: "pick",
    sources: [],
    rationale: "",
    flags: [],
    ...overrides,
  };
}

function item(overrides: Partial<LayoutItem>): LayoutItem {
  nextId += 1;
  return {
    region_id: `r${String(nextId).padStart(4, "0")}`,
    font_role: "dialogue",
    font: "ComicNeue-Bold.ttf",
    size_px: 28,
    lines: [],
    box: { x0: 0, y0: 0, x1: 10, y1: 10 },
    align: "center",
    color: [0, 0, 0],
    stroke_px: 0,
    stroke_color: [255, 255, 255],
    overflow: false,
    ...overrides,
  };
}

describe("textByRegion", () => {
  it("returns {} for an empty list", () => {
    expect(textByRegion([])).toEqual({});
  });

  it("maps every line's region_id to its text", () => {
    const map = textByRegion([line({ region_id: "r0001", text: "Hello" }), line({ region_id: "r0002", text: "World" })]);
    expect(map["r0001"]).toBe("Hello");
    expect(map["r0002"]).toBe("World");
    expect(Object.keys(map).length).toBe(2);
  });

  it("keeps the last duplicate region_id (last write wins, no crash)", () => {
    const map = textByRegion([line({ region_id: "r1", text: "first" }), line({ region_id: "r1", text: "second" })]);
    expect(map["r1"]).toBe("second");
  });
});

describe("overlayStyle", () => {
  it("carries size, colour and alignment into the style", () => {
    const style = overlayStyle(item({ align: "right", color: [200, 30, 30], size_px: 32 }));
    expect(style).toContain("font-size: 32px");
    expect(style).toContain("color: rgb(200, 30, 30)");
    expect(style).toContain("text-align: right");
  });

  it("centers by default and ignores the font filename", () => {
    const style = overlayStyle(item({ align: "center", font: "ComicNeue-Bold.ttf" }));
    expect(style).toContain("text-align: center");
    expect(style).not.toContain("ComicNeue");
    expect(style).toContain("sans-serif");
  });
});