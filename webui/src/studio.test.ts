import { describe, expect, it } from "vitest";

import type { ChapterEdits, FinalLine, Region, SourceFile } from "./api";
import {
  dragBox,
  fitZoom,
  handlePoint,
  hexToRgb,
  maskBounds,
  overlapsPage,
  pageOf,
  pageRectToStrip,
  pageRegions,
  pageToStripPoint,
  regionStatus,
  rgbToHex,
  statusLabels,
  stripToPage,
} from "./studio";

function file(index: number, y0: number, y1: number, height: number, scale = 1): SourceFile {
  return { index, name: `${index}.jpg`, width: 800 / scale, height, y0, y1, scale };
}

function region(id: string, y0: number, y1: number, extra: Partial<Region> = {}): Region {
  return {
    id,
    slice_index: 0,
    kind: "bubble_text",
    bbox: { x0: 10, y0, x1: 110, y1 },
    bubble_bbox: null,
    polygon: null,
    reading_order: 0,
    lang: "ko",
    orientation: "h",
    lines: [],
    text: "안녕",
    confidence: 0.9,
    ocr_alt: null,
    text_color: null,
    stroke_color: null,
    mask_ref: null,
    ...extra,
  };
}

const PAGE2 = file(1, 1000, 1600, 1200, 0.5); // a page twice as tall as its strip rows (scale 0.5)

describe("page <-> strip coordinates", () => {
  it("maps a strip box into a scaled page and back", () => {
    const rect = stripToPage(PAGE2, { x0: 100, y0: 1100, x1: 300, y1: 1200 });
    expect(rect).toEqual({ x: 200, y: 200, width: 400, height: 200 });
    expect(pageToStripPoint(PAGE2, 200, 200)).toEqual({ x: 100, y: 1100 });
    expect(pageRectToStrip(PAGE2, 600, 400, 200, 200)).toEqual({ x0: 100, y0: 1100, x1: 300, y1: 1200 });
  });

  it("rounds a drawn box outward to whole strip pixels", () => {
    expect(pageRectToStrip(PAGE2, 1, 1, 3, 3)).toEqual({ x0: 0, y0: 1000, x1: 2, y1: 1002 });
  });

  it("finds the regions and the page of a box", () => {
    const files = [file(0, 0, 1000, 1000), PAGE2];
    const regions = [
      region("r0002", 1200, 1300, { reading_order: 1 }),
      region("r0001", 1100, 1150, { reading_order: 0 }),
      region("r0003", 100, 200),
      region("r0004", 950, 1050), // spans both pages
    ];
    expect(pageRegions(PAGE2, regions).map((r) => r.id)).toEqual(["r0001", "r0004", "r0002"]);
    expect(overlapsPage(files[0], regions[3].bbox)).toBe(true);
    expect(pageOf(files, regions[0].bbox)).toBe(1);
    expect(pageOf(files, { x0: 0, y0: 5000, x1: 1, y1: 5001 })).toBe(0);
  });
});

describe("dragBox", () => {
  const box = { x0: 100, y0: 100, x1: 200, y1: 150 };

  it("moves the whole box", () => {
    expect(dragBox(box, "move", 10.4, -20.6)).toEqual({ x0: 110, y0: 79, x1: 210, y1: 129 });
  });

  it("resizes only the grabbed edges and never lets them cross", () => {
    expect(dragBox(box, "se", 20, 30)).toEqual({ x0: 100, y0: 100, x1: 220, y1: 180 });
    expect(dragBox(box, "n", 0, -10)).toEqual({ x0: 100, y0: 90, x1: 200, y1: 150 });
    expect(dragBox(box, "w", 500, 0)).toEqual({ x0: 196, y0: 100, x1: 200, y1: 150 });
    expect(dragBox(box, "s", 0, -500)).toEqual({ x0: 100, y0: 100, x1: 200, y1: 104 });
  });

  it("places handles on the corners and edge midpoints", () => {
    const rect = { x: 10, y: 20, width: 100, height: 50 };
    expect(handlePoint(rect, "nw")).toEqual({ x: 10, y: 20 });
    expect(handlePoint(rect, "e")).toEqual({ x: 110, y: 45 });
    expect(handlePoint(rect, "s")).toEqual({ x: 60, y: 70 });
  });
});

describe("regionStatus", () => {
  const edits: ChapterEdits = {
    regions: [],
    translations: [],
    deleted_regions: [],
    edited_region_ids: ["r0001"],
    manual_translation_ids: ["r0001"],
  };
  const line = (flags: string[], text = "Hi"): FinalLine => ({
    region_id: "r0001",
    text,
    decision: "manual",
    sources: [],
    rationale: "",
    flags,
  });

  it("combines the edits summary with the final line", () => {
    const status = regionStatus(region("r0001", 0, 10), edits, line(["source_changed"]));
    expect(statusLabels(status)).toEqual(["edited", "hand-translated", "source changed"]);
  });

  it("flags untranslated, added and low-confidence regions", () => {
    const status = regionStatus(region("m0001", 0, 10, { confidence: 0.2 }), edits, undefined);
    expect(statusLabels(status)).toEqual(["added", "untranslated", "low confidence"]);
  });

  it("never calls a watermark or an empty region untranslated", () => {
    expect(regionStatus(region("r0002", 0, 10, { kind: "watermark" }), null, undefined).untranslated).toBe(false);
    expect(regionStatus(region("r0002", 0, 10, { text: " " }), null, undefined).untranslated).toBe(false);
  });
});

describe("fitZoom", () => {
  it("fits wide pages and never enlarges small ones", () => {
    expect(fitZoom(2000, 1000)).toBe(0.5);
    expect(fitZoom(500, 1000)).toBe(1);
    expect(fitZoom(0, 1000)).toBe(1);
  });
});

describe("cleanup helpers", () => {
  it("finds the painted pixels' bounds", () => {
    const width = 6;
    const height = 4;
    const data = new Uint8ClampedArray(width * height * 4);
    expect(maskBounds(data, width, height)).toBeNull();
    data[(1 * width + 2) * 4 + 3] = 255; // (2, 1)
    data[(3 * width + 4) * 4 + 3] = 30; // (4, 3), faint anti-aliasing still counts
    expect(maskBounds(data, width, height)).toEqual({ x0: 2, y0: 1, x1: 5, y1: 4 });
  });

  it("converts colours both ways", () => {
    expect(hexToRgb("#0a80ff")).toEqual([10, 128, 255]);
    expect(rgbToHex([10, 128, 255])).toBe("#0a80ff");
  });
});
