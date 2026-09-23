import { describe, expect, it } from "vitest";

import type { BBox, SourceFile } from "./api";
import {
  fileAtStripY,
  stackTops,
  stackTotalHeight,
  stackWidth,
  stripBoxToStackBox,
  stripYToStackY,
} from "./strip";

function box(x0: number, y0: number, x1: number, y1: number): BBox {
  return { x0, y0, x1, y1 };
}

function file(overrides: Partial<SourceFile> & { index: number }): SourceFile {
  return {
    name: `${overrides.index}.jpg`,
    width: 800,
    height: 1000,
    y0: 0,
    y1: 1000,
    scale: 1.0,
    ...overrides,
  };
}

// Three scale=1.0 files: strip space and stack space coincide (files stacked in order, no gaps).
const plain: SourceFile[] = [
  file({ index: 0, y0: 0, y1: 1000 }),
  file({ index: 1, y0: 1000, y1: 3000, height: 2000 }),
  file({ index: 2, y0: 3000, y1: 3500, height: 500 }),
];

describe("stripYToStackY with scale=1 files", () => {
  it("reproduces strip-space coordinates as stack coordinates", () => {
    expect(stripYToStackY(plain, 0)).toBe(0);
    expect(stripYToStackY(plain, 500)).toBe(500);
    // At the f0/f1 boundary the last file whose y0 <= yStrip is f1, so this is f1's stack top.
    expect(stripYToStackY(plain, 1000)).toBe(1000);
    expect(stripYToStackY(plain, 2500)).toBe(2500);
    expect(stripYToStackY(plain, 3200)).toBe(3200);
  });

  it("maps yStrip exactly at strip_height to the total stacked height", () => {
    expect(stripYToStackY(plain, 3500)).toBe(stackTotalHeight(plain));
  });
});

describe("proportional mapping for scaled files", () => {
  it("maps a scale=0.5 file's strip midpoint to its natural-height midpoint", () => {
    // Natural 800x2000 scaled by 0.5 -> strip-space span of 1000 rows.
    const half = [file({ index: 0, width: 800, height: 2000, y0: 0, y1: 1000, scale: 0.5 })];
    expect(stripYToStackY(half, 500)).toBe(1000);
    expect(stripYToStackY(half, 250)).toBe(500);
    expect(stripYToStackY(half, 1000)).toBe(2000);
  });

  it("maps proportionally inside a stack after an earlier file", () => {
    const files = [
      file({ index: 0, y0: 0, y1: 1000 }),
      file({ index: 1, y0: 1000, y1: 2000, height: 2000, scale: 0.5 }),
    ];
    // Strip rows 1000..2000 (span 1000) map onto the 2000px-tall second file.
    expect(stripYToStackY(files, 1500)).toBe(1000 + 1000);
  });
});

describe("stackTotalHeight / stackWidth", () => {
  it("sums natural heights and takes the maximum width", () => {
    const files = [
      file({ index: 0, width: 800, height: 1000, y0: 0, y1: 1000 }),
      file({ index: 1, width: 1000, height: 2000, y0: 1000, y1: 3000 }),
      file({ index: 2, width: 700, height: 500, y0: 3000, y1: 3500 }),
    ];
    expect(stackTotalHeight(files)).toBe(3500);
    expect(stackWidth(files)).toBe(1000);
  });
});

describe("degenerate file (y1 === y0)", () => {
  it("does not throw and clamps the fraction to 0", () => {
    const files = [
      file({ index: 0, y0: 0, y1: 1000 }),
      file({ index: 1, y0: 1000, y1: 1000, height: 300 }),
    ];
    expect(() => stripYToStackY(files, 1000)).not.toThrow();
    expect(stripYToStackY(files, 1000)).toBe(1000);
    expect(stripYToStackY(files, 5000)).toBe(1000);
  });
});

describe("stackTops", () => {
  it("is each file's cumulative natural height, starting at 0", () => {
    const files = [
      file({ index: 0, height: 1000, y0: 0, y1: 1000 }),
      file({ index: 1, height: 2000, y0: 1000, y1: 3000 }),
      file({ index: 2, height: 500, y0: 3000, y1: 3500 }),
    ];
    expect(stackTops(files)).toEqual([0, 1000, 3000]);
  });

  it("is empty for no files", () => {
    expect(stackTops([])).toEqual([]);
  });
});

describe("fileAtStripY", () => {
  it("returns the last file whose y0 <= y", () => {
    expect(fileAtStripY(plain, 500)).toBe(plain[0]);
    expect(fileAtStripY(plain, 1000)).toBe(plain[1]);
    expect(fileAtStripY(plain, 3400)).toBe(plain[2]);
    expect(fileAtStripY(plain, 99999)).toBe(plain[2]);
  });

  it("returns the first file when y is above every file's y0", () => {
    expect(fileAtStripY(plain, -5)).toBe(plain[0]);
  });
});

describe("stripBoxToStackBox", () => {
  it("is the identity for a single scale=1 file", () => {
    const files = [file({ index: 0, width: 800, height: 1000, y0: 0, y1: 1000 })];
    expect(stripBoxToStackBox(files, box(100, 50, 300, 150))).toEqual({
      x: 100,
      y: 50,
      width: 200,
      height: 100,
    });
  });

  it("divides x by the scale of the file owning the box's top edge", () => {
    // Natural 800x2000 scaled by 0.5 -> strip rows 0..1000, strip width 400.
    const files = [file({ index: 0, width: 800, height: 2000, y0: 0, y1: 1000, scale: 0.5 })];
    expect(stripBoxToStackBox(files, box(100, 100, 300, 200))).toEqual({
      x: 200,
      y: 200,
      width: 400,
      height: 200,
    });
  });

  it("maps a box on the second file below the first file's stack height", () => {
    const files = [
      file({ index: 0, width: 800, height: 1000, y0: 0, y1: 1000 }),
      file({ index: 1, width: 800, height: 1000, y0: 1000, y1: 2000 }),
    ];
    expect(stripBoxToStackBox(files, box(0, 1200, 100, 1300))).toEqual({
      x: 0,
      y: 1200,
      width: 100,
      height: 100,
    });
  });
});