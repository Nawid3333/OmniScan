import { describe, expect, it } from "vitest";

import type { SourceFile } from "./api";
import { nearestCut, pieceHeights, snapToBand } from "./cuts";
import { stackYToStripY, stripYToStackY } from "./strip";

const band = (y0: number, y1: number) => ({ y0, y1, color: [255, 255, 255] as [number, number, number], is_gradient: false });

describe("snapToBand", () => {
  it("moves a cut to the middle of a close band, else leaves it", () => {
    const bands = [band(100, 140), band(500, 520)];
    expect(snapToBand(90, bands, 30)).toBe(120);
    expect(snapToBand(120, bands, 30)).toBe(120);
    expect(snapToBand(300, bands, 30)).toBe(300);
    expect(snapToBand(535, bands, 30)).toBe(510);
  });
});

describe("nearestCut / pieceHeights", () => {
  it("finds the closest cut within the tolerance", () => {
    expect(nearestCut([100, 200, 210], 206, 8)).toBe(2);
    expect(nearestCut([100], 150, 8)).toBe(-1);
  });

  it("measures the pieces between the cuts", () => {
    expect(pieceHeights([300, 100, 100, 0, 1000], 1000)).toEqual([100, 200, 700]);
  });
});

describe("stackYToStripY", () => {
  const files: SourceFile[] = [
    { index: 0, name: "a", width: 800, height: 1000, y0: 0, y1: 1000, scale: 1 },
    { index: 1, name: "b", width: 400, height: 500, y0: 1000, y1: 2000, scale: 2 },
  ];

  it("inverts stripYToStackY", () => {
    for (const y of [0, 500, 1000, 1500, 2000]) {
      expect(stackYToStripY(files, stripYToStackY(files, y))).toBeCloseTo(y);
    }
    expect(stackYToStripY(files, -10)).toBe(0);
    expect(stackYToStripY(files, 99999)).toBe(2000);
  });
});
