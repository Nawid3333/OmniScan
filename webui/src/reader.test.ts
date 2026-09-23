import { describe, expect, it } from "vitest";

import type { SourceFile } from "./api";
import {
  DEFAULT_COLUMN_WIDTH,
  MAX_COLUMN_WIDTH,
  MIN_COLUMN_WIDTH,
  clampColumnWidth,
  fitColumnWidth,
  layoutStack,
  scrollRatio,
  scrollTopForRatio,
} from "./reader";

function metrics(scrollTop: number, scrollHeight: number, clientHeight: number) {
  return { scrollTop, scrollHeight, clientHeight };
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

describe("scrollRatio", () => {
  it("is 0 at the top and 1 at the bottom of the scrollable range", () => {
    expect(scrollRatio(metrics(0, 2000, 1000))).toBe(0);
    expect(scrollRatio(metrics(1000, 2000, 1000))).toBe(1);
  });

  it("is 0.5 in the middle", () => {
    expect(scrollRatio(metrics(500, 2000, 1000))).toBe(0.5);
  });

  it("is 0 when there is nothing to scroll", () => {
    expect(scrollRatio(metrics(5, 500, 500))).toBe(0);
    expect(scrollRatio(metrics(5, 400, 500))).toBe(0);
  });

  it("clamps scrollTop beyond the range to 1 and negative values to 0", () => {
    expect(scrollRatio(metrics(1500, 2000, 1000))).toBe(1);
    expect(scrollRatio(metrics(-10, 2000, 1000))).toBe(0);
  });
});

describe("scrollTopForRatio", () => {
  it("maps a ratio onto the target's scrollable range", () => {
    expect(scrollTopForRatio(0.5, { scrollHeight: 2000, clientHeight: 1000 })).toBe(500);
    expect(scrollTopForRatio(1, { scrollHeight: 2000, clientHeight: 1000 })).toBe(1000);
  });

  it("clamps the ratio to [0, 1]", () => {
    expect(scrollTopForRatio(1.5, { scrollHeight: 2000, clientHeight: 1000 })).toBe(1000);
    expect(scrollTopForRatio(-1, { scrollHeight: 2000, clientHeight: 1000 })).toBe(0);
  });

  it("returns 0 for an unscrollable target", () => {
    expect(scrollTopForRatio(0.5, { scrollHeight: 500, clientHeight: 500 })).toBe(0);
    expect(scrollTopForRatio(0.5, { scrollHeight: 400, clientHeight: 500 })).toBe(0);
  });

  it("returns an integer", () => {
    expect(scrollTopForRatio(0.3333, { scrollHeight: 1001, clientHeight: 0 })).toBe(334);
  });
});

describe("clampColumnWidth", () => {
  it("falls back to the default for non-finite values", () => {
    expect(clampColumnWidth(NaN)).toBe(DEFAULT_COLUMN_WIDTH);
    expect(clampColumnWidth(Infinity)).toBe(DEFAULT_COLUMN_WIDTH);
  });

  it("clamps to the min/max and rounds", () => {
    expect(clampColumnWidth(50)).toBe(MIN_COLUMN_WIDTH);
    expect(clampColumnWidth(5000)).toBe(MAX_COLUMN_WIDTH);
    expect(clampColumnWidth(333.6)).toBe(334);
  });
});

describe("fitColumnWidth", () => {
  it("gives 'final' the whole viewport and 'compare' half of it minus the gap", () => {
    expect(fitColumnWidth("final", 720, 500)).toBe(500);
    expect(fitColumnWidth("final", 720, 2000)).toBe(720);
    expect(fitColumnWidth("compare", 720, 1000)).toBe(492);
  });

  it("never goes below MIN_COLUMN_WIDTH, even on a tiny viewport", () => {
    expect(fitColumnWidth("compare", 720, 300)).toBe(MIN_COLUMN_WIDTH);
  });

  it("clamps the requested width up to MIN first", () => {
    expect(fitColumnWidth("compare", 100, 2000)).toBe(MIN_COLUMN_WIDTH);
  });
});

describe("layoutStack", () => {
  it("scales tops/heights uniformly from the stack's natural width", () => {
    const files = [
      file({ index: 0, width: 800, height: 1000 }),
      file({ index: 1, width: 800, height: 2000 }),
    ];
    // Natural width 800 -> display width 400: scale 0.5.
    const layout = layoutStack(files, 400);
    expect(layout.tops).toEqual([0, 500]);
    expect(layout.heights).toEqual([500, 1000]);
    expect(layout.totalHeight).toBe(1500);
  });

  it("is the identity when displayWidth equals the natural width", () => {
    const files = [file({ index: 0, width: 800, height: 1000 }), file({ index: 1, width: 800, height: 500 })];
    const layout = layoutStack(files, 800);
    expect(layout).toEqual({ tops: [0, 1000], heights: [1000, 500], totalHeight: 1500 });
  });

  it("heights come from consecutive tops, never from independently rounded scaled heights", () => {
    // Natural height 1000 at scale 1/3 rounds per-file to 333, but three of them must still sum to
    // exactly totalHeight (1000) with no 1px gap or overlap between panels.
    const files = [
      file({ index: 0, width: 300, height: 1000 }),
      file({ index: 1, width: 300, height: 1000 }),
      file({ index: 2, width: 300, height: 1000 }),
    ];
    const layout = layoutStack(files, 100);
    expect(layout.tops).toEqual([0, 333, 667]);
    expect(layout.heights).toEqual([333, 334, 333]);
    expect(layout.heights.reduce((a, b) => a + b, 0)).toBe(layout.totalHeight);
  });

  it("returns empty layout for no files (natural width falls back to 1, not division by zero)", () => {
    const layout = layoutStack([], 400);
    expect(layout).toEqual({ tops: [], heights: [], totalHeight: 0 });
  });
});