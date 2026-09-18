import { describe, expect, it } from "vitest";

import {
  DEFAULT_COLUMN_WIDTH,
  MAX_COLUMN_WIDTH,
  MIN_COLUMN_WIDTH,
  clampColumnWidth,
  fitColumnWidth,
  scrollRatio,
  scrollTopForRatio,
} from "./reader";

function metrics(scrollTop: number, scrollHeight: number, clientHeight: number) {
  return { scrollTop, scrollHeight, clientHeight };
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