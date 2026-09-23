import { describe, expect, it } from "vitest";

import type { InpaintItem } from "./api";
import { cleanClipPath, fillCss, hasStoredPatch, methodColor } from "./inpaint";

let nextId = 0;

function item(overrides: Partial<InpaintItem>): InpaintItem {
  nextId += 1;
  return {
    region_id: `r${String(nextId).padStart(4, "0")}`,
    box: { x0: 0, y0: 0, x1: 10, y1: 10 },
    method: "flat",
    fill: null,
    needs_lama: false,
    mask_px: 0,
    ...overrides,
  };
}

describe("methodColor", () => {
  it("returns three distinct strings", () => {
    const colors = (["flat", "lama", "none"] as const).map(methodColor);
    expect(new Set(colors).size).toBe(3);
  });

  it("pins the exact colors", () => {
    expect(methodColor("flat")).toBe("#1f6feb");
    expect(methodColor("lama")).toBe("#9333ea");
    expect(methodColor("none")).toBe("#6b7280");
  });
});

describe("fillCss", () => {
  it("is transparent for null", () => {
    expect(fillCss(null)).toBe("transparent");
  });

  it("pins the rgb form", () => {
    expect(fillCss([10, 20, 30])).toBe("rgb(10, 20, 30)");
  });
});

describe("cleanClipPath", () => {
  it("pins the exact clip paths at the endpoints and halfway", () => {
    expect(cleanClipPath(0)).toBe("inset(0 100% 0 0)");
    expect(cleanClipPath(50)).toBe("inset(0 50% 0 0)");
    expect(cleanClipPath(100)).toBe("inset(0 0% 0 0)");
  });

  it("clamps out-of-range values to [0, 100]", () => {
    expect(cleanClipPath(-10)).toBe("inset(0 100% 0 0)");
    expect(cleanClipPath(150)).toBe("inset(0 0% 0 0)");
  });
});

describe("hasStoredPatch", () => {
  it("is true when the region id is in the set", () => {
    const stored = item({ region_id: "r0001" });
    expect(hasStoredPatch(stored, new Set(["r0001"]))).toBe(true);
  });

  it("is false when the region id is missing", () => {
    const stored = item({ region_id: "r0001" });
    expect(hasStoredPatch(stored, new Set<string>())).toBe(false);
  });
});