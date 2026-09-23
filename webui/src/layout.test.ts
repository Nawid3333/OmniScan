import { describe, expect, it } from "vitest";

import type { LayoutItem } from "./api";
import { alignLabel, filterItems, fontRoleColor } from "./layout";

let nextId = 0;

function item(overrides: Partial<LayoutItem>): LayoutItem {
  nextId += 1;
  return {
    region_id: `r${String(nextId).padStart(4, "0")}`,
    font_role: "dialogue",
    font: "Bados",
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

describe("fontRoleColor", () => {
  it("returns six distinct strings covering every FontRole", () => {
    const roles = ["dialogue", "thought", "shout", "narration", "free", "sfx"] as const;
    const colors = roles.map(fontRoleColor);
    expect(new Set(colors).size).toBe(6);
  });

  it("pins the exact colors", () => {
    expect(fontRoleColor("dialogue")).toBe("#1f6feb");
    expect(fontRoleColor("thought")).toBe("#0d9488");
    expect(fontRoleColor("shout")).toBe("#dc2626");
    expect(fontRoleColor("narration")).toBe("#9333ea");
    expect(fontRoleColor("free")).toBe("#d97706");
    expect(fontRoleColor("sfx")).toBe("#6b7280");
  });
});

describe("alignLabel", () => {
  it("pins all three alignments", () => {
    expect(alignLabel("center")).toBe("center");
    expect(alignLabel("left")).toBe("left");
    expect(alignLabel("right")).toBe("right");
  });
});

describe("filterItems", () => {
  const list = [
    item({ font_role: "dialogue" }),
    item({ font_role: "sfx" }),
    item({ font_role: "dialogue" }),
    item({ font_role: "narration" }),
  ];

  it("keeps only visible roles and preserves input order", () => {
    const visible = filterItems(list, new Set(["dialogue", "narration"]));
    expect(visible).toEqual([list[0], list[2], list[3]]);
  });

  it("returns [] for an empty set", () => {
    expect(filterItems(list, new Set())).toEqual([]);
  });
});