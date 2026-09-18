import { describe, expect, it } from "vitest";

import type { FilteredChapter, FilteredItem } from "./api";
import { describeItem, itemKey, markRestored, scoreText, summarizeFiltered } from "./filtered";

function item(overrides: Partial<FilteredItem> = {}): FilteredItem {
  return {
    target: "file",
    index: 0,
    state: "filtered",
    score: 0.9,
    matched_example: null,
    method: "phash",
    name: null,
    y0: null,
    y1: null,
    ...overrides,
  };
}

describe("itemKey", () => {
  it("joins chapter, target and index", () => {
    expect(itemKey("Chapter 1", item({ target: "slice", index: 7 }))).toBe("Chapter 1|slice|7");
  });
});

describe("describeItem", () => {
  it("shows a file with and without a name", () => {
    expect(describeItem(item({ target: "file", index: 3, name: "004.jpg" }))).toBe("file 3 (004.jpg)");
    expect(describeItem(item({ target: "file", index: 3, name: null }))).toBe("file 3");
  });

  it("shows a slice with and without a y range (en dash)", () => {
    expect(describeItem(item({ target: "slice", index: 7, y0: 20100, y1: 23000 }))).toBe("slice 7 (y 20100–23000)");
    expect(describeItem(item({ target: "slice", index: 7, y0: null, y1: null }))).toBe("slice 7");
  });
});

describe("scoreText", () => {
  it("joins score, matched example and method", () => {
    expect(scoreText(item({ score: 0.97123, matched_example: "global/end_card.jpg", method: "phash" }))).toBe(
      "0.97 · global/end_card.jpg · phash",
    );
  });

  it("says 'no example' when matched_example is null", () => {
    expect(scoreText(item({ matched_example: null }))).toBe("0.90 · no example · phash");
  });
});

describe("summarizeFiltered", () => {
  it("counts chapters and items by state", () => {
    const chapters: FilteredChapter[] = [
      { chapter: "Chapter 1", items: [item(), item({ state: "restored" }), item({ index: 1 })] },
      { chapter: "Chapter 2", items: [item({ target: "slice", index: 7, state: "restored" })] },
      { chapter: "Chapter 3", items: [] },
    ];
    expect(summarizeFiltered(chapters)).toEqual({ chapters: 3, filtered: 2, restored: 2 });
  });

  it("returns zeros for an empty list", () => {
    expect(summarizeFiltered([])).toEqual({ chapters: 0, filtered: 0, restored: 0 });
  });
});

describe("markRestored", () => {
  const chapters: FilteredChapter[] = [
    { chapter: "Chapter 1", items: [item({ index: 3 }), item({ target: "slice", index: 7 })] },
    { chapter: "Chapter 2", items: [item({ index: 1 })] },
  ];

  it("changes only the addressed item's state", () => {
    const next = markRestored(chapters, "Chapter 1", "slice", 7);
    expect(next[0].items[1].state).toBe("restored");
    expect(next[0].items[0].state).toBe("filtered");
    expect(next[1].items[0].state).toBe("filtered");
    expect(next[1].chapter).toBe("Chapter 2");
  });

  it("does not mutate the input", () => {
    markRestored(chapters, "Chapter 1", "file", 3);
    expect(chapters[0].items[0].state).toBe("filtered");
    expect(chapters[1].items[0].state).toBe("filtered");
  });

  it("returns an equal copy for an unknown chapter or index", () => {
    expect(markRestored(chapters, "Chapter 9", "file", 3)).toEqual(chapters);
    expect(markRestored(chapters, "Chapter 1", "slice", 99)).toEqual(chapters);
  });
});