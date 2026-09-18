/** Pure helpers for the Filtered view (no DOM, no Svelte). */

import type { FilteredChapter, FilteredItem } from "./api";

export function itemKey(chapter: string, item: FilteredItem): string {
  return `${chapter}|${item.target}|${item.index}`;
}

export function summarizeFiltered(chapters: FilteredChapter[]): { chapters: number; filtered: number; restored: number } {
  let filtered = 0;
  let restored = 0;
  for (const c of chapters) {
    for (const item of c.items) {
      if (item.state === "restored") {
        restored += 1;
      } else {
        filtered += 1;
      }
    }
  }
  return { chapters: chapters.length, filtered, restored };
}

export function describeItem(item: FilteredItem): string {
  if (item.target === "file") {
    return `file ${item.index}` + (item.name ? ` (${item.name})` : "");
  }
  return `slice ${item.index}` + (item.y0 !== null && item.y1 !== null ? ` (y ${item.y0}–${item.y1})` : "");
}

export function scoreText(item: FilteredItem): string {
  return `${item.score.toFixed(2)} · ${item.matched_example ?? "no example"} · ${item.method}`;
}

/** A new list with that item's state set to "restored"; the input and every other item are untouched. */
export function markRestored(
  chapters: FilteredChapter[],
  chapter: string,
  target: "file" | "slice",
  index: number,
): FilteredChapter[] {
  return chapters.map((c) =>
    c.chapter === chapter
      ? {
          ...c,
          items: c.items.map((item) =>
            item.target === target && item.index === index ? { ...item, state: "restored" as const } : item,
          ),
        }
      : c,
  );
}