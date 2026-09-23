/** Pure helpers for the Reader view (no DOM, no Svelte). */

import type { SourceFile } from "./api";
import { stackTops, stackTotalHeight, stackWidth } from "./strip";

export interface ScrollMetrics {
  scrollTop: number;
  scrollHeight: number;
  clientHeight: number;
}

export const MIN_COLUMN_WIDTH = 200;
export const MAX_COLUMN_WIDTH = 1600;
export const DEFAULT_COLUMN_WIDTH = 720;
export const COLUMN_GAP = 16;

export type ReaderMode = "final" | "compare";

/** 0 at the top, 1 at the bottom of the scrollable range, clamped to [0, 1];
 *  0 when there is nothing to scroll (scrollHeight <= clientHeight). */
export function scrollRatio(m: ScrollMetrics): number {
  const range = m.scrollHeight - m.clientHeight;
  if (range <= 0) {
    return 0;
  }
  return Math.min(1, Math.max(0, m.scrollTop / range));
}

/** The scrollTop that puts `target` at `ratio` of its scrollable range; ratio is clamped to [0, 1]. */
export function scrollTopForRatio(
  ratio: number,
  target: Pick<ScrollMetrics, "scrollHeight" | "clientHeight">,
): number {
  const clamped = Math.min(1, Math.max(0, ratio));
  return Math.round(clamped * Math.max(0, target.scrollHeight - target.clientHeight));
}

/** Non-finite -> DEFAULT_COLUMN_WIDTH; otherwise Math.round then clamp to [MIN_COLUMN_WIDTH, MAX_COLUMN_WIDTH]. */
export function clampColumnWidth(px: number): number {
  if (!Number.isFinite(px)) {
    return DEFAULT_COLUMN_WIDTH;
  }
  return Math.min(MAX_COLUMN_WIDTH, Math.max(MIN_COLUMN_WIDTH, Math.round(px)));
}

/** Column width fitted to the viewport for `mode`: "final" gets the whole width, "compare" half of it
 *  (minus the gap); never below MIN_COLUMN_WIDTH, even on a tiny viewport. */
export function fitColumnWidth(mode: ReaderMode, requested: number, viewportWidth: number): number {
  const available = mode === "final" ? viewportWidth : Math.floor((viewportWidth - COLUMN_GAP) / 2);
  return clampColumnWidth(Math.min(clampColumnWidth(requested), available));
}

export interface StackLayout {
  /** Each file's top offset, in display pixels. */
  tops: number[];
  /** Each file's height, in display pixels (derived from consecutive `tops`, never from a
   *  separately-rounded scaled height, so pages never gain a gap or an overlap from rounding). */
  heights: number[];
  totalHeight: number;
}

/**
 * Lay out `files` as a vertical stack scaled to `displayWidth` (uniform scale from the stack's
 * natural width, preserving every page's own aspect ratio). Two calls with the same `files` and
 * `displayWidth` always produce identical `tops`/`totalHeight` — this is what lets two independently
 * rendered columns (e.g. raw vs. final pages of the same chapter) share one scrollable coordinate
 * space and sync by copying `scrollTop` directly, with no ratio conversion.
 */
export function layoutStack(files: SourceFile[], displayWidth: number): StackLayout {
  const naturalWidth = stackWidth(files) || 1;
  const scale = displayWidth / naturalWidth;
  const totalHeight = Math.round(stackTotalHeight(files) * scale);
  const tops = stackTops(files).map((top) => Math.round(top * scale));
  const heights = files.map((_file, i) => (i + 1 < tops.length ? tops[i + 1] : totalHeight) - tops[i]);
  return { tops, heights, totalHeight };
}