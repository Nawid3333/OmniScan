/** Pure helpers for the Reader view (no DOM, no Svelte). */

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