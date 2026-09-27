/** Pure helpers behind the Slicer view's output-cut editing (no DOM, no Svelte). */

import type { Band } from "./api";

/** `y` moved to the middle of the nearest uniform band within `maxDistance` rows (a clean place to cut);
 *  `y` itself when no band is that close. */
export function snapToBand(y: number, bands: Band[], maxDistance: number): number {
  let best = y;
  let bestDistance = maxDistance;
  for (const band of bands) {
    const distance = y < band.y0 ? band.y0 - y : y >= band.y1 ? y - (band.y1 - 1) : 0;
    if (distance <= bestDistance) {
      bestDistance = distance;
      best = Math.round((band.y0 + band.y1) / 2);
    }
  }
  return best;
}

/** Index of the cut within `tolerance` rows of `y` (the closest), or -1. */
export function nearestCut(cuts: number[], y: number, tolerance: number): number {
  let index = -1;
  let distance = tolerance;
  cuts.forEach((cut, i) => {
    if (Math.abs(cut - y) <= distance) {
      distance = Math.abs(cut - y);
      index = i;
    }
  });
  return index;
}

/** The heights of the output images the cuts make (strip rows; filtered slices not subtracted). */
export function pieceHeights(cuts: number[], stripHeight: number): number[] {
  const bounds = [0, ...[...new Set(cuts)].filter((c) => c > 0 && c < stripHeight).sort((a, b) => a - b), stripHeight];
  return bounds.slice(1).map((bottom, i) => bottom - bounds[i]);
}
