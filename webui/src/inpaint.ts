/** Pure helpers for rendering inpaint items (no DOM, no Svelte). */

import type { InpaintItem, InpaintMethod } from "./api";

/** Box colour per inpaint method. */
export function methodColor(method: InpaintMethod): string {
  switch (method) {
    case "flat":
      return "#1f6feb";
    case "lama":
      return "#9333ea";
    case "none":
      return "#6b7280";
  }
}

/** CSS background for a flat fill colour; "transparent" when null (nothing was painted). */
export function fillCss(fill: [number, number, number] | null): string {
  return fill === null ? "transparent" : `rgb(${fill[0]}, ${fill[1]}, ${fill[2]})`;
}

/** CSS clip-path revealing the LEFT `revealPercent`% of the clean layer, clamped to [0, 100]. */
export function cleanClipPath(revealPercent: number): string {
  const reveal = Math.min(100, Math.max(0, revealPercent));
  return `inset(0 ${100 - reveal}% 0 0)`;
}

/** True when the item's region id is in the set of ids that have a stored patch. */
export function hasStoredPatch(item: InpaintItem, patchRegionIds: ReadonlySet<string>): boolean {
  return patchRegionIds.has(item.region_id);
}