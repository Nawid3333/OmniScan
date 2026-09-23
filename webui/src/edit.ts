/** Pure helpers for the Edit view's in-place text overlay (no DOM, no Svelte). */

import type { FinalLine, LayoutItem } from "./api";
import { alignLabel } from "./layout";

/**
 * region_id -> current final text. Last entry wins for a duplicate region_id (should not happen
 * from a real final.json); text defaults to "" for regions absent from final.json.
 */
export function textByRegion(lines: FinalLine[]): Record<string, string> {
  return Object.fromEntries(lines.map((line) => [line.region_id, line.text]));
}

/**
 * Rough, readable inline style for one overlay text box (not pixel-exact typesetting): the item's
 * size, colour, alignment and a generic sans-serif. `item.font` is deliberately ignored (a real
 * font preview already exists as the exported page in the Reader view).
 */
export function overlayStyle(item: LayoutItem): string {
  const [r, g, b] = item.color;
  return `font-size: ${item.size_px}px; color: rgb(${r}, ${g}, ${b}); text-align: ${alignLabel(item.align)}; white-space: pre-wrap; font-family: sans-serif;`;
}