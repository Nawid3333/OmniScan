/** Pure helpers for rendering typeset layout items (no DOM, no Svelte). */

import type { FontRole, LayoutItem } from "./api";

/** Box colour per font role. */
export function fontRoleColor(role: FontRole): string {
  switch (role) {
    case "dialogue":
      return "#1f6feb";
    case "thought":
      return "#0d9488";
    case "shout":
      return "#dc2626";
    case "narration":
      return "#9333ea";
    case "free":
      return "#d97706";
    case "sfx":
      return "#6b7280";
  }
}

/** Short label for a layout item's text alignment. */
export function alignLabel(align: "center" | "left" | "right"): string {
  switch (align) {
    case "center":
      return "center";
    case "left":
      return "left";
    case "right":
      return "right";
  }
}

/** Items whose font role is in `visibleRoles`, preserving input order. */
export function filterItems(items: LayoutItem[], visibleRoles: ReadonlySet<FontRole>): LayoutItem[] {
  return items.filter((item) => visibleRoles.has(item.font_role));
}