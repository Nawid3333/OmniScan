/** Pure helpers behind the Studio editor (no DOM, no Svelte): page <-> strip coordinates, box dragging. */

import type { BBox, ChapterEdits, FinalLine, LayoutEdit, LayoutFields, Region, SourceFile } from "./api";

/** A rectangle in one page's natural (unscaled) pixels. */
export interface PageRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Which part of a selected box a drag grabs: the body (move) or one of eight handles (resize). */
export type Handle = "move" | "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";

export const HANDLES: Exclude<Handle, "move">[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

/** Smallest box side (strip pixels) a resize may leave. */
export const MIN_BOX_PX = 4;

function scaleOf(file: SourceFile): number {
  return file.scale > 0 ? file.scale : 1;
}

/** `box` (strip space) as a rectangle in `file`'s natural pixels (may reach outside the page). */
export function stripToPage(file: SourceFile, box: BBox): PageRect {
  const s = scaleOf(file);
  return {
    x: box.x0 / s,
    y: (box.y0 - file.y0) / s,
    width: (box.x1 - box.x0) / s,
    height: (box.y1 - box.y0) / s,
  };
}

/** A point in `file`'s natural pixels -> strip space (unrounded). */
export function pageToStripPoint(file: SourceFile, x: number, y: number): { x: number; y: number } {
  const s = scaleOf(file);
  return { x: x * s, y: file.y0 + y * s };
}

/** The strip box spanned by two page points of `file`, in either order, rounded outward to integers. */
export function pageRectToStrip(file: SourceFile, ax: number, ay: number, bx: number, by: number): BBox {
  const a = pageToStripPoint(file, ax, ay);
  const b = pageToStripPoint(file, bx, by);
  return {
    x0: Math.floor(Math.min(a.x, b.x)),
    y0: Math.floor(Math.min(a.y, b.y)),
    x1: Math.ceil(Math.max(a.x, b.x)),
    y1: Math.ceil(Math.max(a.y, b.y)),
  };
}

/** True when `box` shares at least one strip row with `file`. */
export function overlapsPage(file: SourceFile, box: BBox): boolean {
  return box.y0 < file.y1 && box.y1 > file.y0;
}

/** The regions drawn on `file`'s page, in reading order (slice, then reading_order, then id). */
export function pageRegions(file: SourceFile, regions: Region[]): Region[] {
  return regions
    .filter((region) => overlapsPage(file, region.bbox))
    .sort(
      (a, b) =>
        a.slice_index - b.slice_index || a.reading_order - b.reading_order || a.id.localeCompare(b.id),
    );
}

/** Index of the page holding the centre of `box` (0 when no page does). */
export function pageOf(files: SourceFile[], box: BBox): number {
  const centre = (box.y0 + box.y1) / 2;
  const index = files.findIndex((file) => centre >= file.y0 && centre < file.y1);
  return index < 0 ? 0 : index;
}

/**
 * `box` dragged by (dx, dy) strip pixels at `handle`: "move" shifts the whole box, a side or corner
 * handle moves only its edges. Edges never cross (at least MIN_BOX_PX apart); results are integers.
 */
export function dragBox(box: BBox, handle: Handle, dx: number, dy: number): BBox {
  const rx = Math.round(dx);
  const ry = Math.round(dy);
  if (handle === "move") {
    return { x0: box.x0 + rx, y0: box.y0 + ry, x1: box.x1 + rx, y1: box.y1 + ry };
  }
  let { x0, y0, x1, y1 } = box;
  if (handle.includes("n")) y0 = Math.min(y0 + ry, y1 - MIN_BOX_PX);
  if (handle.includes("s")) y1 = Math.max(y1 + ry, y0 + MIN_BOX_PX);
  if (handle.includes("w")) x0 = Math.min(x0 + rx, x1 - MIN_BOX_PX);
  if (handle.includes("e")) x1 = Math.max(x1 + rx, x0 + MIN_BOX_PX);
  return { x0, y0, x1, y1 };
}

/** Where a handle sits on a page rectangle (for drawing it). */
export function handlePoint(rect: PageRect, handle: Exclude<Handle, "move">): { x: number; y: number } {
  const x = handle.includes("w") ? rect.x : handle.includes("e") ? rect.x + rect.width : rect.x + rect.width / 2;
  const y = handle.includes("n") ? rect.y : handle.includes("s") ? rect.y + rect.height : rect.y + rect.height / 2;
  return { x, y };
}

/** CSS cursor for a handle. */
export function handleCursor(handle: Handle): string {
  if (handle === "move") return "move";
  return `${handle}-resize`;
}

/** True when two boxes are identical. */
export function sameBox(a: BBox, b: BBox): boolean {
  return a.x0 === b.x0 && a.y0 === b.y0 && a.x1 === b.x1 && a.y1 === b.y1;
}

/** What the Studio's region list shows about one region. */
export interface RegionStatus {
  edited: boolean; // carries a hand edit (or was added by hand)
  added: boolean; // drawn by hand
  manualTranslation: boolean; // its English line was written by hand
  sourceChanged: boolean; // the source text changed after the English line was written
  untranslated: boolean; // translatable, but no English line yet
  lowConfidence: boolean; // OCR confidence below 0.5
}

/** A region's status from the edits summary and its final line (if any). */
export function regionStatus(region: Region, edits: ChapterEdits | null, line: FinalLine | undefined): RegionStatus {
  const translatable = region.kind !== "watermark" && region.text.trim() !== "";
  return {
    edited: edits?.edited_region_ids.includes(region.id) ?? false,
    added: region.id.startsWith("m"),
    manualTranslation: edits?.manual_translation_ids.includes(region.id) ?? false,
    sourceChanged: line?.flags.includes("source_changed") ?? false,
    untranslated: translatable && (line === undefined || line.text.trim() === ""),
    lowConfidence: region.confidence < 0.5,
  };
}

/** Short labels for a region's status, in display order. */
export function statusLabels(status: RegionStatus): string[] {
  const labels: string[] = [];
  if (status.added) labels.push("added");
  else if (status.edited) labels.push("edited");
  if (status.manualTranslation) labels.push("hand-translated");
  if (status.sourceChanged) labels.push("source changed");
  if (status.untranslated) labels.push("untranslated");
  if (status.lowConfidence) labels.push("low confidence");
  return labels;
}

/** Zoom steps offered by the Studio toolbar. */
export const ZOOMS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3];

/** The zoom that fits a page of `pageWidth` natural pixels into `available` screen pixels (capped at 1). */
export function fitZoom(pageWidth: number, available: number): number {
  if (pageWidth <= 0 || available <= 0) return 1;
  return Math.min(1, available / pageWidth);
}

/** Bounding box [x0, x1) x [y0, y1) of the pixels with any alpha in RGBA `data` (width x height); null
 *  when nothing is painted. */
export function maskBounds(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): { x0: number; y0: number; x1: number; y1: number } | null {
  let x0 = width;
  let y0 = height;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < height; y++) {
    const row = y * width * 4;
    for (let x = 0; x < width; x++) {
      if (data[row + x * 4 + 3] > 0) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { x0, y0, x1: x1 + 1, y1: y1 + 1 };
}

/** "#rrggbb" -> [r, g, b]. */
export function hexToRgb(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.replace("#", ""), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

/** [r, g, b] -> "#rrggbb". */
export function rgbToHex(rgb: [number, number, number]): string {
  return `#${rgb.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
}

/** The hand lettering fields actually set on `edit` (nulls dropped), for re-sending with one change. */
export function setFields(edit: LayoutEdit | undefined): LayoutFields {
  if (edit === undefined) return {};
  const fields: LayoutFields = {};
  for (const key of ["font", "size_px", "color", "stroke_px", "stroke_color", "align", "angle", "box", "lines"] as const) {
    if (edit[key] !== null) (fields as Record<string, unknown>)[key] = edit[key];
  }
  if (edit.hidden) fields.hidden = true;
  return fields;
}

/** Line breaks typed one per row -> the lines to send (none when the text is blank: automatic breaks). */
export function typedLines(text: string): string[] | undefined {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "");
  return lines.length > 0 ? lines : undefined;
}
