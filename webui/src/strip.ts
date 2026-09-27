/** Coordinate mapping between strip space (ingest.json/slices.json) and the stacked raw-image view. */

import type { BBox, SourceFile } from "./api";

/**
 * Convert a strip-space y-coordinate (as used in ingest.json/slices.json) into a y-offset in the DOM
 * where each file's raw image is stacked at its own NATURAL (unscaled) height, in file order, with no
 * gaps. Finds the file whose [y0, y1) strip-space range contains yStrip (last file whose y0 <= yStrip
 * if yStrip is beyond every file's y1, e.g. exactly at strip_height); maps proportionally within that
 * file. yStrip before the first file's y0 clamps to 0.
 */
export function stripYToStackY(files: SourceFile[], yStrip: number): number {
  let stackTop = 0;
  let result = 0;
  let found = false;
  for (const file of files) {
    if (file.y0 > yStrip) break;
    const span = file.y1 - file.y0;
    const fraction = span === 0 ? 0 : (yStrip - file.y0) / span;
    result = stackTop + fraction * file.height;
    stackTop += file.height;
    found = true;
  }
  return found ? result : 0;
}

/** Total stacked height: sum of every file's natural `height`. */
export function stackTotalHeight(files: SourceFile[]): number {
  return files.reduce((total, file) => total + file.height, 0);
}

/** Each file's top offset in the stack (cumulative sum of the earlier files' natural heights). */
export function stackTops(files: SourceFile[]): number[] {
  const tops: number[] = [];
  let top = 0;
  for (const file of files) {
    tops.push(top);
    top += file.height;
  }
  return tops;
}

/** The natural (unscaled) width to use for the stack container: the maximum `width` across `files`. */
export function stackWidth(files: SourceFile[]): number {
  return files.reduce((max, file) => Math.max(max, file.width), 0);
}

export interface StackBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * The file that owns strip row `yStrip`: the LAST file whose y0 <= yStrip; the FIRST file if yStrip
 * is above every file's y0. Precondition: files is non-empty.
 */
export function fileAtStripY(files: SourceFile[], yStrip: number): SourceFile {
  let owner = files[0];
  for (const file of files) {
    if (file.y0 > yStrip) break;
    owner = file;
  }
  return owner;
}

/**
 * Map a strip-space box into the stacked raw-image view.
 *   y      = stripYToStackY(files, box.y0)
 *   height = max(0, stripYToStackY(files, box.y1) - y)
 *   file   = fileAtStripY(files, box.y0)
 *   s      = file.scale > 0 ? file.scale : 1
 *   x      = box.x0 / s            (strip x -> that page's natural x)
 *   width  = (box.x1 - box.x0) / s
 * (The x scale of the page that contains the box's TOP edge is used for the whole box.)
 */
export function stripBoxToStackBox(files: SourceFile[], box: BBox): StackBox {
  const y = stripYToStackY(files, box.y0);
  const file = fileAtStripY(files, box.y0);
  const s = file.scale > 0 ? file.scale : 1;
  return {
    x: box.x0 / s,
    y,
    width: (box.x1 - box.x0) / s,
    height: Math.max(0, stripYToStackY(files, box.y1) - y),
  };
}
/**
 * Inverse of `stripYToStackY`: a y-offset in the stacked raw-image view -> the strip row (unrounded).
 * Offsets above the stack clamp to the first file's y0, below it to the last file's y1.
 */
export function stackYToStripY(files: SourceFile[], yStack: number): number {
  let top = 0;
  for (const file of files) {
    if (yStack < top + file.height || file === files[files.length - 1]) {
      const fraction = file.height === 0 ? 0 : Math.min(Math.max((yStack - top) / file.height, 0), 1);
      return file.y0 + fraction * (file.y1 - file.y0);
    }
    top += file.height;
  }
  return 0;
}
