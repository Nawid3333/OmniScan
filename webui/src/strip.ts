/** Coordinate mapping between strip space (ingest.json/slices.json) and the stacked raw-image view. */

import type { SourceFile } from "./api";

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

/** The natural (unscaled) width to use for the stack container: the maximum `width` across `files`. */
export function stackWidth(files: SourceFile[]): number {
  return files.reduce((max, file) => Math.max(max, file.width), 0);
}