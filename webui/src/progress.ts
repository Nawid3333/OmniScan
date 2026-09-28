/** Pure helpers for the Progress view (no DOM, no Svelte). */

import type { ChapterProgress } from "./api";

/** How far the pipeline got, in words: "failed at inpaint", "through typeset" or "not started". */
export function pipelineState(progress: ChapterProgress): string {
  if (progress.failed !== null) return `failed at ${progress.failed}`;
  return progress.last_stage !== null ? `through ${progress.last_stage}` : "not started";
}

/** The export state in words: "outdated", "exported" or "not exported". */
export function exportState(progress: ChapterProgress): string {
  if (!progress.exported) return "not exported";
  return progress.outdated ? "outdated" : "exported";
}

/** `part` of `whole` as a whole percentage (100 for an empty whole: nothing is left). */
export function percent(part: number, whole: number): number {
  return whole === 0 ? 100 : Math.round((100 * part) / whole);
}

/** Whether a chapter is ready to release: everything translated and checked, no open problem, exported and current. */
export function isDone(progress: ChapterProgress): boolean {
  return (
    progress.lines > 0 &&
    progress.translated === progress.lines &&
    progress.checked === progress.lines &&
    progress.problems === 0 &&
    progress.exported &&
    !progress.outdated
  );
}

/** Totals over the series: lines, translated, checked, open problems and chapters done. */
export function totals(chapters: ChapterProgress[]): {
  lines: number;
  translated: number;
  checked: number;
  problems: number;
  done: number;
} {
  return chapters.reduce(
    (sum, c) => ({
      lines: sum.lines + c.lines,
      translated: sum.translated + c.translated,
      checked: sum.checked + c.checked,
      problems: sum.problems + c.problems,
      done: sum.done + (isDone(c) ? 1 : 0),
    }),
    { lines: 0, translated: 0, checked: 0, problems: 0, done: 0 },
  );
}
