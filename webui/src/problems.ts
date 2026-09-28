/** Pure helpers for the Problems view (no DOM, no Svelte). */

import type { Problem } from "./api";

/** A problem kind in words, for the filter and the list. */
export const KIND_LABELS: Record<string, string> = {
  untranslated: "no English",
  source_left: "source text left",
  watermark_left: "watermark left",
  overflow: "lettering overflows",
  uncertain: "judge unsure",
  glossary: "glossary term missing",
  too_long: "much too long",
  typo: "possible typo",
};

/** The label of `kind` (the kind itself when it has none). */
export function kindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? kind;
}

/** The problems to show: of `kind` ("" for all), without the checked lines' when `hideChecked`. */
export function filterProblems(problems: Problem[], kind: string, hideChecked: boolean): Problem[] {
  return problems.filter(
    (problem) => (kind === "" || problem.kind === kind) && !(hideChecked && problem.status === "checked"),
  );
}

/** How many problems each kind has, in the order the kinds first appear. */
export function kindCounts(problems: Problem[]): { kind: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const problem of problems) counts.set(problem.kind, (counts.get(problem.kind) ?? 0) + 1);
  return [...counts].map(([kind, count]) => ({ kind, count }));
}

/** How many regions have at least one problem. */
export function regionCount(problems: Problem[]): number {
  return new Set(problems.map((problem) => problem.region_id)).size;
}
