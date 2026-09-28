import type { ConsistencyReport, Divergence, SeriesTypo, TermMiss } from "./api";

/** The report narrowed to the entries whose source, English, term or word contains `query` (any case); all when blank. */
export function filterReport(report: ConsistencyReport, query: string): ConsistencyReport {
  const q = query.trim().toLowerCase();
  if (q === "") return report;
  const has = (text: string) => text.toLowerCase().includes(q);
  return {
    divergences: report.divergences.filter(
      (d: Divergence) => has(d.source) || d.renderings.some((r) => has(r.english)),
    ),
    term_misses: report.term_misses.filter((m: TermMiss) => has(m.term) || has(m.target) || has(m.english)),
    typos: report.typos.filter((t: SeriesTypo) => has(t.word) || has(t.english)),
  };
}

/** How many regions a divergence spans. */
export function placeCount(divergence: Divergence): number {
  return divergence.renderings.reduce((total, rendering) => total + rendering.places.length, 0);
}
