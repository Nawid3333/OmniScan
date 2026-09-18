/** Pure helpers for the translation review table (no DOM, no Svelte). */

import type {
  CandidateRun,
  FinalArtifact,
  FinalLine,
  GlossaryHit,
  GlossaryHits,
  Region,
  RegionsArtifact,
} from "./api";

export interface ReviewRow {
  region: Region;
  candidates: Record<string, string | null>; // run_id -> candidate text for this region (null = that run has none)
  final: FinalLine | null;
  hits: GlossaryHit[];
  violations: GlossaryHit[]; // hits with status "locked" && target_in_final === false
  disagree: boolean; // see candidatesDisagree
  needsAttention: boolean;
}

/** Sort key order: slice_index asc, then reading_order asc, then id by numeric-aware comparison. Does not mutate. */
export function sortRegions(regions: Region[]): Region[] {
  return [...regions].sort(
    (a, b) => a.slice_index - b.slice_index || a.reading_order - b.reading_order || compareIds(a.id, b.id),
  );
}

/** Numeric-aware string comparison: digit runs compare as numbers, everything else as text ("r2" < "r10"). */
function compareIds(a: string, b: string): number {
  const partsA = a.split(/(\d+)/);
  const partsB = b.split(/(\d+)/);
  const length = Math.max(partsA.length, partsB.length);
  for (let i = 0; i < length; i++) {
    const isDigit = i % 2 === 1;
    const va = i < partsA.length ? (isDigit ? Number(partsA[i]) : partsA[i]) : isDigit ? 0 : "";
    const vb = i < partsB.length ? (isDigit ? Number(partsB[i]) : partsB[i]) : isDigit ? 0 : "";
    if (va < vb) return -1;
    if (va > vb) return 1;
  }
  return 0;
}

/** True when at least 2 runs have a non-null candidate AND their texts differ after trim() + toLowerCase(). */
export function candidatesDisagree(candidates: Record<string, string | null>): boolean {
  const texts = Object.values(candidates).filter((text): text is string => text !== null);
  if (texts.length < 2) {
    return false;
  }
  const first = texts[0].trim().toLowerCase();
  return texts.some((text) => text.trim().toLowerCase() !== first);
}

/** One row per region of `ocr`, in sortRegions order. `runs` may be empty, `final` and `hits` may be null. */
export function buildRows(
  ocr: RegionsArtifact,
  runs: CandidateRun[],
  final: FinalArtifact | null,
  hits: GlossaryHits | null,
): ReviewRow[] {
  const candidatesByRegion = new Map<string, Record<string, string | null>>();
  for (const run of runs) {
    const candidates: Record<string, string | null> = {};
    const seen = new Set<string>();
    for (const candidate of run.candidates) {
      if (!seen.has(candidate.region_id)) {
        seen.add(candidate.region_id);
        candidates[candidate.region_id] = candidate.text;
      }
    }
    candidatesByRegion.set(run.run_id, candidates);
  }
  const finalsByRegion = new Map<string, FinalLine>();
  for (const line of final?.lines ?? []) {
    if (!finalsByRegion.has(line.region_id)) {
      finalsByRegion.set(line.region_id, line);
    }
  }
  return sortRegions(ocr.regions).map((region) => {
    const candidates: Record<string, string | null> = {};
    for (const run of runs) {
      candidates[run.run_id] = candidatesByRegion.get(run.run_id)?.[region.id] ?? null;
    }
    const rowHits = hits?.regions[region.id] ?? [];
    const violations = rowHits.filter((hit) => hit.status === "locked" && hit.target_in_final === false);
    const disagree = candidatesDisagree(candidates);
    const rowFinal = finalsByRegion.get(region.id) ?? null;
    const noCandidateText = Object.values(candidates).every((text) => text === null);
    return {
      region,
      candidates,
      final: rowFinal,
      hits: rowHits,
      violations,
      disagree,
      needsAttention:
        (rowFinal !== null && rowFinal.flags.length > 0) ||
        violations.length > 0 ||
        disagree ||
        (rowFinal === null && runs.length > 0 && noCandidateText),
    };
  });
}

export interface Segment {
  text: string;
  hit: GlossaryHit | null;
}

/** Split `text` into consecutive segments so that each hit's [start, end) becomes one segment carrying that hit
 *  and the gaps become segments with hit === null. Hits are sorted by start; any hit that overlaps an earlier one or
 *  lies outside [0, text.length] is ignored; end is clamped to text.length. Concatenating all segment texts always
 *  reproduces `text` exactly. Empty text -> []. */
export function highlightSource(text: string, hits: GlossaryHit[]): Segment[] {
  if (text.length === 0) {
    return [];
  }
  const accepted: { hit: GlossaryHit; end: number }[] = [];
  let cursor = 0;
  for (const hit of [...hits].sort((a, b) => a.start - b.start)) {
    if (hit.start < 0 || hit.start > text.length || hit.start < cursor) {
      continue;
    }
    const end = Math.min(hit.end, text.length);
    if (end <= hit.start) {
      continue;
    }
    accepted.push({ hit: { ...hit, end }, end });
    cursor = end;
  }
  const segments: Segment[] = [];
  let position = 0;
  for (const { hit, end } of accepted) {
    if (hit.start > position) {
      segments.push({ text: text.slice(position, hit.start), hit: null });
    }
    segments.push({ text: text.slice(hit.start, end), hit });
    position = end;
  }
  if (position < text.length) {
    segments.push({ text: text.slice(position), hit: null });
  }
  return segments;
}

/** Counts over `rows`: total, with a final line, flagged, glossary violations, engine-run disagreements. */
export function summarizeReview(rows: ReviewRow[]): {
  total: number;
  withFinal: number;
  flagged: number;
  violations: number;
  disagreements: number;
} {
  return {
    total: rows.length,
    withFinal: rows.filter((row) => row.final !== null).length,
    flagged: rows.filter((row) => row.needsAttention).length,
    violations: rows.reduce((count, row) => count + row.violations.length, 0),
    disagreements: rows.filter((row) => row.disagree).length,
  };
}

/** Rows needing attention only when asked, always preserving input order. */
export function filterRows(rows: ReviewRow[], onlyAttention: boolean): ReviewRow[] {
  return onlyAttention ? rows.filter((row) => row.needsAttention) : rows;
}

/** Chip/mark colour for a glossary hit. */
export function hitColor(hit: GlossaryHit): string {
  if (hit.status === "locked" && hit.target_in_final === false) {
    return "#dc2626";
  }
  if (hit.status === "locked") {
    return "#16a34a";
  }
  if (hit.status === "proposed") {
    return "#d97706";
  }
  return "#6b7280";
}