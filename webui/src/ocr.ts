/** Pure helpers for rendering OCR regions (no DOM, no Svelte). */

import type { Region, RegionKind } from "./api";

export const LOW_CONFIDENCE = 0.5;

/** Box colour per region kind. */
export function kindColor(kind: RegionKind): string {
  switch (kind) {
    case "bubble_text":
      return "#1f6feb";
    case "free_text":
      return "#d97706";
    case "sfx":
      return "#9333ea";
    case "watermark":
      return "#6b7280";
  }
}

/** True when the second-opinion reading exists and differs from the primary text. */
export function disagrees(region: Region): boolean {
  return region.ocr_alt !== null && region.ocr_alt !== region.text;
}

/** True when the region's confidence is strictly below `threshold`. */
export function isLowConfidence(region: Region, threshold: number = LOW_CONFIDENCE): boolean {
  return region.confidence < threshold;
}

/** Regions whose kind is in `visibleKinds`, preserving input order. */
export function filterRegions(regions: Region[], visibleKinds: ReadonlySet<RegionKind>): Region[] {
  return regions.filter((region) => visibleKinds.has(region.kind));
}

/** Counts over `regions`: total, low-confidence, engine disagreements. */
export function summarize(regions: Region[]): {
  total: number;
  lowConfidence: number;
  disagreements: number;
} {
  return {
    total: regions.length,
    lowConfidence: regions.filter((r) => isLowConfidence(r)).length,
    disagreements: regions.filter((r) => disagrees(r)).length,
  };
}