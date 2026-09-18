/** Typed fetch wrappers for the OmniScan debug API (`omniscan serve`, default port 8000). */

export interface SourceFile {
  index: number;
  name: string;
  width: number;
  height: number;
  y0: number;
  y1: number;
  scale: number;
}

export interface IngestArtifact {
  strip_width: number;
  strip_height: number;
  files: SourceFile[];
}

export interface Band {
  y0: number;
  y1: number;
  color: [number, number, number];
  is_gradient: boolean;
}

export interface Slice {
  index: number;
  y0: number;
  y1: number;
  blank: boolean;
  forced_cut: boolean;
  filtered: boolean;
  source_files: number[];
}

export interface SlicesArtifact {
  strip_width: number;
  strip_height: number;
  bands: Band[];
  slices: Slice[];
}

export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type RegionKind = "bubble_text" | "free_text" | "sfx" | "watermark";

export interface OcrLine {
  bbox: BBox;
  polygon: [number, number][] | null;
  text: string;
  score: number;
  engine: string;
}

export interface Region {
  id: string;
  slice_index: number;
  kind: RegionKind;
  bbox: BBox;
  bubble_bbox: BBox | null;
  polygon: [number, number][] | null;
  reading_order: number;
  lang: "ko" | "zh" | "ja" | "en";
  orientation: "h" | "v";
  lines: OcrLine[];
  text: string;
  confidence: number;
  ocr_alt: string | null;
  text_color: [number, number, number] | null;
  stroke_color: [number, number, number] | null;
  mask_ref: string | null;
}

export interface RegionsArtifact {
  schema_version: number;
  regions: Region[];
}

const BASE = "/api";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} — ${path}`);
  }
  return (await res.json()) as T;
}

export async function listSeries(): Promise<string[]> {
  return getJson<string[]>(`${BASE}/series`);
}

export async function listChapters(series: string): Promise<string[]> {
  return getJson<string[]>(`${BASE}/series/${encodeURIComponent(series)}/chapters`);
}

export async function getIngest(series: string, chapter: string): Promise<IngestArtifact> {
  return getJson<IngestArtifact>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/ingest`,
  );
}

export async function getSlices(series: string, chapter: string): Promise<SlicesArtifact> {
  return getJson<SlicesArtifact>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/slices`,
  );
}

export async function getOcr(series: string, chapter: string): Promise<RegionsArtifact> {
  return getJson<RegionsArtifact>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/ocr`,
  );
}

export function pageImageUrl(series: string, chapter: string, index: number): string {
  return `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/pages/${index}`;
}