/** Typed fetch wrappers for the OmniScan debug API (`omniscan serve`, default port 8000). */

export interface SourceFile {
  index: number;
  name: string;
  width: number;
  height: number;
  y0: number;
  y1: number;
  scale: number;
  /** removed by the file-level promo pre-check (not part of the strip); absent in pre-B12 artifacts */
  filtered?: boolean;
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

export interface Candidate {
  region_id: string;
  text: string;
  notes: string | null;
}

export interface CandidateRun {
  schema_version: number;
  run_id: string;
  profile: string;
  model: string;
  created_at: string;
  candidates: Candidate[];
  usage: Record<string, number>;
}

export interface FinalLine {
  region_id: string;
  text: string;
  decision: "pick" | "merge" | "rewrite" | "manual";
  sources: string[];
  rationale: string;
  flags: string[];
}

export interface FinalArtifact {
  schema_version: number;
  judge_model: string;
  created_at: string;
  lines: FinalLine[];
}

export interface GlossaryEntry {
  id: number | null;
  source: string;
  target: string;
  type: string;
  gender: string;
  pronouns: string | null;
  aliases: string[];
  notes: string | null;
  status: "proposed" | "locked" | "rejected";
  origin: "llm" | "reference" | "user";
  first_seen_chapter: number | null;
  count: number;
}

export interface GlossaryHit {
  entry_id: number;
  source: string;
  target: string;
  status: string;
  start: number;
  end: number;
  particle: string | null;
  target_in_final: boolean | null;
}

export interface GlossaryHits {
  regions: Record<string, GlossaryHit[]>;
}

export interface FilteredItem {
  target: "file" | "slice";
  index: number;
  state: "filtered" | "restored";
  score: number;
  matched_example: string | null;
  method: string;
  name: string | null;
  y0: number | null;
  y1: number | null;
}

export interface FilteredChapter {
  chapter: string;
  items: FilteredItem[];
}

export type InpaintMethod = "flat" | "lama" | "none";

export interface InpaintItem {
  region_id: string;
  box: BBox;
  method: InpaintMethod;
  fill: [number, number, number] | null;
  needs_lama: boolean;
  mask_px: number;
}

export interface InpaintArtifact {
  schema_version: number;
  items: InpaintItem[];
}

export type FontRole = "dialogue" | "thought" | "shout" | "narration" | "free" | "sfx";

export interface LayoutItem {
  region_id: string;
  font_role: FontRole;
  font: string;
  size_px: number;
  lines: string[];
  box: BBox;
  align: "center" | "left" | "right";
  color: [number, number, number];
  stroke_px: number;
  stroke_color: [number, number, number];
  overflow: boolean;
}

export interface LayoutArtifact {
  schema_version: number;
  items: LayoutItem[];
}

const BASE = "/api";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText} — ${path}`);
  }
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  return sendJson<T>(path, "POST", body);
}

async function putJson<T>(path: string, body: unknown): Promise<T> {
  return sendJson<T>(path, "PUT", body);
}

async function sendJson<T>(path: string, method: "POST" | "PUT", body: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = "";
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data.detail === "string") {
        detail = ` — ${data.detail}`;
      }
    } catch {
      // body was not JSON: nothing to add
    }
    throw new Error(`${res.status} ${res.statusText} — ${path}${detail}`);
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

export async function listTranslations(series: string, chapter: string): Promise<string[]> {
  return getJson<string[]>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/translations`,
  );
}

export async function getTranslation(series: string, chapter: string, runId: string): Promise<CandidateRun> {
  return getJson<CandidateRun>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/translations/${encodeURIComponent(runId)}`,
  );
}

export async function getFinal(series: string, chapter: string): Promise<FinalArtifact> {
  return getJson<FinalArtifact>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/final`,
  );
}

export async function getGlossary(series: string): Promise<GlossaryEntry[]> {
  return getJson<GlossaryEntry[]>(`${BASE}/series/${encodeURIComponent(series)}/glossary`);
}

export async function getGlossaryHits(series: string, chapter: string): Promise<GlossaryHits> {
  return getJson<GlossaryHits>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/glossary-hits`,
  );
}

export async function listOutput(series: string, chapter: string): Promise<string[]> {
  return getJson<string[]>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/output`,
  );
}

export function outputImageUrl(series: string, chapter: string, name: string): string {
  return `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/output/${encodeURIComponent(name)}`;
}

export async function listFiltered(series: string): Promise<FilteredChapter[]> {
  return getJson<FilteredChapter[]>(`${BASE}/series/${encodeURIComponent(series)}/filtered`);
}

export async function getInpaint(series: string, chapter: string): Promise<InpaintArtifact> {
  return getJson<InpaintArtifact>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/inpaint`,
  );
}

export function inpaintPatchUrl(series: string, chapter: string, regionId: string): string {
  return `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/inpaint/patches/${encodeURIComponent(regionId)}.png`;
}

export async function getLayout(series: string, chapter: string): Promise<LayoutArtifact> {
  return getJson<LayoutArtifact>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/layout`,
  );
}

export async function restoreFiltered(
  series: string,
  chapter: string,
  target: "file" | "slice",
  index: number,
): Promise<void> {
  await postJson<unknown>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/filter/restore`,
    { target, index },
  );
}

export interface RunResult {
  job_id: number;
  stages: string[];
}

/** Queue every pipeline stage through `through` (inclusive, e.g. "ocr", "typeset", "export") for
 *  one chapter. Only actually executes while the server was started with a worker (the real
 *  `omniscan serve`); poll `getJob` with the returned id for progress. */
export async function postRun(series: string, chapter: string, through: string): Promise<RunResult> {
  return postJson<RunResult>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/run`,
    { through },
  );
}

export type JobStatus = "queued" | "running" | "paused" | "done" | "failed" | "cancelled";

export interface Job {
  id: number;
  series: string;
  chapters: string[] | null;
  stages: string[];
  status: JobStatus;
  attempts: number;
  max_attempts: number;
  error: string | null;
}

export async function getJob(jobId: number): Promise<Job> {
  return getJson<Job>(`${BASE}/jobs/${jobId}`);
}

/** Overwrite one region's final line; the server sets its decision to "manual". */
export async function putFinalLine(series: string, chapter: string, regionId: string, text: string): Promise<FinalLine> {
  return putJson<FinalLine>(
    `${BASE}/series/${encodeURIComponent(series)}/chapters/${encodeURIComponent(chapter)}/final/${encodeURIComponent(regionId)}`,
    { text },
  );
}