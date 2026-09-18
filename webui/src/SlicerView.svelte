<script lang="ts">
  import { getIngest, getSlices, pageImageUrl } from "./api";
  import { stackTotalHeight, stackWidth, stripYToStackY } from "./strip";
  import type { Slice, SourceFile, SlicesArtifact } from "./api";

  let { series, chapter }: { series: string; chapter: string } = $props();

  let files = $state<SourceFile[]>([]);
  let slicesArtifact = $state<SlicesArtifact | null>(null);
  let loaded = $state(false);
  let error = $state("");

  let width = $derived(files.length > 0 ? stackWidth(files) : 0);
  let totalHeight = $derived(stackTotalHeight(files));

  // Top of each file's image in the stack: sum of all earlier files' natural heights.
  let stackTops = $derived.by(() => {
    const tops: number[] = [];
    let top = 0;
    for (const file of files) {
      tops.push(top);
      top += file.height;
    }
    return tops;
  });

  interface BandRect {
    y: number;
    height: number;
  }

  let bandRects = $derived.by((): BandRect[] => {
    if (!slicesArtifact || files.length === 0) return [];
    return slicesArtifact.bands.map((band) => {
      const y = stripYToStackY(files, band.y0);
      return { y, height: Math.max(0, stripYToStackY(files, band.y1) - y) };
    });
  });

  interface CutLine {
    y: number;
    forced: boolean;
    label: string | null;
  }

  let cutLines = $derived.by((): CutLine[] => {
    if (!slicesArtifact || files.length === 0) return [];
    const lines: CutLine[] = [];
    const addBoundary = (yStrip: number, slice: Slice | null): void => {
      lines.push({
        y: stripYToStackY(files, yStrip),
        forced: slice !== null && slice.forced_cut,
        label: slice === null ? null : String(slice.y1 - slice.y0),
      });
    };
    for (const slice of slicesArtifact.slices) {
      addBoundary(slice.y0, slice);
    }
    const last = slicesArtifact.slices.at(-1);
    if (last) {
      addBoundary(last.y1, null);
    }
    return lines;
  });

  // Original raw-file boundaries: every file after the first.
  let fileBoundaries = $derived(
    files.slice(1).map((file) => stripYToStackY(files, file.y0)),
  );

  $effect(() => {
    const s = series;
    const c = chapter;
    error = "";
    loaded = false;
    files = [];
    slicesArtifact = null;
    void load(s, c);
  });

  async function load(series: string, chapter: string): Promise<void> {
    try {
      const [ingest, slices] = await Promise.all([
        getIngest(series, chapter),
        getSlices(series, chapter),
      ]);
      files = ingest.files;
      slicesArtifact = slices;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      files = [];
      slicesArtifact = null;
    }
    loaded = true;
  }
</script>

{#if error}
  <p>{error} — no ingest.json/slices.json yet? run ingest and slice first.</p>
{:else if !loaded}
  <p>loading…</p>
{:else if files.length === 0}
  <p>no pages</p>
{:else}
  <div style="position: relative; width: {width}px; height: {totalHeight}px;">
    {#each files as file, i (file.index)}
      <img
        src={pageImageUrl(series, chapter, file.index)}
        alt={file.name}
        width={file.width}
        height={file.height}
        style="position: absolute; left: 0; top: {stackTops[i]}px;"
      />
    {/each}
    <svg
      width={width}
      height={totalHeight}
      viewBox="0 0 {width} {totalHeight}"
      style="position: absolute; left: 0; top: 0; pointer-events: none;"
    >
      {#each bandRects as band, i (i)}
        <rect x="0" y={band.y} width={width} height={band.height} fill="rgba(0, 190, 0, 0.15)" />
      {/each}
      {#each cutLines as line, i (i)}
        <line
          x1="0"
          x2={width}
          y1={line.y}
          y2={line.y}
          stroke={line.forced ? "#cc0000" : "#0a1a4a"}
          stroke-width="2"
        />
        {#if line.label !== null}
          <text x="8" y={Math.max(12, line.y - 4)} fill={line.forced ? "#cc0000" : "#0a1a4a"} font-size="14">
            {line.label}
          </text>
        {/if}
      {/each}
      {#each fileBoundaries as y, i (i)}
        <line x1="0" x2={width} y1={y} y2={y} stroke="gray" stroke-width="1" stroke-dasharray="8 4" />
      {/each}
    </svg>
  </div>
{/if}