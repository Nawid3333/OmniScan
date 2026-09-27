<script lang="ts">
  import RunButton from "./RunButton.svelte";
  import { getCuts, getIngest, getSlices, pageImageUrl, putCuts } from "./api";
  import { nearestCut, pieceHeights, snapToBand } from "./cuts";
  import { stackTotalHeight, stackWidth, stackYToStripY, stripYToStackY } from "./strip";
  import type { CutsState, Slice, SourceFile, SlicesArtifact } from "./api";

  const SNAP_ROWS = 60; // a new or moved cut jumps to a uniform band this close
  const GRAB_PX = 8; // a click this close to a cut line grabs it

  let { series, chapter }: { series: string; chapter: string } = $props();

  let files = $state<SourceFile[]>([]);
  let slicesArtifact = $state<SlicesArtifact | null>(null);
  let loaded = $state(false);
  let error = $state("");
  let cutsState = $state<CutsState | null>(null);
  let editing = $state(false);
  let snap = $state(true);
  let draft = $state<number[]>([]); // the cuts being edited (strip rows)
  let moving: number | null = null; // index in draft of the cut being dragged
  let saving = $state(false);
  let cutError = $state("");
  let svgEl = $state<SVGSVGElement | null>(null);

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
    const slices = slicesArtifact.slices;
    // Slice.forced_cut means "the cut at THIS slice's y1 was forced" (core/schemas.py), so the line at
    // slice[i].y0 is red when the PREVIOUS slice's forced_cut is set. The first line has no preceding cut.
    slices.forEach((slice: Slice, i: number) => {
      lines.push({
        y: stripYToStackY(files, slice.y0),
        forced: i > 0 && slices[i - 1].forced_cut,
        label: String(slice.y1 - slice.y0),
      });
    });
    const last = slices.at(-1);
    if (last) {
      lines.push({ y: stripYToStackY(files, last.y1), forced: last.forced_cut, label: null });
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
    cutsState = null;
    editing = false;
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
      try {
        cutsState = await getCuts(series, chapter);
      } catch {
        cutsState = null;
      }
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      files = [];
      slicesArtifact = null;
    }
    loaded = true;
  }

  let handCuts = $derived(cutsState?.cuts ?? null);
  let shownCuts = $derived(editing ? draft : (handCuts ?? []));
  let crossingAt = $derived(new Map((cutsState?.crossings ?? []).map((c) => [c.cut, c.region_id])));
  let heights = $derived(cutsState ? pieceHeights(shownCuts, cutsState.strip_height) : []);

  function startEditing(): void {
    draft = [...(handCuts ?? cutsState?.auto ?? [])];
    editing = true;
    cutError = "";
  }

  async function save(cuts: number[] | null): Promise<void> {
    saving = true;
    cutError = "";
    try {
      cutsState = await putCuts(series, chapter, cuts);
      draft = [...(cutsState.cuts ?? cutsState.auto)];
    } catch (e) {
      cutError = e instanceof Error ? e.message : String(e);
    } finally {
      saving = false;
    }
  }

  function stripRowAt(event: PointerEvent): number {
    const rect = svgEl!.getBoundingClientRect();
    return Math.round(stackYToStripY(files, event.clientY - rect.top));
  }

  function placed(y: number): number {
    const row = snap && slicesArtifact ? snapToBand(y, slicesArtifact.bands, SNAP_ROWS) : y;
    return Math.min(Math.max(row, 1), (cutsState?.strip_height ?? 2) - 1);
  }

  function onDown(event: PointerEvent): void {
    if (!editing || saving) return;
    const y = stripRowAt(event);
    const tolerance = Math.abs(stackYToStripY(files, GRAB_PX) - stackYToStripY(files, 0));
    const index = nearestCut(draft, y, tolerance);
    if (index >= 0) {
      moving = index;
      svgEl!.setPointerCapture(event.pointerId);
    } else {
      void save([...draft, placed(y)]);
    }
  }

  function onMove(event: PointerEvent): void {
    if (moving === null) return;
    const next = [...draft];
    next[moving] = Math.round(stackYToStripY(files, event.clientY - svgEl!.getBoundingClientRect().top));
    draft = next;
  }

  function onUp(): void {
    if (moving === null) return;
    const next = [...draft];
    next[moving] = placed(next[moving]);
    moving = null;
    void save(next);
  }

  function removeCut(index: number): void {
    void save(draft.filter((_, i) => i !== index));
  }
</script>

{#if error}
  <p>{error} — no ingest.json/slices.json yet? run ingest and slice first.</p>
  <RunButton {series} {chapter} through="slice" onDone={() => load(series, chapter)} />
{:else if !loaded}
  <p>loading…</p>
{:else if files.length === 0}
  <p>no pages</p>
{:else}
  <p style="margin: 4px 0;">
    {#if editing}
      <b>Editing output cuts</b> — click to add a cut, drag a cut to move it, × removes it.
      <label><input type="checkbox" bind:checked={snap} /> snap to calm rows</label>
      <button onclick={() => void save(null)} disabled={saving}>Reset to one image per slice</button>
      <button onclick={() => (editing = false)}>Done</button>
      {#if saving}<span>saving…</span>{/if}
    {:else}
      <button onclick={startEditing} disabled={cutsState === null}>Edit output cuts</button>
      {#if handCuts !== null}
        <span style="color: #c2410c;">the export splits at {handCuts.length} hand-set cuts (orange)</span>
      {/if}
    {/if}
    {#if cutsState !== null && (editing || handCuts !== null)}
      <span>· {heights.length} images, {Math.min(...heights)}–{Math.max(...heights)} px tall</span>
    {/if}
    {#if (cutsState?.crossings.length ?? 0) > 0}
      <span style="color: #dc2626;">· {cutsState?.crossings.length} cut(s) run through lettering (red)</span>
    {/if}
  </p>
  {#if cutError}<p style="color: #dc2626;">{cutError}</p>{/if}
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
      bind:this={svgEl}
      width={width}
      height={totalHeight}
      viewBox="0 0 {width} {totalHeight}"
      style="position: absolute; left: 0; top: 0; pointer-events: {editing ? 'all' : 'none'}; cursor: {editing ? 'row-resize' : 'default'};"
      onpointerdown={onDown}
      onpointermove={onMove}
      onpointerup={onUp}
      role="application"
      aria-label="output cuts"
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
      {#each shownCuts as cut, i (i)}
        {@const y = stripYToStackY(files, cut)}
        {@const crossed = crossingAt.get(cut)}
        <line
          x1="0"
          x2={width}
          y1={y}
          y2={y}
          stroke={crossed ? "#dc2626" : "#ea580c"}
          stroke-width="3"
          stroke-dasharray="12 6"
          aria-label="output cut {cut}"
        />
        <text x={width - 150} y={Math.max(14, y - 5)} fill={crossed ? "#dc2626" : "#ea580c"} font-size="14">
          {crossed ? `cuts through ${crossed}` : `cut at ${cut}`}
        </text>
        {#if editing}
          <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
          <text
            x={width - 22}
            y={y + 5}
            fill="#dc2626"
            font-size="20"
            style="cursor: pointer;"
            onpointerdown={(e) => {
              e.stopPropagation();
              removeCut(i);
            }}
            aria-label="remove cut {cut}">×</text
          >
        {/if}
      {/each}
    </svg>
  </div>
{/if}