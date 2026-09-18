<script lang="ts">
  import { getIngest, getOcr, pageImageUrl } from "./api";
  import { stackTotalHeight, stackWidth, stripBoxToStackBox } from "./strip";
  import { disagrees, filterRegions, isLowConfidence, kindColor, summarize } from "./ocr";
  import type { Region, RegionKind, SourceFile } from "./api";

  let { series, chapter }: { series: string; chapter: string } = $props();

  const KINDS: RegionKind[] = ["bubble_text", "free_text", "sfx", "watermark"];

  let files = $state<SourceFile[]>([]);
  let regions = $state<Region[]>([]);
  let loaded = $state(false);
  let error = $state("");
  let selectedId = $state<string | null>(null);
  let visibleKinds = $state<Set<RegionKind>>(new Set(KINDS));

  let width = $derived(files.length > 0 ? stackWidth(files) : 0);
  let totalHeight = $derived(stackTotalHeight(files));
  let visible = $derived(filterRegions(regions, visibleKinds));
  let summary = $derived(summarize(visible));
  let selected = $derived(visible.find((r) => r.id === selectedId) ?? null);

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

  function toggleKind(kind: RegionKind): void {
    const next = new Set(visibleKinds);
    if (next.has(kind)) {
      next.delete(kind);
    } else {
      next.add(kind);
    }
    visibleKinds = next;
  }

  $effect(() => {
    const s = series;
    const c = chapter;
    error = "";
    loaded = false;
    files = [];
    regions = [];
    selectedId = null;
    visibleKinds = new Set(KINDS);
    void load(s, c);
  });

  async function load(series: string, chapter: string): Promise<void> {
    try {
      const [ingest, ocr] = await Promise.all([getIngest(series, chapter), getOcr(series, chapter)]);
      files = ingest.files;
      regions = ocr.regions;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      files = [];
      regions = [];
    }
    loaded = true;
  }
</script>

{#if error}
  <p>{error} — no ocr.json yet — run ocr first.</p>
{:else if !loaded}
  <p>loading…</p>
{:else if files.length === 0}
  <p>no pages</p>
{:else}
  <p style="margin: 4px 0;">
    {#each KINDS as kind (kind)}
      <label style="margin-right: 12px; color: {kindColor(kind)};">
        <input type="checkbox" checked={visibleKinds.has(kind)} onchange={() => toggleKind(kind)} />
        {kind}
      </label>
    {/each}
    <span>{summary.total} regions · {summary.lowConfidence} low-confidence · {summary.disagreements} disagreement</span>
  </p>
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
      {#each visible as region (region.id)}
        {@const box = stripBoxToStackBox(files, region.bbox)}
        {@const color = kindColor(region.kind)}
        {@const isSelected = region.id === selectedId}
        {#if region.bubble_bbox !== null}
          {@const bubble = stripBoxToStackBox(files, region.bubble_bbox)}
          <rect
            x={bubble.x}
            y={bubble.y}
            width={bubble.width}
            height={bubble.height}
            fill="none"
            stroke="#9ca3af"
            stroke-width="1"
            stroke-dasharray="2 3"
          />
        {/if}
        <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
        <rect
          x={box.x}
          y={box.y}
          width={box.width}
          height={box.height}
          fill={color}
          fill-opacity="0.08"
          stroke={color}
          stroke-width={isSelected ? 4 : 2}
          stroke-dasharray={isLowConfidence(region) ? "6 3" : undefined}
          pointer-events="all"
          onclick={() => (selectedId = selectedId === region.id ? null : region.id)}
        />
        {#if disagrees(region)}
          <rect
            x={box.x - 3}
            y={box.y - 3}
            width={box.width + 6}
            height={box.height + 6}
            fill="none"
            stroke="#eab308"
            stroke-width="3"
          />
        {/if}
        <text x={box.x} y={Math.max(13, box.y - 4)} fill={color} font-size="13">
          {region.reading_order} · {region.confidence.toFixed(2)}{disagrees(region) ? " ≠" : ""}
        </text>
      {/each}
    </svg>
  </div>
  {#if selected}
    <aside
      style="position: fixed; right: 0; top: 0; width: 320px; height: 100vh; overflow: auto; background: white; border-left: 1px solid #d1d5db; padding: 8px; box-sizing: border-box;"
    >
      <p style="margin-top: 0;">
        <button onclick={() => (selectedId = null)}>clear</button>
      </p>
      <dl>
        <dt>id</dt>
        <dd>{selected.id}</dd>
        <dt>kind</dt>
        <dd style="color: {kindColor(selected.kind)}">{selected.kind}</dd>
        <dt>lang</dt>
        <dd>{selected.lang}</dd>
        <dt>orientation</dt>
        <dd>{selected.orientation}</dd>
        <dt>reading_order</dt>
        <dd>{selected.reading_order}</dd>
        <dt>confidence</dt>
        <dd>{selected.confidence.toFixed(3)}</dd>
        <dt>slice_index</dt>
        <dd>{selected.slice_index}</dd>
        <dt>text</dt>
        <dd style="white-space: pre-wrap;">{selected.text}</dd>
        {#if selected.ocr_alt !== null}
          <dt>alt</dt>
          <dd style="white-space: pre-wrap;">{selected.ocr_alt}</dd>
        {/if}
        {#each selected.lines as line, i (i)}
          <dt>line {i}</dt>
          <dd>{line.engine} · {line.score.toFixed(3)} · {line.text}</dd>
        {/each}
      </dl>
    </aside>
  {/if}
{/if}