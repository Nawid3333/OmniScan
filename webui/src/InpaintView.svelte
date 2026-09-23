<script lang="ts">
  import { getIngest, getInpaint, inpaintPatchUrl, pageImageUrl } from "./api";
  import { stackTotalHeight, stackWidth, stripBoxToStackBox } from "./strip";
  import { cleanClipPath, fillCss, hasStoredPatch, methodColor } from "./inpaint";
  import type { InpaintItem, SourceFile } from "./api";

  let { series, chapter }: { series: string; chapter: string } = $props();

  let files = $state<SourceFile[]>([]);
  let items = $state<InpaintItem[]>([]);
  let loaded = $state(false);
  let error = $state("");
  let selectedId = $state<string | null>(null);
  let reveal = $state(50);
  let showOutlines = $state(true);
  // Region ids whose patch <img> failed to load (no entry in patches.npz): they fall back to the
  // flat-fill colour for that one item instead of predicting which ids are stored.
  let patchFailed = $state<Set<string>>(new Set());

  let width = $derived(files.length > 0 ? stackWidth(files) : 0);
  let totalHeight = $derived(stackTotalHeight(files));
  let visible = $derived(showOutlines ? items : []);
  let needsLama = $derived(items.filter((item) => item.needs_lama).length);
  let patchIds = $derived(
    new Set(
      items
        .filter((item) => item.method !== "none" && !patchFailed.has(item.region_id))
        .map((item) => item.region_id),
    ),
  );
  let selected = $derived(items.find((item) => item.region_id === selectedId) ?? null);

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

  function markPatchFailed(regionId: string): void {
    const next = new Set(patchFailed);
    next.add(regionId);
    patchFailed = next;
  }

  $effect(() => {
    const s = series;
    const c = chapter;
    error = "";
    loaded = false;
    files = [];
    items = [];
    selectedId = null;
    patchFailed = new Set();
    void load(s, c);
  });

  async function load(series: string, chapter: string): Promise<void> {
    try {
      const [ingest, inpaint] = await Promise.all([getIngest(series, chapter), getInpaint(series, chapter)]);
      files = ingest.files;
      items = inpaint.items;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      files = [];
      items = [];
    }
    loaded = true;
  }
</script>

{#if error}
  <p>{error} — no inpaint.json yet — run inpaint first.</p>
{:else if !loaded}
  <p>loading…</p>
{:else if files.length === 0}
  <p>no pages</p>
{:else}
  <p style="margin: 4px 0;">
    <input type="range" min="0" max="100" bind:value={reveal} />
    <span>raw ↔ clean ({reveal}% clean)</span>
    <label style="margin-left: 12px;">
      <input type="checkbox" bind:checked={showOutlines} />
      show patch outlines
    </label>
    <span style="margin-left: 12px;">{items.length} regions · {needsLama} need lama</span>
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
    <!-- clean layer: patches positioned where export would apply them -->
    <div style="position: absolute; inset: 0; clip-path: {cleanClipPath(reveal)};">
      {#each items as item (item.region_id)}
        {@const box = stripBoxToStackBox(files, item.box)}
        {#if item.method !== "none"}
          {#if hasStoredPatch(item, patchIds)}
            <img
              src={inpaintPatchUrl(series, chapter, item.region_id)}
              alt={item.region_id}
              style="position: absolute; left: {box.x}px; top: {box.y}px; width: {box.width}px; height: {box.height}px;"
              onerror={() => markPatchFailed(item.region_id)}
            />
          {:else}
            <div
              style="position: absolute; left: {box.x}px; top: {box.y}px; width: {box.width}px; height: {box.height}px; background: {fillCss(item.fill)};"
            ></div>
          {/if}
        {/if}
      {/each}
    </div>
    <svg
      width={width}
      height={totalHeight}
      viewBox="0 0 {width} {totalHeight}"
      style="position: absolute; left: 0; top: 0; pointer-events: none;"
    >
      {#each visible as item (item.region_id)}
        {@const box = stripBoxToStackBox(files, item.box)}
        {@const color = methodColor(item.method)}
        {@const isSelected = item.region_id === selectedId}
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
          stroke-dasharray={item.method === "none" ? "6 3" : undefined}
          pointer-events="all"
          onclick={() => (selectedId = selectedId === item.region_id ? null : item.region_id)}
        />
        <text x={box.x} y={Math.max(13, box.y - 4)} fill={color} font-size="13">
          {item.method}{item.needs_lama ? " · needs lama" : ""}
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
        <dt>region_id</dt>
        <dd>{selected.region_id}</dd>
        <dt>method</dt>
        <dd><span style="color: {methodColor(selected.method)}">●</span> {selected.method}</dd>
        <dt>needs_lama</dt>
        <dd>{selected.needs_lama}</dd>
        <dt>mask_px</dt>
        <dd>{selected.mask_px}</dd>
        {#if selected.fill !== null}
          <dt>fill</dt>
          <dd>
            <span
              style="display: inline-block; width: 14px; height: 14px; background: {fillCss(selected.fill)}; vertical-align: -2px; border: 1px solid #d1d5db;"
            ></span>
            {fillCss(selected.fill)}
          </dd>
        {/if}
        {#if hasStoredPatch(selected, patchIds)}
          <dt>patch</dt>
          <dd><img src={inpaintPatchUrl(series, chapter, selected.region_id)} alt={selected.region_id} style="max-width: 300px;" /></dd>
        {/if}
      </dl>
    </aside>
  {/if}
{/if}