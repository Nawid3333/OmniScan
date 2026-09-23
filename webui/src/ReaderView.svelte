<script lang="ts">
  import RunButton from "./RunButton.svelte";
  import { getIngest, listOutput, outputImageUrl, pageImageUrl } from "./api";
  import type { SourceFile } from "./api";
  import {
    COLUMN_GAP,
    DEFAULT_COLUMN_WIDTH,
    MAX_COLUMN_WIDTH,
    MIN_COLUMN_WIDTH,
    fitColumnWidth,
    layoutStack,
    scrollRatio,
    scrollTopForRatio,
  } from "./reader";
  import type { ReaderMode, ScrollMetrics } from "./reader";

  let { series, chapter }: { series: string; chapter: string } = $props();

  let names = $state<string[]>([]);
  let rawFiles = $state<SourceFile[]>([]);
  let rawFailed = $state(false);
  let rawLoaded = $state(false);
  let loaded = $state(false);
  let mode = $state<ReaderMode>("final");
  let width = $state(DEFAULT_COLUMN_WIDTH);
  let viewportWidth = $state(window.innerWidth);
  let syncScroll = $state(true);

  let finalBox = $state<HTMLDivElement>();
  let rawBox = $state<HTMLDivElement>();
  let syncing = $state(false);

  let columnWidth = $derived(fitColumnWidth(mode, width, viewportWidth));
  // Both columns are laid out from the SAME file list and width, so they always land on identical
  // tops/totalHeight (page N's panel starts at the same y in both columns, even when a page in
  // `outputNames` is missing — that page's slot in the final column is just left blank, not
  // collapsed, so later pages never drift out of alignment). This is also what makes `scrollTop`
  // syncing exact below: with equal scrollHeight on both sides, copying the raw value is correct.
  let layout = $derived(rawFiles.length > 0 ? layoutStack(rawFiles, columnWidth) : null);
  let outputNames = $derived(new Set(names));

  $effect(() => {
    const s = series;
    const c = chapter;
    names = [];
    rawFiles = [];
    rawFailed = false;
    rawLoaded = false;
    loaded = false;
    mode = "final";
    void load(s, c);
  });

  $effect(() => {
    if (mode === "compare" && !rawFailed && !rawLoaded) {
      void loadRaw(series, chapter);
    }
  });

  async function load(s: string, c: string): Promise<void> {
    try {
      names = await listOutput(s, c);
    } catch {
      names = [];
    }
    loaded = true;
  }

  async function loadRaw(s: string, c: string): Promise<void> {
    try {
      const ingest = await getIngest(s, c);
      rawFiles = ingest.files.filter((file) => file.filtered !== true);
      rawLoaded = true;
    } catch {
      rawFailed = true;
      mode = "final";
    }
  }

  function metrics(box: HTMLDivElement): ScrollMetrics {
    return {
      scrollTop: box.scrollTop,
      scrollHeight: box.scrollHeight,
      clientHeight: box.clientHeight,
    };
  }

  function onScroll(source: "final" | "raw"): void {
    if (mode !== "compare" || !syncScroll || syncing) {
      return;
    }
    const sourceBox = source === "final" ? finalBox : rawBox;
    const targetBox = source === "final" ? rawBox : finalBox;
    if (!sourceBox || !targetBox) {
      return;
    }
    const scrollTop = scrollTopForRatio(scrollRatio(metrics(sourceBox)), metrics(targetBox));
    if (Math.abs(targetBox.scrollTop - scrollTop) > 1) {
      syncing = true;
      targetBox.scrollTop = scrollTop;
      requestAnimationFrame(() => (syncing = false));
    }
  }
</script>

<svelte:window onresize={() => (viewportWidth = window.innerWidth)} />

{#if !loaded}
  <p>loading…</p>
{:else if names.length === 0}
  <p>no output yet — the export stage has not produced images for this chapter</p>
  <RunButton {series} {chapter} through="export" onDone={() => load(series, chapter)} />
{:else}
  <p class="controls">
    <label>
      <input type="radio" name="reader-mode" value="final" bind:group={mode} />
      final only
    </label>
    <label>
      <input type="radio" name="reader-mode" value="compare" bind:group={mode} />
      raw | final
    </label>
    <label>
      width
      <input type="range" min={MIN_COLUMN_WIDTH} max={MAX_COLUMN_WIDTH} step={20} bind:value={width} />
      {width} px
    </label>
    {#if mode === "compare"}
      <label>
        <input type="checkbox" bind:checked={syncScroll} />
        sync scroll
      </label>
    {/if}
    <span>{names.length} output image(s)</span>
  </p>
  {#if rawFailed}
    <p style="color: #dc2626;">raw pages unavailable (no ingest.json) — showing final only</p>
  {/if}
  {#if mode === "final"}
    <div class="column" style="width: {columnWidth}px;" bind:this={finalBox}>
      {#each names as name (name)}
        <img src={outputImageUrl(series, chapter, name)} alt={name} />
      {/each}
    </div>
  {:else if layout === null}
    <p>loading raw pages…</p>
  {:else}
    <div class="pair" style="gap: {COLUMN_GAP}px;">
      <div
        class="column"
        style="width: {columnWidth}px;"
        bind:this={rawBox}
        onscroll={() => onScroll("raw")}
      >
        <div class="stack" style="width: {columnWidth}px; height: {layout.totalHeight}px;">
          {#each rawFiles as file, i (file.index)}
            <img
              src={pageImageUrl(series, chapter, file.index)}
              alt={file.name}
              style="top: {layout.tops[i]}px; width: {columnWidth}px; height: {layout.heights[i]}px;"
            />
          {/each}
        </div>
      </div>
      <div
        class="column"
        style="width: {columnWidth}px;"
        bind:this={finalBox}
        onscroll={() => onScroll("final")}
      >
        <div class="stack" style="width: {columnWidth}px; height: {layout.totalHeight}px;">
          {#each rawFiles as file, i (file.index)}
            {#if outputNames.has(file.name)}
              <img
                src={outputImageUrl(series, chapter, file.name)}
                alt={file.name}
                style="top: {layout.tops[i]}px; width: {columnWidth}px; height: {layout.heights[i]}px;"
              />
            {/if}
          {/each}
        </div>
      </div>
    </div>
  {/if}
{/if}

<style>
  .controls {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 4px 0;
  }
  .pair {
    display: flex;
    justify-content: center;
  }
  .column {
    flex-shrink: 0;
    overflow-y: auto;
    height: calc(100vh - 140px);
    margin: 0 auto;
    background: #f9fafb;
  }
  .column img {
    display: block;
    width: 100%;
    height: auto;
  }
  .stack {
    position: relative;
  }
  .stack img {
    position: absolute;
    left: 0;
  }
</style>