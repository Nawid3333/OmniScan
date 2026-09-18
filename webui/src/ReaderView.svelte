<script lang="ts">
  import { getIngest, listOutput, outputImageUrl, pageImageUrl } from "./api";
  import type { SourceFile } from "./api";
  import {
    COLUMN_GAP,
    DEFAULT_COLUMN_WIDTH,
    MAX_COLUMN_WIDTH,
    MIN_COLUMN_WIDTH,
    fitColumnWidth,
    scrollRatio,
    scrollTopForRatio,
  } from "./reader";
  import type { ReaderMode, ScrollMetrics } from "./reader";

  let { series, chapter }: { series: string; chapter: string } = $props();

  let names = $state<string[]>([]);
  let rawFiles = $state<SourceFile[]>([]);
  let rawFailed = $state(false);
  let loaded = $state(false);
  let mode = $state<ReaderMode>("final");
  let width = $state(DEFAULT_COLUMN_WIDTH);
  let viewportWidth = $state(window.innerWidth);

  let finalBox = $state<HTMLDivElement>();
  let rawBox = $state<HTMLDivElement>();
  let syncing = $state(false);

  let columnWidth = $derived(fitColumnWidth(mode, width, viewportWidth));

  $effect(() => {
    const s = series;
    const c = chapter;
    names = [];
    rawFiles = [];
    rawFailed = false;
    loaded = false;
    mode = "final";
    void load(s, c);
  });

  $effect(() => {
    if (mode === "compare" && !rawFailed && rawFiles.length === 0) {
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
    if (mode !== "compare" || syncing) {
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
  {:else}
    <div class="pair" style="gap: {COLUMN_GAP}px;">
      <div
        class="column"
        style="width: {columnWidth}px;"
        bind:this={rawBox}
        onscroll={() => onScroll("raw")}
      >
        {#each rawFiles as file (file.index)}
          <img src={pageImageUrl(series, chapter, file.index)} alt={file.name} />
        {/each}
      </div>
      <div
        class="column"
        style="width: {columnWidth}px;"
        bind:this={finalBox}
        onscroll={() => onScroll("final")}
      >
        {#each names as name (name)}
          <img src={outputImageUrl(series, chapter, name)} alt={name} />
        {/each}
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
</style>