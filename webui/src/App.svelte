<script lang="ts">
  import { onMount } from "svelte";

  import OcrView from "./OcrView.svelte";
  import ReaderView from "./ReaderView.svelte";
  import SlicerView from "./SlicerView.svelte";
  import TranslationView from "./TranslationView.svelte";
  import { listChapters, listSeries } from "./api";

  let view = $state<"slicer" | "ocr" | "translation" | "reader">("slicer");
  let series = $state("");
  let chapter = $state("");
  let seriesList = $state<string[]>([]);
  let chapterList = $state<string[]>([]);
  let loadError = $state("");

  onMount(async () => {
    try {
      seriesList = await listSeries();
    } catch (e) {
      loadError = `could not load series list: ${e instanceof Error ? e.message : String(e)}`;
    }
  });

  async function onSeriesChange(): Promise<void> {
    chapter = "";
    chapterList = [];
    loadError = "";
    if (!series) return;
    try {
      chapterList = await listChapters(series);
    } catch (e) {
      loadError = `could not load chapters: ${e instanceof Error ? e.message : String(e)}`;
    }
  }
</script>

<main>
  <h1>OmniScan debug</h1>
  <p>
    <select bind:value={series} onchange={onSeriesChange}>
      <option value="">— series —</option>
      {#each seriesList as s (s)}
        <option value={s}>{s}</option>
      {/each}
    </select>
    <select bind:value={chapter}>
      <option value="">— chapter —</option>
      {#each chapterList as c (c)}
        <option value={c}>{c}</option>
      {/each}
    </select>
  </p>
  {#if loadError}
    <p>{loadError}</p>
  {/if}
  {#if series && chapter}
    <p>
      <button onclick={() => (view = "slicer")} disabled={view === "slicer"}>Slicer</button>
      <button onclick={() => (view = "ocr")} disabled={view === "ocr"}>OCR</button>
      <button onclick={() => (view = "translation")} disabled={view === "translation"}>Translation</button>
      <button onclick={() => (view = "reader")} disabled={view === "reader"}>Reader</button>
    </p>
    {#if view === "slicer"}
      <SlicerView {series} {chapter} />
    {:else if view === "ocr"}
      <OcrView {series} {chapter} />
    {:else if view === "reader"}
      <ReaderView {series} {chapter} />
    {:else}
      <TranslationView {series} {chapter} />
    {/if}
  {:else}
    <p>pick a series and chapter (no ingest.json yet for a chapter? run ingest first)</p>
  {/if}
</main>