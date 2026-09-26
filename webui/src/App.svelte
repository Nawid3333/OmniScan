<script lang="ts">
  import { onMount } from "svelte";

  import EditView from "./EditView.svelte";
  import FilteredView from "./FilteredView.svelte";
  import InpaintView from "./InpaintView.svelte";
  import LayoutView from "./LayoutView.svelte";
  import OcrView from "./OcrView.svelte";
  import ReaderView from "./ReaderView.svelte";
  import SlicerView from "./SlicerView.svelte";
  import StudioView from "./StudioView.svelte";
  import TranslationView from "./TranslationView.svelte";
  import { listChapters, listSeries } from "./api";

  let view = $state<
    "studio" | "slicer" | "ocr" | "translation" | "reader" | "inpaint" | "layout" | "edit" | "filtered"
  >("studio");
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
  <h1>OmniScan</h1>
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
  {#if series}
    <p>
      {#if chapter}
        <button onclick={() => (view = "studio")} disabled={view === "studio"}>Studio</button>
        <button onclick={() => (view = "slicer")} disabled={view === "slicer"}>Slicer</button>
        <button onclick={() => (view = "ocr")} disabled={view === "ocr"}>OCR</button>
        <button onclick={() => (view = "translation")} disabled={view === "translation"}>Translation</button>
        <button onclick={() => (view = "reader")} disabled={view === "reader"}>Reader</button>
        <button onclick={() => (view = "inpaint")} disabled={view === "inpaint"}>Inpaint</button>
        <button onclick={() => (view = "layout")} disabled={view === "layout"}>Layout</button>
        <button onclick={() => (view = "edit")} disabled={view === "edit"}>Edit</button>
      {/if}
      <button onclick={() => (view = "filtered")} disabled={view === "filtered"}>Filtered</button>
    </p>
    {#if view === "filtered"}
      <FilteredView {series} />
    {:else if chapter}
      {#if view === "studio"}
        <StudioView {series} {chapter} />
      {:else if view === "slicer"}
        <SlicerView {series} {chapter} />
      {:else if view === "ocr"}
        <OcrView {series} {chapter} />
      {:else if view === "inpaint"}
        <InpaintView {series} {chapter} />
      {:else if view === "layout"}
        <LayoutView {series} {chapter} />
      {:else if view === "edit"}
        <EditView {series} {chapter} />
      {:else if view === "reader"}
        <ReaderView {series} {chapter} />
      {:else}
        <TranslationView {series} {chapter} />
      {/if}
    {:else}
      <p>pick a series and chapter (no ingest.json yet for a chapter? run ingest first)</p>
    {/if}
  {:else}
    <p>pick a series and chapter (no ingest.json yet for a chapter? run ingest first)</p>
  {/if}
</main>