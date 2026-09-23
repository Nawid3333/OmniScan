<script lang="ts">
  import RunButton from "./RunButton.svelte";
  import {
    getFinal,
    getIngest,
    getInpaint,
    getLayout,
    inpaintPatchUrl,
    pageImageUrl,
    putFinalLine,
  } from "./api";
  import type { FinalLine, InpaintItem, LayoutItem, SourceFile } from "./api";
  import { stackTops as computeStackTops, stackTotalHeight, stackWidth, stripBoxToStackBox } from "./strip";
  import { fillCss, hasStoredPatch } from "./inpaint";
  import { overlayStyle, textByRegion } from "./edit";

  let { series, chapter }: { series: string; chapter: string } = $props();

  let files = $state<SourceFile[]>([]);
  let items = $state<LayoutItem[]>([]); // layout.json: one overlay box per typeset region
  let patches = $state<InpaintItem[]>([]); // inpaint.json: the clean layer under the text
  let inpaintOk = $state(true);
  let lines = $state<FinalLine[]>([]);
  let loaded = $state(false);
  let error = $state("");

  // Click-to-edit state for exactly one region at a time.
  let editingId = $state<string | null>(null);
  let draft = $state("");
  let savingId = $state<string | null>(null);
  let saveError = $state<string | null>(null);

  let width = $derived(files.length > 0 ? stackWidth(files) : 0);
  let totalHeight = $derived(stackTotalHeight(files));
  let texts = $derived(textByRegion(lines));
  // Region ids whose patch <img> failed to load (no entry in patches.npz): they fall back to the
  // flat-fill colour for that one item instead of predicting which ids are stored.
  let patchFailed = $state<Set<string>>(new Set());
  let patchIds = $derived(
    new Set(
      patches
        .filter((patch) => patch.method !== "none" && !patchFailed.has(patch.region_id))
        .map((patch) => patch.region_id),
    ),
  );

  // Top of each file's image in the stack: sum of all earlier files' natural heights.
  let stackTops = $derived(computeStackTops(files));

  function markPatchFailed(regionId: string): void {
    const next = new Set(patchFailed);
    next.add(regionId);
    patchFailed = next;
  }

  function beginEdit(item: LayoutItem): void {
    editingId = item.region_id;
    draft = texts[item.region_id] ?? "";
    saveError = null;
  }

  function cancelEdit(): void {
    editingId = null; // the div re-renders the last-saved text, so the draft is discarded
    saveError = null;
  }

  async function commitEdit(item: LayoutItem): Promise<void> {
    if (savingId !== null || editingId !== item.region_id) return;
    savingId = item.region_id;
    saveError = null;
    try {
      const saved = await putFinalLine(series, chapter, item.region_id, draft);
      lines = lines.map((line) => (line.region_id === saved.region_id ? saved : line));
      if (editingId === item.region_id) {
        editingId = null; // swap back to the read-only div, which now shows the saved text
      }
    } catch (e) {
      if (editingId === item.region_id) {
        // keep the textarea open with the attempted text still in it, so the edit isn't lost
        saveError = e instanceof Error ? e.message : String(e);
      }
    } finally {
      savingId = null;
    }
  }

  function onKeydown(event: KeyboardEvent, item: LayoutItem): void {
    if (event.key === "Escape") {
      cancelEdit();
    } else if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void commitEdit(item);
    }
  }

  /** Focus the edit textarea (caret at the end) when it mounts. */
  function focusOnMount(node: HTMLTextAreaElement): void {
    node.focus();
    node.selectionStart = node.selectionEnd = node.value.length;
  }

  $effect(() => {
    const s = series;
    const c = chapter;
    error = "";
    loaded = false;
    files = [];
    items = [];
    patches = [];
    inpaintOk = true;
    lines = [];
    editingId = null;
    savingId = null;
    saveError = null;
    patchFailed = new Set();
    void load(s, c);
  });

  async function load(series: string, chapter: string): Promise<void> {
    try {
      const [ingest, layout] = await Promise.all([getIngest(series, chapter), getLayout(series, chapter)]);
      files = ingest.files;
      items = layout.items;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      files = [];
      items = [];
      loaded = true;
      return;
    }
    // final.json and inpaint.json are both optional here: a missing final line only means a blank
    // (still editable) overlay, and missing inpaint falls back to raw pages with a run button.
    const [finalResult, inpaintResult] = await Promise.allSettled([
      getFinal(series, chapter),
      getInpaint(series, chapter),
    ]);
    lines = finalResult.status === "fulfilled" ? finalResult.value.lines : [];
    patches = inpaintResult.status === "fulfilled" ? inpaintResult.value.items : [];
    inpaintOk = inpaintResult.status === "fulfilled";
    loaded = true;
  }
</script>

{#if error}
  <p>{error} — no layout.json yet — run typeset first.</p>
  <RunButton {series} {chapter} through="typeset" onDone={() => load(series, chapter)} />
{:else if !loaded}
  <p>loading…</p>
{:else if files.length === 0}
  <p>no pages</p>
{:else}
  {#if !inpaintOk}
    <p style="margin: 4px 0;">no inpaint.json yet — showing raw pages (the original text is still visible underneath).</p>
    <RunButton {series} {chapter} through="inpaint" onDone={() => load(series, chapter)} />
  {/if}
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
    <!-- clean layer: patches positioned where export would apply them; always fully clean (the
         InpaintView reveal slider is permanently at 100%), so no clip-path is needed -->
    {#if inpaintOk}
      <div style="position: absolute; inset: 0;">
        {#each patches as patch (patch.region_id)}
          {@const box = stripBoxToStackBox(files, patch.box)}
          {#if patch.method !== "none"}
            {#if hasStoredPatch(patch, patchIds)}
              <img
                src={inpaintPatchUrl(series, chapter, patch.region_id)}
                alt={patch.region_id}
                style="position: absolute; left: {box.x}px; top: {box.y}px; width: {box.width}px; height: {box.height}px;"
                onerror={() => markPatchFailed(patch.region_id)}
              />
            {:else}
              <div
                style="position: absolute; left: {box.x}px; top: {box.y}px; width: {box.width}px; height: {box.height}px; background: {fillCss(patch.fill)};"
              ></div>
            {/if}
          {/if}
        {/each}
      </div>
    {/if}
    <!-- final-text overlay: one editable box per layout item -->
    {#each items as item (item.region_id)}
      {@const box = stripBoxToStackBox(files, item.box)}
      {#if editingId === item.region_id}
        <textarea
          use:focusOnMount
          bind:value={draft}
          onkeydown={(event) => onKeydown(event, item)}
          onblur={() => void commitEdit(item)}
          style="{overlayStyle(item)} position: absolute; left: {box.x}px; top: {box.y}px; width: {box.width}px; height: {box.height}px; box-sizing: border-box; margin: 0; padding: 0; background: rgba(255, 255, 255, 0.85); border: 1px solid #1f6feb; resize: none;"
        ></textarea>
      {:else}
        <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
        <div
          onclick={() => beginEdit(item)}
          style="{overlayStyle(item)} position: absolute; left: {box.x}px; top: {box.y}px; width: {box.width}px; height: {box.height}px; background: rgba(255, 255, 255, 0.85); overflow: hidden; cursor: text;"
        >{texts[item.region_id] ?? ""}</div>
      {/if}
      {#if savingId === item.region_id}
        <span style="position: absolute; left: {box.x}px; top: {Math.max(0, box.y - 18)}px; font-size: 12px; color: #1f6feb;">saving…</span>
      {:else if editingId === item.region_id && saveError !== null}
        <span style="position: absolute; left: {box.x}px; top: {Math.max(0, box.y - 18)}px; color:#dc2626;">{saveError}</span>
      {/if}
    {/each}
  </div>
{/if}