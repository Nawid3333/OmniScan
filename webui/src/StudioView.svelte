<script lang="ts">
  import RunButton from "./RunButton.svelte";
  import {
    addRegion,
    deleteRegion,
    getEdits,
    getFinal,
    getIngest,
    getInpaint,
    getOcr,
    inpaintPatchUrl,
    listProfiles,
    pageImageUrl,
    patchRegion,
    putFinalLine,
    revertFinalLine,
    revertRegion,
    translateRegions,
  } from "./api";
  import type {
    BBox,
    ChapterEdits,
    FinalLine,
    InpaintItem,
    Region,
    RegionKind,
    SourceFile,
    Suggestion,
    TranslationProfile,
  } from "./api";
  import { kindColor } from "./ocr";
  import {
    HANDLES,
    ZOOMS,
    dragBox,
    fitZoom,
    handleCursor,
    handlePoint,
    overlapsPage,
    pageOf,
    pageRectToStrip,
    pageRegions,
    regionStatus,
    sameBox,
    statusLabels,
    stripToPage,
  } from "./studio";
  import type { Handle } from "./studio";

  let { series, chapter }: { series: string; chapter: string } = $props();

  const KINDS: RegionKind[] = ["bubble_text", "free_text", "sfx", "watermark"];
  const NUDGE_COMMIT_MS = 400;

  let files = $state<SourceFile[]>([]);
  let regions = $state<Region[]>([]);
  let lines = $state<FinalLine[]>([]);
  let edits = $state<ChapterEdits | null>(null);
  let patches = $state<InpaintItem[]>([]);
  let hasOcr = $state(false);
  let loaded = $state(false);
  let error = $state("");
  let busy = $state(false);
  let actionError = $state("");

  let pageIndex = $state(0);
  let zoom = $state(1);
  let tool = $state<"select" | "draw">("select");
  let newKind = $state<RegionKind>("bubble_text");
  let showBoxes = $state(true);
  let showBubbles = $state(true);
  let showClean = $state(false);
  let selectedId = $state<string | null>(null);
  let sourceDraft = $state("");
  let translationDraft = $state("");
  let profiles = $state<TranslationProfile[]>([]);
  let profileChoice = $state(""); // "" = every enabled profile
  let suggestions = $state<Suggestion[]>([]);
  let kept = $state<Suggestion | null>(null); // the suggestion the English draft was taken from
  let translating = $state(false);

  let svgEl = $state<SVGSVGElement | null>(null);
  let canvasEl = $state<HTMLDivElement | null>(null);
  let drag: { id: string; handle: Handle; startX: number; startY: number; box: BBox } | null = null;
  let preview = $state<{ id: string; box: BBox } | null>(null);
  let drawing = $state<{ ax: number; ay: number; bx: number; by: number } | null>(null);
  let nudgeTimer: ReturnType<typeof setTimeout> | null = null;

  let page = $derived(files[pageIndex] ?? null);
  let onPage = $derived(page ? pageRegions(page, regions) : []);
  let lineById = $derived(new Map(lines.map((line) => [line.region_id, line])));
  let selected = $derived(regions.find((region) => region.id === selectedId) ?? null);
  let selectedLine = $derived(selected ? lineById.get(selected.id) : undefined);
  let selectedStatus = $derived(selected ? regionStatus(selected, edits, selectedLine) : null);
  let deletedOnPage = $derived(
    page && edits ? edits.deleted_regions.filter((region) => overlapsPage(page, region.bbox)) : [],
  );
  let cleanOnPage = $derived(
    page && showClean ? patches.filter((patch) => patch.method !== "none" && overlapsPage(page, patch.box)) : [],
  );
  let todo = $derived(regions.filter((region) => regionStatus(region, edits, lineById.get(region.id)).untranslated).length);

  $effect(() => {
    const s = series;
    const c = chapter;
    loaded = false;
    error = "";
    actionError = "";
    selectedId = null;
    pageIndex = 0;
    files = [];
    regions = [];
    lines = [];
    edits = null;
    patches = [];
    void load(s, c);
  });

  async function load(s: string, c: string): Promise<void> {
    try {
      files = (await getIngest(s, c)).files;
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      loaded = true;
      return;
    }
    await refresh();
    try {
      profiles = await listProfiles();
    } catch {
      profiles = []; // translation stays available with the enabled profiles
    }
    if (canvasEl !== null && files.length > 0) {
      zoom = fitZoom(files[0].width, canvasEl.clientWidth - 24);
    }
    loaded = true;
  }

  /** Re-read everything an edit can change (regions, lines, edits summary, clean patches). */
  async function refresh(): Promise<void> {
    const [ocr, final, summary, inpaint] = await Promise.allSettled([
      getOcr(series, chapter),
      getFinal(series, chapter),
      getEdits(series, chapter),
      getInpaint(series, chapter),
    ]);
    hasOcr = ocr.status === "fulfilled";
    regions = ocr.status === "fulfilled" ? ocr.value.regions : [];
    lines = final.status === "fulfilled" ? final.value.lines : [];
    edits = summary.status === "fulfilled" ? summary.value : null;
    patches = inpaint.status === "fulfilled" ? inpaint.value.items : [];
    if (selectedId !== null && !regions.some((region) => region.id === selectedId)) {
      selectedId = null;
    }
  }

  /** Run one edit, then refresh; errors are shown instead of thrown. */
  async function act<T>(operation: () => Promise<T>): Promise<T | null> {
    busy = true;
    actionError = "";
    try {
      const result = await operation();
      await refresh();
      return result;
    } catch (e) {
      actionError = e instanceof Error ? e.message : String(e);
      return null;
    } finally {
      busy = false;
    }
  }

  function select(id: string | null): void {
    selectedId = id;
    suggestions = [];
    kept = null;
    const region = regions.find((r) => r.id === id);
    sourceDraft = region?.text ?? "";
    translationDraft = region ? (lineById.get(region.id)?.text ?? "") : "";
  }

  function goToPage(index: number): void {
    if (index < 0 || index >= files.length) return;
    pageIndex = index;
    if (selected !== null && page !== null && !overlapsPage(page, selected.bbox)) select(null);
  }

  function selectFromList(region: Region): void {
    select(region.id);
    const index = pageOf(files, region.bbox);
    if (index !== pageIndex) pageIndex = index;
  }

  function boxOf(region: Region): BBox {
    return preview !== null && preview.id === region.id ? preview.box : region.bbox;
  }

  function toPagePoint(event: PointerEvent): { x: number; y: number } {
    const rect = svgEl!.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / zoom, y: (event.clientY - rect.top) / zoom };
  }

  function startDrag(event: PointerEvent, region: Region, handle: Handle): void {
    if (tool !== "select" || busy) return;
    event.stopPropagation();
    event.preventDefault();
    if (selectedId !== region.id) select(region.id);
    const point = toPagePoint(event);
    drag = { id: region.id, handle, startX: point.x, startY: point.y, box: region.bbox };
    svgEl!.setPointerCapture(event.pointerId);
  }

  function onBackgroundDown(event: PointerEvent): void {
    if (busy) return;
    if (tool === "draw") {
      const point = toPagePoint(event);
      drawing = { ax: point.x, ay: point.y, bx: point.x, by: point.y };
      svgEl!.setPointerCapture(event.pointerId);
    } else {
      select(null);
    }
  }

  function onPointerMove(event: PointerEvent): void {
    if (page === null) return;
    const point = toPagePoint(event);
    if (drag !== null) {
      const s = page.scale > 0 ? page.scale : 1;
      const box = dragBox(drag.box, drag.handle, (point.x - drag.startX) * s, (point.y - drag.startY) * s);
      preview = { id: drag.id, box };
    } else if (drawing !== null) {
      drawing = { ...drawing, bx: point.x, by: point.y };
    }
  }

  async function onPointerUp(): Promise<void> {
    if (drag !== null) {
      const started = drag;
      const moved = preview;
      drag = null;
      if (moved !== null && !sameBox(moved.box, started.box)) {
        await act(() => patchRegion(series, chapter, started.id, { bbox: moved.box }));
      }
      preview = null;
    } else if (drawing !== null && page !== null) {
      const drawn = drawing;
      drawing = null;
      const box = pageRectToStrip(page, drawn.ax, drawn.ay, drawn.bx, drawn.by);
      if (box.x1 - box.x0 >= 4 && box.y1 - box.y0 >= 4) {
        const region = await act(() => addRegion(series, chapter, { bbox: box, kind: newKind }));
        if (region !== null) {
          tool = "select";
          select(region.id);
        }
      }
    }
  }

  function nudge(dx: number, dy: number): void {
    if (selected === null || page === null) return;
    const s = page.scale > 0 ? page.scale : 1;
    const base = preview !== null && preview.id === selected.id ? preview.box : selected.bbox;
    preview = { id: selected.id, box: dragBox(base, "move", dx * s, dy * s) };
    if (nudgeTimer !== null) clearTimeout(nudgeTimer);
    const id = selected.id;
    nudgeTimer = setTimeout(() => {
      nudgeTimer = null;
      const moved = preview;
      if (moved === null || moved.id !== id) return;
      void act(() => patchRegion(series, chapter, id, { bbox: moved.box })).then(() => {
        preview = null;
      });
    }, NUDGE_COMMIT_MS);
  }

  function onKeydown(event: KeyboardEvent): void {
    const target = event.target as HTMLElement | null;
    if (target !== null && (target.tagName === "TEXTAREA" || target.tagName === "INPUT" || target.tagName === "SELECT")) {
      return;
    }
    const step = event.shiftKey ? 10 : 1;
    if (event.key === "Escape") {
      drawing = null;
      tool = "select";
      select(null);
    } else if (event.key === "b" || event.key === "B") {
      tool = "draw";
    } else if (event.key === "v" || event.key === "V") {
      tool = "select";
    } else if (event.key === "PageDown") {
      goToPage(pageIndex + 1);
    } else if (event.key === "PageUp") {
      goToPage(pageIndex - 1);
    } else if ((event.key === "Delete" || event.key === "Backspace") && selected !== null) {
      void removeSelected();
    } else if (event.key.startsWith("Arrow") && selected !== null) {
      event.preventDefault();
      const dx = event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0;
      const dy = event.key === "ArrowUp" ? -step : event.key === "ArrowDown" ? step : 0;
      nudge(dx, dy);
    } else {
      return;
    }
    event.preventDefault();
  }

  async function saveSource(): Promise<void> {
    if (selected === null) return;
    const region = await act(() => patchRegion(series, chapter, selected!.id, { text: sourceDraft }));
    if (region !== null) sourceDraft = region.text;
  }

  async function saveTranslation(): Promise<void> {
    if (selected === null) return;
    const by = kept !== null && kept.text === translationDraft ? kept.profile : undefined;
    const line = await act(() => putFinalLine(series, chapter, selected!.id, translationDraft, by));
    if (line !== null) translationDraft = line.text;
  }

  /** Ask the translation profiles for the selected region (nothing is saved until Save English). */
  async function translateSelected(): Promise<void> {
    if (selected === null) return;
    const id = selected.id;
    translating = true;
    actionError = "";
    try {
      const result = await translateRegions(series, chapter, [id], { profile: profileChoice || undefined });
      if (selectedId !== id) return;
      suggestions = result.suggestions;
      if (translationDraft.trim() === "" && suggestions.length > 0) useSuggestion(suggestions[0]);
    } catch (e) {
      actionError = e instanceof Error ? e.message : String(e);
    } finally {
      translating = false;
    }
  }

  function useSuggestion(suggestion: Suggestion): void {
    translationDraft = suggestion.text;
    kept = suggestion;
  }

  async function keepSuggestion(suggestion: Suggestion): Promise<void> {
    useSuggestion(suggestion);
    await saveTranslation();
  }

  /** Translate every untranslated region of the page and keep each first suggestion as its English. */
  async function translatePage(): Promise<void> {
    const ids = onPage
      .filter((region) => regionStatus(region, edits, lineById.get(region.id)).untranslated)
      .map((region) => region.id);
    if (ids.length === 0) {
      actionError = "every region on this page already has an English line";
      return;
    }
    translating = true;
    await act(() => translateRegions(series, chapter, ids, { profile: profileChoice || undefined, apply: true }));
    translating = false;
    if (selected !== null) translationDraft = lineById.get(selected.id)?.text ?? translationDraft;
  }

  async function setKind(kind: RegionKind): Promise<void> {
    if (selected === null || kind === selected.kind) return;
    await act(() => patchRegion(series, chapter, selected!.id, { kind }));
  }

  async function setCoordinate(key: keyof BBox, value: number): Promise<void> {
    if (selected === null || !Number.isFinite(value)) return;
    await act(() => patchRegion(series, chapter, selected!.id, { bbox: { ...selected!.bbox, [key]: Math.round(value) } }));
  }

  async function removeSelected(): Promise<void> {
    if (selected === null) return;
    const id = selected.id;
    const done = await act(() => deleteRegion(series, chapter, id));
    if (done !== null) select(null);
  }

  async function revertSelectedRegion(): Promise<void> {
    if (selected === null) return;
    const id = selected.id;
    const region = await act(() => revertRegion(series, chapter, id));
    select(region === null ? null : id);
  }

  async function revertSelectedTranslation(): Promise<void> {
    if (selected === null) return;
    const line = await act(() => revertFinalLine(series, chapter, selected!.id));
    translationDraft = line?.text ?? "";
  }

  async function restore(region: Region): Promise<void> {
    await act(() => revertRegion(series, chapter, region.id));
    select(region.id);
  }

  function saveOnCtrlEnter(event: KeyboardEvent, save: () => Promise<void>): void {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      void save();
    }
  }

  function snippet(text: string, max = 40): string {
    const flat = text.replace(/\s+/g, " ").trim();
    return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
  }
</script>

<svelte:window onkeydown={onKeydown} />

{#if error}
  <p>{error} — no ingest.json yet — run ingest first.</p>
  <RunButton {series} {chapter} through="slice" onDone={() => load(series, chapter)} />
{:else if !loaded}
  <p>loading…</p>
{:else if files.length === 0}
  <p>no pages</p>
{:else}
  <div class="toolbar">
    <button onclick={() => goToPage(pageIndex - 1)} disabled={pageIndex === 0} title="previous page (PageUp)">◀</button>
    <select value={pageIndex} onchange={(e) => goToPage(Number((e.target as HTMLSelectElement).value))}>
      {#each files as file, i (file.index)}
        <option value={i}>page {i + 1} / {files.length} · {file.name}</option>
      {/each}
    </select>
    <button onclick={() => goToPage(pageIndex + 1)} disabled={pageIndex >= files.length - 1} title="next page (PageDown)">▶</button>
    <select bind:value={zoom} title="zoom">
      {#each ZOOMS as z (z)}
        <option value={z}>{Math.round(z * 100)}%</option>
      {/each}
      {#if !ZOOMS.includes(zoom)}
        <option value={zoom}>{Math.round(zoom * 100)}% (fit)</option>
      {/if}
    </select>
    <span class="sep"></span>
    <button class:active={tool === "select"} onclick={() => (tool = "select")} title="select, move and resize boxes (V)">Select</button>
    <button class:active={tool === "draw"} onclick={() => (tool = "draw")} title="draw a new text box (B)">Draw box</button>
    <select bind:value={newKind} title="kind of the next drawn box">
      {#each KINDS as kind (kind)}
        <option value={kind}>{kind}</option>
      {/each}
    </select>
    <span class="sep"></span>
    <label><input type="checkbox" bind:checked={showBoxes} /> boxes</label>
    <label><input type="checkbox" bind:checked={showBubbles} /> bubbles</label>
    <label title="show the cleaned patches (inpaint) over the raw page"><input type="checkbox" bind:checked={showClean} /> cleaned</label>
    <span class="sep"></span>
    <select bind:value={profileChoice} title="translation profile for the Translate buttons">
      <option value="">enabled profiles</option>
      {#each profiles as profile (profile.name)}
        <option value={profile.name}>{profile.name}{profile.enabled ? "" : " (off)"}</option>
      {/each}
    </select>
    <button onclick={() => void translatePage()} disabled={busy || translating} title="translate every untranslated region of this page and keep the first suggestion">Translate page</button>
    <span class="sep"></span>
    <RunButton {series} {chapter} through="export" startStage="inpaint" label="Render (inpaint → export)" onDone={refresh} />
    {#if translating}<span class="muted">translating…</span>{:else if busy}<span class="muted">saving…</span>{/if}
  </div>
  {#if actionError}
    <p class="error">{actionError}</p>
  {/if}
  {#if !hasOcr}
    <p class="muted">
      No OCR yet: draw the text boxes by hand and type their text, or run the pipeline through OCR.
      <RunButton {series} {chapter} through="ocr" onDone={refresh} />
    </p>
  {/if}

  <div class="studio">
    <div class="canvas" bind:this={canvasEl}>
      {#if page !== null}
        <div class="page" style="width: {page.width * zoom}px; height: {page.height * zoom}px;">
          <img src={pageImageUrl(series, chapter, page.index)} alt={page.name} width={page.width * zoom} height={page.height * zoom} draggable="false" />
          <svg
            bind:this={svgEl}
            width={page.width * zoom}
            height={page.height * zoom}
            viewBox="0 0 {page.width} {page.height}"
            style="cursor: {tool === 'draw' ? 'crosshair' : 'default'};"
            onpointerdown={onBackgroundDown}
            onpointermove={onPointerMove}
            onpointerup={() => void onPointerUp()}
            role="application"
            aria-label="page editor"
          >
            {#each cleanOnPage as patch (patch.region_id)}
              {@const rect = stripToPage(page, patch.box)}
              <image href={inpaintPatchUrl(series, chapter, patch.region_id)} x={rect.x} y={rect.y} width={rect.width} height={rect.height} preserveAspectRatio="none" pointer-events="none" />
            {/each}
            {#if showBoxes}
              {#each onPage as region (region.id)}
                {@const rect = stripToPage(page, boxOf(region))}
                {@const color = kindColor(region.kind)}
                {@const isSelected = region.id === selectedId}
                {#if showBubbles && region.bubble_bbox !== null}
                  {@const bubble = stripToPage(page, region.bubble_bbox)}
                  <rect x={bubble.x} y={bubble.y} width={bubble.width} height={bubble.height} fill="none" stroke="#9ca3af" stroke-width={1 / zoom} stroke-dasharray="{3 / zoom} {3 / zoom}" pointer-events="none" />
                {/if}
                <rect
                  x={rect.x}
                  y={rect.y}
                  width={rect.width}
                  height={rect.height}
                  fill={color}
                  fill-opacity={isSelected ? 0.18 : 0.06}
                  stroke={color}
                  stroke-width={(isSelected ? 3 : 1.5) / zoom}
                  style="cursor: {tool === 'select' ? 'move' : 'crosshair'};"
                  onpointerdown={(e) => (tool === "select" ? startDrag(e, region, "move") : undefined)}
                  role="button"
                  tabindex="-1"
                  aria-label={region.id}
                />
                <text x={rect.x + 2 / zoom} y={rect.y - 3 / zoom} fill={color} font-size={12 / zoom} pointer-events="none">{region.reading_order + 1}</text>
                {#if isSelected && tool === "select"}
                  {#each HANDLES as handle (handle)}
                    {@const point = handlePoint(rect, handle)}
                    <rect
                      x={point.x - 5 / zoom}
                      y={point.y - 5 / zoom}
                      width={10 / zoom}
                      height={10 / zoom}
                      fill="white"
                      stroke={color}
                      stroke-width={1.5 / zoom}
                      style="cursor: {handleCursor(handle)};"
                      onpointerdown={(e) => startDrag(e, region, handle)}
                      role="button"
                      tabindex="-1"
                      aria-label="resize {handle}"
                    />
                  {/each}
                {/if}
              {/each}
            {/if}
            {#if drawing !== null}
              <rect
                x={Math.min(drawing.ax, drawing.bx)}
                y={Math.min(drawing.ay, drawing.by)}
                width={Math.abs(drawing.bx - drawing.ax)}
                height={Math.abs(drawing.by - drawing.ay)}
                fill={kindColor(newKind)}
                fill-opacity="0.15"
                stroke={kindColor(newKind)}
                stroke-width={2 / zoom}
                stroke-dasharray="{4 / zoom} {3 / zoom}"
                pointer-events="none"
              />
            {/if}
          </svg>
        </div>
      {/if}
    </div>

    <aside class="panel">
      {#if selected !== null && selectedStatus !== null}
        <section>
          <h3>
            <span class="chip" style="background: {kindColor(selected.kind)}"></span>
            {selected.id}
            <button class="link" onclick={() => select(null)}>close</button>
          </h3>
          {#if statusLabels(selectedStatus).length > 0}
            <p class="labels">{statusLabels(selectedStatus).join(" · ")}</p>
          {/if}
          <label>
            kind
            <select value={selected.kind} onchange={(e) => void setKind((e.target as HTMLSelectElement).value as RegionKind)} disabled={busy}>
              {#each KINDS as kind (kind)}
                <option value={kind}>{kind}</option>
              {/each}
            </select>
          </label>
          <div class="coords">
            {#each ["x0", "y0", "x1", "y1"] as const as key (key)}
              <label>
                {key}
                <input type="number" value={selected.bbox[key]} onchange={(e) => void setCoordinate(key, Number((e.target as HTMLInputElement).value))} disabled={busy} />
              </label>
            {/each}
          </div>
          <label>
            source text <span class="muted">(OCR {selected.confidence.toFixed(2)} · Ctrl+Enter saves)</span>
            <textarea rows="3" bind:value={sourceDraft} onkeydown={(e) => saveOnCtrlEnter(e, saveSource)} lang={selected.lang}></textarea>
          </label>
          {#if selected.ocr_alt !== null}
            <p class="muted">
              second reading: {selected.ocr_alt}
              <button class="link" onclick={() => (sourceDraft = selected!.ocr_alt ?? sourceDraft)}>use</button>
            </p>
          {/if}
          <button onclick={() => void saveSource()} disabled={busy || sourceDraft === selected.text}>Save source</button>
          <label>
            English <span class="muted">(Ctrl+Enter saves)</span>
            <textarea rows="3" bind:value={translationDraft} onkeydown={(e) => saveOnCtrlEnter(e, saveTranslation)} lang="en"></textarea>
          </label>
          {#if selectedLine !== undefined && selectedLine.decision !== "manual" && selectedLine.rationale}
            <p class="muted">judge: {selectedLine.rationale}</p>
          {/if}
          <button onclick={() => void saveTranslation()} disabled={busy || translationDraft === (selectedLine?.text ?? "")}>Save English</button>
          <button onclick={() => void translateSelected()} disabled={busy || translating || selected.kind === "watermark" || selected.text.trim() === ""} title="ask the translation profile(s); nothing is saved until you keep a suggestion">
            {translating ? "Translating…" : "Translate"}
          </button>
          {#if suggestions.length > 0}
            <ul class="suggestions">
              {#each suggestions as suggestion, i (i)}
                <li>
                  <span class="muted">{suggestion.profile}</span>
                  <span>{suggestion.text}</span>
                  <button class="link" onclick={() => useSuggestion(suggestion)}>use</button>
                  <button class="link" onclick={() => void keepSuggestion(suggestion)} disabled={busy}>keep</button>
                </li>
              {/each}
            </ul>
          {/if}
          <div class="actions">
            {#if selectedStatus.manualTranslation}
              <button onclick={() => void revertSelectedTranslation()} disabled={busy}>Revert English</button>
            {/if}
            {#if selectedStatus.edited && !selectedStatus.added}
              <button onclick={() => void revertSelectedRegion()} disabled={busy}>Revert box and text</button>
            {/if}
            <button class="danger" onclick={() => void removeSelected()} disabled={busy} title="Delete">Delete region</button>
          </div>
        </section>
      {:else}
        <section>
          <h3>Page {pageIndex + 1}</h3>
          <p class="muted">
            Click a box to edit it, drag it to move, drag its handles to resize. <b>B</b> draws a new box,
            <b>Del</b> removes the selected one, arrows nudge it (Shift = 10 px), PageUp/PageDown change page.
            {regions.length} regions in the chapter, {todo} untranslated.
          </p>
        </section>
      {/if}
      <section>
        <h4>Regions on this page</h4>
        {#if onPage.length === 0}
          <p class="muted">none</p>
        {/if}
        <ol class="list">
          {#each onPage as region (region.id)}
            {@const line = lineById.get(region.id)}
            {@const labels = statusLabels(regionStatus(region, edits, line))}
            <li>
              <button class="row" class:selected={region.id === selectedId} onclick={() => selectFromList(region)}>
                <span class="chip" style="background: {kindColor(region.kind)}"></span>
                <span class="texts">
                  <span lang={region.lang}>{snippet(region.text) || "—"}</span>
                  <span class="en">{snippet(line?.text ?? "") || "—"}</span>
                  {#if labels.length > 0}<span class="labels">{labels.join(" · ")}</span>{/if}
                </span>
              </button>
            </li>
          {/each}
        </ol>
      </section>
      {#if deletedOnPage.length > 0}
        <section>
          <h4>Deleted on this page</h4>
          <ul class="list">
            {#each deletedOnPage as region (region.id)}
              <li>
                <span lang={region.lang}>{snippet(region.text) || region.id}</span>
                <button class="link" onclick={() => void restore(region)} disabled={busy}>restore</button>
              </li>
            {/each}
          </ul>
        </section>
      {/if}
    </aside>
  </div>
{/if}

<style>
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin: 4px 0 8px;
  }
  .sep {
    width: 1px;
    height: 20px;
    background: #d1d5db;
  }
  button.active {
    background: #1f6feb;
    color: white;
  }
  .studio {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 380px;
    gap: 8px;
    height: calc(100vh - 170px);
  }
  .canvas {
    overflow: auto;
    background: #e5e7eb;
    padding: 12px;
  }
  .page {
    position: relative;
    margin: 0 auto;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
  }
  .page img,
  .page svg {
    position: absolute;
    left: 0;
    top: 0;
    user-select: none;
  }
  .page svg {
    touch-action: none;
  }
  .panel {
    overflow: auto;
    border-left: 1px solid #d1d5db;
    padding: 0 8px;
  }
  .panel label {
    display: block;
    margin: 6px 0;
  }
  .panel textarea {
    width: 100%;
    box-sizing: border-box;
    font-size: 15px;
  }
  .coords {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
  }
  .coords input {
    width: 100%;
    box-sizing: border-box;
  }
  .actions {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin: 10px 0;
  }
  .danger {
    color: #b91c1c;
  }
  .chip {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 4px;
    flex: none;
  }
  .list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .row {
    display: flex;
    width: 100%;
    text-align: left;
    gap: 6px;
    align-items: flex-start;
    background: none;
    border: 1px solid transparent;
    padding: 4px;
    cursor: pointer;
  }
  .row.selected {
    border-color: #1f6feb;
    background: #eff6ff;
  }
  .texts {
    display: flex;
    flex-direction: column;
  }
  .en {
    color: #374151;
  }
  .labels {
    color: #9a3412;
    font-size: 12px;
  }
  .muted {
    color: #6b7280;
    font-size: 13px;
  }
  .error {
    color: #dc2626;
  }
  .suggestions {
    list-style: none;
    padding: 0;
    margin: 6px 0;
  }
  .suggestions li {
    border: 1px solid #e5e7eb;
    padding: 4px;
    margin-bottom: 4px;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    align-items: baseline;
  }
  button.link {
    background: none;
    border: none;
    color: #1f6feb;
    cursor: pointer;
    padding: 0 4px;
  }
</style>
