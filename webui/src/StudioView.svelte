<script lang="ts">
  import RunButton from "./RunButton.svelte";
  import {
    addCleanup,
    addRegion,
    cleanupPatchUrl,
    deleteCleanup,
    deleteLayout,
    deleteRegion,
    getCleanup,
    getEdits,
    getFinal,
    getIngest,
    getInpaint,
    getLiveLayout,
    getOcr,
    inpaintPatchUrl,
    listFonts,
    listProfiles,
    pageImageUrl,
    patchRegion,
    previewUrl,
    putLayout,
    putFinalLine,
    revertFinalLine,
    revertRegion,
    translateRegions,
  } from "./api";
  import type {
    BBox,
    ChapterEdits,
    CleanupMethod,
    CleanupPatch,
    FinalLine,
    InpaintItem,
    LayoutEdit,
    LayoutFields,
    LayoutItem,
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
    hexToRgb,
    maskBounds,
    overlapsPage,
    pageOf,
    pageRectToStrip,
    pageRegions,
    regionStatus,
    rgbToHex,
    sameBox,
    setFields,
    statusLabels,
    stripToPage,
    typedLines,
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
  let tool = $state<"select" | "draw" | "clean">("select");
  let newKind = $state<RegionKind>("bubble_text");
  let showBoxes = $state(true);
  let showBubbles = $state(true);
  let showClean = $state(false);
  let showPreview = $state(false);
  let showLettering = $state(true);
  let fonts = $state<string[]>([]);
  let liveItems = $state<LayoutItem[]>([]);
  let handSet = $state<string[]>([]);
  let previewVersion = $state(0);
  let letter = $state({
    font: "",
    size: 24,
    color: "#000000",
    strokePx: 0,
    strokeColor: "#ffffff",
    align: "center" as "center" | "left" | "right",
    angle: 0,
    lines: "",
    hidden: false,
  });
  let selectedId = $state<string | null>(null);
  let sourceDraft = $state("");
  let translationDraft = $state("");
  let profiles = $state<TranslationProfile[]>([]);
  let profileChoice = $state(""); // "" = every enabled profile
  let suggestions = $state<Suggestion[]>([]);
  let kept = $state<Suggestion | null>(null); // the suggestion the English draft was taken from
  let translating = $state(false);
  let cleanups = $state<CleanupPatch[]>([]);
  let cleanVersion = $state(0); // bumped on every refresh: patch ids are reused after a delete
  let cleanMethod = $state<CleanupMethod>("inpaint");
  let brush = $state(24);
  let fillColor = $state("#ffffff");
  let autoColor = $state(true);
  let cloneSource = $state<{ x: number; y: number } | null>(null);
  let cloneOffset: [number, number] | null = null;
  let painting: { x: number; y: number } | null = null;
  let painted = $state(false);
  let pointer = $state<{ x: number; y: number } | null>(null);
  let maskEl = $state<HTMLCanvasElement | null>(null);

  let svgEl = $state<SVGSVGElement | null>(null);
  let canvasEl = $state<HTMLDivElement | null>(null);
  type DragKind = "region" | "lettering";
  let drag: { id: string; kind: DragKind; handle: Handle; startX: number; startY: number; box: BBox } | null = null;
  let preview = $state<{ id: string; kind: DragKind; box: BBox } | null>(null);
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
  let cleanupsOnPage = $derived(page ? cleanups.filter((patch) => overlapsPage(page, patch.box)) : []);
  let liveById = $derived(new Map(liveItems.map((item) => [item.region_id, item])));
  let letteringOnPage = $derived(page && showLettering ? liveItems.filter((item) => overlapsPage(page, item.box)) : []);
  let selectedItem = $derived(selected ? liveById.get(selected.id) : undefined);
  let selectedLayout = $derived<LayoutEdit | undefined>(
    selected ? edits?.layout?.find((edit) => edit.region_id === selected!.id) : undefined,
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
    try {
      fonts = await listFonts();
    } catch {
      fonts = [];
    }
    if (canvasEl !== null && files.length > 0) {
      zoom = fitZoom(files[0].width, canvasEl.clientWidth - 24);
    }
    loaded = true;
  }

  /** Re-read everything an edit can change (regions, lines, edits summary, clean patches). */
  async function refresh(): Promise<void> {
    const [ocr, final, summary, inpaint, cleanup, layout] = await Promise.allSettled([
      getOcr(series, chapter),
      getFinal(series, chapter),
      getEdits(series, chapter),
      getInpaint(series, chapter),
      getCleanup(series, chapter),
      getLiveLayout(series, chapter),
    ]);
    liveItems = layout.status === "fulfilled" ? layout.value.items : [];
    handSet = layout.status === "fulfilled" ? layout.value.hand_set : [];
    previewVersion += 1;
    cleanups = cleanup.status === "fulfilled" ? cleanup.value.patches : [];
    cleanVersion += 1;
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
    resetLetterDraft(id);
    const region = regions.find((r) => r.id === id);
    sourceDraft = region?.text ?? "";
    translationDraft = region ? (lineById.get(region.id)?.text ?? "") : "";
  }

  /** The lettering form from the region's hand lettering where set, else from the typesetter's item. */
  function resetLetterDraft(id: string | null): void {
    const item = id === null ? undefined : liveById.get(id);
    const edit = id === null ? undefined : edits?.layout?.find((e) => e.region_id === id);
    letter = {
      font: edit?.font ?? item?.font ?? fonts[0] ?? "",
      size: edit?.size_px ?? item?.size_px ?? 24,
      color: rgbToHex(edit?.color ?? item?.color ?? [0, 0, 0]),
      strokePx: edit?.stroke_px ?? item?.stroke_px ?? 0,
      strokeColor: rgbToHex(edit?.stroke_color ?? item?.stroke_color ?? [255, 255, 255]),
      align: edit?.align ?? item?.align ?? "center",
      angle: edit?.angle ?? item?.angle ?? 0,
      lines: (edit?.lines ?? []).join("\n"),
      hidden: edit?.hidden ?? false,
    };
  }

  /** Send the lettering form as the region's hand lettering (keeping a box dragged earlier). */
  async function applyLettering(): Promise<void> {
    if (selected === null) return;
    const id = selected.id;
    const fields: LayoutFields = letter.hidden
      ? { hidden: true }
      : {
          font: letter.font || undefined,
          size_px: Math.round(letter.size),
          color: hexToRgb(letter.color),
          stroke_px: Math.round(letter.strokePx),
          stroke_color: hexToRgb(letter.strokeColor),
          align: letter.align,
          angle: Number(letter.angle) || 0,
          lines: typedLines(letter.lines),
          box: selectedLayout?.box ?? undefined,
        };
    await act(() => putLayout(series, chapter, id, fields));
    resetLetterDraft(id);
  }

  async function revertLettering(): Promise<void> {
    if (selected === null) return;
    const id = selected.id;
    await act(() => deleteLayout(series, chapter, id));
    resetLetterDraft(id);
  }

  function goToPage(index: number): void {
    if (index < 0 || index >= files.length) return;
    discardStrokes();
    cloneSource = null;
    pageIndex = index;
    if (selected !== null && page !== null && !overlapsPage(page, selected.bbox)) select(null);
  }

  function selectFromList(region: Region): void {
    select(region.id);
    const index = pageOf(files, region.bbox);
    if (index !== pageIndex) pageIndex = index;
  }

  function boxOf(region: Region): BBox {
    return preview !== null && preview.kind === "region" && preview.id === region.id ? preview.box : region.bbox;
  }

  function letteringBoxOf(item: LayoutItem): BBox {
    return preview !== null && preview.kind === "lettering" && preview.id === item.region_id ? preview.box : item.box;
  }

  function toPagePoint(event: PointerEvent): { x: number; y: number } {
    const rect = svgEl!.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / zoom, y: (event.clientY - rect.top) / zoom };
  }

  function startDrag(
    event: PointerEvent,
    id: string,
    handle: Handle,
    kind: DragKind = "region",
    box: BBox | null = null,
  ): void {
    if (tool !== "select" || busy) return;
    event.stopPropagation();
    event.preventDefault();
    if (selectedId !== id) select(id);
    const start = box ?? regions.find((region) => region.id === id)?.bbox;
    if (start === undefined) return;
    const point = toPagePoint(event);
    drag = { id, kind, handle, startX: point.x, startY: point.y, box: start };
    svgEl!.setPointerCapture(event.pointerId);
  }

  function onBackgroundDown(event: PointerEvent): void {
    if (busy) return;
    if (tool === "clean") {
      startPaint(event);
      return;
    }
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
    pointer = point;
    if (painting !== null) {
      paintTo(point);
    } else if (drag !== null) {
      const s = page.scale > 0 ? page.scale : 1;
      const box = dragBox(drag.box, drag.handle, (point.x - drag.startX) * s, (point.y - drag.startY) * s);
      preview = { id: drag.id, kind: drag.kind, box };
    } else if (drawing !== null) {
      drawing = { ...drawing, bx: point.x, by: point.y };
    }
  }

  async function onPointerUp(): Promise<void> {
    if (painting !== null) {
      painting = null;
    } else if (drag !== null) {
      const started = drag;
      const moved = preview;
      drag = null;
      if (moved !== null && !sameBox(moved.box, started.box)) {
        if (started.kind === "lettering") {
          const fields = { ...setFields(edits?.layout?.find((e) => e.region_id === started.id)), box: moved.box };
          await act(() => putLayout(series, chapter, started.id, fields));
        } else {
          await act(() => patchRegion(series, chapter, started.id, { bbox: moved.box }));
        }
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

  function maskContext(): CanvasRenderingContext2D | null {
    return maskEl?.getContext("2d") ?? null;
  }

  function startPaint(event: PointerEvent): void {
    const point = toPagePoint(event);
    if (cleanMethod === "clone" && event.altKey) {
      cloneSource = point;
      cloneOffset = null;
      return;
    }
    if (cleanMethod === "clone" && cloneSource === null) {
      actionError = "Alt+click the spot to copy from first";
      return;
    }
    if (cleanMethod === "clone" && cloneOffset === null && cloneSource !== null) {
      cloneOffset = [Math.round(cloneSource.x - point.x), Math.round(cloneSource.y - point.y)];
    }
    actionError = "";
    painting = point;
    svgEl!.setPointerCapture(event.pointerId);
    paintTo(point);
  }

  function paintTo(point: { x: number; y: number }): void {
    const ctx = maskContext();
    if (ctx === null || painting === null) return;
    ctx.strokeStyle = "#ff0050";
    ctx.fillStyle = "#ff0050";
    ctx.lineWidth = brush;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(painting.x, painting.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(point.x, point.y, brush / 2, 0, Math.PI * 2);
    ctx.fill();
    painting = point;
    painted = true;
  }

  function discardStrokes(): void {
    const ctx = maskContext();
    if (ctx !== null && maskEl !== null) ctx.clearRect(0, 0, maskEl.width, maskEl.height);
    painted = false;
    cloneOffset = null;
  }

  /** Send the painted strokes as one cleanup patch of the chosen method. */
  async function applyStrokes(): Promise<void> {
    const ctx = maskContext();
    if (ctx === null || maskEl === null || page === null) return;
    const bounds = maskBounds(ctx.getImageData(0, 0, maskEl.width, maskEl.height).data, maskEl.width, maskEl.height);
    if (bounds === null) return;
    const out = document.createElement("canvas");
    out.width = bounds.x1 - bounds.x0;
    out.height = bounds.y1 - bounds.y0;
    out.getContext("2d")!.drawImage(maskEl, bounds.x0, bounds.y0, out.width, out.height, 0, 0, out.width, out.height);
    const stroke = {
      page: page.index,
      box: bounds,
      mask: out.toDataURL("image/png"),
      method: cleanMethod,
      color: cleanMethod === "fill" && !autoColor ? hexToRgb(fillColor) : null,
      offset: cleanMethod === "clone" ? cloneOffset : null,
    };
    const patch = await act(() => addCleanup(series, chapter, stroke));
    if (patch !== null) {
      discardStrokes();
      showClean = true;
    }
  }

  async function removeCleanup(patchId: string): Promise<void> {
    await act(() => deleteCleanup(series, chapter, patchId));
  }

  /** Hand the keyboard back to the page after a toolbar control changed, so Enter, [ ] and Alt+click reach
   *  the editor instead of the control. */
  function blurControl(event: Event): void {
    (event.currentTarget as HTMLElement).blur();
  }

  function setTool(next: "select" | "draw" | "clean"): void {
    if (tool === "clean" && next !== "clean") discardStrokes();
    tool = next;
    if (next === "clean") {
      showClean = true;
      select(null);
    }
  }

  function nudge(dx: number, dy: number): void {
    if (selected === null || page === null) return;
    const s = page.scale > 0 ? page.scale : 1;
    const base = preview !== null && preview.kind === "region" && preview.id === selected.id ? preview.box : selected.bbox;
    preview = { id: selected.id, kind: "region", box: dragBox(base, "move", dx * s, dy * s) };
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
    if (tool === "clean" && event.key === "Enter") {
      void applyStrokes();
    } else if (tool === "clean" && event.key === "Escape" && painted) {
      discardStrokes();
    } else if (event.key === "[") {
      brush = Math.max(2, brush - 4);
    } else if (event.key === "]") {
      brush = Math.min(120, brush + 4);
    } else if (event.key === "c" || event.key === "C") {
      setTool("clean");
    } else if (event.key === "Escape") {
      drawing = null;
      tool = "select";
      select(null);
    } else if (event.key === "b" || event.key === "B") {
      setTool("draw");
    } else if (event.key === "v" || event.key === "V") {
      setTool("select");
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
    <button class:active={tool === "select"} onclick={() => setTool("select")} title="select, move and resize boxes (V)">Select</button>
    <button class:active={tool === "draw"} onclick={() => setTool("draw")} title="draw a new text box (B)">Draw box</button>
    <button class:active={tool === "clean"} onclick={() => setTool("clean")} title="paint over what to clean or restore (C)">Clean</button>
    <select bind:value={newKind} title="kind of the next drawn box">
      {#each KINDS as kind (kind)}
        <option value={kind}>{kind}</option>
      {/each}
    </select>
    <span class="sep"></span>
    <label><input type="checkbox" bind:checked={showBoxes} /> boxes</label>
    <label><input type="checkbox" bind:checked={showBubbles} /> bubbles</label>
    <label title="show the cleaned patches (inpaint) over the raw page"><input type="checkbox" bind:checked={showClean} /> cleaned</label>
    <label title="show the page as the release will look: cleaned and lettered, your edits included"><input type="checkbox" bind:checked={showPreview} /> preview</label>
    <label title="show where each English line is lettered (drag the selected one to move or resize it)"><input type="checkbox" bind:checked={showLettering} /> lettering</label>
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
  {#if tool === "clean"}
    <div class="toolbar cleanbar">
      <select bind:value={cleanMethod} onchange={blurControl} title="what the painted pixels become">
        <option value="inpaint">inpaint (rebuild from the surroundings)</option>
        <option value="fill">fill with a colour</option>
        <option value="clone">clone (Alt+click the source first)</option>
        <option value="restore">restore the raw page</option>
      </select>
      <label title="brush size ([ and ])">brush <input type="range" min="2" max="120" bind:value={brush} onchange={blurControl} /> {brush}px</label>
      {#if cleanMethod === "fill"}
        <label><input type="checkbox" bind:checked={autoColor} /> colour around the stroke</label>
        {#if !autoColor}<input type="color" bind:value={fillColor} title="fill colour" />{/if}
      {/if}
      {#if cleanMethod === "clone"}
        <span class="muted">{cloneSource === null ? "Alt+click where to copy from" : "source set — paint to copy"}</span>
      {/if}
      <button onclick={() => void applyStrokes()} disabled={busy || !painted} title="Enter">Apply</button>
      <button onclick={discardStrokes} disabled={!painted} title="Esc">Discard strokes</button>
    </div>
  {/if}
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
          <img
            src={showPreview ? previewUrl(series, chapter, page.index, previewVersion) : pageImageUrl(series, chapter, page.index)}
            alt={page.name}
            width={page.width * zoom}
            height={page.height * zoom}
            draggable="false"
          />
          <canvas
            bind:this={maskEl}
            width={page.width}
            height={page.height}
            class="mask"
            style="width: {page.width * zoom}px; height: {page.height * zoom}px;"
          ></canvas>
          <svg
            bind:this={svgEl}
            width={page.width * zoom}
            height={page.height * zoom}
            viewBox="0 0 {page.width} {page.height}"
            style="cursor: {tool === 'select' ? 'default' : tool === 'clean' ? 'none' : 'crosshair'};"
            onpointerleave={() => (pointer = null)}
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
            {#if showClean}
              {#each cleanupsOnPage as patch (patch.id)}
                {@const rect = stripToPage(page, patch.box)}
                <image href={cleanupPatchUrl(series, chapter, patch.id, cleanVersion)} x={rect.x} y={rect.y} width={rect.width} height={rect.height} preserveAspectRatio="none" pointer-events="none" />
              {/each}
            {/if}
            {#if showBoxes && tool !== "clean"}
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
                  onpointerdown={(e) => (tool === "select" ? startDrag(e, region.id, "move") : undefined)}
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
                      onpointerdown={(e) => startDrag(e, region.id, handle)}
                      role="button"
                      tabindex="-1"
                      aria-label="resize {handle}"
                    />
                  {/each}
                {/if}
              {/each}
            {/if}
            {#if tool === "select"}
              {#each letteringOnPage as item (item.region_id)}
                {@const rect = stripToPage(page, letteringBoxOf(item))}
                {@const mine = item.region_id === selectedId}
                <rect
                  x={rect.x}
                  y={rect.y}
                  width={rect.width}
                  height={rect.height}
                  fill="none"
                  stroke={item.overflow ? "#dc2626" : "#0d9488"}
                  stroke-width={(mine ? 2 : 1) / zoom}
                  stroke-dasharray="{5 / zoom} {3 / zoom}"
                  pointer-events={mine ? "all" : "none"}
                  style="cursor: move;"
                  onpointerdown={(e) => startDrag(e, item.region_id, "move", "lettering", item.box)}
                  role="button"
                  tabindex="-1"
                  aria-label="lettering {item.region_id}"
                />
                {#if mine}
                  {#each HANDLES as handle (handle)}
                    {@const point = handlePoint(rect, handle)}
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r={4 / zoom}
                      fill="#0d9488"
                      style="cursor: {handleCursor(handle)};"
                      onpointerdown={(e) => startDrag(e, item.region_id, handle, "lettering", item.box)}
                      role="button"
                      tabindex="-1"
                      aria-label="lettering resize {handle}"
                    />
                  {/each}
                {/if}
              {/each}
            {/if}
            {#if tool === "clean" && pointer !== null}
              <circle cx={pointer.x} cy={pointer.y} r={brush / 2} fill="none" stroke="#ff0050" stroke-width={1.5 / zoom} pointer-events="none" />
            {/if}
            {#if tool === "clean" && cleanMethod === "clone" && cloneSource !== null}
              <g pointer-events="none" stroke="#1f6feb" stroke-width={2 / zoom}>
                <line x1={cloneSource.x - 8 / zoom} y1={cloneSource.y} x2={cloneSource.x + 8 / zoom} y2={cloneSource.y} />
                <line x1={cloneSource.x} y1={cloneSource.y - 8 / zoom} x2={cloneSource.x} y2={cloneSource.y + 8 / zoom} />
              </g>
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
        {#if selectedItem !== undefined || selectedLayout?.hidden}
          <section class="lettering">
            <h4>
              Lettering
              {#if handSet.includes(selected.id)}<span class="labels">hand-set</span>{/if}
              {#if selectedItem?.overflow}<span class="error">overflows its box</span>{/if}
            </h4>
            <div class="grid2">
              <label>
                font
                <select bind:value={letter.font}>
                  {#each fonts as font (font)}
                    <option value={font}>{font.replace(/\.(ttf|otf)$/i, "")}</option>
                  {/each}
                  {#if letter.font && !fonts.includes(letter.font)}<option value={letter.font}>{letter.font}</option>{/if}
                </select>
              </label>
              <label>size <input type="number" min="4" max="400" bind:value={letter.size} /></label>
              <label>colour <input type="color" bind:value={letter.color} /></label>
              <label>outline <input type="number" min="0" max="40" bind:value={letter.strokePx} /></label>
              <label>outline colour <input type="color" bind:value={letter.strokeColor} /></label>
              <label>
                align
                <select bind:value={letter.align}>
                  <option value="center">centre</option>
                  <option value="left">left</option>
                  <option value="right">right</option>
                </select>
              </label>
              <label>angle <input type="number" min="-180" max="180" step="1" bind:value={letter.angle} /></label>
              <label><input type="checkbox" bind:checked={letter.hidden} /> no lettering</label>
            </div>
            <label>
              line breaks <span class="muted">(one line per row; empty = automatic)</span>
              <textarea rows="3" bind:value={letter.lines} lang="en"></textarea>
            </label>
            <p class="muted">Drag the teal dashed box on the page to move or resize the lettering.</p>
            <div class="actions">
              <button onclick={() => void applyLettering()} disabled={busy}>Apply lettering</button>
              {#if handSet.includes(selected.id)}
                <button onclick={() => void revertLettering()} disabled={busy}>Revert lettering</button>
              {/if}
            </div>
          </section>
        {/if}
      {:else if tool === "clean"}
        <section>
          <h3>Clean page {pageIndex + 1}</h3>
          <p class="muted">
            Paint over leftover lettering or damaged art, then <b>Apply</b> (Enter). <i>inpaint</i> rebuilds the
            pixels from what surrounds them, <i>fill</i> paints one colour (by default the colour around the
            stroke), <i>clone</i> copies from the spot you Alt+clicked, <i>restore</i> brings back the raw page
            where the automatic cleaning went too far. <b>[</b> and <b>]</b> change the brush. Your patches go on
            after the automatic cleaning at export; use Render to see them in the finished pages.
          </p>
          <h4>Hand cleanup on this page</h4>
          {#if cleanupsOnPage.length === 0}<p class="muted">none yet</p>{/if}
          <ul class="list">
            {#each cleanupsOnPage as patch (patch.id)}
              <li>
                {patch.id} · {patch.method} · {patch.mask_px} px
                <button class="link" onclick={() => void removeCleanup(patch.id)} disabled={busy}>delete</button>
              </li>
            {/each}
          </ul>
          {#if cleanups.length > 0}
            <button onclick={() => void removeCleanup(cleanups[cleanups.length - 1].id)} disabled={busy}>Undo last cleanup</button>
          {/if}
        </section>
      {:else}
        <section>
          <h3>Page {pageIndex + 1}</h3>
          <p class="muted">
            Click a box to edit it, drag it to move, drag its handles to resize. <b>B</b> draws a new box, <b>C</b> cleans,
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
  .page canvas.mask {
    position: absolute;
    left: 0;
    top: 0;
    opacity: 0.45;
    pointer-events: none;
  }
  .cleanbar {
    background: #fff1f2;
    padding: 4px;
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
  .grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2px 8px;
  }
  .grid2 input[type="number"],
  .grid2 select {
    width: 100%;
    box-sizing: border-box;
  }
  .lettering {
    border-top: 1px solid #e5e7eb;
    margin-top: 8px;
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
