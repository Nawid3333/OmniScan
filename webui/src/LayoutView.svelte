<script lang="ts">
  import RunButton from "./RunButton.svelte";
  import { getIngest, getLayout, pageImageUrl } from "./api";
  import { stackTops as computeStackTops, stackTotalHeight, stackWidth, stripBoxToStackBox } from "./strip";
  import { alignLabel, filterItems, fontRoleColor } from "./layout";
  import type { FontRole, LayoutItem, SourceFile } from "./api";

  let { series, chapter }: { series: string; chapter: string } = $props();

  const ROLES: FontRole[] = ["dialogue", "thought", "shout", "narration", "free", "sfx"];

  let files = $state<SourceFile[]>([]);
  let items = $state<LayoutItem[]>([]);
  let loaded = $state(false);
  let error = $state("");
  let selectedId = $state<string | null>(null);
  let visibleRoles = $state<Set<FontRole>>(new Set(ROLES));

  let width = $derived(files.length > 0 ? stackWidth(files) : 0);
  let totalHeight = $derived(stackTotalHeight(files));
  let visible = $derived(filterItems(items, visibleRoles));
  let selected = $derived(items.find((item) => item.region_id === selectedId) ?? null);

  // Top of each file's image in the stack: sum of all earlier files' natural heights.
  let stackTops = $derived(computeStackTops(files));

  function toggleRole(role: FontRole): void {
    const next = new Set(visibleRoles);
    if (next.has(role)) {
      next.delete(role);
    } else {
      next.add(role);
    }
    visibleRoles = next;
  }

  $effect(() => {
    const s = series;
    const c = chapter;
    error = "";
    loaded = false;
    files = [];
    items = [];
    selectedId = null;
    visibleRoles = new Set(ROLES);
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
    }
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
  <p style="margin: 4px 0;">
    {#each ROLES as role (role)}
      <label style="margin-right: 12px; color: {fontRoleColor(role)};">
        <input type="checkbox" checked={visibleRoles.has(role)} onchange={() => toggleRole(role)} />
        {role}
      </label>
    {/each}
    <span>{visible.length} of {items.length} items</span>
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
      {#each visible as item (item.region_id)}
        {@const box = stripBoxToStackBox(files, item.box)}
        {@const color = fontRoleColor(item.font_role)}
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
          stroke-dasharray={item.overflow ? "6 3" : undefined}
          pointer-events="all"
          onclick={() => (selectedId = selectedId === item.region_id ? null : item.region_id)}
        />
        <text x={box.x} y={Math.max(13, box.y - 4)} fill={color} font-size="13">
          {item.font_role} · {item.font} {item.size_px}px{item.overflow ? " · overflow" : ""}
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
        <dt>font_role</dt>
        <dd style="color: {fontRoleColor(selected.font_role)}">{selected.font_role}</dd>
        <dt>font</dt>
        <dd>{selected.font}</dd>
        <dt>size_px</dt>
        <dd>{selected.size_px}</dd>
        <dt>align</dt>
        <dd>{alignLabel(selected.align)}</dd>
        <dt>lines</dt>
        <dd style="white-space: pre-wrap;">{selected.lines.join("\n")}</dd>
        <dt>color</dt>
        <dd>
          <span
            style="display: inline-block; width: 14px; height: 14px; background: rgb({selected.color[0]}, {selected.color[1]}, {selected.color[2]}); vertical-align: -2px; border: 1px solid #d1d5db;"
          ></span>
          rgb({selected.color.join(", ")})
        </dd>
        <dt>stroke</dt>
        <dd>
          <span
            style="display: inline-block; width: 14px; height: 14px; background: rgb({selected.stroke_color[0]}, {selected.stroke_color[1]}, {selected.stroke_color[2]}); vertical-align: -2px; border: 1px solid #d1d5db;"
          ></span>
          rgb({selected.stroke_color.join(", ")}) · {selected.stroke_px}px
        </dd>
        <dt>overflow</dt>
        <dd style={selected.overflow ? "color: #dc2626; font-weight: bold;" : ""}>{selected.overflow}</dd>
      </dl>
    </aside>
  {/if}
{/if}