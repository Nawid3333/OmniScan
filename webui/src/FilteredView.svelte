<script lang="ts">
  import { listFiltered, pageImageUrl, restoreFiltered } from "./api";
  import type { FilteredChapter, FilteredItem } from "./api";
  import { describeItem, itemKey, markRestored, scoreText, summarizeFiltered } from "./filtered";

  let { series }: { series: string } = $props();

  let chapters = $state<FilteredChapter[]>([]);
  let loaded = $state(false);
  let error = $state("");
  let busyKey = $state<string | null>(null);
  let itemErrors = $state<Record<string, string>>({});

  let summary = $derived(summarizeFiltered(chapters));

  $effect(() => {
    const s = series;
    error = "";
    loaded = false;
    chapters = [];
    itemErrors = {};
    busyKey = null;
    void load(s);
  });

  async function load(s: string): Promise<void> {
    try {
      chapters = await listFiltered(s);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      chapters = [];
    }
    loaded = true;
  }

  async function onRestore(chapter: string, item: FilteredItem): Promise<void> {
    const key = itemKey(chapter, item);
    itemErrors = { ...itemErrors, [key]: "" };
    busyKey = key;
    try {
      await restoreFiltered(series, chapter, item.target, item.index);
      chapters = markRestored(chapters, chapter, item.target, item.index);
    } catch (e) {
      itemErrors = { ...itemErrors, [key]: e instanceof Error ? e.message : String(e) };
    } finally {
      busyKey = null;
    }
  }
</script>

{#if error}
  <p>{error}</p>
{:else if !loaded}
  <p>loading…</p>
{:else if chapters.length === 0}
  <p>nothing was filtered in this series</p>
{:else}
  <p>{summary.chapters} chapters · {summary.filtered} filtered · {summary.restored} restored</p>
  {#each chapters as c (c.chapter)}
    <h3>{c.chapter}</h3>
    <ul>
      {#each c.items as item (itemKey(c.chapter, item))}
        {@const key = itemKey(c.chapter, item)}
        <li class="item">
          {#if item.target === "file"}
            <img
              src={pageImageUrl(series, c.chapter, item.index)}
              alt={item.name ?? `page ${item.index}`}
              loading="lazy"
              style="max-width: 160px; max-height: 220px;"
            />
          {/if}
          <div>
            <p>
              {describeItem(item)}
              <span class="badge {item.state}">{item.state}</span>
            </p>
            <p>{scoreText(item)}</p>
            {#if item.state === "filtered"}
              <button onclick={() => onRestore(c.chapter, item)} disabled={busyKey === key}>Restore</button>
            {/if}
            {#if itemErrors[key]}
              <span class="error">{itemErrors[key]}</span>
            {/if}
          </div>
        </li>
      {/each}
    </ul>
  {/each}
{/if}

<style>
  .item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin: 8px 0;
    list-style: none;
  }
  .item img {
    border: 1px solid #d1d5db;
  }
  .badge {
    display: inline-block;
    margin-left: 8px;
    padding: 1px 8px;
    border-radius: 999px;
    color: #fff;
    font-size: 12px;
  }
  .badge.filtered {
    background: #dc2626;
  }
  .badge.restored {
    background: #16a34a;
  }
  .error {
    margin-left: 8px;
    color: #dc2626;
  }
</style>