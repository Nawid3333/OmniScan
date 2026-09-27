<script lang="ts">
  import { getConsistency } from "./api";
  import type { ConsistencyReport } from "./api";
  import { filterReport, placeCount } from "./consistency";

  let { series, onOpen }: { series: string; onOpen: (chapter: string, regionId: string) => void } = $props();

  let report = $state<ConsistencyReport | null>(null);
  let error = $state("");
  let busy = $state(false);
  let query = $state("");

  let shown = $derived(report ? filterReport(report, query) : null);

  $effect(() => {
    const s = series;
    report = null;
    query = "";
    void load(s);
  });

  async function load(s: string): Promise<void> {
    busy = true;
    try {
      report = await getConsistency(s);
      error = "";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
</script>

<section class="consistency">
  <p class="intro">
    Lines of this series said again but translated differently, and lines missing a locked glossary term's English.
    Open a place to fix it in the Studio (its Find &amp; replace fixes many at once).
    <button onclick={() => load(series)} disabled={busy}>Check again</button>
    <input placeholder="filter" bind:value={query} />
  </p>
  {#if error}
    <p class="error">{error}</p>
  {/if}
  {#if shown === null}
    <p>{busy ? "checking…" : ""}</p>
  {:else}
    <h2>Translated differently ({shown.divergences.length})</h2>
    {#if shown.divergences.length === 0}
      <p class="muted">Every repeated line reads the same everywhere.</p>
    {/if}
    {#each shown.divergences as divergence (divergence.source)}
      <div class="item">
        <p class="source">{divergence.source} <span class="muted">· {placeCount(divergence)} places</span></p>
        <ul>
          {#each divergence.renderings as rendering (rendering.english)}
            <li>
              <span class="english">{rendering.english}</span>
              {#each rendering.places as [chapter, regionId] (`${chapter}/${regionId}`)}
                <button class="place" onclick={() => onOpen(chapter, regionId)} title="open in the Studio">{chapter} · {regionId}</button>
              {/each}
            </li>
          {/each}
        </ul>
      </div>
    {/each}
    <h2>Glossary terms missing ({shown.term_misses.length})</h2>
    {#if shown.term_misses.length === 0}
      <p class="muted">Every translated line with a locked term uses its English.</p>
    {/if}
    <ul>
      {#each shown.term_misses as miss (`${miss.chapter}/${miss.region_id}/${miss.term}`)}
        <li>
          <button class="place" onclick={() => onOpen(miss.chapter, miss.region_id)} title="open in the Studio">{miss.chapter} · {miss.region_id}</button>
          <strong>{miss.term}</strong> → <em>{miss.target}</em>: <span class="english">{miss.english}</span>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .item {
    border-top: 1px solid #e5e7eb;
    padding: 4px 0;
  }
  .source {
    font-weight: bold;
    margin: 4px 0;
  }
  .english {
    margin-right: 8px;
  }
  .place {
    font-size: 12px;
    margin: 0 4px 2px 0;
  }
  .muted {
    color: #6b7280;
    font-weight: normal;
  }
  .error {
    color: #b91c1c;
  }
</style>
