<script lang="ts">
  import RunButton from "./RunButton.svelte";
  import { getFinal, getGlossaryHits, getOcr, getTranslation, listTranslations } from "./api";
  import type { CandidateRun, FinalArtifact, GlossaryHit, GlossaryHits, RegionsArtifact } from "./api";
  import { buildRows, filterRows, highlightSource, hitColor, summarizeReview } from "./translation";
  import type { ReviewRow } from "./translation";

  let { series, chapter }: { series: string; chapter: string } = $props();

  let ocr = $state<RegionsArtifact | null>(null);
  let runs = $state<CandidateRun[]>([]);
  let final = $state<FinalArtifact | null>(null);
  let hits = $state<GlossaryHits | null>(null);
  let failedRuns = $state<string[]>([]);
  let loaded = $state(false);
  let missingOcr = $state(false);
  let onlyAttention = $state(false);

  let rows = $derived(ocr ? buildRows(ocr, runs, final, hits) : []);
  let summary = $derived(summarizeReview(rows));
  let visible = $derived(filterRows(rows, onlyAttention));

  function isViolation(hit: GlossaryHit): boolean {
    return hit.status === "locked" && hit.target_in_final === false;
  }

  function notesFor(run: CandidateRun, row: ReviewRow): string | null {
    const candidate = run.candidates.find((c) => c.region_id === row.region.id);
    return candidate?.notes ?? null;
  }

  $effect(() => {
    const s = series;
    const c = chapter;
    ocr = null;
    runs = [];
    final = null;
    hits = null;
    failedRuns = [];
    loaded = false;
    missingOcr = false;
    void load(s, c);
  });

  async function load(series: string, chapter: string): Promise<void> {
    const [ocrResult, runIdsResult, finalResult, hitsResult] = await Promise.allSettled([
      getOcr(series, chapter),
      listTranslations(series, chapter),
      getFinal(series, chapter),
      getGlossaryHits(series, chapter),
    ]);
    if (ocrResult.status === "rejected") {
      missingOcr = true;
      loaded = true;
      return;
    }
    ocr = ocrResult.value;
    final = finalResult.status === "fulfilled" ? finalResult.value : null;
    hits = hitsResult.status === "fulfilled" ? hitsResult.value : null;
    const runIds = runIdsResult.status === "fulfilled" ? runIdsResult.value : [];
    const results = await Promise.allSettled(runIds.map((runId) => getTranslation(series, chapter, runId)));
    const ok: CandidateRun[] = [];
    const failed: string[] = [];
    results.forEach((result, i) => {
      if (result.status === "fulfilled") {
        ok.push(result.value);
      } else {
        failed.push(runIds[i]);
      }
    });
    runs = ok;
    failedRuns = failed;
    loaded = true;
  }
</script>

{#if missingOcr}
  <p>no ocr.json yet — run ocr first</p>
  <RunButton {series} {chapter} through="ocr" onDone={() => load(series, chapter)} />
{:else if !loaded}
  <p>loading…</p>
{:else if ocr}
  <p style="margin: 4px 0;">
    <label style="margin-right: 12px;">
      <input type="checkbox" bind:checked={onlyAttention} />
      only rows needing attention
    </label>
    <span>
      {summary.total} regions · {summary.withFinal} with final · {summary.flagged} flagged ·
      {summary.violations} glossary violation · {summary.disagreements} disagreements
    </span>
  </p>
  {#each failedRuns as runId (runId)}
    <p style="color: #dc2626;">could not load run {runId}</p>
  {/each}
  <table class="review">
    <thead>
      <tr>
        <th>#</th>
        <th>source</th>
        {#each runs as run (run.run_id)}
          <th title="{run.profile} · {run.model}">{run.run_id}</th>
        {/each}
        <th>final</th>
        <th>glossary</th>
      </tr>
    </thead>
    <tbody>
      {#each visible as row (row.region.id)}
        {@const background =
          row.violations.length > 0 ? "#fef2f2" : row.needsAttention ? "#fffbeb" : "transparent"}
        <tr style="background: {background};">
          <td style="white-space: nowrap;">
            {row.region.id}<br />
            slice {row.region.slice_index}<br />
            order {row.region.reading_order}<br />
            {row.region.kind}
          </td>
          <td style="white-space: pre-wrap;">
            {#each highlightSource(row.region.text, row.hits) as segment, i (i)}
              {#if segment.hit}
                <mark
                  style="background: {hitColor(segment.hit)}33; color: inherit; border-radius: 2px;"
                  title="{segment.hit.source} → {segment.hit.target} ({segment.hit.status})"
                >{segment.text}</mark
                >
              {:else}{segment.text}{/if}
            {/each}
            {#if row.region.ocr_alt !== null && row.region.ocr_alt !== row.region.text}
              <div style="color: #6b7280;">alt: {row.region.ocr_alt}</div>
            {/if}
          </td>
          {#each runs as run (run.run_id)}
            {@const text = row.candidates[run.run_id] ?? null}
            {@const notes = notesFor(run, row)}
            <td style="white-space: pre-wrap;" title={notes ?? undefined}>
              {#if text !== null}{text}{:else}<span style="color: #6b7280;">—</span>{/if}
            </td>
          {/each}
          <td style="white-space: pre-wrap;">
            {#if row.final}
              {row.final.text}
              <span class="badge">{row.final.decision}</span>
              {#each row.final.flags as flag (flag)}
                <span
                  class="badge"
                  style="background: {flag === 'glossary_violation' ? '#dc2626' : '#f59e0b'};"
                >{flag}</span
                >
              {/each}
              {#if row.final.rationale}
                <details>
                  <summary>why</summary>
                  {row.final.rationale}
                </details>
              {/if}
            {:else}
              <span style="color: #6b7280;">—</span>
            {/if}
          </td>
          <td style="white-space: pre-wrap;">
            {#each row.hits as hit, i (i)}
              <span
                class="chip"
                style="border-color: {hitColor(hit)}; color: {hitColor(hit)};"
              >{isViolation(hit) ? "✗ " : ""}{hit.source} → {hit.target}</span
              >
            {/each}
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
{/if}

<style>
  table.review {
    border-collapse: collapse;
    width: 100%;
  }
  table.review th,
  table.review td {
    border: 1px solid #d1d5db;
    padding: 4px 8px;
    text-align: left;
    vertical-align: top;
    font-size: 14px;
  }
  table.review thead th {
    position: sticky;
    top: 0;
    background: white;
    z-index: 1;
  }
  .badge {
    display: inline-block;
    margin: 2px 2px 0 0;
    padding: 0 6px;
    border-radius: 8px;
    background: #6b7280;
    color: white;
    font-size: 11px;
  }
  .chip {
    display: inline-block;
    margin: 0 6px 2px 0;
    padding: 0 6px;
    border: 1px solid;
    border-radius: 8px;
    font-size: 12px;
  }
</style>