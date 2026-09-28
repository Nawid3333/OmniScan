<script lang="ts">
  import { getProgress } from "./api";
  import type { ChapterProgress } from "./api";
  import { exportState, isDone, percent, pipelineState, totals } from "./progress";

  let { series, onOpen }: { series: string; onOpen: (chapter: string) => void } = $props();

  let chapters = $state<ChapterProgress[] | null>(null);
  let error = $state("");
  let busy = $state(false);

  let sum = $derived(totals(chapters ?? []));

  $effect(() => {
    const s = series;
    chapters = null;
    void load(s);
  });

  async function load(s: string): Promise<void> {
    busy = true;
    try {
      chapters = await getProgress(s);
      error = "";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
</script>

<section class="progress">
  <p class="intro">
    Where each chapter stands: how far the pipeline got, the lines translated and checked, the open problems on the
    lines not checked yet, and whether the pages are exported and current. Open a chapter to see its problems.
    <button onclick={() => load(series)} disabled={busy}>Check again</button>
  </p>
  {#if error}
    <p class="error">{error}</p>
  {/if}
  {#if chapters === null}
    {#if !error}<p>loading…</p>{/if}
  {:else if chapters.length === 0}
    <p>No chapters yet.</p>
  {:else}
    <p class="hint">
      {sum.done} of {chapters.length} chapter(s) ready · {percent(sum.translated, sum.lines)}% translated ·
      {percent(sum.checked, sum.lines)}% checked · {sum.problems} open problem(s)
    </p>
    <table>
      <thead>
        <tr><th>Chapter</th><th>Pipeline</th><th>English</th><th>Checked</th><th>Problems</th><th>Pages</th></tr>
      </thead>
      <tbody>
        {#each chapters as chapter (chapter.chapter)}
          <tr class:done={isDone(chapter)}>
            <td><button class="place" onclick={() => onOpen(chapter.chapter)} title="open this chapter's problems">{chapter.chapter}</button></td>
            <td class:failed={chapter.failed !== null}>{pipelineState(chapter)}</td>
            <td>
              <span class="bar"><span style="width: {percent(chapter.translated, chapter.lines)}%"></span></span>
              {chapter.translated}/{chapter.lines}
            </td>
            <td>
              <span class="bar checked"><span style="width: {percent(chapter.checked, chapter.lines)}%"></span></span>
              {chapter.checked}/{chapter.lines}
            </td>
            <td class:warn={chapter.problems > 0}>{chapter.problems}</td>
            <td class:warn={chapter.outdated}>{exportState(chapter)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</section>

<style>
  .progress table {
    border-collapse: collapse;
  }
  .progress td,
  .progress th {
    padding: 4px 10px;
    border-bottom: 1px solid #e5e7eb;
    text-align: left;
    white-space: nowrap;
  }
  .progress tr.done td {
    background: #f0fdf4;
  }
  .bar {
    display: inline-block;
    width: 60px;
    height: 6px;
    background: #e5e7eb;
    border-radius: 3px;
    vertical-align: middle;
    margin-right: 4px;
    overflow: hidden;
  }
  .bar > span {
    display: block;
    height: 100%;
    background: #2563eb;
  }
  .bar.checked > span {
    background: #16a34a;
  }
  .failed,
  .error {
    color: #dc2626;
  }
  .warn {
    color: #b45309;
    font-weight: 600;
  }
  .hint {
    color: #4b5563;
  }
</style>
