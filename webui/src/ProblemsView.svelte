<script lang="ts">
  import { allowTypoWord, getProblems } from "./api";
  import type { Problem } from "./api";
  import { filterProblems, kindCounts, kindLabel, regionCount } from "./problems";

  let {
    series,
    chapter,
    onOpen,
  }: { series: string; chapter: string; onOpen: (regionId: string) => void } = $props();

  let problems = $state<Problem[] | null>(null);
  let error = $state("");
  let busy = $state(false);
  let kind = $state("");
  let hideChecked = $state(true);

  let shown = $derived(problems ? filterProblems(problems, kind, hideChecked) : []);
  let kinds = $derived(kindCounts(problems ? filterProblems(problems, "", hideChecked) : []));

  $effect(() => {
    const s = series;
    const c = chapter;
    problems = null;
    kind = "";
    void load(s, c);
  });

  async function load(s: string, c: string): Promise<void> {
    busy = true;
    try {
      problems = await getProblems(s, c);
      error = "";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  async function notATypo(word: string): Promise<void> {
    busy = true;
    try {
      await allowTypoWord(series, word);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      busy = false;
      return;
    }
    await load(series, chapter);
  }
</script>

<section class="problems">
  <p class="intro">
    What to look at before this chapter goes out: lines with no English, source text left in the English, lettering
    that overflows its balloon, lines the judge was unsure of or that miss a locked glossary term, much too long lines,
    possible typos, and what the last <code>omniscan qa</code> still read on the finished pages. Open a line to fix it
    in the Studio.
    <button onclick={() => load(series, chapter)} disabled={busy}>Check again</button>
  </p>
  {#if error}
    <p class="error">{error}</p>
  {/if}
  {#if problems === null}
    {#if !error}<p>loading…</p>{/if}
  {:else}
    <p>
      <select bind:value={kind} aria-label="kind filter">
        <option value="">every kind ({filterProblems(problems, "", hideChecked).length})</option>
        {#each kinds as entry (entry.kind)}
          <option value={entry.kind}>{kindLabel(entry.kind)} ({entry.count})</option>
        {/each}
      </select>
      <label><input type="checkbox" bind:checked={hideChecked} /> hide checked lines</label>
    </p>
    {#if shown.length === 0}
      <p class="ok">
        {problems.length === 0 ? "No problems found." : "Nothing left with this filter."}
      </p>
    {:else}
      <p class="hint">{shown.length} problem(s) on {regionCount(shown)} line(s).</p>
      <ul>
        {#each shown as problem, i (`${problem.region_id}/${problem.kind}/${problem.finished_page}/${i}`)}
          <li class:checked={problem.status === "checked"}>
            <button class="place" onclick={() => onOpen(problem.region_id)} title="open in the Studio">
              {problem.region_id}
            </button>
            <span class="kind">{kindLabel(problem.kind)}</span>
            {#if problem.finished_page}<span class="badge">finished page</span>{/if}
            {#if problem.status === "checked"}<span class="badge checked">checked</span>{/if}
            <span class="message">{problem.message}</span>
            {#if problem.read}<span class="muted">read: “{problem.read}”</span>{/if}
            {#if problem.kind === "typo" && problem.word}
              <button class="place" onclick={() => notATypo(problem.word)} disabled={busy} title="accept this word everywhere in the series">
                not a typo
              </button>
            {/if}
          </li>
        {/each}
      </ul>
    {/if}
  {/if}
</section>

<style>
  .problems ul {
    list-style: none;
    padding: 0;
  }
  .problems li {
    padding: 4px 0;
    border-bottom: 1px solid #e5e7eb;
  }
  .problems li.checked {
    color: #6b7280;
  }
  .kind {
    font-weight: 600;
    margin: 0 6px;
  }
  .badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    background: #9ca3af;
    color: #fff;
    font-size: 12px;
    margin-right: 6px;
  }
  .badge.checked {
    background: #16a34a;
  }
  .muted,
  .hint {
    color: #6b7280;
  }
  .ok {
    color: #16a34a;
  }
  .error {
    color: #dc2626;
  }
</style>
