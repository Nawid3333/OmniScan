<script lang="ts">
  import { getMemory, rebuildMemory, setRuleEnabled } from "./api";
  import type { LearnedRule, SeriesMemory } from "./api";
  import { describeRule, filterMemory, groupRules, replaceRule, ruleState } from "./learned";

  let { series }: { series: string } = $props();

  let memory = $state<SeriesMemory | null>(null);
  let error = $state("");
  let busy = $state(false);
  let query = $state("");

  let groups = $derived(memory ? groupRules(memory.rules) : []);
  let lines = $derived(memory ? filterMemory(memory.translations, query) : []);

  $effect(() => {
    const s = series;
    memory = null;
    error = "";
    query = "";
    void load(() => getMemory(s));
  });

  async function load(fetchMemory: () => Promise<SeriesMemory>): Promise<void> {
    busy = true;
    try {
      memory = await fetchMemory();
      error = "";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  async function toggle(rule: LearnedRule): Promise<void> {
    if (!memory) return;
    busy = true;
    try {
      const updated = await setRuleEnabled(series, rule.id, !rule.enabled);
      memory = { ...memory, rules: replaceRule(memory.rules, updated) };
      error = "";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }
</script>

<section class="learned">
  <p class="intro">
    What your hand edits in this series taught OmniScan. The next chapters use it: OCR fixes and deletions apply when
    the OCR runs, translation memory and preferred wording when the translation runs.
    <button onclick={() => load(() => rebuildMemory(series))} disabled={busy}>Rebuild from edits</button>
  </p>
  {#if error}
    <p class="error">{error}</p>
  {/if}
  {#if memory === null}
    {#if !error}<p>loading…</p>{/if}
  {:else}
    {#if !memory.enabled}
      <p class="warn">Learning is switched off for this series (series.toml: [learn] enabled = false).</p>
    {/if}
    <h3>Rules</h3>
    {#if groups.length === 0}
      <p>
        No rules yet. Correct OCR text, delete false detections, mark watermarks or sound effects, or rewrite a
        translation in the Studio — a rule starts acting once {memory.min_count} matching corrections show it.
      </p>
    {:else}
      <p class="hint">
        A rule acts after {memory.min_count} matching corrections (one for watermark and sound-effect labels). Switch off
        any rule that is wrong; it stays off when the memory is rebuilt.
      </p>
      {#each groups as group (group.kind)}
        <h4>{group.title}</h4>
        <table>
          <tbody>
            {#each group.rules as rule (rule.id)}
              {@const state = ruleState(rule, memory.min_count)}
              <tr class:off={!rule.active}>
                <td>{describeRule(rule)}</td>
                <td class="count">seen {rule.count}×</td>
                <td><span class="badge" class:on={state === "on"} class:disabled={state === "off"}>{state}</span></td>
                <td>
                  <button onclick={() => toggle(rule)} disabled={busy}>{rule.enabled ? "Switch off" : "Switch on"}</button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/each}
    {/if}
    <h3>Translation memory ({memory.translations.length} {memory.translations.length === 1 ? "line" : "lines"})</h3>
    {#if memory.translations.length === 0}
      <p>No lines yet: every English line you write or keep in the Studio is remembered for the next chapters.</p>
    {:else}
      <p class="hint">
        A line that comes up again is translated exactly this way; similar lines are shown to the translation model as
        examples of your style.
      </p>
      <input type="search" placeholder="search source or English" bind:value={query} />
      <table>
        <thead>
          <tr><th>Source</th><th>English</th><th>Chapter</th><th></th></tr>
        </thead>
        <tbody>
          {#each lines as line (line.source)}
            <tr>
              <td>{line.source}</td>
              <td>{line.english}</td>
              <td>{line.chapter}</td>
              <td class="count">{line.count > 1 ? `${line.count}×` : ""} {line.typed ? "typed" : "kept suggestion"}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  {/if}
</section>

<style>
  .learned table {
    border-collapse: collapse;
    margin-bottom: 12px;
  }
  .learned td,
  .learned th {
    padding: 4px 10px;
    border-bottom: 1px solid #e5e7eb;
    text-align: left;
    vertical-align: top;
  }
  .learned tr.off td:first-child {
    color: #6b7280;
  }
  .count {
    color: #6b7280;
    white-space: nowrap;
  }
  .badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    background: #9ca3af;
    color: #fff;
    font-size: 12px;
    white-space: nowrap;
  }
  .badge.on {
    background: #16a34a;
  }
  .badge.disabled {
    background: #dc2626;
  }
  .hint {
    color: #4b5563;
  }
  .warn {
    color: #b45309;
  }
  .error {
    color: #dc2626;
  }
</style>
