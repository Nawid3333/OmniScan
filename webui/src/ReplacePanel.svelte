<script lang="ts">
  import { replaceText } from "./api";
  import type { ReplaceChange, ReplaceRequest } from "./api";
  import { changeSummary, replaceRequest, sameRule } from "./replace";
  import type { ReplaceForm } from "./replace";

  let { series, chapter, onApplied }: { series: string; chapter: string; onApplied: () => Promise<void> } = $props();

  let form = $state<ReplaceForm>({
    find: "",
    replace: "",
    target: "english",
    scope: "chapter",
    wholeWord: false,
    matchCase: true,
    regex: false,
  });
  let previewed = $state<ReplaceRequest | null>(null);
  let changes = $state<ReplaceChange[]>([]);
  let working = $state(false);
  let message = $state("");
  let failure = $state("");

  const current = $derived(replaceRequest(form, chapter, false));
  const ready = $derived(previewed !== null && sameRule(previewed, current) && changes.length > 0);

  async function preview(): Promise<void> {
    await run(async () => {
      const request = replaceRequest(form, chapter, true);
      changes = (await replaceText(series, request)).changes;
      previewed = request;
      message = changes.length === 0 ? "nothing matches" : `would change ${changeSummary(changes, form.target)}`;
    });
  }

  async function apply(): Promise<void> {
    await run(async () => {
      const done = (await replaceText(series, current)).changes;
      message = `changed ${changeSummary(done, form.target)} (Undo takes back this chapter's part)`;
      changes = [];
      previewed = null;
      await onApplied();
    });
  }

  async function run(operation: () => Promise<void>): Promise<void> {
    working = true;
    failure = "";
    try {
      await operation();
    } catch (e) {
      failure = e instanceof Error ? e.message : String(e);
    } finally {
      working = false;
    }
  }
</script>

<div class="replace">
  <input placeholder="find" bind:value={form.find} onkeydown={(e) => e.key === "Enter" && void preview()} />
  <input placeholder="replace with" bind:value={form.replace} onkeydown={(e) => e.key === "Enter" && void preview()} />
  <select bind:value={form.target} title="which texts to search">
    <option value="english">English lines</option>
    <option value="source">source texts</option>
  </select>
  <select bind:value={form.scope} title="where to search">
    <option value="chapter">this chapter</option>
    <option value="series">whole series</option>
  </select>
  <label title="whole words only"><input type="checkbox" bind:checked={form.wholeWord} /> word</label>
  <label title="off: any case, and the replacement takes each match's case (JINWOO → JIN-WOO)"><input type="checkbox" bind:checked={form.matchCase} /> case</label>
  <label title="find is a regular expression; \1 … in the replacement are its groups"><input type="checkbox" bind:checked={form.regex} /> regex</label>
  <button onclick={() => void preview()} disabled={working || form.find === ""}>Preview</button>
  <button onclick={() => void apply()} disabled={working || !ready} title="preview first; every change becomes a hand edit">Replace all</button>
  {#if message}<span class="muted">{message}</span>{/if}
  {#if failure}<span class="error">{failure}</span>{/if}
</div>
{#if changes.length > 0}
  <ol class="changes">
    {#each changes as change (`${change.chapter}/${change.region_id}`)}
      <li>
        <span class="muted">{change.chapter} · {change.region_id}</span>
        <del>{change.before}</del> → <ins>{change.after}</ins>
      </li>
    {/each}
  </ol>
{/if}

<style>
  .replace {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin: 0 0 8px;
    padding: 6px;
    background: #f3f4f6;
    border: 1px solid #d1d5db;
  }
  .changes {
    max-height: 180px;
    overflow: auto;
    margin: 0 0 8px;
    font-size: 14px;
  }
  del {
    color: #b91c1c;
  }
  ins {
    color: #047857;
    text-decoration: none;
  }
  .muted {
    color: #6b7280;
  }
  .error {
    color: #b91c1c;
  }
</style>
