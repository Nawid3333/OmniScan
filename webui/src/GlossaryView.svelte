<script lang="ts">
  import { addTerm, changeTerm, getGlossary, removeTerm } from "./api";
  import type { GlossaryEntry, TermChange, TermType } from "./api";
  import {
    STATUS_HINTS,
    TERM_TYPES,
    draftOf,
    filterTerms,
    parseAliases,
    replaceTerm,
    statusCounts,
    termChange,
  } from "./glossary";
  import type { StatusFilter, TermDraft } from "./glossary";

  let { series }: { series: string } = $props();

  let entries = $state<GlossaryEntry[] | null>(null);
  let error = $state("");
  let busy = $state(false);
  let status = $state<StatusFilter>("all");
  let query = $state("");
  let editing = $state<number | null>(null);
  let draft = $state<TermDraft>({ source: "", target: "", type: "other", notes: "", aliases: "" });
  let fresh = $state<TermDraft>({ source: "", target: "", type: "other", notes: "", aliases: "" });
  let lockNew = $state(true);

  let shown = $derived(entries ? filterTerms(entries, status, query) : []);
  let counts = $derived(statusCounts(entries ?? []));

  $effect(() => {
    const s = series;
    entries = null;
    error = "";
    editing = null;
    void run(async () => {
      entries = await getGlossary(s);
    });
  });

  async function run(action: () => Promise<void>): Promise<void> {
    busy = true;
    try {
      await action();
      error = "";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  function add(): Promise<void> {
    return run(async () => {
      const added = await addTerm(series, {
        source: fresh.source,
        target: fresh.target,
        type: fresh.type,
        status: lockNew ? "locked" : "proposed",
        notes: fresh.notes.trim() || null,
        aliases: parseAliases(fresh.aliases),
      });
      entries = [...(entries ?? []), added];
      fresh = { source: "", target: "", type: fresh.type, notes: "", aliases: "" };
    });
  }

  function change(entry: GlossaryEntry, body: TermChange): Promise<void> {
    return run(async () => {
      if (entry.id === null) return;
      const updated = await changeTerm(series, entry.id, body);
      entries = replaceTerm(entries ?? [], updated);
      editing = null;
    });
  }

  function remove(entry: GlossaryEntry): Promise<void> {
    return run(async () => {
      if (entry.id === null) return;
      await removeTerm(series, entry.id);
      entries = (entries ?? []).filter((e) => e.id !== entry.id);
    });
  }

  function edit(entry: GlossaryEntry): void {
    editing = entry.id;
    draft = draftOf(entry);
  }

  function save(entry: GlossaryEntry): Promise<void> {
    const body = termChange(entry, draft);
    if (Object.keys(body).length === 0) {
      editing = null;
      return Promise.resolve();
    }
    return change(entry, body);
  }
</script>

<section class="glossary">
  <p class="intro">
    Names and terms this series must translate the same way every time. A locked term's English is binding for the
    translation and the judge; a proposed one is a suggestion. The next translation redoes just the lines holding a
    term you change.
  </p>
  {#if error}
    <p class="error">{error}</p>
  {/if}
  <form
    class="add"
    onsubmit={(e) => {
      e.preventDefault();
      void add();
    }}
  >
    <input placeholder="source (as the raw writes it)" bind:value={fresh.source} required />
    <input placeholder="English" bind:value={fresh.target} required />
    <select bind:value={fresh.type} aria-label="type of the new term">
      {#each TERM_TYPES as type (type)}
        <option value={type}>{type}</option>
      {/each}
    </select>
    <input placeholder="other spellings, comma-separated" bind:value={fresh.aliases} />
    <input placeholder="notes" bind:value={fresh.notes} />
    <label><input type="checkbox" bind:checked={lockNew} /> locked</label>
    <button type="submit" disabled={busy}>Add term</button>
  </form>
  {#if entries === null}
    {#if !error}<p>loading…</p>{/if}
  {:else}
    <p>
      <select bind:value={status} aria-label="status filter">
        <option value="all">all ({counts.all})</option>
        <option value="locked">locked ({counts.locked})</option>
        <option value="proposed">proposed ({counts.proposed})</option>
        <option value="rejected">rejected ({counts.rejected})</option>
      </select>
      <input type="search" placeholder="search" bind:value={query} />
    </p>
    {#if entries.length === 0}
      <p>
        No terms yet. Add one above, or let OmniScan propose some: <code>omniscan glossary propose</code> (from the
        series' own text) or <code>omniscan reference</code> (from official English chapters).
      </p>
    {:else}
      <table>
        <thead>
          <tr><th>Source</th><th>English</th><th>Type</th><th>Other spellings</th><th>Notes</th><th>Status</th><th></th></tr>
        </thead>
        <tbody>
          {#each shown as entry (entry.id)}
            <tr class:rejected={entry.status === "rejected"}>
              {#if editing === entry.id}
                <td><input bind:value={draft.source} aria-label="source" /></td>
                <td><input bind:value={draft.target} aria-label="English" /></td>
                <td>
                  <select bind:value={draft.type} aria-label="type">
                    {#each TERM_TYPES as type (type)}
                      <option value={type}>{type}</option>
                    {/each}
                  </select>
                </td>
                <td><input bind:value={draft.aliases} aria-label="other spellings" /></td>
                <td><input bind:value={draft.notes} aria-label="notes" /></td>
                <td><span class="badge {entry.status}">{entry.status}</span></td>
                <td class="actions">
                  <button onclick={() => save(entry)} disabled={busy}>Save</button>
                  <button onclick={() => (editing = null)} disabled={busy}>Cancel</button>
                </td>
              {:else}
                <td>{entry.source}</td>
                <td>{entry.target}</td>
                <td>{entry.type}</td>
                <td>{entry.aliases.join(", ")}</td>
                <td class="notes">{entry.notes ?? ""}</td>
                <td>
                  <span class="badge {entry.status}" title={STATUS_HINTS[entry.status]}>{entry.status}</span>
                  {#if entry.origin === "user"}<span class="origin">yours</span>{/if}
                </td>
                <td class="actions">
                  <button onclick={() => edit(entry)} disabled={busy}>Edit</button>
                  {#if entry.status !== "locked"}
                    <button onclick={() => change(entry, { status: "locked" })} disabled={busy}>Lock</button>
                  {/if}
                  {#if entry.status !== "rejected"}
                    <button onclick={() => change(entry, { status: "rejected" })} disabled={busy}>Reject</button>
                  {:else}
                    <button onclick={() => change(entry, { status: "proposed" })} disabled={busy}>Propose</button>
                  {/if}
                  <button onclick={() => remove(entry)} disabled={busy}>Remove</button>
                </td>
              {/if}
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  {/if}
</section>

<style>
  .glossary table {
    border-collapse: collapse;
  }
  .glossary td,
  .glossary th {
    padding: 4px 10px;
    border-bottom: 1px solid #e5e7eb;
    text-align: left;
    vertical-align: top;
  }
  .glossary tr.rejected td {
    color: #6b7280;
  }
  .add {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    margin-bottom: 12px;
  }
  .notes {
    color: #4b5563;
  }
  .actions {
    white-space: nowrap;
  }
  .badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 999px;
    background: #9ca3af;
    color: #fff;
    font-size: 12px;
  }
  .badge.locked {
    background: #16a34a;
  }
  .badge.rejected {
    background: #dc2626;
  }
  .origin {
    color: #6b7280;
    font-size: 12px;
    margin-left: 4px;
  }
  .error {
    color: #dc2626;
  }
</style>
