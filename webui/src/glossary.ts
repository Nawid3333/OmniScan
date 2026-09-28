/** Pure helpers for the Glossary view (no DOM, no Svelte). */

import type { GlossaryEntry, TermChange, TermType } from "./api";

export const TERM_TYPES: TermType[] = [
  "person",
  "place",
  "org",
  "skill",
  "item",
  "rank",
  "title",
  "honorific",
  "sfx",
  "other",
];

export type StatusFilter = "all" | GlossaryEntry["status"];

/** What each status does to a translation, as the view explains it. */
export const STATUS_HINTS: Record<GlossaryEntry["status"], string> = {
  locked: "every translation must use this English",
  proposed: "shown to the model as a suggestion",
  rejected: "never shown to the model, never proposed again",
};

/** The entries with `status` (all for "all") whose source, English, aliases or notes contain `query`. */
export function filterTerms(entries: GlossaryEntry[], status: StatusFilter, query: string): GlossaryEntry[] {
  const q = query.trim().toLowerCase();
  return entries.filter(
    (e) =>
      (status === "all" || e.status === status) &&
      (!q ||
        [e.source, e.target, e.notes ?? "", ...e.aliases].some((text) => text.toLowerCase().includes(q))),
  );
}

/** How many entries have each status, and all of them. */
export function statusCounts(entries: GlossaryEntry[]): Record<StatusFilter, number> {
  const counts: Record<StatusFilter, number> = { all: entries.length, locked: 0, proposed: 0, rejected: 0 };
  for (const e of entries) counts[e.status] += 1;
  return counts;
}

/** The spellings in a comma-separated field (Latin, CJK and Japanese commas), trimmed, empty ones dropped. */
export function parseAliases(text: string): string[] {
  return text
    .split(/[,，、]/)
    .map((alias) => alias.trim())
    .filter((alias) => alias !== "");
}

/** A row's editable fields as the form holds them. */
export interface TermDraft {
  source: string;
  target: string;
  type: TermType;
  notes: string;
  aliases: string;
}

/** The form fields of an entry. */
export function draftOf(entry: GlossaryEntry): TermDraft {
  return {
    source: entry.source,
    target: entry.target,
    type: entry.type as TermType,
    notes: entry.notes ?? "",
    aliases: entry.aliases.join(", "),
  };
}

/** The PATCH body for what the draft changes in `entry` ({} when nothing does). */
export function termChange(entry: GlossaryEntry, draft: TermDraft): TermChange {
  const change: TermChange = {};
  if (draft.source.trim() !== entry.source) change.source = draft.source;
  if (draft.target.trim() !== entry.target) change.target = draft.target;
  if (draft.type !== entry.type) change.type = draft.type;
  if (draft.notes.trim() !== (entry.notes ?? "")) change.notes = draft.notes;
  const aliases = parseAliases(draft.aliases);
  if (aliases.join("\n") !== entry.aliases.join("\n")) change.aliases = aliases;
  return change;
}

/** A new list with that entry replaced by `updated` (matched by id). */
export function replaceTerm(entries: GlossaryEntry[], updated: GlossaryEntry): GlossaryEntry[] {
  return entries.map((e) => (e.id === updated.id ? updated : e));
}
