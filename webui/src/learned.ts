/** Pure helpers for the Learned view (no DOM, no Svelte). */

import type { LearnedRule, LearnKind, MemoryEntry } from "./api";

export const KIND_ORDER: LearnKind[] = ["ocr_fix", "preferred_term", "drop_text", "watermark_text", "sfx_text"];

export const KIND_TITLES: Record<LearnKind, string> = {
  ocr_fix: "OCR fixes",
  preferred_term: "Preferred wording",
  drop_text: "Always deleted",
  watermark_text: "Marked as watermark",
  sfx_text: "Marked as sound effect",
};

/** What a rule does to a later chapter, in words. */
export function describeRule(rule: LearnedRule): string {
  switch (rule.kind) {
    case "ocr_fix":
      return `read “${rule.wrong}” → write “${rule.right}”`;
    case "preferred_term":
      return `“${rule.wrong}” → “${rule.right}” in translations`;
    case "drop_text":
      return `delete regions reading “${rule.wrong}”`;
    case "watermark_text":
      return `regions reading “${rule.wrong}” are a watermark`;
    case "sfx_text":
      return `regions reading “${rule.wrong}” are a sound effect`;
  }
}

/** A rule's state: on (acting), off (switched off by hand) or waiting for `minCount` corrections. */
export function ruleState(rule: LearnedRule, minCount: number): string {
  if (!rule.enabled) return "off";
  if (rule.active) return "on";
  return `needs ${minCount - rule.count} more`;
}

/** The rules grouped by kind in KIND_ORDER, most seen first; kinds without rules are left out. */
export function groupRules(rules: LearnedRule[]): { kind: LearnKind; title: string; rules: LearnedRule[] }[] {
  return KIND_ORDER.map((kind) => ({
    kind,
    title: KIND_TITLES[kind],
    rules: rules.filter((rule) => rule.kind === kind).sort((a, b) => b.count - a.count),
  })).filter((group) => group.rules.length > 0);
}

/** A new list with that rule replaced by `updated` (matched by id). */
export function replaceRule(rules: LearnedRule[], updated: LearnedRule): LearnedRule[] {
  return rules.map((rule) => (rule.id === updated.id ? updated : rule));
}

/** The memory lines whose source or English contains `query` (case-insensitive); all of them for "". */
export function filterMemory(entries: MemoryEntry[], query: string): MemoryEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) return entries;
  return entries.filter((e) => e.source.toLowerCase().includes(q) || e.english.toLowerCase().includes(q));
}
